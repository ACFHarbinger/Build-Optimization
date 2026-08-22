"""
Local screenshot OCR transcription for the STS2 reward advisor (SA8).

Consumes a screenshot (disk or raw bytes) plus a ``recognition`` Hydra config,
locates the configured crop boxes (16:9 and 16:10 variants; no detection model
this slice), crops each offer's name-banner sub-region, OCRs *only that crop*,
and returns ``core.recognition.RecognizedName`` records for the ``{offer1,
offer2, offer3}`` regions.

Design rules inherited from the roadmap/bus:
  - card names are art-integrated, so **whole-image OCR will not work** — the
    recognizer must crop before OCR. Configured crop boxes are the single
    source of layout truth shared with the fixture generator.
  - never the raw screenshot to a cloud model in this phase (SA11); this layer
    is local-only.
  - real screenshots / game assets are never committed; only deterministic
    synthetic fixtures (.generator output) are.

This layer imports Pillow + pytesseract. It degrades with a clear
``ImportError`` when they are absent, and callers/tests skip the OCR-dependent
portion rather than failing (mirrors core.native_backend's graceful-skip
precedent). The pure matching/confidence seam (``core.recognition``) has no
such dependency.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from core.recognition import (
    ConfidencePolicy,
    NameMatcher,
    RecognizedName,
    resolve_name,
)

try:
    from PIL import Image, ImageOps

    _PIL_AVAILABLE = True
except ImportError:  # pragma: no cover - guarded by skip at the call site
    _PIL_AVAILABLE = False

try:
    import pytesseract

    _TESSERACT_AVAILABLE = True
except ImportError:  # pragma: no cover - guarded by skip at the call site
    _TESSERACT_AVAILABLE = False


def ocr_available() -> bool:
    """True when Pillow + pytesseract import *and* the tesseract binary runs.

    Image.transcribe and the OCR tests depend on the actual tesseract binary
    (pytesseract is just a wrapper); when the binary is missing we degrade to a
    clean skip rather than failing, mirroring core.native_backend's precedent.
    """
    if not (_PIL_AVAILABLE and _TESSERACT_AVAILABLE):
        return False
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Crop-box geometry
# ---------------------------------------------------------------------------


def _box4(box: Sequence) -> Tuple[float, float, float, float]:  # noqa: ANN001 - unstructured config box
    """Cast any 4-item box (list/tuple) to a fixed 4-float tuple.

    Raises ValueError if the box doesn't have exactly four elements, since a
    2-D crop needs (left, top, right, bottom).
    """
    values = tuple(float(v) for v in box)
    if len(values) != 4:
        raise ValueError(f"crop box must have 4 values (l,t,r,b), got {values!r}")
    return values  # type: ignore[return-value]


@dataclass(frozen=True)
class CropBoxes:
    """Relative crop rectangles for the reward offer regions and name banner.

    ``offers`` are fractions of the screenshot width/height and locate the
    three card-shaped offer regions. ``name_banner`` is a *sub-region* given as
    fractions of an individual offer box, locating the high-contrast name strip
    inside each card. Both are resolution-independent, so a single definition
    works at any size and across the 16:9 / 16:10 variants.
    """

    offers: List[Tuple[float, float, float, float]]
    name_banner: Tuple[float, float, float, float] = (0.10, 0.82, 0.90, 0.95)
    deck_grid: Optional[Tuple[float, float, float, float]] = None

    def to_absolute(self, width: int, height: int) -> List[Tuple[int, int, int, int]]:
        """Convert the relative offer boxes to absolute pixel (l, t, r, b)."""
        boxes: List[Tuple[int, int, int, int]] = []
        for left, top, right, bottom in self.offers:
            x0 = max(0, min(width, int(left * width)))
            y0 = max(0, min(height, int(top * height)))
            x1 = max(0, min(width, int(right * width)))
            y1 = max(0, min(height, int(bottom * height)))
            if x1 > x0 and y1 > y0:
                boxes.append((x0, y0, x1, y1))
        return boxes

    def banner_boxes(self, width: int, height: int) -> List[Tuple[int, int, int, int]]:
        """The name-banner crop for each offer, in absolute pixels.

        For each offer box, the banner sub-box is computed relative to that
        offer's own width/height (so a single banner definition applies to all
        three cards regardless of aspect).
        """
        bl, bt, br, bb = self.name_banner
        result: List[Tuple[int, int, int, int]] = []
        for left, top, right, bottom in self.to_absolute(width, height):
            bw = right - left
            bh = bottom - top
            x0 = left + int(bl * bw)
            y0 = top + int(bt * bh)
            x1 = left + int(br * bw)
            y1 = top + int(bb * bh)
            if x1 > x0 and y1 > y0:
                result.append((x0, y0, x1, y1))
        return result

    @classmethod
    def from_config(cls, config: Dict):  # noqa: ANN001 - config is unstructured dict
        """Build CropBoxes from a Hydra/OmegaConf dict.

        Accepts a normalized ``offers`` list of four numbers (l, t, r, b) and an
        optional ``name_banner`` box.
        """
        offers = [_box4(box) for box in config.get("offers", [])]
        banner_cfg = config.get("name_banner")
        banner = _box4(banner_cfg) if banner_cfg else (0.10, 0.82, 0.90, 0.95)
        deck = config.get("deck_grid")
        deck_box = _box4(deck) if deck else None
        return cls(offers=offers, name_banner=banner, deck_grid=deck_box)


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------


def load_image(source: Path) -> "Optional[Image.Image]":
    """Load a screenshot from disk into a Pillow RGB image (grayscale-able).

    Returns ``None`` if Pillow isn't importable so callers can degrade.
    """
    if not _PIL_AVAILABLE:
        return None
    with Image.open(source) as img:
        return ImageOps.exif_transpose(img).convert("RGB")


def load_image_bytes(data: bytes) -> "Optional[Image.Image]":
    """Load a screenshot from raw bytes (e.g. clipboard PNG) into an image."""
    if not _PIL_AVAILABLE:
        return None
    with Image.open(io.BytesIO(data)) as img:
        return ImageOps.exif_transpose(img).convert("RGB")


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------


class OcrTranscriber:
    """Crops the configured offer regions and OCRs each name-banner crop.

    Args:
        boxes: ``CropBoxes`` for the reward (and optionally deck-grid) screen.
        matcher: ``NameMatcher`` resolving transcribed text to catalogue cards.
        accept_threshold: Per-user confidence threshold (SA10 policy).
        psm: tesseract page-segmentation mode — single-line (7) suits a
             name-banner strip.
    """

    def __init__(
        self,
        boxes: CropBoxes,
        matcher: NameMatcher,
        accept_threshold: float,
        psm: int = 7,
    ) -> None:
        if not ocr_available():  # pragma: no cover - guarded at the call site
            raise RuntimeError("OcrTranscriber requires Pillow and pytesseract")
        self.boxes = boxes
        self.matcher = matcher
        self.accept_threshold = accept_threshold
        self.psm = psm

    def transcribe(self, image: "Image.Image") -> List[Tuple[RecognizedName, ConfidencePolicy]]:
        """OCR the offer-name crops and resolve each to a RecognizedName.

        Args:
            image: A loaded screenshot (RGB).

        Returns:
            List of (RecognizedName, ConfidencePolicy) — one per configured
            offer region, in region order. Regions without usable content are
            still returned (with ``needs_dataset_entry=True`` via the matcher
            resolving empty → None) so the caller can show all three slots.
        """
        width, height = image.size
        results: List[Tuple[RecognizedName, ConfidencePolicy]] = []
        banner_boxes = self.boxes.banner_boxes(width, height)
        for idx, (left, top, right, bottom) in enumerate(banner_boxes, 1):
            region_id = f"offer{idx}"
            # Crop the offer region first, then the name-banner sub-region so
            # we OCR ONLY the high-contrast name strip, never the card art.
            crop = image.crop((left, top, right, bottom))
            raw_text = self._ocr(crop)
            # Down-weight empty/noisy text so it never passes a threshold.
            confidence = self._post_ocr_confidence(raw_text)
            name, policy = resolve_name(
                self.matcher, region_id, raw_text, confidence, self.accept_threshold
            )
            results.append((name, policy))
        return results

    def _ocr(self, crop: "Image.Image") -> str:
        """OCR a single crop using tesseract; returns a cleaned single line."""
        # UPSCALE + grayscale helps tesseract on small banner text.
        width, height = crop.size
        if width * height == 0:
            return ""
        scale = max(2.0, 240.0 / max(width, 1))
        enlarged = crop.resize((int(width * scale), int(height * scale)))
        gray = enlarged.convert("L")
        try:
            text = pytesseract.image_to_string(gray, config=f"--psm {self.psm}")
        except pytesseract.TesseractNotFoundError:  # pragma: no cover - extern
            return ""
        return text.replace("\n", " ").strip()

    def _post_ocr_confidence(self, raw_text: str) -> float:
        """Heuristic [0,1] confidence from OCR output length/quality.

        Not a substitute for a real confidence signal from the OCR engine; a
        deliberately simple proxy for this slice: non-empty single line = high,
        empty = zero, run-on=lower. Tuned so a *clean* fixture transcription
        passes the default accept threshold and junk does not.
        """
        text = raw_text.strip()
        if not text:
            return 0.0
        words = text.split()
        if len(words) == 1 and len(text) <= 24:
            return 0.98
        if len(words) <= 2 and len(text) <= 32:
            return 0.90
        return 0.75

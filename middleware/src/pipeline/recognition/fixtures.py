"""
Deterministic synthetic reward-screen fixture generator (SA8).

Builds small mocked screenshots that exercise the crop-and-OCR pipeline shape
withOUT touching real STS2 assets (MegaCrit/Kobold Studios IP — never committed
into this public AGPL repo). Each fixture renders three card-shaped regions
with a plain background, abstract "art" blocks so naive whole-image OCR can't
read the name, and a distinct name-banner strip at a fixed relative position
holding the card name in high-contrast text.

The banner crop-box (the same fractions the recognizer uses) and this generator
share one layout source of truth — change the layout once and both stay in
sync. The generator is seeded, so the same seed ⇒ byte-identical PNGs (the
kind of thing that must be reproducible for the fixtures test).

Only Pillow is required to generate; the generated PNGs are repository-owned.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING, List, Sequence, Tuple, cast

if TYPE_CHECKING:
    from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Layout — the single source of truth
# ---------------------------------------------------------------------------


class RewardLayout:
    """Relative geometry + palette for a synthetic three-offer reward screen.

    All positions are fractions of width/height. The three offer boxes and the
    name-banner sub-region inside each one are defined here so the fixture
    generator and the recognizer's ``CropBoxes`` agree.
    """

    OFFERS: Tuple[Tuple[float, float, float, float], ...] = (
        (0.06, 0.18, 0.31, 0.95),  # offer 1 (l, t, r, b)
        (0.36, 0.18, 0.61, 0.95),  # offer 2
        (0.66, 0.18, 0.91, 0.95),  # offer 3
    )
    # Name-banner strip inside each card: lower third, high-contrast band.
    BANNER: Tuple[float, float, float, float] = (0.10, 0.82, 0.90, 0.95)

    BACKGROUND: Tuple[int, int, int] = (38, 40, 48)
    CARD_BG: Tuple[int, int, int] = (58, 60, 70)
    BANNER_BG: Tuple[int, int, int] = (222, 224, 230)
    BANNER_FG: Tuple[int, int, int] = (18, 18, 22)
    FRAME: Tuple[Tuple[int, int, int], ...] = (
        (120, 120, 128),  # common frame
        (70, 130, 180),   # uncommon frame
        (190, 150, 40),   # rare frame
    )

    @classmethod
    def crop_boxes(cls) -> "RewardLayout":
        return cls()

    def offer_boxes(self) -> List[Tuple[int, int, int, int]]:
        """Offer boxes as absolute pixels — used to draw (not OCR)."""
        return [_frac_to_abs(b, 1, 1) for b in self.OFFERS]


def _frac_to_abs(
    box: Tuple[float, float, float, float], width: int, height: int
) -> Tuple[int, int, int, int]:
    left, top, right, bottom = box
    return (
        int(left * width),
        int(top * height),
        int(right * width),
        int(bottom * height),
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_reward_screen(
    names: Sequence[str],
    width: int = 1280,
    height: int = 720,
    seed: int = 42,
    degrade: bool = False,
) -> "Image.Image":
    """Render a deterministic synthetic three-offer reward screen.

    Args:
        names: Three card names to write in the banner strips (must be len 3).
        width/height: Screenshot size. A 16:9 vs 16:10 variant is just a
            different height here; box fractions are resolution-independent.
        seed: Seeded RNG so the abstract art for each card is reproducible.
        degrade: When True, blur the banners simulating a low-quality
            screenshot (the "low-confidence" fixture variant).

    Returns:
        A Pillow RGB image.
    """
    from PIL import Image, ImageDraw, ImageFilter  # runtime dep for rendering

    if len(names) != 3:
        raise ValueError("render_reward_screen requires exactly 3 names")

    rng = random.Random(seed)
    image = Image.new("RGB", (width, height), RewardLayout.BACKGROUND)
    draw = ImageDraw.Draw(image)

    for idx, (box, name) in enumerate(zip(RewardLayout.OFFERS, names)):
        left, top, right, bottom = _frac_to_abs(box, width, height)
        frame = RewardLayout.FRAME[idx % len(RewardLayout.FRAME)]
        draw.rounded_rectangle((left, top, right, bottom), radius=14, fill=RewardLayout.CARD_BG, outline=frame, width=4)

        # Abstract "art" block — gradient-ish fill so naive whole-image OCR
        # (which would read a jumble) can't read the name from here.
        art_left, art_top, art_right, art_bottom = left + 12, top + 12, right - 12, int(height * 0.62)
        draw.rectangle((art_left, art_top, art_right, art_bottom), fill=tuple(
            (c + rng.randint(-14, 14)) % 256 for c in (70, 72, 92)
        ))
        for _ in range(8):  # noise streaks
            sx0 = art_left + rng.randint(0, art_right - art_left)
            sy0 = art_top + rng.randint(0, art_bottom - art_top)
            draw.line((sx0, sy0, sx0 + rng.randint(20, 120), sy0 + rng.randint(-30, 30)), fill=tuple(
                rng.randint(90, 180) for _ in range(3)
            ), width=2)

        # Name-banner strip, high contrast, at the SAME fractions the
        # recognizer crops.
        b_left, b_top, b_right, b_bottom = _frac_to_abs(RewardLayout.BANNER, right - left, bottom - top)
        banner = (left + b_left, top + b_top, left + b_right, top + b_bottom)
        draw.rectangle(banner, fill=RewardLayout.BANNER_BG)

        font = _load_font(draw, name, banner)
        text = name
        if degrade:
            text = _degraded_text(name, rng)
        bbox = draw.textbbox((0, 0), text, font=font)
        tx = banner[0] + (banner[2] - banner[0] - (bbox[2] - bbox[0])) // 2
        ty = banner[1] + (banner[3] - banner[1] - (bbox[3] - bbox[1])) // 2 - 4
        draw.text((tx, ty), text, fill=RewardLayout.BANNER_FG, font=font)

    if degrade:
        image = image.filter(ImageFilter.GaussianBlur(radius=2))

    return image


def _load_font(draw: "ImageDraw.ImageDraw", text: str, banner: Tuple[int, int, int, int]) -> "ImageFont.ImageFont":
    """Pick a scalable mono font sized to fit the banner, else the default."""
    import glob

    from PIL import ImageFont
    candidates = sorted(glob.glob("/usr/share/fonts/**/*.ttf", recursive=True)) + sorted(
        glob.glob("/usr/share/fonts/**/*.ttf", recursive=True)
    )
    target_h = banner[3] - banner[1]
    size = max(16, int(target_h * 0.5))
    if candidates:
        # prefer a clear sans/mono font
        pref = [c for c in candidates if "Mono" in c or "Liberation" in c or "DejaVu" in c]
        path = (pref or candidates)[0]
        try:
            return cast("ImageFont.ImageFont", ImageFont.truetype(path, size))
        except Exception:
            pass
    return cast("ImageFont.ImageFont", ImageFont.load_default())


def _degraded_text(name: str, rng: random.Random) -> str:
    """Cheap corruption for the 'low-confidence' fixture variant."""
    chars = []
    for ch in name:
        if ch.isspace():
            chars.append(ch)
        elif rng.random() < 0.12:
            chars.append(rng.choice("lI|#@8O0"))
        else:
            chars.append(ch)
    return "".join(chars)


def generate_fixtures(
    output_dir: Path,
    width: int = 1280,
    height: int = 720,
    seed: int = 42,
) -> dict:
    """Render and write the committed fixture set to ``output_dir``.

    Writes:
        reward_clean.png        — three clean, high-confidence names.
        reward_degraded.png     — a blurred/occluded variant (low-confidence).
        reward_unknown.png      — a name that is NOT in the catalogue (drives
                                  the ``needs_dataset_entry`` blocking branch).

    Returns:
        A small manifest dict of filenames → names used.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    clean_names = ["Carnage", "Inflame", "Cleave"]
    clean = render_reward_screen(clean_names, width=width, height=height, seed=seed)
    clean.save(output_dir / "reward_clean.png")

    degraded = render_reward_screen(clean_names, width=width, height=height, seed=seed, degrade=True)
    degraded.save(output_dir / "reward_degraded.png")

    unknown_names = ["Carnage", "Qliphoth", "Cleave"]
    unknown = render_reward_screen(unknown_names, width=width, height=height, seed=seed)
    unknown.save(output_dir / "reward_unknown.png")

    return {
        "reward_clean.png": clean_names,
        "reward_degraded.png": clean_names,
        "reward_unknown.png": unknown_names,
    }


if __name__ == "__main__":  # pragma: no cover - ad-hoc generation via CLI
    import sys

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("middleware/tests/fixtures")
    manifest = generate_fixtures(out)
    print(f"Wrote fixtures to {out}:")
    for name in manifest:
        print(f"  - {name}")

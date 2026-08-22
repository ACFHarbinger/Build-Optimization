"""
Pipeline: STS2 screenshot recognition (SA8/SA10).

Local screenshot → crop-box OCR → ``core.recognition.RecognizedName`` records.
This is the recognition seam the advisor consumes; it never produces a card
silently (see ``core.recognition``), and never uploads the raw image (SA11).

Only the OCR transcription path depends on Pillow/pytesseract; the pure
matching/confidence contract in ``core.recognition`` does not.
"""

from .fixtures import RewardLayout, generate_fixtures, render_reward_screen
from .transcribe import CropBoxes, OcrTranscriber, load_image, load_image_bytes, ocr_available

__all__ = [
    "RewardLayout",
    "generate_fixtures",
    "render_reward_screen",
    "CropBoxes",
    "OcrTranscriber",
    "load_image",
    "load_image_bytes",
    "ocr_available",
]

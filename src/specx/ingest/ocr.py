"""Bonus: OCR for image-only pages (off by default; the core pipeline is text-only).

Requires the Tesseract binary (``brew install tesseract`` / ``apt install tesseract-ocr``).
Orientation is detected first because scanned spec charts are often rotated 90°.
"""

from __future__ import annotations

import io
import logging
import re
import shutil

import pymupdf

log = logging.getLogger(__name__)


def ocr_available() -> bool:
    return shutil.which("tesseract") is not None


def ocr_page(page: pymupdf.Page, dpi: int = 300) -> str:
    if not ocr_available():
        log.warning("OCR requested but tesseract is not installed; skipping page %s", page.number + 1)
        return ""
    import pytesseract
    from PIL import Image

    pix = page.get_pixmap(dpi=dpi)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    try:
        osd = pytesseract.image_to_osd(img)
        angle = int(re.search(r"Rotate: (\d+)", osd).group(1))
        if angle:
            img = img.rotate(-angle, expand=True)
    except Exception:  # OSD fails on sparse pages; OCR as-is
        pass
    return pytesseract.image_to_string(img, config="--psm 6")

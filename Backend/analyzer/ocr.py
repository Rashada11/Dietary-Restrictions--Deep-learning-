"""Label OCR with OpenCV preprocessing and EasyOCR.

EasyOCR is installed with the Python dependencies, so no separate desktop OCR
application or PATH configuration is required.
"""
import cv2
import easyocr
import numpy as np


_reader = None


def _get_reader():
    """Create the OCR reader only when an image is first analysed."""
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _reader


def extract_text(image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError("The uploaded file is not a readable image.")
    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, 12, 7, 21)
    processed = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    results = _get_reader().readtext(processed, detail=0, paragraph=True)
    return " ".join(" ".join(results).split())

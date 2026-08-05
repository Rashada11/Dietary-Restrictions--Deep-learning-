"""Shared, dependency-light NLP utilities for API/model integration."""
import re


def normalize_ingredient_text(text):
    """Normalize OCR text while preserving meaningful ingredient words."""
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9,;()\-\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

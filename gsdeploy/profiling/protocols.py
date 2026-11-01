"""Compatibility imports for the versioned OCR preparation protocol."""

from ..tasks import (
    OCR_PROTOCOL, page_ocr_text, parse_sroie_annotation, restore_sroie_reference,
)

__all__ = ["OCR_PROTOCOL", "page_ocr_text", "parse_sroie_annotation", "restore_sroie_reference"]

"""PDF pages -> PIL images (PyMuPDF; no Poppler needed on Windows)."""
from __future__ import annotations

from PIL import Image


def pdf_to_images(data: bytes, dpi: int = 200, max_pages: int | None = None) -> list[Image.Image]:
    try:
        import pymupdf as fitz  # PyMuPDF >= 1.24
    except ImportError:
        import fitz  # older PyMuPDF

    images: list[Image.Image] = []
    with fitz.open(stream=data, filetype="pdf") as doc:
        if doc.needs_pass:
            raise ValueError("PDF is password-protected")
        zoom = dpi / 72.0
        for i, page in enumerate(doc):
            if max_pages and i >= max_pages:
                break
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            images.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
    return images

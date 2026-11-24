"""Upload decoding and optional preprocessing."""
from __future__ import annotations

import io
from typing import Any

from PIL import Image, ImageFilter, ImageOps, UnidentifiedImageError

from utils.file_utils import extension
from utils.pdf_utils import pdf_to_images

IMAGE_EXTS = {"jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff", "gif"}


class UnsupportedFileError(ValueError):
    pass


def _to_rgb(img: Image.Image) -> Image.Image:
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[-1])
        return bg
    return img.convert("RGB") if img.mode != "RGB" else img


def decode_upload(name: str, data: bytes, pdf_dpi: int = 200, gif_all_frames: bool = False) -> list[tuple[str, int, Image.Image]]:
    """Return [(file name, page number, RGB image)] for one uploaded file."""
    ext = extension(name)
    if ext == "pdf":
        try:
            pages = pdf_to_images(data, dpi=pdf_dpi)
        except Exception as exc:
            raise UnsupportedFileError(f"{name}: cannot read PDF ({type(exc).__name__})") from exc
        if not pages:
            raise UnsupportedFileError(f"{name}: PDF has no pages")
        return [(name, i, p) for i, p in enumerate(pages, 1)]
    if ext not in IMAGE_EXTS:
        raise UnsupportedFileError(f"{name}: unsupported file type '.{ext}'")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise UnsupportedFileError(f"{name}: corrupted or unreadable image") from exc
    n_frames = getattr(img, "n_frames", 1)
    multi = ext in ("tif", "tiff") or (ext == "gif" and gif_all_frames)
    out = []
    for i in range(n_frames if multi else 1):
        img.seek(i)
        frame = ImageOps.exif_transpose(img.copy()) if i == 0 else img.copy()
        out.append((name, i + 1, _to_rgb(frame)))
    return out


def deskew(img: Image.Image, max_angle: float = 5.0, step: float = 0.5) -> Image.Image:
    """Small-angle deskew by maximising horizontal projection variance."""
    gray = ImageOps.grayscale(img)
    small = gray.copy()
    small.thumbnail((800, 800))
    inv = ImageOps.invert(small)
    best_angle, best_score = 0.0, -1.0
    angle = -max_angle
    while angle <= max_angle:
        rot = inv.rotate(angle, expand=False, fillcolor=0)
        hist = [sum(row) for row in _rows(rot)]
        mean = sum(hist) / len(hist)
        score = sum((h - mean) ** 2 for h in hist)
        if score > best_score:
            best_angle, best_score = angle, score
        angle += step
    if abs(best_angle) < 0.25:
        return img
    return img.rotate(best_angle, expand=True, fillcolor=(255, 255, 255), resample=Image.BICUBIC)


def _rows(img: Image.Image):
    w, h = img.size
    data = img.tobytes()
    for y in range(h):
        yield data[y * w:(y + 1) * w]


def preprocess(img: Image.Image, options: dict[str, Any] | None) -> Image.Image:
    options = options or {}
    out = _to_rgb(img)
    if options.get("deskew"):
        out = deskew(out)
    if options.get("contrast"):
        out = ImageOps.autocontrast(out, cutoff=1)
        out = out.filter(ImageFilter.SHARPEN)
    max_side = int(options.get("max_side") or 0)
    if max_side and max(out.size) > max_side:
        out = out.copy()
        out.thumbnail((max_side, max_side), Image.LANCZOS)
    return out


def thumbnail_bytes(img: Image.Image, size: int = 360) -> bytes:
    t = img.copy()
    t.thumbnail((size, size))
    buf = io.BytesIO()
    t.save(buf, format="PNG")
    return buf.getvalue()

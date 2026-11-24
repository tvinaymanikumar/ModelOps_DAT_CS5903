"""Upload, decode and preview documents."""
from __future__ import annotations

from typing import Any

import streamlit as st

from config.settings import SUPPORTED_UPLOAD_TYPES
from utils.image_utils import UnsupportedFileError, decode_upload, thumbnail_bytes

MAX_PREVIEW = 24


def render_upload(settings: dict[str, Any]) -> None:
    files = st.file_uploader("Images or PDFs", type=SUPPORTED_UPLOAD_TYPES, accept_multiple_files=True,
                             help="Images (JPG, PNG, WEBP, BMP, TIFF, GIF) or PDF. Each page is processed separately.")
    signature = (tuple((f.name, f.size) for f in files or []), settings["pdf_dpi"], settings["gif_all"])
    if signature != st.session_state["upload_signature"]:
        pages, thumbs, errors = [], [], []
        with st.spinner("Reading files"):
            for f in files or []:
                try:
                    decoded = decode_upload(f.name, f.getvalue(), settings["pdf_dpi"], settings["gif_all"])
                except UnsupportedFileError as exc:
                    errors.append(str(exc))
                    continue
                multi = len(decoded) > 1
                for name, page, img in decoded:
                    pages.append((name, page, img))
                    if len(thumbs) < MAX_PREVIEW:
                        thumbs.append((f"{name} (page {page})" if multi else name, thumbnail_bytes(img, 300)))
        st.session_state.update(pages=pages, thumbs=thumbs, upload_errors=errors, upload_signature=signature)

    for err in st.session_state["upload_errors"]:
        st.error(err)
    pages = st.session_state["pages"]
    if not pages:
        return
    st.caption(f"{len(pages)} image(s) ready. Each one is processed separately.")
    thumbs = st.session_state["thumbs"]
    cols = st.columns(8)
    for i, (label, data) in enumerate(thumbs):
        cols[i % 8].image(data, caption=label, width="stretch")
    if len(pages) > MAX_PREVIEW:
        st.caption(f"Showing the first {MAX_PREVIEW} pages.")

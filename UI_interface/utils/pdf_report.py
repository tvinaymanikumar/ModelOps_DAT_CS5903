"""ReportLab PDF report for one item. Uses a Unicode TTF font when available
(DejaVu on Linux, Arial / Segoe UI on Windows) so non-ASCII text renders."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image as RLImage, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

_FONT = None
_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\arial.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
]


def _font() -> str:
    global _FONT
    if _FONT:
        return _FONT
    for path in [os.getenv("PDF_FONT_PATH", "")] + _CANDIDATES:
        if path and Path(path).exists():
            try:
                pdfmetrics.registerFont(TTFont("ReportFont", path))
                _FONT = "ReportFont"
                return _FONT
            except Exception:
                continue
    _FONT = "Helvetica"
    return _FONT


def build_item_pdf(path: Path, envelope: dict[str, Any], image_path: str | None) -> None:
    font = _font()
    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["BodyText"], fontName=font, fontSize=9, leading=12)
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontName=font, fontSize=15, leading=19, spaceAfter=4)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontName=font, fontSize=11, leading=14, spaceBefore=8)
    p = lambda t, s=body: Paragraph(escape(str(t)).replace("\n", "<br/>"), s)  # noqa: E731

    story: list[Any] = [p(f"Extraction report: {envelope.get('image')}", h1)]
    meta = [["Task", f"{envelope.get('task_name')} ({envelope.get('mode')})"], ["Model", envelope.get("model")],
            ["Status", envelope.get("status")], ["Page", envelope.get("page")],
            ["Latency (ms)", (envelope.get("metrics") or {}).get("latency_ms")],
            ["Created", envelope.get("created_at")]]
    t = Table([[p(a), p("" if b is None else b)] for a, b in meta], colWidths=[35 * mm, 135 * mm])
    t.setStyle(TableStyle([("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#4A5A6A")),
                           ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story += [t, Spacer(1, 4 * mm)]

    if image_path and Path(image_path).exists():
        from PIL import Image
        with Image.open(image_path) as im:
            w, h = im.size
        max_w, max_h = 170 * mm, 95 * mm
        scale = min(max_w / w, max_h / h)
        story += [RLImage(image_path, width=w * scale, height=h * scale), Spacer(1, 4 * mm)]

    result = envelope.get("result")
    if envelope.get("mode") == "transcription" and result:
        story += [p("Transcription", h2), p(result.get("transcription", ""))]
    elif result:
        story.append(p("Extracted fields", h2))
        rows = [[p("Field"), p("Value")]] + [[p(k), p(v)] for k, v in result.items()]
        ft = Table(rows, colWidths=[60 * mm, 110 * mm], repeatRows=1)
        ft.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E3ECF2")),
                                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#B7C4CF")),
                                ("VALIGN", (0, 0), (-1, -1), "TOP")]))
        story.append(ft)
    else:
        story += [p("No result", h2), p(envelope.get("error") or "")]

    flags = (envelope.get("validation") or {}).get("flags") or []
    if flags:
        story.append(p("Review flags", h2))
        for f in flags:
            story.append(p(f"• [{f.get('rule')}] {f.get('field') + ': ' if f.get('field') else ''}{f.get('detail')}"))

    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm,
                            title=f"Report {envelope.get('image')}")
    doc.build(story)

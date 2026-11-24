"""MinerU2.5-only invoice extraction for a fixed schema.

Schema (all values are strings): company, date, address, total.
Uses the official mineru-vl-utils HTTP client against an OpenAI-compatible
MinerU2.5 endpoint. Images are processed sequentially and one JSON file is
written per image.

Install:
    pip install --upgrade mineru-vl-utils pillow

Run:
    python mineru2_5_invoice_schema_batch.py
"""

from __future__ import annotations

import html
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps
from mineru_vl_utils import MinerUClient


# ============================================================
# 1. CONFIGURATION
# ============================================================

MINERU_URL = "http://10.68.0.39:9008/v1"
MINERU_MODEL = "mineru2.5"
TIMEOUT_SECONDS = 300
MAX_RETRIES = 2

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"
}

SCHEMA_KEYS = ("company", "date", "address", "total")

# The official helper performs layout detection and text/table recognition.
mineru_client = MinerUClient(
    backend="http-client",
    server_url=MINERU_URL,
    model_name=MINERU_MODEL,
    http_timeout=TIMEOUT_SECONDS,
    max_retries=MAX_RETRIES,
    use_tqdm=False,
    isolate_block_errors=True,
)


def field(obj: Any, name: str, default=None):
    """Read fields from MinerU ContentBlock objects or dictionaries."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def empty_schema() -> dict[str, str]:
    return {key: "" for key in SCHEMA_KEYS}


def clean_text(value: Any) -> str:
    return " ".join(html.unescape(str(value or "")).split()).strip()


def tidy_value(value: Any) -> str:
    value = clean_text(value)
    value = re.sub(r"^[\s:;,|.\-–—]+", "", value)
    value = re.sub(r"[\s|_=.·•;,:\-–—]+$", "", value)
    return value.strip()


class TableTextParser(HTMLParser):
    """Preserve table cell boundaries when MinerU returns HTML tables."""
    def __init__(self):
        super().__init__()
        self.parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.parts.append("\n")
        elif tag in ("td", "th"):
            self.parts.append(" | ")
        elif tag == "br":
            self.parts.append(" ")

    def handle_endtag(self, tag):
        if tag == "tr":
            self.parts.append("\n")

    def handle_data(self, data):
        self.parts.append(data)


def content_to_text(block: Any) -> str:
    content = str(field(block, "content", "") or "")
    block_type = str(field(block, "type", "text") or "text").lower()
    if block_type == "table" or "<table" in content.lower() or "<tr" in content.lower():
        parser = TableTextParser()
        try:
            parser.feed(content)
            content = "".join(parser.parts)
        except Exception:
            content = re.sub(r"<[^>]+>", " ", content)
    return html.unescape(content).strip()


def blocks_to_lines(blocks: list[Any]) -> list[dict[str, Any]]:
    """Build OCR lines with block coordinates for nearby label/value matching."""
    lines: list[dict[str, Any]] = []
    for block in blocks or []:
        text = content_to_text(block)
        if not text:
            continue

        bbox = field(block, "bbox", None) or [0, 0, 1, 1]
        try:
            x1, y1, x2, y2 = [float(v) for v in bbox]
        except (TypeError, ValueError):
            x1, y1, x2, y2 = 0.0, 0.0, 1.0, 1.0

        block_type = str(field(block, "type", "text") or "text").lower()
        for raw_line in text.splitlines():
            line = tidy_value(raw_line)
            if line:
                lines.append({
                    "text": line,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "yc": (y1 + y2) / 2.0,
                    "type": block_type,
                })

    lines.sort(key=lambda item: (round(item["yc"], 3), item["x1"]))
    return lines


def is_date_value(text: str) -> bool:
    text = clean_text(text)
    patterns = [
        r"\b\d{1,4}[./-]\d{1,2}[./-]\d{1,4}\b",
        r"\b\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}\b",
        r"\b[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{2,4}\b",
    ]
    return any(re.search(pattern, text, re.I) for pattern in patterns)


def extract_date_token(text: str) -> str:
    """Return a date substring without changing its printed format."""
    patterns = [
        r"\b\d{1,4}[./-]\d{1,2}[./-]\d{1,4}\b",
        r"\b\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}\b",
        r"\b[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{2,4}\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            return match.group(0).strip()
    return ""


def line_is_label(text: str) -> bool:
    low = clean_text(text).lower()
    label_patterns = [
        r"^(?:invoice|tax invoice|receipt|bill|proforma invoice)$",
        r"^(?:invoice\s*(?:no|number|#)|bill\s*(?:no|number|#)|po\s*(?:no|number|#))\b",
        r"^(?:invoice\s+date|date|due\s+date|payment\s+terms)\s*[:#\-]?$",
        r"^(?:company|company name|supplier|vendor|seller|from|issued by)\s*[:#\-]?$",
        r"^(?:address|company address|supplier address|vendor address|seller address|registered address)\s*[:#\-]?$",
        r"^(?:bill to|billed to|ship to|shipping address|billing address)\s*[:#\-]?$",
        r"^(?:subtotal|sub total|tax|vat|gst|discount|total|grand total|amount due|balance due)\s*[:#\-]?$",
    ]
    return any(re.search(pattern, low, re.I) for pattern in label_patterns)


def looks_like_value(text: str) -> bool:
    text = tidy_value(text)
    if not text or line_is_label(text):
        return False
    if re.fullmatch(r"[\s|:;,.\-_=]+", text):
        return False
    return bool(re.search(r"[A-Za-z0-9]", text))


def nearby_value(lines: list[dict[str, Any]], index: int, max_y_gap: float = 0.09) -> str:
    """Find a value to the right on the same row, otherwise below the label."""
    current = lines[index]
    candidates = []
    for j, candidate in enumerate(lines):
        if j == index or not looks_like_value(candidate["text"]):
            continue
        dy = candidate["yc"] - current["yc"]
        if abs(dy) > max_y_gap:
            continue
        same_row = abs(dy) <= 0.035
        to_right = candidate["x1"] >= current["x1"] - 0.01
        if same_row and to_right:
            dx = max(0.0, candidate["x1"] - current["x2"])
            score = abs(dy) * 10 + dx * 0.5
        elif 0 <= dy <= max_y_gap:
            # Prefer a value under the label and roughly aligned with it.
            dx = abs(candidate["x1"] - current["x1"])
            score = 1.0 + dy * 10 + dx * 0.25
        else:
            continue
        candidates.append((score, j, candidate["text"]))
    if not candidates:
        return ""
    candidates.sort(key=lambda item: (item[0], item[1]))
    return tidy_value(candidates[0][2])


def labelled_value(
    lines: list[dict[str, Any]],
    patterns: list[str],
    validator=None,
    max_y_gap: float = 0.10,
) -> tuple[str, int | None]:
    """Extract a value following a known label or from its nearby block."""
    compiled = [re.compile(pattern, re.I) for pattern in patterns]
    for i, item in enumerate(lines):
        text = item["text"]
        for pattern in compiled:
            match = pattern.search(text)
            if not match:
                continue
            # Label-only regexes can have no capture group.
            value = tidy_value(match.group(1)) if match.lastindex else ""
            if not looks_like_value(value) or (validator and not validator(value)):
                value = nearby_value(lines, i, max_y_gap=max_y_gap)
            if value and (validator is None or validator(value)):
                return value, i
    return "", None


def clean_company_value(value: str) -> str:
    value = tidy_value(value)
    value = re.sub(r"^(?:company(?:\s+name)?|supplier|vendor|seller|from|issued\s+by)\s*[:#\-]?\s*", "", value, flags=re.I)
    return tidy_value(value)


def extract_company(lines: list[dict[str, Any]]) -> str:
    """Identify the invoice issuer/company, using labels then top-page fallback."""
    issuer_label = re.compile(
        r"^\s*(?:company(?:\s+name)?|supplier(?:\s+name)?|vendor(?:\s+name)?|"
        r"seller(?:\s+name)?|issued\s+by|from)\s*[:：#\-]?\s*(.*?)\s*$",
        re.I,
    )

    # First, look for an explicitly labelled value, either inline or next to the label.
    for i, item in enumerate(lines):
        match = issuer_label.match(item["text"])
        if not match:
            continue
        value = clean_company_value(match.group(1))
        if not value or not re.search(r"[A-Za-z]", value) or is_date_value(value):
            value = clean_company_value(nearby_value(lines, i, max_y_gap=0.08))
        if value and re.search(r"[A-Za-z]", value) and not re.search(
            r"\b(?:address|invoice|date|total|phone|email|gstin|vat|bill\s+to|ship\s+to)\b",
            value,
            re.I,
        ):
            return value

    # Some invoices have the issuer name as an unlabelled title/logo near the top.
    excluded = re.compile(
        r"\b(invoice|tax invoice|receipt|bill to|billed to|ship to|shipping|billing|customer|"
        r"company name|supplier name|vendor name|seller name|date|address|phone|tel|email|"
        r"www\.|http|gstin|vat|tax id|invoice no|invoice number|subtotal|grand total|"
        r"total|amount due|page|purchase order|po number)\b",
        re.I,
    )
    candidates = []
    for item in lines:
        text = tidy_value(item["text"])
        if item["y1"] > 0.38 or not text or len(text) < 2:
            continue
        if excluded.search(text) or is_date_value(text):
            continue
        if not re.search(r"[A-Za-z]", text):
            continue
        if re.search(r"(?:\S+@\S+|https?://|www\.)", text, re.I):
            continue
        if re.search(r"\b(?:tel|phone|fax)\b", text, re.I):
            continue
        # Prefer uppermost business-like lines; avoid short generic labels.
        score = 2.0 - item["y1"] * 3
        score += 0.6 if re.search(
            r"\b(inc|ltd|limited|llc|corp|corporation|company|co\.)\b",
            text,
            re.I,
        ) else 0
        score += 0.2 if len(text.split()) >= 2 else 0
        score -= 0.5 if len(text.split()) <= 1 else 0
        candidates.append((score, item["y1"], text))

    if candidates:
        candidates.sort(key=lambda x: (-x[0], x[1]))
        return clean_company_value(candidates[0][2])
    return ""


def extract_date(lines: list[dict[str, Any]]) -> str:
    # Strong labels first; deliberately excludes due/delivery/shipping dates.
    patterns = [
        r"\b(?:invoice\s+date|date\s+of\s+(?:issue|invoice)|date\s+issued|issued\s+on|bill\s+date|document\s+date)\s*[:：#\-]?\s*(.*)$",
        r"\binvoice\s+date\b\s*[:：#\-]?\s*(.*)$",
    ]
    value, _ = labelled_value(
        lines,
        patterns,
        validator=is_date_value,
        max_y_gap=0.08,
    )
    if value:
        token = extract_date_token(value)
        if token:
            return token

    # Generic date label next, avoiding due date, delivery date, and expiry.
    for i, item in enumerate(lines):
        low = item["text"].lower()
        if re.search(r"\b(due|delivery|ship(?:ping)?|expiry|expiration|payment)\s+date\b", low):
            continue
        if re.search(r"\bdate\b", low):
            token = extract_date_token(item["text"])
            if token:
                return token
            candidate = nearby_value(lines, i, max_y_gap=0.09)
            token = extract_date_token(candidate)
            if token:
                return token

    # Last resort: the first date-looking string that is not clearly a due date.
    for item in sorted(lines, key=lambda x: (x["y1"], x["x1"])):
        low = item["text"].lower()
        if re.search(r"\b(due|delivery|ship(?:ping)?|expiry|expiration|payment)\s+date\b", low):
            continue
        token = extract_date_token(item["text"])
        if token:
            return token
    return ""


def extract_address(lines: list[dict[str, Any]], company: str) -> str:
    specific_patterns = [
        r"\b(?:company|supplier|vendor|seller|registered|office)\s+address\s*[:：#\-]?\s*(.*)$",
        r"\baddress\s*[:：#\-]\s*(.*)$",
        r"\baddress\s*$",
    ]
    value, idx = labelled_value(
        lines,
        specific_patterns,
        validator=lambda x: bool(re.search(r"[A-Za-z0-9]", x)),
        max_y_gap=0.08,
    )

    anchor = idx
    # If address is explicitly present, append nearby continuation lines such as city/postcode.
    if value and anchor is not None:
        base_y = lines[anchor]["yc"]
        base_x = lines[anchor]["x1"]
        pieces = [value]
        for j, item in enumerate(lines):
            if j == anchor:
                continue
            if item["yc"] <= base_y or item["yc"] - base_y > 0.11:
                continue
            if abs(item["x1"] - base_x) > 0.18:
                continue
            text = tidy_value(item["text"])
            if not text or is_date_value(text):
                continue
            if re.search(
                r"\b(invoice|invoice no|date|total|subtotal|amount due|gstin|vat|phone|tel|email|www\.)\b",
                text,
                re.I,
            ):
                break
            if re.search(r"\b(?:address|bill to|ship to|customer)\b", text, re.I):
                break
            if looks_like_value(text) and text not in pieces:
                pieces.append(text)
            if len(pieces) >= 3:
                break
        return tidy_value(", ".join(pieces))

    # Fallback: collect upper-left lines that resemble address text and are close to company.
    candidates = []
    for item in lines:
        text = tidy_value(item["text"])
        low = text.lower()
        if not text or item["y1"] > 0.48:
            continue
        if company and text.casefold() == company.casefold():
            continue
        if is_date_value(text):
            continue
        if re.search(
            r"\b(invoice|tax invoice|receipt|bill to|billed to|ship to|customer|"
            r"invoice no|invoice number|date|due|total|subtotal|amount due|"
            r"phone|tel|email|www\.|http|gstin|vat|tax id|terms)\b",
            low,
            re.I,
        ):
            continue
        # Address-like cues: street names, suite/unit, postal code, comma-separated locality.
        address_cue = bool(re.search(
            r"\b(address|road|rd\.?|street|st\.?|avenue|ave\.?|lane|ln\.?|"
            r"suite|unit|floor|building|blvd|boulevard|pincode|postcode|zip|"
            r"district|state|country|city|nagar|colony|sector)\b|\d{4,6}",
            text,
            re.I,
        )) or text.count(",") >= 1
        if address_cue and re.search(r"[A-Za-z]", text):
            candidates.append(item)

    if candidates:
        candidates.sort(key=lambda x: (x["y1"], x["x1"]))
        # Keep only nearby lines to avoid combining unrelated address blocks.
        chosen = [candidates[0]]
        for item in candidates[1:]:
            if item["y1"] - chosen[-1]["y1"] <= 0.08 and len(chosen) < 3:
                chosen.append(item)
        return tidy_value(", ".join(item["text"] for item in chosen))
    return ""


def extract_money(value: str) -> str:
    """Extract the amount token while preserving currency and decimal formatting."""
    value = tidy_value(value)
    # Prefer an explicitly currency-tagged amount.
    currency_pattern = re.compile(
        r"(?:₹|\$|€|£|¥|\b(?:INR|USD|EUR|GBP|Rs\.?|Rupees)\b)\s*"
        r"(?:\(?-?\s*)?\d[\d,]*(?:\.\d{1,2})?\)?(?:\s*/-)?",
        re.I,
    )
    match = currency_pattern.search(value)
    if match:
        return tidy_value(match.group(0))

    # Otherwise select the numeric amount and keep original grouping/decimals.
    number_pattern = re.compile(
        r"(?<![\w/.-])\(?-?\d[\d,]*(?:\.\d{1,2})?\)?(?:\s*/-)?(?![\w/.-])"
    )
    matches = list(number_pattern.finditer(value))
    if matches:
        # In total fields, a numeric-looking token in the same line is most likely the amount.
        return tidy_value(matches[-1].group(0))
    return ""


def extract_total(lines: list[dict[str, Any]]) -> str:
    # Strongest labels first. Explicitly skip subtotal and tax-only rows.
    patterns = [
        r"\b(?:grand\s+total|invoice\s+total|total\s+payable|net\s+payable|amount\s+due|balance\s+due|total\s+due|total\s+amount)\s*[:：#\-|]?\s*(.*)$",
        r"\btotal\b\s*[:：#\-|]?\s*(.*)$",
    ]
    candidates = []
    for priority, pattern in enumerate(patterns):
        regex = re.compile(pattern, re.I)
        for i, item in enumerate(lines):
            text = item["text"]
            low = text.lower()
            if re.search(r"\b(?:sub\s*total|subtotal|tax total|total tax|total qty|total quantity|total items)\b", low):
                continue
            match = regex.search(text)
            if not match:
                continue
            tail = tidy_value(match.group(1))
            amount = extract_money(tail)
            if not amount:
                amount = extract_money(nearby_value(lines, i, max_y_gap=0.10))
            if not amount:
                # Invoice rows sometimes place label and amount in separate table cells.
                tail_after_pipe = text.split("|", 1)[1] if "|" in text else ""
                amount = extract_money(tail_after_pipe)
            if amount:
                # Grand total/invoice total labels rank before a generic Total.
                label_priority = 0 if priority == 0 else 1
                position_score = 0 if item["y1"] > 0.55 else 1
                candidates.append((label_priority, position_score, -item["y1"], amount))
        if candidates:
            break

    if candidates:
        candidates.sort(key=lambda row: (row[0], row[1], row[2]))
        return candidates[0][3]

    # Last resort: currency-marked amounts on lower half of the invoice.
    money_candidates = []
    for item in lines:
        text = item["text"]
        low = text.lower()
        if re.search(r"\b(?:subtotal|tax|vat|gst|invoice no|invoice number|phone|account)\b", low):
            continue
        amount = extract_money(text)
        if amount and re.search(r"₹|\$|€|£|¥|\bINR\b|\bUSD\b|\bEUR\b|\bGBP\b|\bRs\.?\b", text, re.I):
            money_candidates.append((item["y1"], amount))
    if money_candidates:
        money_candidates.sort(key=lambda row: row[0], reverse=True)
        return money_candidates[0][1]
    return ""


def extract_invoice_schema(lines: list[dict[str, Any]]) -> dict[str, str]:
    """Extract the invoice fields into exactly the requested schema."""
    company = extract_company(lines)
    date = extract_date(lines)
    address = extract_address(lines, company)
    total = extract_total(lines)
    return {
        "company": company or "",
        "date": date or "",
        "address": address or "",
        "total": total or "",
    }


def is_inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def save_json(path: Path, data: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        json.dump(data, output, indent=4, ensure_ascii=False)
        output.write("\n")


def process_one_image(image_path: Path, input_dir: Path, output_dir: Path, index: int, total: int) -> bool:
    relative_path = image_path.relative_to(input_dir)
    output_path = output_dir / relative_path.parent / f"{image_path.stem}.json"

    print("\n" + "=" * 76)
    print(f"IMAGE {index}/{total}: {relative_path}")
    print("=" * 76)

    try:
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")

        # MinerUClient handles the model's layout + text/table recognition stages.
        blocks = mineru_client.two_step_extract(image)
        lines = blocks_to_lines(blocks)
        result = extract_invoice_schema(lines)

        print("\nMINERU RECOGNIZED TEXT:")
        if lines:
            for item in lines:
                print(f"y={item['yc']:.3f}, x={item['x1']:.3f}: {item['text']}")
        else:
            print("[No readable block content returned]")

        print("\nINVOICE JSON:")
        print(json.dumps(result, indent=4, ensure_ascii=False))
        save_json(output_path, result)

        populated = sum(bool(value) for value in result.values())
        print(f"Saved: {output_path}")
        print(f"Populated fields: {populated}/{len(SCHEMA_KEYS)}")
        print("Status:", "SUCCESS" if populated else "NO FIELDS FOUND")
        return populated > 0

    except Exception as error:
        result = empty_schema()
        save_json(output_path, result)
        print(f"ERROR: {type(error).__name__}: {error}")
        print("Saved schema-shaped fallback JSON:")
        print(json.dumps(result, indent=4, ensure_ascii=False))
        print("Saved:", output_path)
        return False


def main():
    input_text = input("Enter input invoice image folder path: ").strip().strip('"').strip("'")
    output_text = input("Enter result folder path: ").strip().strip('"').strip("'")

    input_dir = Path(input_text).expanduser().resolve()
    output_dir = Path(output_text).expanduser().resolve()

    if not input_dir.is_dir():
        print(f"ERROR: Input folder not found: {input_dir}")
        return

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        print(f"ERROR: Cannot create output folder: {error}")
        return

    image_files = sorted(
        (
            path for path in input_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
            and not (
                is_inside(output_dir, input_dir)
                and is_inside(path, output_dir)
            )
        ),
        key=lambda path: str(path.relative_to(input_dir)).lower(),
    )

    if not image_files:
        print(f"No supported invoice images found in: {input_dir}")
        return

    print("\n" + "=" * 76)
    print("MINERU2.5 INVOICE SCHEMA BATCH EXTRACTION")
    print("=" * 76)
    print("Endpoint:", MINERU_URL)
    print("Model:", MINERU_MODEL)
    print("Input folder:", input_dir)
    print("Output folder:", output_dir)
    print("Images found:", len(image_files))
    print("Schema:", list(SCHEMA_KEYS))
    print("Values: strings; missing/blank/unreadable fields use empty strings")
    print("Mode: sequential, one image at a time")

    successes = 0
    for index, image_path in enumerate(image_files, start=1):
        if process_one_image(image_path, input_dir, output_dir, index, len(image_files)):
            successes += 1

    print("\n" + "=" * 76)
    print("BATCH PROCESSING FINISHED")
    print("=" * 76)
    print("Total images:", len(image_files))
    print("Images with at least one field extracted:", successes)
    print("Images with no fields/errors:", len(image_files) - successes)
    print("Result folder:", output_dir)


if __name__ == "__main__":
    main()

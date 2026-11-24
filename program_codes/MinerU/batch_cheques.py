"""MinerU2.5-only cheque extraction (fixed ContentBlock handling).

Uses the official mineru-vl-utils two-stage extraction client against an
OpenAI-compatible MinerU2.5 endpoint. Processes images sequentially and writes
one filename.json per image with the fixed cheque schema:
  payee_name, amount_in_words, amount, year, account_no

Install:
    pip install --upgrade mineru-vl-utils pillow
Run:
    python mineru2_5_cheque_schema_batch_fixed.py
"""

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

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
SCHEMA_KEYS = ("payee_name", "amount_in_words", "amount", "year", "account_no")

# Official helper returns ContentBlock objects (attributes), not dictionaries.
# This script explicitly supports those objects and dictionary-like test data.
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
    """Read a value from a MinerU ContentBlock or a dict."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def empty_schema():
    return {
        "payee_name": "",
        "amount_in_words": "",
        "amount": "",
        "year": "",
        "account_no": "",
    }


def clean_text(value):
    return " ".join(html.unescape(str(value or "")).split()).strip()


def normalize_amount(value):
    """Cheque schema: convert amount formatting to an integer; blank stays ''."""
    raw = clean_text(value)
    digits = re.sub(r"\D", "", raw)
    return int(digits) if digits else ""


class _HTMLTextParser(HTMLParser):
    """Convert table HTML into readable rows without losing cell boundaries."""
    def __init__(self):
        super().__init__()
        self.parts = []

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


def content_to_text(block):
    """Read block content from either a ContentBlock object or a dictionary."""
    content = field(block, "content", "") or ""
    block_type = str(field(block, "type", "text") or "text").lower()
    content = str(content)

    if block_type == "table" or "<table" in content.lower() or "<tr" in content.lower():
        parser = _HTMLTextParser()
        try:
            parser.feed(content)
            content = "".join(parser.parts)
        except Exception:
            content = re.sub(r"<[^>]+>", " ", content)

    return html.unescape(content).strip()


def blocks_to_lines(blocks):
    """Build OCR lines while retaining normalized block coordinates.

    Important fix: MinerU's two_step_extract returns ContentBlock instances.
    The previous script skipped every block by requiring isinstance(block, dict).
    """
    lines = []

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

        # Use line breaks emitted by table parsing and recognizer output.
        for raw_line in text.splitlines():
            line = clean_text(raw_line).strip("| ")
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

    # Stable approximate reading order: top-to-bottom, then left-to-right.
    lines.sort(key=lambda item: (round(item["yc"], 3), item["x1"]))
    return lines


def is_field_label(text):
    """Identify common cheque labels so they are not mistaken for values."""
    value = clean_text(text).lower().lstrip(".: -")
    return bool(re.match(
        r"^(?:pay(?:\s+to\b|ee\b)?|beneficiary\b|name\b|"
        r"rupees?\b|amount\b|sum\s+of\b|date\b|dtd\b|"
        r"a\s*/\s*c\b|account\b|acct\b|ifsc\b|micr\b|"
        r"branch\b|bank\b|cheque\b|check\b|no\.?\b|"
        r"serial\b|please\b)", value
    ))


def _candidate_is_value(text):
    text = clean_text(text)
    if not text or is_field_label(text):
        return False
    if re.fullmatch(r"[_\-.=:/\\ ]+", text):
        return False
    return bool(re.search(r"[A-Za-z0-9]", text))


def next_value(lines, index, max_y_gap=0.12):
    """Find a nearby value by geometry, preferring the same row to the right."""
    current = lines[index]
    candidates = []

    for j, candidate in enumerate(lines):
        if j == index or not _candidate_is_value(candidate["text"]):
            continue

        vertical_delta = candidate["yc"] - current["yc"]
        if abs(vertical_delta) > max_y_gap:
            continue

        # Prefer same-row candidates to the right of the label.
        same_row = abs(vertical_delta) <= 0.035
        to_right = candidate["x1"] >= current["x1"] - 0.01

        if same_row and to_right:
            horizontal_gap = max(0.0, candidate["x1"] - current["x2"])
            score = abs(vertical_delta) * 10 + horizontal_gap * 0.5
        elif vertical_delta >= 0:
            # A value can be on the next line below a label.
            score = 1.0 + vertical_delta * 10
        else:
            # Do not usually grab content above a label.
            continue

        # Prevent selecting an unrelated field label/value very far away.
        candidates.append((score, j, candidate["text"]))

    if not candidates:
        return ""

    candidates.sort(key=lambda row: (row[0], row[1]))
    return clean_text(candidates[0][2])


def tidy_value(value):
    value = clean_text(value)
    value = re.sub(r"^[\s:;,.\-–—]+", "", value)
    value = re.sub(r"[\s_=.·•]+$", "", value)
    return value.strip()


def labelled_value(lines, patterns, validator=None, max_y_gap=0.12):
    """Read a value on the same line as a label or from a nearby OCR block."""
    compiled = [re.compile(p, re.IGNORECASE) for p in patterns]

    for index, item in enumerate(lines):
        text = item["text"]
        for pattern in compiled:
            match = pattern.search(text)
            if not match:
                continue

            value = tidy_value(match.group(1) if match.lastindex else "")
            if not _candidate_is_value(value):
                value = tidy_value(next_value(lines, index, max_y_gap=max_y_gap))

            if value and (validator is None or validator(value)):
                return value

    return ""


def extract_payee(lines):
    patterns = [
        # Handles the common cheque wording and OCR variants.
        r"\bpay\s*to\s*(?:the\s+order\s+of\s*)?[:：\-]?\s*(.*)$",
        r"\bpayee(?:\s+name)?\s*[:：\-]?\s*(.*)$",
        r"\bbeneficiary(?:\s+name)?\s*[:：\-]?\s*(.*)$",
    ]

    def valid(value):
        return (
            bool(re.search(r"[A-Za-z]", value))
            and not re.search(r"\b(?:rupees?|only|account|a/c|date|branch|bank)\b", value, re.I)
        )

    value = labelled_value(lines, patterns, valid, max_y_gap=0.08)
    if value:
        return value

    # Some cheque layouts print just "Pay" at the left of the payee line.
    value = labelled_value(
        lines,
        [r"^\s*pay\s*[:：\-]?\s*(.*)$"],
        valid,
        max_y_gap=0.06,
    )
    return value


def extract_amount_in_words(lines):
    patterns = [
        r"\bamount\s+in\s+words?\s*[:：\-]?\s*(.*)$",
        r"\bsum\s+of\s+rupees?\s*[:：\-]?\s*(.*)$",
        r"\brupees?\s*[:：\-]?\s*(.*)$",
    ]

    def valid(value):
        return bool(re.search(r"[A-Za-z]", value)) and not re.fullmatch(
            r"(?:rupees?|amount\s+in\s+words?|only|rs\.?)[ .]*", value, re.I
        )

    value = labelled_value(lines, patterns, valid, max_y_gap=0.14)
    if value:
        return value

    # Many cheques have no explicit label; the amount-in-words line often ends in "only".
    candidates = []
    for item in lines:
        text = tidy_value(item["text"])
        if not re.search(r"\bonly\b", text, re.I):
            continue
        if not re.search(r"[A-Za-z]", text):
            continue
        if re.search(r"\b(?:pay|beneficiary|bank|branch|account|cheque|check|date|ifsc|micr)\b", text, re.I):
            continue
        score = 0
        if re.search(r"\b(?:rupees?|hundred|thousand|lakh|million|crore)\b", text, re.I):
            score += 10
        if item["y1"] >= 0.25:
            score += 1
        candidates.append((score, len(text), text))

    if candidates:
        candidates.sort(key=lambda row: (-row[0], -row[1]))
        return candidates[0][2]
    return ""


def is_date_text(text):
    return bool(re.search(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b", text))


def extract_amount(lines):
    """Choose the best numeric amount candidate while excluding dates and IDs."""
    candidates = []
    money_pattern = re.compile(
        r"(?<!\d)(?:₹\s*|\bINR\s*|\bRs\.?\s*)?"
        r"([0-9][0-9, ]*(?:\.\d{1,2})?)\s*(/-|/\-|/−|−|-)?",
        re.IGNORECASE,
    )

    for item in lines:
        text = item["text"]
        low = text.lower()

        if is_date_text(text):
            continue
        if re.search(r"\b(?:account|a\s*/\s*c|acct|ifsc|micr|branch|cheque|check|serial)\b", low):
            continue
        if re.search(r"amount\s+in\s+words|rupees?.*\bonly\b", low):
            continue

        for match in money_pattern.finditer(text):
            raw_number = match.group(1).strip()
            digits = re.sub(r"\D", "", raw_number)
            if not digits or len(digits) > 10:
                continue

            has_currency = bool(re.search(r"₹|\bINR\b|\bRs\.?", text, re.I))
            has_amount_label = bool(re.search(r"\bamount(?:\s+in\s+figures)?\b", low))
            has_suffix = bool(match.group(2))
            comma_grouped = "," in raw_number

            score = 0
            score += 100 if has_amount_label else 0
            score += 45 if has_currency else 0
            score += 35 if has_suffix else 0
            score += 15 if comma_grouped else 0
            score += 10 if len(digits) >= 3 else 0
            score += 20 if item["x1"] >= 0.45 else 0
            score += 10 if item["y1"] <= 0.70 else 0
            score -= 20 if len(digits) >= 9 else 0

            # Reject likely dates/IDs unless there is explicit monetary evidence.
            if score >= 25:
                candidates.append((score, len(digits), item["y1"], -item["x1"], digits, text))

    if not candidates:
        return ""

    candidates.sort(key=lambda row: (-row[0], -row[1], row[2], row[3]))
    return candidates[0][4]


def extract_year(lines):
    """Extract a four-digit year from a cheque date or a date-area block."""
    candidates = []
    year_pattern = re.compile(r"\b((?:19|20)\d{2})\b")

    for item in lines:
        text = item["text"]
        low = text.lower()
        years = year_pattern.findall(text)
        if not years:
            continue
        if re.search(r"\b(?:account|a\s*/\s*c|ifsc|micr|amount|rupees)\b", low):
            continue

        date_cue = bool(re.search(r"\bdate\b|\bdtd\b|\bon\b", low)) or is_date_text(text)
        position_cue = item["x1"] >= 0.60 and item["y1"] <= 0.45
        score = 100 if date_cue else (35 if position_cue else 0)
        if score:
            for year in years:
                candidates.append((score, item["y1"], year))

    if not candidates:
        return ""

    candidates.sort(key=lambda row: (-row[0], row[1]))
    return candidates[0][2]


def extract_account_no(lines):
    """Extract only an explicitly labelled account number; avoid MICR guessing."""
    patterns = [
        r"\baccount\s*(?:no\.?|number|#)\s*[:：\-]?\s*([A-Z0-9][A-Z0-9\s\-]{3,24})",
        r"\ba\s*/\s*c\s*(?:no\.?|number|#)?\s*[:：\-]?\s*([A-Z0-9][A-Z0-9\s\-]{3,24})",
        r"\bacct\.?\s*(?:no\.?|number|#)\s*[:：\-]?\s*([A-Z0-9][A-Z0-9\s\-]{3,24})",
    ]
    value = labelled_value(lines, patterns, validator=None, max_y_gap=0.08)

    if not value:
        # Label-only cases: inspect immediately adjacent value block.
        label_re = re.compile(r"\b(?:account\s*(?:no\.?|number|#)|a\s*/\s*c|acct\.?\s*(?:no\.?|number|#))\b", re.I)
        for index, item in enumerate(lines):
            if label_re.search(item["text"]):
                value = next_value(lines, index, max_y_gap=0.08)
                if value:
                    break

    digits = re.sub(r"\D", "", value)
    if 5 <= len(digits) <= 20:
        # Keep OCR spacing/formatting where possible.
        return tidy_value(value)
    return ""


def extract_cheque_schema(lines):
    output = empty_schema()
    output["payee_name"] = extract_payee(lines)
    output["amount_in_words"] = extract_amount_in_words(lines)
    output["amount"] = normalize_amount(extract_amount(lines))
    output["year"] = extract_year(lines)
    output["account_no"] = extract_account_no(lines)
    return output


# ============================================================
# 2. BATCH PROCESSING / OUTPUT
# ============================================================
def is_inside(path, parent):
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        json.dump(data, output, indent=4, ensure_ascii=False)
        output.write("\n")


def process_one_image(image_path, input_dir, output_dir, index, total):
    relative_path = image_path.relative_to(input_dir)
    output_path = output_dir / relative_path.parent / f"{image_path.stem}.json"

    print("\n" + "=" * 75)
    print(f"IMAGE {index}/{total}: {relative_path}")
    print("=" * 75)

    try:
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")

        blocks = mineru_client.two_step_extract(image)
        lines = blocks_to_lines(blocks)
        result = extract_cheque_schema(lines)

        # Print OCR text so mismatches can be diagnosed without extra output files.
        print("\nMINERU RECOGNIZED LINES:")
        if lines:
            for line in lines:
                print(f"y={line['yc']:.3f}, x={line['x1']:.3f}: {line['text']}")
        else:
            print("[No readable block content returned]")

        save_json(output_path, result)
        print("\nCHEQUE SCHEMA JSON:")
        print(json.dumps(result, indent=4, ensure_ascii=False))
        populated = sum(bool(value) for value in result.values())
        print(f"Saved: {output_path}")
        print(f"Populated fields: {populated}/{len(SCHEMA_KEYS)}")
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
    input_text = input("Enter input cheque image folder path: ").strip().strip('"').strip("'")
    output_text = input("Enter result folder path: ").strip().strip('"').strip("'")

    input_dir = Path(input_text).expanduser().resolve()
    output_dir = Path(output_text).expanduser().resolve()

    if not input_dir.is_dir():
        print(f"ERROR: Input folder not found: {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
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
        print("No supported cheque images found.")
        return

    print("\n" + "=" * 75)
    print("MINERU2.5 CHEQUE SCHEMA EXTRACTION")
    print("=" * 75)
    print("Endpoint:", MINERU_URL)
    print("Model:", MINERU_MODEL)
    print("Images found:", len(image_files))
    print("Schema:", ", ".join(SCHEMA_KEYS))
    print("Output: one filename.json per image")
    print("Processing: sequential images; MinerU two-step extraction")

    successes = 0
    for index, image_path in enumerate(image_files, start=1):
        if process_one_image(image_path, input_dir, output_dir, index, len(image_files)):
            successes += 1

    print("\n" + "=" * 75)
    print("BATCH PROCESSING FINISHED")
    print("=" * 75)
    print("Total images:", len(image_files))
    print("Images with at least one populated field:", successes)
    print("Images with all fields blank / errors:", len(image_files) - successes)
    print("Output folder:", output_dir)


if __name__ == "__main__":
    main()

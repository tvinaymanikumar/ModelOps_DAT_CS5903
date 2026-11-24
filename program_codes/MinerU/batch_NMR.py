"""MinerU2.5-only extraction for the Nigeria medical-record dataset.

Schema:
    {"name": "", "age": "", "sex": ""}

Uses the official mineru-vl-utils HTTP client for MinerU2.5 layout + OCR.
Images are processed sequentially; one flat JSON file is written per image.

Install:
    pip install --upgrade mineru-vl-utils pillow

Run:
    python mineru2_5_nigeria_medical_batch.py
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
MAX_IMAGE_SIDE = 3000

SCHEMA_KEYS = ("name", "age", "sex")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

# Official MinerU two-stage client: layout detection + region recognition.
mineru_client = MinerUClient(
    backend="http-client",
    server_url=MINERU_URL,
    model_name=MINERU_MODEL,
    http_timeout=TIMEOUT_SECONDS,
    max_retries=MAX_RETRIES,
    use_tqdm=False,
    isolate_block_errors=True,
)


# ============================================================
# 2. NORMALIZATION
# ============================================================
def clean_text(value: Any) -> str:
    return " ".join(html.unescape(str(value or "")).replace("\u00a0", " ").split()).strip()


def normalize_sex(value: Any) -> str:
    """Normalize common sex abbreviations, preserving unrecognized text."""
    raw = clean_text(value)
    if not raw:
        return ""
    token = re.sub(r"[^a-z]", "", raw.lower())
    if token in {"m", "male"}:
        return "male"
    if token in {"f", "female", "fem"}:
        return "female"
    return raw


def normalize_age(value: Any) -> str:
    """Normalize examples like 25, 25y, 25year, 4m, 4 mon, 4 months."""
    raw = clean_text(value)
    if not raw:
        return ""

    # Handle combined age/sex values such as 12/f or 5 years / male.
    combined = re.fullmatch(
        r"\s*(\d+(?:\.\d+)?\s*(?:years?|yrs?|yr|y|months?|mons?|mon|mos?|mo|m)?)"
        r"\s*[/|]\s*(?:m|male|f|female|fem)\s*",
        raw,
        flags=re.IGNORECASE,
    )
    if combined:
        raw = combined.group(1).strip()

    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(.*?)\s*", raw, flags=re.IGNORECASE)
    if not match:
        return raw

    number = match.group(1)
    unit_text = re.sub(r"\bold\b", "", match.group(2).lower()).strip()
    unit = re.sub(r"[\s./_\-]+", "", unit_text)

    if not unit:
        return f"{number} years"

    year_units = {"y", "yr", "yrs", "year", "years", "yea", "yo", "yro", "yearold", "yearsold"}
    month_units = {"m", "mo", "mos", "mon", "mons", "month", "months", "mth", "mths", "monthold", "monthsold"}

    if unit in year_units or "year" in unit or unit in {"yea"}:
        return f"{number} years"
    if unit in month_units or "month" in unit or "mon" in unit:
        return f"{number} months"
    return raw


def normalize_medical_fields(fields: dict[str, Any]) -> dict[str, str]:
    """Apply schema and normalize age/sex, including combined age-sex fields."""
    output = {key: clean_text(fields.get(key, "")) for key in SCHEMA_KEYS}

    # Some records put both values in one field, e.g. age='12/f' or '5 years/m'.
    combined = re.fullmatch(
        r"\s*(\d+(?:\.\d+)?\s*(?:years?|yrs?|yr|y|months?|mons?|mon|mos?|mo|m)?)"
        r"\s*[/|]\s*(m|male|f|female|fem)\s*",
        output["age"],
        flags=re.IGNORECASE,
    )
    if combined:
        output["age"] = combined.group(1).strip()
        if not output["sex"]:
            output["sex"] = combined.group(2)

    output["age"] = normalize_age(output["age"])
    output["sex"] = normalize_sex(output["sex"])
    return output


# ============================================================
# 3. READ MINERU OUTPUT BLOCKS
# ============================================================
def get_attr(obj: Any, key: str, default=None):
    """Supports MinerU ContentBlock dict objects and ordinary dictionaries."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


class _TableTextParser(HTMLParser):
    """Convert table HTML to text while retaining row/cell separators."""
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.parts.append("\n")
        elif tag in {"td", "th"}:
            self.parts.append(" | ")
        elif tag == "br":
            self.parts.append(" ")

    def handle_endtag(self, tag):
        if tag == "tr":
            self.parts.append("\n")

    def handle_data(self, data):
        self.parts.append(data)


def block_text(block: Any) -> str:
    text = get_attr(block, "content", "") or ""
    text = str(text)
    block_type = str(get_attr(block, "type", "text") or "text").lower()

    if block_type == "table" or "<table" in text.lower() or "<tr" in text.lower():
        parser = _TableTextParser()
        try:
            parser.feed(text)
            text = "".join(parser.parts)
        except Exception:
            text = re.sub(r"<[^>]+>", " ", text)

    return html.unescape(text).strip()


def extract_lines(blocks: list[Any]) -> list[dict[str, Any]]:
    """Return non-empty OCR lines plus block coordinates for neighboring values."""
    lines = []
    for block in blocks or []:
        text = block_text(block)
        if not text:
            continue
        bbox = get_attr(block, "bbox", [0.0, 0.0, 1.0, 1.0]) or [0.0, 0.0, 1.0, 1.0]
        try:
            x1, y1, x2, y2 = [float(v) for v in bbox]
        except (TypeError, ValueError):
            x1, y1, x2, y2 = 0.0, 0.0, 1.0, 1.0

        for raw in text.splitlines():
            line = clean_text(raw).strip("| ")
            if line:
                lines.append({
                    "text": line,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "yc": (y1 + y2) / 2.0,
                    "type": str(get_attr(block, "type", "text") or "text").lower(),
                })

    lines.sort(key=lambda item: (round(item["yc"], 3), item["x1"]))
    return lines


# ============================================================
# 4. SCHEMA-SPECIFIC FIELD EXTRACTION
# ============================================================
# Common label spellings for this dataset. Keep these lists narrow enough to
# avoid confusing unrelated medical terms with the requested fields.
NAME_LABEL = r"(?:patient\s+name|name\s+of\s+patient|name\s+of\s+person|full\s+name|name)"
AGE_LABEL = r"(?:age\s*/\s*sex|age\s+and\s+sex|age|yrs?\.?|years?\s+old)"
SEX_LABEL = r"(?:sex|gender)"
ALL_LABELS = rf"(?:{NAME_LABEL}|{AGE_LABEL}|{SEX_LABEL})"

NAME_PATTERNS = [
    re.compile(rf"^\s*{NAME_LABEL}\s*[:#=\-]?\s*(.*?)\s*$", re.IGNORECASE),
    re.compile(r"^\s*(?:patient|client)\s*[:#=\-]\s*(.*?)\s*$", re.IGNORECASE),
]
AGE_VALUE_RE = re.compile(
    r"(\d+(?:\.\d+)?\s*(?:years?|yrs?|yr|y|yea|months?|mons?|mon|mos?|mo|m)?"
    r"\s*(?:[/|]\s*(?:m|male|f|female|fem))?)",
    re.IGNORECASE,
)
SEX_VALUE_RE = re.compile(r"\b(male|female|fem|m|f)\b", re.IGNORECASE)


def clean_name_candidate(value: str) -> str:
    value = clean_text(value).strip(" :#=-|,;.")
    # Stop before another known field if several labels were OCRed on one line.
    value = re.split(rf"\s+(?={ALL_LABELS}\b\s*[:#=\-]?)", value, maxsplit=1, flags=re.IGNORECASE)[0]
    return value.strip(" :#=-|,;.")


def value_after_label(text: str, field_name: str) -> str:
    """Find a field's value when label and value are on the same OCR line."""
    aliases = {
        "name": NAME_LABEL,
        "age": AGE_LABEL,
        "sex": SEX_LABEL,
    }[field_name]
    # Match a label anywhere within a line and stop capture at the next label.
    match = re.search(
        rf"(?<![A-Za-z])(?:{aliases})\b\s*[:#=\-]?\s*(.*?)"
        rf"(?=\s+(?:{ALL_LABELS})\b\s*[:#=\-]?|$)",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return ""
    value = clean_text(match.group(1)).strip(" :#=-|,;.")

    if field_name == "name":
        return clean_name_candidate(value)
    if field_name == "age":
        age_match = AGE_VALUE_RE.search(value)
        return age_match.group(1).strip() if age_match else ""
    sex_match = SEX_VALUE_RE.search(value)
    return sex_match.group(1) if sex_match else ""


def is_label_line(text: str, field_name: str) -> bool:
    aliases = {"name": NAME_LABEL, "age": AGE_LABEL, "sex": SEX_LABEL}[field_name]
    return bool(re.fullmatch(rf"\s*{aliases}\s*[:#=\-]?\s*", text, re.IGNORECASE))


def is_value_like(text: str, field_name: str) -> bool:
    value = clean_text(text)
    if not value or value_after_label(value, "name") or value_after_label(value, "age") or value_after_label(value, "sex"):
        # Lines that include labels should be parsed directly, not used as neighbors.
        if re.search(rf"(?i)\b{ALL_LABELS}\b", value):
            return False
    if re.fullmatch(r"[\s:;,_./|=\-]+", value):
        return False
    if field_name == "age":
        return AGE_VALUE_RE.fullmatch(value) is not None
    if field_name == "sex":
        return SEX_VALUE_RE.fullmatch(value) is not None
    return bool(re.search(r"[A-Za-z]", value)) and len(value) <= 100


def neighbor_value(lines: list[dict[str, Any]], label_index: int, field_name: str) -> str:
    """Find a likely adjacent value for a label-only block using layout geometry."""
    label = lines[label_index]
    candidates = []

    for idx, candidate in enumerate(lines):
        if idx == label_index:
            continue
        value = candidate["text"]
        if not is_value_like(value, field_name):
            continue

        # Prefer a value on the same row and to the right of the label.
        y_gap = abs(candidate["yc"] - label["yc"])
        if candidate["x1"] >= label["x2"] - 0.005 and y_gap <= 0.035:
            candidates.append((0, y_gap, candidate["x1"] - label["x2"], value))
            continue

        # Otherwise permit the closest next line below when horizontally aligned.
        vertical_gap = candidate["y1"] - label["y2"]
        x_overlap = min(candidate["x2"], label["x2"]) - max(candidate["x1"], label["x1"])
        if 0 <= vertical_gap <= 0.08 and x_overlap >= -0.04:
            candidates.append((1, vertical_gap, abs(candidate["x1"] - label["x1"]), value))

    if not candidates:
        return ""
    candidates.sort(key=lambda x: (x[0], x[1], x[2]))
    return candidates[0][3]


def extract_from_lines(lines: list[dict[str, Any]]) -> dict[str, str]:
    output = {key: "" for key in SCHEMA_KEYS}
    texts = [line["text"] for line in lines]

    # First: same-line label/value extraction, including multiple fields per line.
    for text in texts:
        if not output["name"]:
            output["name"] = value_after_label(text, "name")
        if not output["age"]:
            output["age"] = value_after_label(text, "age")
        if not output["sex"]:
            output["sex"] = value_after_label(text, "sex")

        # Common compact combined notation: "12/f", "5 years/m".
        if not output["age"] or not output["sex"]:
            combo = re.search(
                r"\b(\d+(?:\.\d+)?\s*(?:years?|yrs?|yr|y|months?|mons?|mon|mos?|mo|m)?)"
                r"\s*[/|]\s*(female|male|fem|f|m)\b",
                text,
                re.IGNORECASE,
            )
            if combo:
                if not output["age"]:
                    output["age"] = combo.group(1).strip()
                if not output["sex"]:
                    output["sex"] = combo.group(2)

    # Second: labels and values separated into neighboring layout blocks.
    label_aliases = {"name": NAME_LABEL, "age": AGE_LABEL, "sex": SEX_LABEL}
    for index, line in enumerate(lines):
        for field_name, aliases in label_aliases.items():
            if output[field_name] or not is_label_line(line["text"], field_name):
                continue
            found = neighbor_value(lines, index, field_name)
            if found:
                output[field_name] = found

    # Third: if the record is a simple table (header row includes name/age/sex),
    # read values from subsequent table-like OCR lines in column order.
    if not any(output.values()):
        for idx, text in enumerate(texts[:-1]):
            lower = re.sub(r"[^a-z]+", " ", text.lower()).split()
            if {"name", "age", "sex"}.issubset(set(lower)):
                next_line = texts[idx + 1]
                cells = [c.strip() for c in re.split(r"\s{2,}|\s*\|\s*|\t", next_line) if c.strip()]
                if len(cells) >= 3:
                    output["name"], output["age"], output["sex"] = cells[:3]
                break

    return normalize_medical_fields(output)


# ============================================================
# 5. BATCH PROCESSING AND OUTPUT
# ============================================================
def save_json(path: Path, data: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        f.write("\n")


def is_inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def process_one(image_path: Path, input_dir: Path, output_dir: Path, index: int, total: int) -> bool:
    relative = image_path.relative_to(input_dir)
    output_path = output_dir / relative.parent / f"{image_path.stem}.json"

    print("\n" + "=" * 72)
    print(f"IMAGE {index}/{total}: {relative}")
    print("=" * 72)

    try:
        with Image.open(image_path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))

        # Official MinerU two-stage layout + content extraction.
        blocks = mineru_client.two_step_extract(image)
        lines = extract_lines(blocks)
        print("Recognized OCR lines:")
        if lines:
            for line in lines:
                print("  ", line["text"])
        else:
            print("  [No text recognized]")

        fields = extract_from_lines(lines)
        save_json(output_path, fields)

        print("\nFINAL JSON:")
        print(json.dumps(fields, ensure_ascii=False, indent=4))
        print("Saved:", output_path)
        return any(fields.values())

    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}")
        # Schema datasets always receive the full schema, even on extraction failure.
        empty = {key: "" for key in SCHEMA_KEYS}
        save_json(output_path, empty)
        print("Saved schema-shaped fallback JSON:")
        print(json.dumps(empty, ensure_ascii=False, indent=4))
        print("Saved:", output_path)
        return False


def main() -> None:
    input_text = input("Enter input image folder path: ").strip().strip('"').strip("'")
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

    images = sorted(
        (
            p for p in input_dir.rglob("*")
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS
            and not (is_inside(output_dir, input_dir) and is_inside(p, output_dir))
        ),
        key=lambda p: str(p.relative_to(input_dir)).lower(),
    )

    if not images:
        print(f"No supported images found in: {input_dir}")
        return

    print("\n" + "=" * 72)
    print("NIGERIA MEDICAL RECORDS - MINERU2.5 ONLY")
    print("=" * 72)
    print("Endpoint:", MINERU_URL)
    print("Model:", MINERU_MODEL)
    print("Input folder:", input_dir)
    print("Output folder:", output_dir)
    print("Images found:", len(images))
    print("Schema:", list(SCHEMA_KEYS))
    print("Processing: sequential, one image at a time")

    success_count = 0
    for i, path in enumerate(images, start=1):
        if process_one(path, input_dir, output_dir, i, len(images)):
            success_count += 1

    print("\n" + "=" * 72)
    print("BATCH FINISHED")
    print("=" * 72)
    print("Total images:", len(images))
    print("Images with at least one extracted field:", success_count)
    print("Images with all values blank or errors:", len(images) - success_count)
    print("Output folder:", output_dir)


if __name__ == "__main__":
    main()

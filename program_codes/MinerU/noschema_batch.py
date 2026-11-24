"""Sequential flat-key extraction using the official MinerU2.5 client.

Install: pip install -U mineru-vl-utils pillow
Run:     python mineru2_5_compatible_batch.py

Uses only MinerU2.5. It does not call Qwen, PaddleOCR, or another LLM.
"""

from __future__ import annotations

import json
import re
from html import unescape
from html.parser import HTMLParser
from pathlib import Path

from PIL import Image, ImageOps
from mineru_vl_utils import MinerUClient


# --------------------------- Configuration ---------------------------
MINERU_URL = "http://10.68.0.39:9008/v1"
MINERU_MODEL = "mineru2.5"
REQUEST_TIMEOUT_SECONDS = 180
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

# Preserve recognized lines which do not have clear label/value boundaries.
# Set False if you want ONLY confidently parsed key-value pairs.
PRESERVE_UNPARSED_LINES = True


# Official MinerU client uses the model's task prompts, preprocessing, and
# two-stage layout/content recognition rather than hand-written API prompts.
client = MinerUClient(
    backend="http-client",
    server_url=MINERU_URL,
    model_name=MINERU_MODEL,
    http_timeout=REQUEST_TIMEOUT_SECONDS,
    use_tqdm=False,
)


# --------------------------- Table parsing ---------------------------
class HTMLTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        if tag == "tr":
            self._row = []
        elif tag in {"td", "th"}:
            self._cell = []
        elif tag == "br" and self._cell is not None:
            self._cell.append(" ")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"td", "th"} and self._cell is not None:
            if self._row is not None:
                self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._row:
                self.rows.append(self._row)
            self._row = None

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)


def parse_table_rows(content: str) -> list[list[str]]:
    """Parse HTML tables (MinerU's documented table content format)."""
    if not content:
        return []
    lowered = content.lower()
    if "<table" in lowered or "<tr" in lowered:
        parser = HTMLTableParser()
        try:
            parser.feed(content)
            return parser.rows
        except Exception:
            return []

    # Fallback for Markdown/pipe-separated output.
    rows: list[list[str]] = []
    for line in content.splitlines():
        if "|" not in line:
            continue
        cells = [unescape(c.strip()) for c in line.strip().strip("|").split("|")]
        if cells and not all(set(c) <= set("-: ") for c in cells):
            rows.append(cells)
    return rows


# --------------------------- Key/value extraction ---------------------------
KEY_PUNCTUATION_RE = re.compile(r"[^\w]+", flags=re.UNICODE)
NUMBERED_PREFIX_RE = re.compile(r"^\s*\d{1,3}\s*[.)-]\s*")

# Handles merged OCR text such as "StudentRITIKA BASU", "CourseMBA",
# "Date of Joining17/09/2023", or "Roll No.504375".
MERGED_BOUNDARY_RE = re.compile(
    r"(?<=[a-z])(?=[A-Z])|(?<=[A-Za-z.])(?=\d)|(?<=\))(?=[A-Z])|(?<=\.)(?=[A-Z]{1,4}\s)"
)


def to_key(label: str) -> str:
    label = NUMBERED_PREFIX_RE.sub("", label.strip())
    key = KEY_PUNCTUATION_RE.sub("_", label.lower()).strip("_")
    key = re.sub(r"_+", "_", key)
    return key or "field"


def add_field(fields: dict[str, str], label: str, value: str) -> None:
    """Add a flat string value, preserving duplicate labels with suffixes."""
    key = to_key(label)
    value = " ".join((value or "").split())

    if key not in fields:
        fields[key] = value
    elif fields[key] == "" and value:
        fields[key] = value
    elif fields[key] != value:
        index = 2
        while f"{key}_{index}" in fields:
            index += 1
        fields[f"{key}_{index}"] = value


def split_merged_line(line: str) -> tuple[str, str] | None:
    """Split a line only where a likely label/value boundary is visible."""
    text = NUMBERED_PREFIX_RE.sub("", line.strip())
    if not text:
        return None

    # Last boundary is usually the join between a label and its value,
    # e.g. "DateOfBirth12-02-1996" -> "DateOfBirth" / "12-02-1996".
    matches = list(MERGED_BOUNDARY_RE.finditer(text))
    matches = [m for m in matches if m.start() > 0 and m.start() < len(text)]
    if not matches:
        return None

    split_at = matches[-1].start()
    label = text[:split_at].strip(" .:-\t")
    value = text[split_at:].strip(" .:-\t")

    if len(label) < 2 or not re.search(r"[A-Za-z]", label):
        return None
    if not value:
        return None
    return label, value


def extract_text_lines(text: str, fields: dict[str, str], unparsed: list[str]) -> None:
    """Extract explicit or recognizable merged fields from OCR text."""
    for original_line in (text or "").splitlines():
        line = " ".join(unescape(original_line).split())
        if not line:
            continue

        # A clear label/value separator is preferable to heuristics.
        if ":" in line or "：" in line:
            label, value = re.split(r"[:：]", line, maxsplit=1)
            label = NUMBERED_PREFIX_RE.sub("", label.strip())
            if label and re.search(r"[A-Za-z]", label):
                add_field(fields, label, value)
                continue

        # Handle a label and value concatenated by OCR without punctuation.
        pair = split_merged_line(line)
        if pair:
            add_field(fields, pair[0], pair[1])
            continue

        # A numbered label with no detectable value is retained as blank.
        numbered = NUMBERED_PREFIX_RE.match(line)
        if numbered:
            label = line[numbered.end():].strip()
            if label and len(label) <= 100 and re.search(r"[A-Za-z]", label):
                # Hash-only OCR artifacts after a label indicate unreadable text.
                label = re.sub(r"[#*]+$", "", label).strip()
                if label:
                    add_field(fields, label, "")
                    continue

        unparsed.append(line)


def extract_table_fields(rows: list[list[str]], fields: dict[str, str]) -> None:
    """Convert two-column tables to pairs and other tables by header."""
    cleaned = [[(cell or "").strip() for cell in row] for row in rows]
    cleaned = [row for row in cleaned if any(row)]
    if not cleaned:
        return

    width = max(len(row) for row in cleaned)
    cleaned = [row + [""] * (width - len(row)) for row in cleaned]

    if width <= 2:
        for row in cleaned:
            if row[0]:
                add_field(fields, row[0].rstrip(":"), row[1] if len(row) > 1 else "")
        return

    headers = cleaned[0]
    for row_no, row in enumerate(cleaned[1:], start=1):
        for column, value in enumerate(row):
            label = headers[column] or f"column_{column + 1}"
            if len(cleaned) > 2:
                label = f"{label}_row_{row_no}"
            add_field(fields, label, value)


def blocks_to_flat_json(blocks) -> tuple[dict[str, str], str]:
    """Convert official MinerU ContentBlock output to a flat JSON object."""
    fields: dict[str, str] = {}
    unparsed_lines: list[str] = []
    page_text_parts: list[str] = []

    for block in blocks:
        # ContentBlock inherits from dict in mineru-vl-utils.
        block_type = str(block.get("type", "text")).lower()
        content = block.get("content") or ""
        content = str(content).strip()
        if not content:
            continue

        page_text_parts.append(content)

        if block_type == "table":
            rows = parse_table_rows(content)
            if rows:
                extract_table_fields(rows, fields)
            else:
                extract_text_lines(content, fields, unparsed_lines)
        elif block_type not in {"image", "image_block", "equation_block"}:
            if block_type in {"title", "doc_title"}:
                add_field(fields, "document_title", content)
            else:
                extract_text_lines(content, fields, unparsed_lines)

    if PRESERVE_UNPARSED_LINES:
        for i, line in enumerate(unparsed_lines, start=1):
            key = f"unparsed_text_line_{i}"
            while key in fields:
                i += 1
                key = f"unparsed_text_line_{i}"
            fields[key] = line

    # JSON scalar values are strings; blanks are "". This is flat by design.
    flat = {
        str(key): ("" if value is None else str(value).strip())
        for key, value in fields.items()
    }
    page_text = "\n".join(page_text_parts)
    return flat, page_text


# --------------------------- File helpers ---------------------------
def save_json(path: Path, data: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
        file.write("\n")


def is_inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def process_image(image_path: Path, input_dir: Path, output_dir: Path, index: int, total: int) -> bool:
    relative = image_path.relative_to(input_dir)
    output_dir_for_image = output_dir / relative.parent
    output_path = output_dir_for_image / f"{image_path.stem}.json"

    print("\n" + "=" * 72)
    print(f"IMAGE {index}/{total}: {relative}")
    print("=" * 72)

    try:
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")

        # This is MinerU's supported high-level two-stage extraction method.
        blocks = client.two_step_extract(image)
        fields, page_text = blocks_to_flat_json(blocks)

        print(f"MinerU blocks returned: {len(blocks)}")
        print("\nRECOGNIZED TEXT:\n" + (page_text or "[No text recognized]"))
        print("\nFLAT JSON:\n" + json.dumps(fields, indent=2, ensure_ascii=False))

        save_json(output_path, fields)
        print("Saved:", output_path)
        print("Status:", "SUCCESS" if page_text.strip() else "NO TEXT RECOGNIZED")
        return bool(page_text.strip())

    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
        print("Saving {} to allow the batch to continue.")
        save_json(output_path, {})
        print("Saved:", output_path)
        return False


def main() -> None:
    input_raw = input("Enter input image folder path: ").strip().strip('"').strip("'")
    output_raw = input("Enter result folder path: ").strip().strip('"').strip("'")

    input_dir = Path(input_raw).expanduser().resolve()
    output_dir = Path(output_raw).expanduser().resolve()

    if not input_dir.is_dir():
        print(f"ERROR: Input folder not found: {input_dir}")
        return

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"ERROR: Cannot create output folder: {exc}")
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
        print("No supported images found:", input_dir)
        return

    print("\n" + "=" * 72)
    print("MINERU2.5-ONLY SEQUENTIAL BATCH EXTRACTION")
    print("Endpoint:", MINERU_URL)
    print("Model:", MINERU_MODEL)
    print("Images:", len(image_files))
    print("Output folder:", output_dir)
    print("Output: one flat-key filename.json per image")
    print("Other models: disabled / not used")

    succeeded = 0
    for index, image_path in enumerate(image_files, start=1):
        if process_image(image_path, input_dir, output_dir, index, len(image_files)):
            succeeded += 1

    print("\n" + "=" * 72)
    print("BATCH FINISHED")
    print("Total images:", len(image_files))
    print("Images with recognized text:", succeeded)
    print("Images with errors or no recognized text:", len(image_files) - succeeded)
    print("Results:", output_dir)


if __name__ == "__main__":
    main()

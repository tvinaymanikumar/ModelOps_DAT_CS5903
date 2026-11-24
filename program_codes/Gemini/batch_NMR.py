"""
Gemini-only predefined-schema document extraction.
Install: python -m pip install -U google-genai pillow
Run: python gemini_schema_batch.py
"""

import json
import os
import re
import time
from io import BytesIO
from pathlib import Path
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

from google import genai
from google.genai import types
from PIL import Image, ImageOps


# ============================================================
# 1. GEMINI CONFIGURATION
# ============================================================

API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip()


TEMPERATURE = 0.0
TOP_P = 1.0
MAX_OUTPUT_TOKENS = 3000
MAX_IMAGE_SIDE = 1600

# Wait between Gemini requests to reduce per-minute rate-limit errors.
REQUEST_DELAY_SECONDS = 6

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"
}

if not API_KEY or API_KEY == "YOUR_ACTUAL_GEMINI_API_KEY":
    raise SystemExit("ERROR: Add your actual Gemini API key to API_KEY.")

client = genai.Client(api_key=API_KEY)


# ============================================================
# 2. SCHEMA CONFIGURATION
# ============================================================

# Default schema from the reference: Nigeria medical records.
SCHEMA = {
    "name": "",
    "age": "",
    "sex": ""
}

# For a cheque dataset, replace the schema above with:
# SCHEMA = {
#     "payee_name": "",
#     "amount_in_words": "",
#     "amount": "",
#     "year": "",
#     "account_no": ""
# }

# For an invoice dataset, replace the schema above with:
# SCHEMA = {
#     "company": "",
#     "date": "",
#     "address": "",
#     "total": ""
# }


# ============================================================
# 3. IMAGE PREPROCESSING
# ============================================================

def image_to_jpeg_bytes(image_path: Path) -> bytes:
    """Orient, resize if necessary, and encode image as JPEG."""
    with Image.open(image_path) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")

    image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


# ============================================================
# 4. JSON PARSING AND FLATTENING
# ============================================================

def parse_json_response(text: str) -> dict:
    """Parse a JSON object, allowing Markdown fences or surrounding text."""
    text = (text or "").strip()

    if text.startswith("```"):
        text = "\n".join(
            line for line in text.splitlines()
            if not line.strip().startswith("```")
        ).strip()

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        result = None
        for position, character in enumerate(text):
            if character != "{":
                continue
            try:
                candidate, _ = decoder.raw_decode(text[position:])
                if isinstance(candidate, dict):
                    result = candidate
                    break
            except json.JSONDecodeError:
                continue
        if result is None:
            raise ValueError("No valid JSON object found in Gemini response.")

    if not isinstance(result, dict):
        raise ValueError("Gemini response must be a JSON object.")
    return result


def flatten_json(data: dict, prefix: str = "") -> dict:
    """Flatten nested JSON using dot-separated keys and string values."""
    flattened = {}
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(flatten_json(value, full_key))
        elif isinstance(value, list):
            flattened[full_key] = json.dumps(value, ensure_ascii=False)
        elif value is None:
            flattened[full_key] = ""
        elif isinstance(value, bool):
            flattened[full_key] = str(value).lower()
        else:
            flattened[full_key] = str(value).strip()
    return flattened


def normalized_key(key: str) -> str:
    """Normalize key formatting without inferring semantic synonyms."""
    return re.sub(r"[^a-z0-9]", "", str(key).lower())


# ============================================================
# 5. AGE AND SEX NORMALIZATION (NIGERIA MEDICAL RECORDS)
# ============================================================

def normalize_sex(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    token = re.sub(r"[^a-z]", "", raw.lower())
    if token in {"m", "male"}:
        return "male"
    if token in {"f", "female", "fem"}:
        return "female"
    return raw


def normalize_age(value: str) -> str:
    """Normalize ages to '<number> years' or '<number> months'."""
    raw = str(value or "").strip()
    if not raw:
        return ""

    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(.*?)\s*", raw,
                         flags=re.IGNORECASE)
    if not match:
        return raw

    number = match.group(1)
    unit_text = re.sub(r"\bold\b", "", match.group(2).lower().strip())
    unit = re.sub(r"[\s./_\-]+", "", unit_text)

    if not unit:
        return f"{number} years"

    year_units = {"y", "yr", "yrs", "year", "years", "yo", "yro"}
    month_units = {
        "m", "mo", "mos", "mon", "mons", "month", "months",
        "mth", "mths", "monthold", "monthsold"
    }

    if unit in year_units:
        return f"{number} years"
    if unit in month_units or "month" in unit or "mon" in unit:
        return f"{number} months"
    return raw


def normalize_medical_fields(fields: dict) -> dict:
    """Normalize age/sex, including compact values such as '12/f'."""
    output = dict(fields or {})
    age = str(output.get("age", "") or "").strip()
    sex = str(output.get("sex", "") or "").strip()

    match = re.fullmatch(
        r"\s*(\d+(?:\.\d+)?\s*(?:(?:years?|yrs?|y|months?|mons?|mon|mos?|m))?)"
        r"\s*[/|]\s*(m|male|f|female|fem)\s*",
        age,
        flags=re.IGNORECASE,
    )
    if match:
        age = match.group(1).strip()
        if not sex:
            sex = match.group(2)

    output["age"] = normalize_age(age)
    output["sex"] = normalize_sex(sex)
    return output


# ============================================================
# 6. SCHEMA ENFORCEMENT
# ============================================================

def enforce_schema(fields: dict) -> dict:
    """Return exactly the configured keys, filling missing values with ''."""
    fields = fields or {}
    lookup = {}
    for key, value in fields.items():
        lookup.setdefault(normalized_key(key), value)

    output = {}
    for schema_key in SCHEMA:
        if schema_key in fields:
            value = fields[schema_key]
        else:
            value = lookup.get(normalized_key(schema_key), "")

        if value is None:
            value = ""
        elif isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        elif isinstance(value, bool):
            value = str(value).lower()
        else:
            value = str(value).strip()
        output[schema_key] = value

    keys = {normalized_key(key) for key in SCHEMA}
    if {"age", "sex"}.issubset(keys):
        output = normalize_medical_fields(output)

    return {key: output.get(key, "") for key in SCHEMA}


# ============================================================
# 7. BUILD SCHEMA-AWARE PROMPT
# ============================================================

def build_prompt() -> str:
    schema_json = json.dumps(SCHEMA, ensure_ascii=False, indent=2)

    prompt = f"""
You are a careful document information extraction system.
Extract values directly from the original document image, which may
contain printed text, handwriting, or both.

REQUIRED PREDEFINED SCHEMA:
{schema_json}

Rules:
1. Return exactly the schema keys shown above, with identical spelling.
2. Do not add any keys outside the schema.
3. The original image is the source of truth.
4. Extract only values supported by the image.
5. If a field is blank, absent, or unreadable, return "".
6. Never guess or fabricate a value.
7. Every value must be a string; do not use null, numbers, or booleans.
8. Return one valid JSON object only, with no Markdown or explanation.
"""
    keys = {normalized_key(key) for key in SCHEMA}
    if {"age", "sex"}.issubset(keys):
        prompt += """
Age and sex normalization:
- M/m -> "male"; F/f -> "female"; also normalize "male", "female", "fem".
- Format ages as "<number> years" or "<number> months".
- 5, 5 y, 5 yr, 5 year, 5 years -> "5 years".
- 5 m, 5 mo, 5 mon, 5 month, 5 months -> "5 months".
- A numeric age without a unit means years for this dataset.
- If age and sex are combined, e.g. "12/f", return age "12 years"
  and sex "female", provided this interpretation is supported by the image.
"""
    return prompt


# ============================================================
# 8. GEMINI EXTRACTION
# ============================================================

def extract_with_gemini(image_bytes: bytes) -> dict:
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            build_prompt(),
        ],
        config=types.GenerateContentConfig(
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            response_mime_type="application/json",
        ),
    )

    if not response.text:
        raise ValueError(
            "Gemini returned no text. Check model name, API key, quota, "
            "and response finish reason."
        )

    parsed = parse_json_response(response.text)
    return enforce_schema(flatten_json(parsed))


# ============================================================
# 9. SAVE JSON
# ============================================================

def save_json(file_path: Path, data: dict) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)
        file.write("\n")


def is_inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


# ============================================================
# 10. PROCESS ONE IMAGE: SAVE ONLY filename.json
# ============================================================

def process_one_image(image_path: Path, output_dir: Path,
                      index: int, total: int) -> bool:
    json_path = output_dir / f"{image_path.stem}.json"

    print("\n" + "=" * 72)
    print(f"IMAGE {index}/{total}: {image_path.name}")
    print("=" * 72)

    try:
        image_bytes = image_to_jpeg_bytes(image_path)
        fields = extract_with_gemini(image_bytes)
        save_json(json_path, fields)

        print("\nEXTRACTED JSON:")
        print(json.dumps(fields, indent=4, ensure_ascii=False))
        print("\nJSON saved to:", json_path)
        return True

    except Exception as error:
        # Print errors to the terminal only; do not create error/raw files.
        print(f"\nFAILED ({type(error).__name__}): {error}")
        return False


# ============================================================
# 11. SEQUENTIAL BATCH PROCESSING
# ============================================================

def main() -> None:
    print("=" * 72)
    print("GEMINI PREDEFINED-SCHEMA DOCUMENT EXTRACTION")
    print("=" * 72)
    print("API key configured:", bool(API_KEY))
    print("Gemini model:", GEMINI_MODEL)
    print("Schema keys:", list(SCHEMA.keys()))
    print("Temperature:", TEMPERATURE)
    print("Processing: sequential, one image at a time")

    input_value = input("\nEnter input image folder path: ").strip().strip('"').strip("'")
    output_value = input("Enter result folder path: ").strip().strip('"').strip("'")

    input_dir = Path(input_value).expanduser().resolve()
    output_dir = Path(output_value).expanduser().resolve()

    if not input_dir.is_dir():
        print("ERROR: Input folder not found:", input_dir)
        return

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        print("ERROR: Cannot create output folder:", error)
        return

    image_files = sorted(
        (
            path for path in input_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
            and not (
                output_dir != input_dir
                and is_inside(output_dir, input_dir)
                and is_inside(path, output_dir)
            )
        ),
        key=lambda path: str(path.relative_to(input_dir)).lower(),
    )

    if not image_files:
        print("No supported images found in:", input_dir)
        return

    # All JSON files go directly into one output folder, so prevent collisions.
    name_counts = {}
    for image_path in image_files:
        stem = image_path.stem.casefold()
        name_counts[stem] = name_counts.get(stem, 0) + 1

    duplicates = sorted(name for name, count in name_counts.items() if count > 1)
    if duplicates:
        print("ERROR: Duplicate image filenames could overwrite output JSON files.")
        print("Duplicate names:", ", ".join(duplicates))
        print("Rename them or process those folders separately.")
        return

    print("\nInput folder:", input_dir)
    print("Output folder:", output_dir)
    print("Images found:", len(image_files))
    print("Required schema keys:", list(SCHEMA.keys()))
    print("=" * 72)

    successes = 0
    failures = 0

    for index, image_path in enumerate(image_files, start=1):
        if process_one_image(image_path, output_dir, index, len(image_files)):
            successes += 1
        else:
            failures += 1

        # Pause between requests (not after the final image).
        if index < len(image_files):
            print(f"\nWaiting {REQUEST_DELAY_SECONDS} seconds before the next image...")
            time.sleep(REQUEST_DELAY_SECONDS)

    print("\n" + "=" * 72)
    print("BATCH PROCESSING FINISHED")
    print("=" * 72)
    print("Total images:", len(image_files))
    print("Successful:", successes)
    print("Failed:", failures)
    print("Result folder:", output_dir)
    print("=" * 72)


if __name__ == "__main__":
    main()

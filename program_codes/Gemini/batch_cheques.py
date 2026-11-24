"""
Gemini-only batch document extraction with optional predefined schema.

Install:
    python -m pip install -U google-genai pillow

Run:
    python gemini_api_batch_with_timer.py

Security:
    Replace YOUR_GEMINI_API_KEY with your own key locally.
    Do not share your API key or commit this file with the real key.
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

# Wait before each Gemini API request to reduce request frequency.
REQUEST_DELAY_SECONDS = 6

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"
}

if not API_KEY or API_KEY == "YOUR_GEMINI_API_KEY":
    raise SystemExit(
        "ERROR: Replace YOUR_GEMINI_API_KEY with your Gemini API key "
        "inside this script."
    )

client = genai.Client(api_key=API_KEY)


# ============================================================
# 2. SCHEMA CONFIGURATION
# ============================================================

# Active example: cheque dataset.
# All fields are strings except "amount", which is converted to an integer
# when a readable amount is present. Blank/unreadable amount remains "".
SCHEMA = {
    "payee_name": "",
    "amount_in_words": "",
    "amount": "",
    "year": "",
    "account_no": ""
}

# For schema-free extraction, change the line above to:
# SCHEMA = None

# Example invoice schema:
# SCHEMA = {
#     "company": "",
#     "date": "",
#     "address": "",
#     "total": ""
# }

# Example Nigeria medical-record schema:
# SCHEMA = {"name": "", "age": "", "sex": ""}


# ============================================================
# 3. IMAGE PREPROCESSING
# ============================================================

def image_to_jpeg_bytes(image_path: Path) -> bytes:
    """Orient, resize if necessary, and encode the image as JPEG."""
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
    """Parse a JSON object, including one wrapped in Markdown fences."""
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
    """Flatten nested JSON using dot-separated keys and scalar strings."""
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
    """Ignore case and formatting differences when matching schema keys."""
    return re.sub(r"[^a-z0-9]", "", str(key).lower())


def enforce_schema(fields: dict) -> dict:
    """Return exactly the configured schema keys; fill missing values with ''."""
    if SCHEMA is None:
        return fields

    fields = fields or {}
    normalized_lookup = {}
    for key, value in fields.items():
        normalized_lookup.setdefault(normalized_key(key), value)

    output = {}
    for schema_key in SCHEMA:
        if schema_key in fields:
            value = fields[schema_key]
        else:
            value = normalized_lookup.get(normalized_key(schema_key), "")

        if value is None:
            value = ""
        elif isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        elif isinstance(value, bool):
            value = str(value).lower()
        else:
            value = str(value).strip()

        output[schema_key] = value

    return output


# ============================================================
# 5. OPTIONAL DATASET-SPECIFIC NORMALIZATION
# ============================================================

def is_cheque_schema() -> bool:
    if SCHEMA is None:
        return False
    required = {"payee_name", "amount_in_words", "amount", "account_no"}
    return required.issubset(set(SCHEMA.keys()))


def normalize_cheque_amount(value):
    """Convert e.g. '1,80,000/-' to 180000; keep blanks as ''."""
    if value is None:
        return ""
    raw = str(value).strip()
    if not raw:
        return ""

    digits = re.sub(r"[^0-9]", "", raw)
    return int(digits) if digits else ""


def normalize_medical_fields(fields: dict) -> dict:
    """Normalize age and sex for the optional name/age/sex schema."""
    output = dict(fields)
    age = str(output.get("age", "") or "").strip()
    sex = str(output.get("sex", "") or "").strip()

    combined = re.fullmatch(
        r"\s*(\d+(?:\.\d+)?\s*(?:years?|yrs?|y|months?|mons?|mon|mos?|m)?)"
        r"\s*[/|]\s*(m|male|f|female|fem)\s*",
        age,
        flags=re.IGNORECASE,
    )
    if combined:
        age = combined.group(1).strip()
        if not sex:
            sex = combined.group(2)

    if age:
        match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(.*?)\s*", age,
                             flags=re.IGNORECASE)
        if match:
            number = match.group(1)
            unit = re.sub(r"[^a-z]", "", match.group(2).lower())
            if not unit or unit in {"y", "yr", "yrs", "year", "years", "yo"}:
                age = f"{number} years"
            elif unit in {
                "m", "mo", "mos", "mon", "mons", "month", "months", "mth", "mths"
            } or "month" in unit or "mon" in unit:
                age = f"{number} months"

    sex_token = re.sub(r"[^a-z]", "", sex.lower())
    if sex_token in {"m", "male"}:
        sex = "male"
    elif sex_token in {"f", "female", "fem"}:
        sex = "female"

    output["age"] = age
    output["sex"] = sex
    return output


def normalize_output_fields(fields: dict) -> dict:
    normalized = enforce_schema(fields or {})

    if is_cheque_schema():
        normalized["amount"] = normalize_cheque_amount(
            normalized.get("amount", "")
        )

    if SCHEMA is not None and {"name", "age", "sex"}.issubset(SCHEMA.keys()):
        normalized = normalize_medical_fields(normalized)

    return normalized


# ============================================================
# 6. BUILD PROMPT
# ============================================================

def build_prompt() -> str:
    base_prompt = """
You are a careful document information extraction system.
Read the original document image, which may contain printed text,
handwriting, or both.

Rules:
1. Use the image as the primary source of truth.
2. Extract only information supported by visible content.
3. Never guess or fabricate values.
4. Include labelled fields even if their values are blank.
5. Use an empty string "" for blank, absent, or unreadable fields.
6. Return one valid JSON object only, without Markdown or explanation.
"""

    if SCHEMA is None:
        return base_prompt + """
No predefined schema is supplied. Discover meaningful fields from the image.
Use visible labels as descriptive snake_case keys. Include labelled blank
fields with value "". Return a flat JSON object where possible. Use strings
for scalar values and do not add unsupported fields.
"""

    prompt = base_prompt + f"""
REQUIRED PREDEFINED SCHEMA:
{json.dumps(SCHEMA, ensure_ascii=False, indent=2)}

Return exactly these keys, with identical spelling. Do not add other keys.
Every value must be a string, except for the cheque schema's amount field,
which must be an integer when readable and "" when blank or unreadable.
Use "" for any other blank, absent, or unreadable field.
"""

    if is_cheque_schema():
        prompt += """
Cheque amount rule: remove commas, currency symbols, spaces, and trailing
notation such as /- or \\- from the numeric amount. Example: "1,80,000/-"
becomes the JSON integer 180000. Do not infer an amount from amount-in-words.
"""
    if {"name", "age", "sex"}.issubset(SCHEMA.keys()):
        prompt += """
For age/sex: normalize M to "male" and F to "female". Format ages as
"<number> years" or "<number> months". A numeric age without a unit means
years for this dataset. For a combined value such as "12/f", return
age "12 years" and sex "female" if supported by the image.
"""
    return prompt


# ============================================================
# 7. GEMINI EXTRACTION WITH TIMER
# ============================================================

def extract_with_gemini(image_bytes: bytes) -> dict:
    """Wait before calling Gemini, then return normalized extracted fields."""
    if REQUEST_DELAY_SECONDS > 0:
        print(f"Waiting {REQUEST_DELAY_SECONDS} seconds before Gemini request...")
        time.sleep(REQUEST_DELAY_SECONDS)

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
            "Gemini returned no text. Check the model name, API key, quota, "
            "and response finish reason."
        )

    parsed = parse_json_response(response.text)
    flattened = flatten_json(parsed)
    return normalize_output_fields(flattened)


# ============================================================
# 8. SAVE JSON
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
# 9. PROCESS ONE IMAGE
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
        # Errors are printed only; no extra error/raw-response files are saved.
        print(f"\nFAILED ({type(error).__name__}): {error}")
        return False


# ============================================================
# 10. SEQUENTIAL BATCH PROCESSING
# ============================================================

def main() -> None:
    print("=" * 72)
    print("GEMINI API BATCH DOCUMENT EXTRACTION")
    print("=" * 72)
    print("Gemini model:", GEMINI_MODEL)
    print("Timer between requests:", REQUEST_DELAY_SECONDS, "seconds")
    print("Schema mode:", "PREDEFINED" if SCHEMA is not None else "SCHEMA-FREE")
    if SCHEMA is not None:
        print("Schema keys:", list(SCHEMA.keys()))

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

    # JSONs are placed in one folder, so duplicate stems could overwrite.
    name_counts = {}
    for image_path in image_files:
        stem = image_path.stem.casefold()
        name_counts[stem] = name_counts.get(stem, 0) + 1

    duplicates = sorted(name for name, count in name_counts.items() if count > 1)
    if duplicates:
        print("ERROR: Duplicate image filenames could overwrite JSON outputs.")
        print("Duplicate names:", ", ".join(duplicates))
        print("Rename them or process the folders separately.")
        return

    print("\nInput folder:", input_dir)
    print("Output folder:", output_dir)
    print("Images found:", len(image_files))
    print("=" * 72)

    successes = 0
    failures = 0

    for index, image_path in enumerate(image_files, start=1):
        if process_one_image(image_path, output_dir, index, len(image_files)):
            successes += 1
        else:
            failures += 1

    print("\n" + "=" * 72)
    print("BATCH PROCESSING FINISHED")
    print("=" * 72)
    print("Total images:", len(image_files))
    print("Successful:", successes)
    print("Failed:", failures)
    print("Result folder:", output_dir)


if __name__ == "__main__":
    main()

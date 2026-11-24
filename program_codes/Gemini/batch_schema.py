
"""
Gemini-only predefined-schema document extraction.
Processes images sequentially and saves only filename.json.

Install:
    python -m pip install -U google-genai pillow

Run:
    python gemini_schema_batch.py
"""

import json
import os
import re
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

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".webp",
    ".bmp", ".tif", ".tiff"
}

if not API_KEY or API_KEY == "YOUR_ACTUAL_GEMINI_API_KEY":
    raise SystemExit(
        "ERROR: Replace YOUR_ACTUAL_GEMINI_API_KEY "
        "with your actual Gemini API key."
    )

try:
    client = genai.Client(api_key=API_KEY)
except Exception as error:
    raise SystemExit(f"ERROR initializing Gemini: {error}")


# ============================================================
# 2. PREDEFINED SCHEMA
# ============================================================

# Select ONE schema for the dataset you are processing.
# Every schema value must be an empty string.

# INVOICE DATASET
SCHEMA = {
    "company": "",
    "date": "",
    "address": "",
    "total": ""
}

# CHEQUE DATASET
# To use this schema, comment out the invoice schema above
# and uncomment the following:
#
# SCHEMA = {
#     "payee_name": "",
#     "amount_in_words": "",
#     "amount": "",
#     "year": "",
#     "account_no": ""
# }

# Example for another dataset:
#
# SCHEMA = {
#     "patient_name": "",
#     "age": "",
#     "sex": "",
#     "date": ""
# }


# ============================================================
# 3. EXTRACTION PROMPT
# ============================================================

def build_prompt() -> str:
    """Build a prompt using the currently selected schema."""

    schema_json = json.dumps(
        SCHEMA,
        indent=4,
        ensure_ascii=False
    )

    return f"""
You are a careful document information extraction system.

Extract the required information from the supplied document image.
The document may contain printed text, handwriting, or both.

REQUIRED SCHEMA:
{schema_json}

STRICT INSTRUCTIONS:

1. Return exactly the keys shown in the required schema.
2. Preserve the spelling of every schema key exactly.
3. Do not add any additional keys.
4. Extract each value from the original image.
5. Use an empty string "" if a field is blank, absent, or unreadable.
6. Never guess or invent a value.
7. Preserve names, dates, numbers, and identifiers as written.
8. Every value must be a string.
9. Do not return null, numbers, booleans, lists, or nested objects.
10. Do not include explanations or Markdown.
11. Return one valid JSON object only.

The original document image is the source of truth.

Example output:
{schema_json}
"""


# ============================================================
# 4. IMAGE PREPROCESSING
# ============================================================

def image_to_jpeg_bytes(image_path: Path) -> bytes:
    """Correct orientation, resize if necessary, and encode as JPEG."""

    with Image.open(image_path) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")

    # Preserve aspect ratio; do not enlarge small images.
    image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))

    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=95)

    return buffer.getvalue()


# ============================================================
# 5. PARSE GEMINI RESPONSE
# ============================================================

def parse_json_response(response_text: str) -> dict:
    """Parse a JSON object from Gemini's response."""

    text = (response_text or "").strip()

    # Remove Markdown code fences if present.
    if text.startswith("```"):
        text = "\n".join(
            line for line in text.splitlines()
            if not line.strip().startswith("```")
        ).strip()

    try:
        result = json.loads(text)

    except json.JSONDecodeError:
        # Find a JSON object if extra text surrounds the response.
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
            raise ValueError(
                "No valid JSON object found in Gemini response."
            )

    if not isinstance(result, dict):
        raise ValueError(
            "Expected Gemini to return a JSON object."
        )

    return result


# ============================================================
# 6. NORMALIZE FIELD KEYS
# ============================================================

def normalized_key(key: str) -> str:
    """
    Normalize key formatting for matching.
    Example: Account No. -> accountno
    """

    return re.sub(r"[^a-z0-9]", "", str(key).lower())


# ============================================================
# 7. ENFORCE THE PREDEFINED SCHEMA
# ============================================================

def enforce_schema(fields: dict) -> dict:
    """
    Return exactly the required schema keys.
    Missing, blank, and unreadable fields become "".
    All output values are strings.
    """

    fields = fields or {}

    normalized_lookup = {}

    for key, value in fields.items():
        normalized_lookup.setdefault(
            normalized_key(key), value
        )

    output = {}

    for schema_key in SCHEMA:
        if schema_key in fields:
            value = fields[schema_key]
        else:
            value = normalized_lookup.get(
                normalized_key(schema_key), ""
            )

        if value is None:
            value = ""

        elif isinstance(value, (dict, list)):
            value = json.dumps(
                value, ensure_ascii=False
            )

        elif isinstance(value, bool):
            value = str(value).lower()

        else:
            value = str(value).strip()

        output[schema_key] = value

    return output


# ============================================================
# 8. CALL GEMINI
# ============================================================

def extract_with_gemini(image_bytes: bytes) -> dict:
    """Send one image to Gemini and enforce the selected schema."""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/jpeg"
            ),
            build_prompt(),
        ],
        config=types.GenerateContentConfig(
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            response_mime_type="application/json",
        ),
    )

    raw_text = response.text

    if not raw_text:
        raise ValueError(
            "Gemini returned no text. Check the model name, "
            "API key, quota, and response finish reason."
        )

    parsed_json = parse_json_response(raw_text)

    # Enforce the schema locally, even if Gemini returns extra keys.
    return enforce_schema(parsed_json)


# ============================================================
# 9. SAVE JSON OUTPUT
# ============================================================

def save_json(file_path: Path, data: dict) -> None:
    """Save the final JSON output."""

    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False
        )
        file.write("\n")


# ============================================================
# 10. PATH HELPER
# ============================================================

def is_inside(path: Path, parent: Path) -> bool:
    """Check whether a path is inside a directory."""

    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


# ============================================================
# 11. PROCESS ONE IMAGE
# ============================================================

def process_one_image(
    image_path: Path,
    output_dir: Path,
    index: int,
    total: int,
) -> bool:

    name = image_path.stem

    # Save only filename.json.
    json_path = output_dir / f"{name}.json"

    print("\n" + "=" * 72)
    print(f"IMAGE {index}/{total}: {image_path.name}")
    print("=" * 72)

    try:
        image_bytes = image_to_jpeg_bytes(image_path)

        fields = extract_with_gemini(image_bytes)

        save_json(json_path, fields)

        print("\nEXTRACTED JSON:")
        print(json.dumps(
            fields,
            indent=4,
            ensure_ascii=False
        ))

        print("\nSaved to:", json_path)

        return True

    except Exception as error:
        # Print errors to the terminal only.
        # No error-log or raw-response files are created.
        print(
            f"\nFAILED ({type(error).__name__}): {error}"
        )
        return False


# ============================================================
# 12. BATCH PROCESSING
# ============================================================

def main() -> None:

    print("=" * 72)
    print("GEMINI PREDEFINED-SCHEMA EXTRACTION")
    print("=" * 72)

    print("API key configured:", bool(API_KEY))
    print("Gemini model:", GEMINI_MODEL)
    print("Schema keys:", list(SCHEMA.keys()))
    print("Temperature:", TEMPERATURE)
    print("Processing: sequential, one image at a time")

    input_value = input(
        "\nEnter input image folder path: "
    ).strip().strip('"').strip("'")

    output_value = input(
        "Enter result folder path: "
    ).strip().strip('"').strip("'")

    input_dir = Path(input_value).expanduser().resolve()
    output_dir = Path(output_value).expanduser().resolve()

    if not input_dir.is_dir():
        print("\nERROR: Input folder not found:", input_dir)
        return

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        print("\nERROR: Cannot create output folder:", error)
        return

    # Discover images recursively.
    # Exclude existing files inside the output directory.
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
        key=lambda path: str(
            path.relative_to(input_dir)
        ).lower(),
    )

    if not image_files:
        print("\nNo supported images found:", input_dir)
        return

    # All outputs are saved directly into one folder as filename.json.
    # Stop if duplicate names could overwrite one another.
    name_counts = {}

    for image_path in image_files:
        key = image_path.stem.casefold()
        name_counts[key] = name_counts.get(key, 0) + 1

    duplicates = sorted(
        name for name, count in name_counts.items()
        if count > 1
    )

    if duplicates:
        print(
            "\nERROR: Duplicate image filenames detected. "
            "Saving filename.json would overwrite results."
        )
        print("Duplicate names:", ", ".join(duplicates))
        print(
            "Rename the duplicate images or process their "
            "folders separately."
        )
        return

    print("\nInput folder:", input_dir)
    print("Output folder:", output_dir)
    print("Images found:", len(image_files))
    print("=" * 72)

    successes = 0
    failures = 0

    # Process each image before moving to the next.
    for index, image_path in enumerate(image_files, start=1):

        success = process_one_image(
            image_path=image_path,
            output_dir=output_dir,
            index=index,
            total=len(image_files),
        )

        if success:
            successes += 1
        else:
            failures += 1

    print("\n" + "=" * 72)
    print("BATCH PROCESSING FINISHED")
    print("=" * 72)
    print("Total images:", len(image_files))
    print("Successful:", successes)
    print("Failed:", failures)
    print("Output folder:", output_dir)
    print("=" * 72)


if __name__ == "__main__":
    main()

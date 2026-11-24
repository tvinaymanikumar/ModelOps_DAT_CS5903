
"""
Gemini-only schema-free document extraction with sequential batch processing.

Install dependencies:
    python -m pip install -U google-genai pillow

Run:
    python noschema_batch.py
"""

import json
import os
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
# 1. CONFIGURATION: API KEY AND MODEL
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
    raise SystemExit(
        f"ERROR: Could not initialize Gemini client: {error}"
    )


# ============================================================
# 2. SCHEMA-FREE EXTRACTION PROMPT
# ============================================================

PROMPT = """
You are a careful document information extraction system.

Extract information directly from the supplied document image.
The document may contain printed text, handwriting, or both.

Do not assume or classify a particular document type.
Do not use a predefined schema.

Instructions:
1. Identify every meaningful field label visible in the image.
2. Extract the corresponding value for each field.
3. Include identifiable fields even when their values are blank.
4. For blank fields, return an empty string "".
5. For unreadable or uncertain values, return "" rather than guessing.
6. Use descriptive snake_case keys in one flat JSON object.
7. Return every value as a string.
8. Preserve names, dates, numbers, and identifiers as written.
9. Do not invent fields or values unsupported by the image.
10. Do not omit an identifiable field just because its value is blank.
11. Do not include explanatory text or Markdown.
12. Return only a valid JSON object.

Example:
{
    "full_name": "John Smith",
    "date_of_birth": "",
    "address": "Example Street",
    "policy_number": ""
}
"""


# ============================================================
# 3. IMAGE PREPROCESSING
# ============================================================

def image_to_jpeg_bytes(image_path: Path) -> bytes:
    """Correct image orientation and convert the image to JPEG."""

    with Image.open(image_path) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")

    image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))

    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=95)

    return buffer.getvalue()


# ============================================================
# 4. PARSE GEMINI JSON RESPONSE
# ============================================================

def parse_json_response(response_text: str) -> dict:
    """Parse the response and require a JSON object."""

    text = (response_text or "").strip()

    # Remove Markdown code fences if present.
    if text.startswith("```"):
        lines = [
            line for line in text.splitlines()
            if not line.strip().startswith("```")
        ]
        text = "\n".join(lines).strip()

    try:
        result = json.loads(text)

    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end < start:
            raise ValueError(
                "No JSON object found in Gemini response."
            )

        result = json.loads(text[start:end + 1])

    if not isinstance(result, dict):
        raise ValueError(
            "Gemini response must be a JSON object."
        )

    return result


# ============================================================
# 5. FLATTEN JSON AND CONVERT VALUES TO STRINGS
# ============================================================

def flatten_json(data: dict, prefix: str = "") -> dict:
    """Flatten nested objects and convert values to strings."""

    flattened = {}

    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)

        if isinstance(value, dict):
            flattened.update(
                flatten_json(value, full_key)
            )

        elif isinstance(value, list):
            flattened[full_key] = json.dumps(
                value, ensure_ascii=False
            )

        elif value is None:
            flattened[full_key] = ""

        elif isinstance(value, bool):
            flattened[full_key] = str(value).lower()

        else:
            flattened[full_key] = str(value).strip()

    return flattened


# ============================================================
# 6. CALL GEMINI FOR ONE IMAGE
# ============================================================

def extract_with_gemini(image_bytes: bytes) -> dict:
    """Send one image to Gemini and return extracted fields."""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/jpeg"
            ),
            PROMPT,
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

    return flatten_json(parsed_json)


# ============================================================
# 7. SAVE JSON
# ============================================================

def save_json(path: Path, data: dict) -> None:
    """Save extracted fields as a JSON file."""

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False
        )


# ============================================================
# 8. PATH HELPER
# ============================================================

def is_inside(path: Path, parent: Path) -> bool:
    """Return True if path is inside parent."""

    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


# ============================================================
# 9. PROCESS ONE IMAGE
# ============================================================

def process_one_image(
    image_path: Path,
    output_dir: Path,
    index: int,
    total: int,
) -> bool:
    """Process one image and save only filename.json."""

    name = image_path.stem

    # Output contains only the original image filename plus .json.
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

        print("\nJSON saved to:", json_path)

        return True

    except Exception as error:
        # Print the error but do not create an error-log file.
        print(
            f"\nFAILED ({type(error).__name__}): {error}"
        )
        return False


# ============================================================
# 10. BATCH PROCESSING
# ============================================================

def main() -> None:

    print("=" * 72)
    print("GEMINI-ONLY SCHEMA-FREE DOCUMENT EXTRACTION")
    print("=" * 72)

    print("API key configured:", bool(API_KEY))
    print("Gemini model:", GEMINI_MODEL)
    print("Temperature:", TEMPERATURE)
    print("Processing: one image at a time")

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
        print("\nERROR: Cannot create result folder:", error)
        return

    # Find supported images, including images in subfolders.
    # Skip the output folder if it is inside the input folder.
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
        print("\nNo supported images found in:", input_dir)
        return

    # Because output filenames must be filename.json, detect
    # duplicate image names before processing to avoid overwriting.
    output_names = [path.stem.lower() for path in image_files]
    duplicate_names = {
        name for name in output_names
        if output_names.count(name) > 1
    }

    if duplicate_names:
        print(
            "\nERROR: Duplicate image filenames were found "
            "in the input folder or its subfolders."
        )
        print(
            "Saving only filename.json would overwrite files "
            "with identical names."
        )
        print("Duplicate names:", ", ".join(sorted(duplicate_names)))
        print(
            "Please rename the duplicate images or separate "
            "them into different input batches."
        )
        return

    print("\nInput folder:", input_dir)
    print("Output folder:", output_dir)
    print("Images found:", len(image_files))
    print("=" * 72)

    successes = 0
    failures = 0

    # Sequential processing: one image per request.
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
    print("Results saved in:", output_dir)
    print("=" * 72)


if __name__ == "__main__":
    main()

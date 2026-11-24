import base64
import json
from io import BytesIO
from pathlib import Path
import re

from PIL import Image, ImageOps
from openai import OpenAI


# ============================================================
# 1. REMOTE MODEL CONFIGURATION
# ============================================================

PADDLE_URL = "http://10.68.0.39:9006/v1"
PADDLE_MODEL = "paddleocr-vl"

QWEN_URL = "http://10.68.0.39:9005/v1"
QWEN_MODEL = "qwen2.5-vl-7b"

TEMPERATURE = 0.0
TOP_P = 1.0
MAX_TOKENS = 3000
TIMEOUT = 180
MAX_IMAGE_SIDE = 1600

# ------------------------------------------------------------
# SCHEMA CONFIGURATION
# ------------------------------------------------------------
# For a dataset WITH a predefined schema, list its required keys.
# Every output value will be a string. Missing/blank/unreadable
# values will be represented as "".
#
# Default: schema-free extraction. Set this to a dictionary for a
# dataset WITH a predefined schema.
# SCHEMA = None


# schema for invoice data sets
SCHEMA = {
  "company": "",
  "date": "",
  "address": "",
  "total": ""
}

# Example: uncomment and customize for a cheque dataset.
# SCHEMA = {
#     "payee_name": "",
#     "amount_in_words": "",
#     "amount": "",
#     "year": "",
#     "account_no": "",
# }

# When SCHEMA = None, the model discovers field names from the image.
#
# IMPORTANT: Use one schema per dataset. Change SCHEMA before
# running the script on a different dataset.

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".webp",
    ".bmp", ".tif", ".tiff"
}

paddle_client = OpenAI(
    base_url=PADDLE_URL,
    api_key="not-needed",
    timeout=TIMEOUT,
)

qwen_client = OpenAI(
    base_url=QWEN_URL,
    api_key="not-needed",
    timeout=TIMEOUT,
)


# ============================================================
# 2. IMAGE PREPROCESSING
# ============================================================

def image_to_base64(image_path):
    """Orient, resize if needed, and encode an image as JPEG Base64."""
    with Image.open(image_path) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")

    # thumbnail() preserves the aspect ratio and does not enlarge images.
    image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))

    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ============================================================
# 3. COMMON VISION API CALL
# ============================================================

def get_response_text(content):
    """Convert common OpenAI-compatible response content to text."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
            else:
                text = getattr(item, "text", None)
            if text is not None:
                parts.append(str(text))
        return "\n".join(parts)

    return "" if content is None else str(content)


def call_vision_model(client, model, prompt, image_data, image_first=False):
    """Call a remote OpenAI-compatible vision endpoint."""
    image_block = {
        "type": "image_url",
        "image_url": {
            "url": "data:image/jpeg;base64," + image_data,
        },
    }

    text_block = {
        "type": "text",
        "text": prompt,
    }

    content = (
        [image_block, text_block]
        if image_first
        else [text_block, image_block]
    )

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
    )

    return get_response_text(response.choices[0].message.content)


# ============================================================
# 4. APPROACH 1: PADDLEOCR ONLY
# ============================================================

def run_paddle_only(image_data):
    """Return the raw recognized text from the PaddleOCR endpoint."""
    return call_vision_model(
        client=paddle_client,
        model=PADDLE_MODEL,
        prompt="OCR:",
        image_data=image_data,
        image_first=True,
    )


def is_unusable_ocr(text):
    if not text or not text.strip():
        return True

    compact = "".join(text.split())
    return bool(compact) and set(compact) <= {"+"}


# ============================================================
# 5. JSON PARSING, FLATTENING, AND SCHEMA ENFORCEMENT
# ============================================================

def parse_json_response(text):
    """Parse a JSON object, allowing surrounding Markdown fences/text."""
    text = (text or "").strip()

    if text.startswith("```"):
        lines = [
            line for line in text.splitlines()
            if not line.strip().startswith("```")
        ]
        text = "\n".join(lines).strip()

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        # Locate a complete JSON object within surrounding text.
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
            raise ValueError("No valid JSON object found in model response.")

    if not isinstance(result, dict):
        raise ValueError("Expected the model response to be a JSON object.")

    return result


def flatten_json(data, prefix=""):
    """Flatten nested JSON into dot-separated keys and string values."""
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


def normalized_key(key):
    """Normalize formatting differences in keys, not their meanings."""
    return re.sub(r"[^a-z0-9]", "", str(key).lower())


def enforce_schema(fields):
    """Keep exactly the schema keys and use empty strings for missing ones.

    Key matching is case-insensitive and ignores spaces, hyphens, and
    underscores. It does not attempt to infer semantic synonyms.
    """
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


def add_schema_instructions(base_prompt):
    """Append output instructions for the selected dataset schema."""
    if SCHEMA is None:
        return base_prompt + """

        No predefined schema is supplied. Discover meaningful fields
        from the image and use visible labels as descriptive snake_case
        JSON keys. Include labelled blank fields with value "". Do not
        invent fields or values. Use a flat JSON object and make all
        scalar values strings. Return valid JSON only.
        """

    return base_prompt + f"""

    REQUIRED PREDEFINED SCHEMA:
    {json.dumps(SCHEMA, ensure_ascii=False, indent=2)}

    Return exactly these schema keys, with exactly the same spelling.
    Do not add keys outside the schema. Every value must be a string.
    If a field is blank, absent, or unreadable, return an empty string "".
    Do not guess or fabricate values. Return one valid JSON object only.
    """


def ensure_candidate_keys(final_fields, qwen_candidate):
    """Apply the schema, or preserve Qwen-only keys if no schema exists."""
    if SCHEMA is not None:
        return enforce_schema(final_fields)

    completed = dict(final_fields or {})
    for key in (qwen_candidate or {}):
        completed.setdefault(key, "")
    return completed


# ============================================================
# 6. APPROACH 2: QWEN ONLY
# ============================================================

def run_qwen_only(image_data):
    prompt = """
    You are a precise document information extraction system.

    Extract information directly from the original image. The document
    may contain printed or handwritten information. Do not assume a
    particular document type unless the visible content establishes it.

    Instructions:
    1. Treat the original image as the source of truth.
    2. Include all requested fields, or all identifiable fields when no
       predefined schema is supplied.
    3. Include labelled fields even if their values are blank.
    4. For blank, absent, or unreadable values, use an empty string "".
    5. Preserve names, dates, numbers, and identifiers as written.
    6. Do not guess, invent values, or add unsupported information.
    7. Use descriptive snake_case keys when no schema is supplied.
    8. Return all scalar values as strings; do not use null.
    9. Return a flat JSON object where possible.
    10. Return valid JSON only, with no Markdown or explanation.
    """
    prompt = add_schema_instructions(prompt)

    raw_response = call_vision_model(
        client=qwen_client,
        model=QWEN_MODEL,
        prompt=prompt,
        image_data=image_data,
    )

    parsed = parse_json_response(raw_response)
    fields = flatten_json(parsed)
    fields = enforce_schema(fields)
    return fields, raw_response


# ============================================================
# 7. APPROACH 3: PADDLEOCR + QWEN RECONCILIATION
# ============================================================

def run_combined_pipeline(image_data, paddle_text, qwen_candidate):
    qwen_candidate_text = json.dumps(
        qwen_candidate or {},
        indent=2,
        ensure_ascii=False,
    )

    prompt = f"""
    You are a careful document extraction and reconciliation system.

    You have three sources:
    1. The original document image.
    2. Text recognized by PaddleOCR.
    3. An independent Qwen2.5-VL extraction, if available.

    Neither model output is guaranteed to be correct.

    PADDLEOCR CANDIDATE:
    <paddle_ocr_candidate>
    {paddle_text[:12000]}
    </paddle_ocr_candidate>

    INDEPENDENT QWEN CANDIDATE:
    <qwen_candidate>
    {qwen_candidate_text[:10000]}
    </qwen_candidate>

    IMPORTANT RULES:
    1. Examine the original image directly; it is the primary evidence.
    2. Do not automatically prefer PaddleOCR or the first Qwen output.
    3. If candidates agree and the image supports the value, retain it.
    4. If candidates disagree, inspect the visible characters and choose
       the value best supported by the original image.
    5. Never guess unreadable handwriting or invent information.
    6. Include labelled blank fields with value "".
    7. All scalar output values must be strings, not null.
    8. Return one valid JSON object only, without Markdown or explanation.
    """

    prompt = add_schema_instructions(prompt)

    raw_response = call_vision_model(
        client=qwen_client,
        model=QWEN_MODEL,
        prompt=prompt,
        image_data=image_data,
    )

    parsed = parse_json_response(raw_response)
    fields = flatten_json(parsed)
    fields = enforce_schema(fields)
    return fields, raw_response


# ============================================================
# 8. FILE-SAVING HELPERS
# ============================================================

def save_text(file_path, text):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(text or "", encoding="utf-8")


def save_json(file_path, data):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)
        file.write("\n")


def save_error(output_dir, name, stage, error):
    error_path = output_dir / f"{name}_{stage}_error.txt"
    save_text(error_path, str(error))
    print(f"Error details saved to: {error_path}")


# ============================================================
# 9. PROCESS ONE IMAGE THROUGH ALL APPROACHES
# ============================================================

def process_one_image(image_path, input_dir, output_dir, index, total):
    relative_path = image_path.relative_to(input_dir)
    image_output_dir = output_dir / relative_path.parent
    image_output_dir.mkdir(parents=True, exist_ok=True)
    name = image_path.stem

    paddle_txt = image_output_dir / f"{name}_paddle_only.txt"
    qwen_json_path = image_output_dir / f"{name}_qwen_only.json"
    qwen_raw_path = image_output_dir / f"{name}_qwen_only_raw.txt"
    pipeline_json_path = image_output_dir / f"{name}_pipeline.json"
    pipeline_raw_path = image_output_dir / f"{name}_pipeline_raw.txt"

    status = {"paddle": False, "qwen": False, "pipeline": False}

    print("\n\n" + "#" * 75)
    print(f"IMAGE {index}/{total}: {relative_path}")
    print("#" * 75)

    try:
        image_data = image_to_base64(image_path)
        print("Image preprocessing completed.")
    except Exception as error:
        print(f"Image preprocessing failed: {error}")
        save_error(image_output_dir, name, "image", error)
        return status

    paddle_text = None
    qwen_fields = None

    # --------------------------------------------------------
    # APPROACH 1: PADDLEOCR ONLY
    # --------------------------------------------------------
    print("\n" + "=" * 65)
    print(f"[{index}/{total}] PADDLEOCR ONLY")
    print("=" * 65)

    try:
        paddle_text = run_paddle_only(image_data)
        save_text(paddle_txt, paddle_text)
        print("\nPADDLEOCR OUTPUT:")
        print(paddle_text)
        print("\nSaved to:", paddle_txt)

        if is_unusable_ocr(paddle_text):
            print(
                "WARNING: PaddleOCR returned empty text or only '+' characters. "
                "The combined pipeline will be skipped."
            )
            paddle_text = None
        else:
            status["paddle"] = True
    except Exception as error:
        paddle_text = None
        print("PaddleOCR request failed:", error)
        save_error(image_output_dir, name, "paddle", error)

    # --------------------------------------------------------
    # APPROACH 2: QWEN ONLY
    # --------------------------------------------------------
    print("\n" + "=" * 65)
    print(f"[{index}/{total}] QWEN ONLY")
    print("=" * 65)

    try:
        qwen_fields, qwen_raw = run_qwen_only(image_data)
        save_text(qwen_raw_path, qwen_raw)
        save_json(qwen_json_path, qwen_fields)

        print("\nQWEN-ONLY JSON OUTPUT:")
        print(json.dumps(qwen_fields, indent=4, ensure_ascii=False))
        print("\nRaw response:", qwen_raw_path)
        print("JSON output:", qwen_json_path)
        status["qwen"] = True
    except Exception as error:
        qwen_fields = None
        print("Qwen-only extraction failed:", error)
        save_error(image_output_dir, name, "qwen_only", error)

    # --------------------------------------------------------
    # APPROACH 3: PADDLEOCR + QWEN PIPELINE
    # --------------------------------------------------------
    print("\n" + "=" * 65)
    print(f"[{index}/{total}] PADDLEOCR + QWEN PIPELINE")
    print("=" * 65)

    if paddle_text is None:
        print("Combined pipeline skipped because PaddleOCR returned unusable text or failed.")
    else:
        try:
            combined_fields, combined_raw = run_combined_pipeline(
                image_data=image_data,
                paddle_text=paddle_text,
                qwen_candidate=qwen_fields or {},
            )

            combined_fields = ensure_candidate_keys(
                combined_fields,
                qwen_fields or {},
            )

            save_text(pipeline_raw_path, combined_raw)
            save_json(pipeline_json_path, combined_fields)

            print("\nFINAL RECONCILED JSON:")
            print(json.dumps(combined_fields, indent=4, ensure_ascii=False))
            print("\nRaw response:", pipeline_raw_path)
            print("JSON output:", pipeline_json_path)
            status["pipeline"] = True
        except Exception as error:
            print("Combined pipeline failed:", error)
            save_error(image_output_dir, name, "pipeline", error)

    # --------------------------------------------------------
    # PER-IMAGE SUMMARY
    # --------------------------------------------------------
    print("\n" + "-" * 65)
    print(f"COMPLETED IMAGE {index}/{total}: {relative_path}")
    print("-" * 65)
    print("PaddleOCR only:", "SUCCESS" if status["paddle"] else "FAILED")
    print("Qwen only:", "SUCCESS" if status["qwen"] else "FAILED")
    print("Combined:", "SUCCESS" if status["pipeline"] else "FAILED")
    print("Results saved in:", image_output_dir)
    return status


# ============================================================
# 10. BATCH PROCESSING
# ============================================================

def is_inside(path, parent):
    """Return True if path is inside parent (or equals parent)."""
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def main():
    folder_input = input("Enter input image folder path: ").strip().strip('"').strip("'")
    result_input = input("Enter result folder path: ").strip().strip('"').strip("'")

    input_dir = Path(folder_input).expanduser().resolve()
    output_dir = Path(result_input).expanduser().resolve()

    if not input_dir.is_dir():
        print(f"ERROR: Input folder not found: {input_dir}")
        return

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        print(f"Cannot create result folder: {error}")
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
        print(f"No supported images found in: {input_dir}")
        return

    print("\n" + "=" * 75)
    print("BATCH DOCUMENT EXTRACTION")
    print("=" * 75)
    print("Input folder:", input_dir)
    print("Result folder:", output_dir)
    print("Images found:", len(image_files))
    print("PaddleOCR model:", PADDLE_MODEL)
    print("Qwen model:", QWEN_MODEL)
    print("Temperature:", TEMPERATURE)
    print("Top-p:", TOP_P)
    print("Max output tokens:", MAX_TOKENS)
    print("Processing mode: SEQUENTIAL")
    print("Schema mode:", "PREDEFINED" if SCHEMA is not None else "SCHEMA-FREE")
    if SCHEMA is not None:
        print("Required schema keys:", list(SCHEMA.keys()))

    totals = {"paddle": 0, "qwen": 0, "pipeline": 0}

    # Process each image completely before starting the next image.
    for index, image_path in enumerate(image_files, start=1):
        status = process_one_image(
            image_path=image_path,
            input_dir=input_dir,
            output_dir=output_dir,
            index=index,
            total=len(image_files),
        )

        for key in totals:
            if status[key]:
                totals[key] += 1

    print("\n\n" + "=" * 75)
    print("BATCH PROCESSING FINISHED")
    print("=" * 75)
    print("Total images:", len(image_files))
    print("PaddleOCR successes:", totals["paddle"])
    print("Qwen-only successes:", totals["qwen"])
    print("Combined pipeline successes:", totals["pipeline"])
    print("Result folder:", output_dir)


if __name__ == "__main__":
    main()

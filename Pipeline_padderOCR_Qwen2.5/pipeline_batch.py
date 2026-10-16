# after processing


import base64
import json
from io import BytesIO
from pathlib import Path

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

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".webp",
    ".bmp", ".tif", ".tiff"
}

paddle_client = OpenAI(
    base_url=PADDLE_URL,
    api_key="not-needed",
    timeout=TIMEOUT
)

qwen_client = OpenAI(
    base_url=QWEN_URL,
    api_key="not-needed",
    timeout=TIMEOUT
)

# ============================================================
# 2. IMAGE PREPROCESSING
# ============================================================

def image_to_base64(image_path):
    """Convert an image into a Base64 JPEG string."""

    with Image.open(image_path) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")

    image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))

    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=95)

    return base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

# ============================================================
# 3. COMMON VISION API CALL
# ============================================================

def get_response_text(content):

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

def call_vision_model(
    client,
    model,
    prompt,
    image_data,
    image_first=False
):
    """Call a remote OpenAI-compatible vision endpoint."""

    image_block = {
        "type": "image_url",
        "image_url": {
            "url": "data:image/jpeg;base64," + image_data
        }
    }

    text_block = {
        "type": "text",
        "text": prompt
    }

    content = (
        [image_block, text_block]
        if image_first
        else [text_block, image_block]
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": content
            }
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS
    )

    return get_response_text(
        response.choices[0].message.content
    )

# ============================================================
# 4. APPROACH 1: PADDLEOCR ONLY
# ============================================================

def run_paddle_only(image_data):

    prompt = "OCR:"

    return call_vision_model(
        client=paddle_client,
        model=PADDLE_MODEL,
        prompt=prompt,
        image_data=image_data,
        image_first=True
    )

def is_unusable_ocr(text):

    if not text or not text.strip():
        return True

    compact = "".join(text.split())

    return bool(compact) and set(compact) <= {"+"}

# ============================================================
# 5. JSON PARSING AND FLATTENING
# ============================================================

def parse_json_response(text):

    text = text.strip()

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
                "No JSON object found in model response."
            )

        result = json.loads(text[start:end + 1])

    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object.")

    return result

def flatten_json(data, prefix=""):

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


def ensure_candidate_keys(final_fields, qwen_candidate):
    """Retain Qwen-only keys if reconciliation omits them; fill with ""."""
    completed = dict(final_fields)
    for key in (qwen_candidate or {}):
        completed.setdefault(key, "")
    return completed

# ============================================================
# 6. APPROACH 2: QWEN ONLY
# ============================================================

def run_qwen_only(image_data):

    prompt = """
    You are a document information extraction system.

    Extract meaningful fields and their corresponding values
    directly from the original image.

    The document may contain printed or handwritten information.
    Do not assume a particular document type.

    Instructions:
    1. Identify every meaningful field label visible on the form.
    2. Include every identifiable field even when its value is blank.
    3. If a field is blank on the form, return its value as an empty string "".
    4. If a field is present but unreadable, return "" rather than guessing.
    5. Use descriptive snake_case keys and one flat JSON object.
    6. Return every value as a string; do not use null.
    7. Preserve names, dates, numbers and identifiers as written.
    8. Do not invent values or include unsupported fields.
    9. Do not include explanations or Markdown.
    10. Return valid JSON only.

    Return only the extracted key-value pairs.
    """

    raw_response = call_vision_model(
        client=qwen_client,
        model=QWEN_MODEL,
        prompt=prompt,
        image_data=image_data
    )

    parsed = parse_json_response(raw_response)

    return flatten_json(parsed), raw_response

# ============================================================
# 7. APPROACH 3: PADDLEOCR + QWEN RECONCILIATION
# ============================================================

def run_combined_pipeline(
    image_data,
    paddle_text,
    qwen_candidate
):

    qwen_candidate_text = json.dumps(
        qwen_candidate or {},
        indent=2,
        ensure_ascii=False
    )

    prompt = f"""
    You are a careful document extraction and reconciliation system.

    You have three sources:
    1. The original document image.
    2. Text recognized by PaddleOCR.
    3. An independent Qwen2.5-VL extraction.

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
    4. If they disagree, inspect the visible characters and choose the
       value best supported by the image.
    5. Include every identifiable field label visible on the form.
    6. Retain fields from either candidate when they belong to this form.
    7. If a field is blank on the form, keep its key and return "".
    8. If a field is present but unreadable or unresolved, return "".
    9. Never omit a field simply because its value is empty.
    10. Do not use null or invent missing information.
    11. Use descriptive snake_case keys and a flat JSON object.
    12. Return all values as strings.
    13. Do not include explanations or Markdown. Return valid JSON only.

    Reconcile both candidates against the original image and
    return the most accurate key-value extraction possible.
    """

    raw_response = call_vision_model(
        client=qwen_client,
        model=QWEN_MODEL,
        prompt=prompt,
        image_data=image_data
    )

    parsed = parse_json_response(raw_response)

    return flatten_json(parsed), raw_response

# ============================================================
# 8. FILE-SAVING HELPERS
# ============================================================

def save_text(file_path, text):

    file_path.parent.mkdir(parents=True, exist_ok=True)

    file_path.write_text(
        text,
        encoding="utf-8"
    )

def save_json(file_path, data):

    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False
        )

def save_error(output_dir, name, stage, error):

    error_path = output_dir / f"{name}_{stage}_error.txt"

    save_text(error_path, str(error))

    print(f"Error details saved to: {error_path}")

# ============================================================
# 9. PROCESS ONE IMAGE
# Complete all three approaches before moving to the next image.
# ============================================================

def process_one_image(image_path, input_dir, output_dir, index, total):

    relative_path = image_path.relative_to(input_dir)

    # Preserve input subfolders in the output folder.
    image_output_dir = output_dir / relative_path.parent
    image_output_dir.mkdir(parents=True, exist_ok=True)

    name = image_path.stem

    # Output paths for this image.
    paddle_txt = image_output_dir / f"{name}_paddle_only.txt"

    qwen_json_path = image_output_dir / f"{name}_qwen_only.json"
    qwen_raw_path = image_output_dir / f"{name}_qwen_only_raw.txt"

    pipeline_json_path = image_output_dir / f"{name}_pipeline.json"
    pipeline_raw_path = image_output_dir / f"{name}_pipeline_raw.txt"

    status = {
        "paddle": False,
        "qwen": False,
        "pipeline": False
    }

    print("\n\n")
    print("#" * 75)
    print(f"IMAGE {index}/{total}: {relative_path}")
    print("#" * 75)

    # --------------------------------------------------------
    # PREPROCESS IMAGE
    # --------------------------------------------------------

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

        # Save the raw endpoint response first.
        save_text(paddle_txt, paddle_text)

        print("\nPADDLEOCR OUTPUT:")
        print(paddle_text)
        print("\nSaved to:", paddle_txt)

        if is_unusable_ocr(paddle_text):
            print(
                "WARNING: PaddleOCR returned empty text or only "
                "'+' characters. The combined pipeline will be skipped."
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
        print(json.dumps(
            qwen_fields,
            indent=4,
            ensure_ascii=False
        ))

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
        print(
            "Combined pipeline skipped because PaddleOCR "
            "returned unusable text or failed."
        )

    else:
        try:
            combined_fields, combined_raw = run_combined_pipeline(
                image_data=image_data,
                paddle_text=paddle_text,
                qwen_candidate=qwen_fields or {}
            )

            # Preserve keys found by Qwen-only if reconciliation omits them.
            # When a key is missing from the final response, its value is "".
            combined_fields = ensure_candidate_keys(
                combined_fields,
                qwen_fields or {}
            )

            save_text(pipeline_raw_path, combined_raw)
            save_json(pipeline_json_path, combined_fields)

            print("\nFINAL RECONCILED JSON:")
            print(json.dumps(
                combined_fields,
                indent=4,
                ensure_ascii=False
            ))

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
    """Return True if path is inside parent."""

    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False

def main():

    folder_input = input(
        "Enter input image folder path: "
    ).strip().strip('"').strip("'")

    result_input = input(
        "Enter result folder path: "
    ).strip().strip('"').strip("'")

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

    # Find supported images in the input folder and subfolders.
    # Results are processed in sorted order, one image at a time.
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
        key=lambda path: str(path.relative_to(input_dir)).lower()
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
    print("Processing mode: SEQUENTIAL")

    totals = {
        "paddle": 0,
        "qwen": 0,
        "pipeline": 0
    }

    # Do not submit the entire folder to either model.
    # Process an image completely, then continue to the next.
    for index, image_path in enumerate(image_files, start=1):

        status = process_one_image(
            image_path=image_path,
            input_dir=input_dir,
            output_dir=output_dir,
            index=index,
            total=len(image_files)
        )

        for key in totals:
            if status[key]:
                totals[key] += 1

    # --------------------------------------------------------
    # FINAL BATCH SUMMARY
    # --------------------------------------------------------

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

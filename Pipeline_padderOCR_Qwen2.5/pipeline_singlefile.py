# import base64
# import json
# from io import BytesIO
# from pathlib import Path

# from PIL import Image, ImageOps
# from openai import OpenAI


# # ============================================================
# # 1. REMOTE MODEL CONFIGURATION
# # No PaddleOCR installation required on this computer.
# # ============================================================

# PADDLE_URL = "http://10.68.0.39:9006/v1"
# PADDLE_MODEL = "paddleocr-vl"

# QWEN_URL = "http://10.68.0.39:9005/v1"
# QWEN_MODEL = "qwen2.5-vl-7b"

# TEMPERATURE = 0.0
# TOP_P = 1.0
# MAX_TOKENS = 3000
# TIMEOUT = 180
# MAX_IMAGE_SIDE = 1600

# paddle_client = OpenAI(
#     base_url=PADDLE_URL,
#     api_key="not-needed",
#     timeout=TIMEOUT
# )

# qwen_client = OpenAI(
#     base_url=QWEN_URL,
#     api_key="not-needed",
#     timeout=TIMEOUT
# )


# # ============================================================
# # 2. IMAGE PREPROCESSING
# # ============================================================

# def image_to_base64(image_path):
#     """Prepare the image for remote vision model requests."""

#     with Image.open(image_path) as original:
#         image = ImageOps.exif_transpose(original).convert("RGB")

#     image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))

#     buffer = BytesIO()
#     image.save(buffer, format="JPEG", quality=95)

#     return base64.b64encode(
#         buffer.getvalue()
#     ).decode("utf-8")


# # ============================================================
# # 3. COMMON REMOTE VISION REQUEST
# # ============================================================

# def call_vision_model(
#     client,
#     model,
#     prompt,
#     image_data,
#     image_first=False
# ):
#     image_block = {
#         "type": "image_url",
#         "image_url": {
#             "url": f"data:image/jpeg;base64,{image_data}"
#         }
#     }

#     text_block = {
#         "type": "text",
#         "text": prompt
#     }

#     # PaddleOCR-VL's documented example sends the image first.
#     # Qwen receives the prompt first.
#     content = (
#         [image_block, text_block]
#         if image_first
#         else [text_block, image_block]
#     )

#     response = client.chat.completions.create(
#         model=model,
#         messages=[
#             {
#                 "role": "user",
#                 "content": content
#             }
#         ],
#         temperature=TEMPERATURE,
#         top_p=TOP_P,
#         max_tokens=MAX_TOKENS
#     )

#     result = response.choices[0].message.content

#     if isinstance(result, str):
#         return result

#     if isinstance(result, list):
#         text_parts = []

#         for item in result:
#             if isinstance(item, dict):
#                 text = item.get("text")
#             else:
#                 text = getattr(item, "text", None)

#             if text is not None:
#                 text_parts.append(str(text))

#         return "\n".join(text_parts)

#     return str(result)


# # ============================================================
# # 4. APPROACH 1: PADDLEOCR ONLY
# # Image -> remote PaddleOCR endpoint -> recognized text
# # ============================================================

# def run_paddle_only(image_data):

#     # Use the OCR task prompt for PaddleOCR-VL.
#     prompt = "OCR:"

#     return call_vision_model(
#         client=paddle_client,
#         model=PADDLE_MODEL,
#         prompt=prompt,
#         image_data=image_data,
#         image_first=True
#     )


# # ============================================================
# # 5. JSON PARSING AND FLATTENING
# # ============================================================

# def parse_json_response(text):
#     text = text.strip()

#     # Remove Markdown code fences if present.
#     if text.startswith("```"):
#         lines = text.splitlines()
#         lines = [
#             line for line in lines
#             if not line.strip().startswith("```")
#         ]
#         text = "\n".join(lines).strip()

#     try:
#         result = json.loads(text)

#     except json.JSONDecodeError:
#         start = text.find("{")
#         end = text.rfind("}")

#         if start == -1 or end < start:
#             raise ValueError(
#                 "No JSON object found in model response."
#             )

#         result = json.loads(text[start:end + 1])

#     if not isinstance(result, dict):
#         raise ValueError("Expected a JSON object.")

#     return result


# def flatten_json(data, prefix=""):
#     """Make nested objects into flat key-value pairs."""

#     flattened = {}

#     for key, value in data.items():
#         full_key = f"{prefix}.{key}" if prefix else str(key)

#         if isinstance(value, dict):
#             flattened.update(
#                 flatten_json(value, full_key)
#             )

#         elif isinstance(value, list):
#             flattened[full_key] = json.dumps(
#                 value, ensure_ascii=False
#             )

#         elif value is None:
#             flattened[full_key] = ""

#         elif isinstance(value, bool):
#             flattened[full_key] = str(value).lower()

#         else:
#             flattened[full_key] = str(value).strip()

#     return flattened


# # ============================================================
# # 6. APPROACH 2: QWEN ONLY
# # Image -> remote Qwen endpoint -> key-value JSON
# # ============================================================

# def run_qwen_only(image_data):

#     prompt = """
#     Extract meaningful fields and their corresponding values
#     directly from this document image.

#     The document may contain printed or handwritten text.
#     Do not assume a particular document type.

#     Rules:
#     1. Use descriptive snake_case keys.
#     2. Return one flat JSON object.
#     3. Return every value as a string.
#     4. Preserve names, dates, numbers and identifiers as written.
#     5. Do not invent or guess missing values.
#     6. Do not include explanations or Markdown.
#     7. Return valid JSON only.

#     Return only the extracted key-value pairs.
#     """

#     raw_response = call_vision_model(
#         client=qwen_client,
#         model=QWEN_MODEL,
#         prompt=prompt,
#         image_data=image_data
#     )

#     parsed = parse_json_response(raw_response)

#     return flatten_json(parsed), raw_response


# # ============================================================
# # 7. APPROACH 3: PADDLEOCR + QWEN PIPELINE
# # Image -> PaddleOCR text -> Qwen + original image -> JSON
# # ============================================================

# def run_combined_pipeline(image_data, paddle_text):

#     prompt = f"""
#     You are a document information extraction system.

#     Extract meaningful fields and their values from the
#     supplied document image.

#     Use the original image as the primary evidence.
#     Use the PaddleOCR transcription as supplementary evidence.
#     Verify OCR values against the image whenever possible.

#     Rules:
#     1. Do not assume a particular document type.
#     2. Use descriptive snake_case JSON keys.
#     3. Return a flat JSON object.
#     4. Return every value as a string.
#     5. Preserve names, dates, amounts and identifiers.
#     6. Do not invent missing or unreadable information.
#     7. Do not include explanations or Markdown.
#     8. Return valid JSON only.

#     PaddleOCR transcription:
#     <ocr_text>
#     {paddle_text[:15000]}
#     </ocr_text>

#     Return only the extracted key-value JSON object.
#     """

#     raw_response = call_vision_model(
#         client=qwen_client,
#         model=QWEN_MODEL,
#         prompt=prompt,
#         image_data=image_data
#     )

#     parsed = parse_json_response(raw_response)

#     return flatten_json(parsed), raw_response


# # ============================================================
# # 8. FILE OUTPUT HELPERS
# # ============================================================

# def save_text(path, text):
#     path.parent.mkdir(parents=True, exist_ok=True)

#     path.write_text(
#         text,
#         encoding="utf-8"
#     )


# def save_json(path, data):
#     path.parent.mkdir(parents=True, exist_ok=True)

#     with path.open("w", encoding="utf-8") as file:
#         json.dump(
#             data,
#             file,
#             indent=4,
#             ensure_ascii=False
#         )


# # ============================================================
# # 9. MAIN PROGRAM
# # ============================================================

# def main():

#     image_input = input(
#         "Enter input image path: "
#     ).strip().strip('"').strip("'")

#     folder_input = input(
#         "Enter result folder path: "
#     ).strip().strip('"').strip("'")

#     image_path = Path(image_input).expanduser()
#     output_dir = Path(folder_input).expanduser()

#     if not image_path.is_file():
#         print(f"ERROR: Image not found: {image_path}")
#         return

#     try:
#         output_dir.mkdir(parents=True, exist_ok=True)
#     except OSError as error:
#         print(f"Cannot create output folder: {error}")
#         return

#     name = image_path.stem

#     paddle_txt = output_dir / f"{name}_paddle_only.txt"

#     qwen_json_path = output_dir / f"{name}_qwen_only.json"
#     qwen_raw_path = output_dir / f"{name}_qwen_only_raw.txt"

#     pipeline_json_path = output_dir / f"{name}_pipeline.json"
#     pipeline_raw_path = output_dir / f"{name}_pipeline_raw.txt"

#     print("\n" + "=" * 60)
#     print("IMAGE EXTRACTION COMPARISON")
#     print("=" * 60)
#     print("Input image:", image_path)
#     print("Result folder:", output_dir.resolve())
#     print("PaddleOCR model:", PADDLE_MODEL)
#     print("Qwen model:", QWEN_MODEL)
#     print("Temperature:", TEMPERATURE)
#     print("Top-p:", TOP_P)
#     print("Local PaddleOCR installation: NOT REQUIRED")

#     # Prepare the image once for the comparison.
#     try:
#         image_data = image_to_base64(image_path)
#     except Exception as error:
#         print(f"Image preprocessing failed: {error}")
#         return

#     paddle_text = None

#     # --------------------------------------------------------
#     # APPROACH 1: PADDLEOCR ONLY
#     # --------------------------------------------------------

#     print("\n\n" + "=" * 60)
#     print("1. PADDLEOCR ONLY")
#     print("=" * 60)

#     try:
#         paddle_text = run_paddle_only(image_data)

#         # Save the raw recognized text, including unexpected
#         # responses, so they can be inspected.
#         save_text(paddle_txt, paddle_text)

#         print(paddle_text)
#         print("\nSaved to:", paddle_txt)

#         # A response containing only plus signs isn't usable OCR.
#         if not paddle_text.strip() or set(
#             paddle_text.strip()
#         ) <= {"+"}:
#             print(
#                 "\nWARNING: PaddleOCR returned no usable OCR text. "
#                 "The combined pipeline will be skipped."
#             )
#             paddle_text = None

#     except Exception as error:
#         error_path = output_dir / f"{name}_paddle_error.txt"
#         save_text(error_path, str(error))

#         print("PaddleOCR request failed:", error)
#         print("Error saved to:", error_path)

#     # --------------------------------------------------------
#     # APPROACH 2: QWEN ONLY
#     # --------------------------------------------------------

#     print("\n\n" + "=" * 60)
#     print("2. QWEN ONLY")
#     print("=" * 60)

#     try:
#         qwen_fields, qwen_raw = run_qwen_only(image_data)

#         save_text(qwen_raw_path, qwen_raw)
#         save_json(qwen_json_path, qwen_fields)

#         print(json.dumps(
#             qwen_fields,
#             indent=4,
#             ensure_ascii=False
#         ))

#         print("\nRaw response saved to:", qwen_raw_path)
#         print("JSON saved to:", qwen_json_path)

#     except Exception as error:
#         error_path = output_dir / f"{name}_qwen_only_error.txt"
#         save_text(error_path, str(error))

#         print("Qwen-only request failed:", error)
#         print("Error saved to:", error_path)

#     # --------------------------------------------------------
#     # APPROACH 3: PADDLEOCR + QWEN
#     # --------------------------------------------------------

#     print("\n\n" + "=" * 60)
#     print("3. PADDLEOCR + QWEN PIPELINE")
#     print("=" * 60)

#     if paddle_text is None:
#         print(
#             "Skipped because PaddleOCR did not return usable text. "
#             "Check the PaddleOCR-only output and server."
#         )

#     else:
#         try:
#             combined_fields, combined_raw = run_combined_pipeline(
#                 image_data,
#                 paddle_text
#             )

#             save_text(pipeline_raw_path, combined_raw)
#             save_json(pipeline_json_path, combined_fields)

#             print(json.dumps(
#                 combined_fields,
#                 indent=4,
#                 ensure_ascii=False
#             ))

#             print("\nRaw response saved to:", pipeline_raw_path)
#             print("JSON saved to:", pipeline_json_path)

#         except Exception as error:
#             error_path = output_dir / f"{name}_pipeline_error.txt"
#             save_text(error_path, str(error))

#             print("Combined pipeline failed:", error)
#             print("Error saved to:", error_path)

#     # --------------------------------------------------------
#     # FINAL SUMMARY
#     # --------------------------------------------------------

#     print("\n\n" + "=" * 60)
#     print("OUTPUT SUMMARY")
#     print("=" * 60)
#     print("PaddleOCR TXT :", paddle_txt)
#     print("Qwen-only JSON:", qwen_json_path)
#     print("Qwen raw TXT  :", qwen_raw_path)
#     print("Pipeline JSON :", pipeline_json_path)
#     print("Pipeline raw  :", pipeline_raw_path)
#     print("Output folder :", output_dir.resolve())


# if __name__ == "__main__":
#     main()






# After improvement

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

# Model generation parameters
TEMPERATURE = 0.0
TOP_P = 1.0
MAX_TOKENS = 3000
TIMEOUT = 180

# Resize large images before sending them to the endpoints
MAX_IMAGE_SIDE = 1600

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
    """
    Open the image, correct its orientation, convert it to RGB,
    resize it if necessary, and encode it as a JPEG Base64 string.
    """

    with Image.open(image_path) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")

    image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))

    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=95)

    return base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")


# ============================================================
# 3. COMMON REMOTE VISION API CALL
# ============================================================

def get_response_text(content):
    """Convert model response content to a string."""

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
    """
    Send the image and prompt to a remote OpenAI-compatible
    vision endpoint.
    """

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

    # PaddleOCR-VL is sent the image first.
    # Qwen receives the instruction first.
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
# Image -> PaddleOCR endpoint -> OCR text
# ============================================================

def run_paddle_only(image_data):
    """
    Calls the remote PaddleOCR-VL endpoint.
    No local PaddleOCR package is used.
    """

    prompt = "OCR:"

    return call_vision_model(
        client=paddle_client,
        model=PADDLE_MODEL,
        prompt=prompt,
        image_data=image_data,
        image_first=True
    )


def is_unusable_ocr(text):
    """
    Detect empty output or a response consisting only of '+' signs,
    such as the reported '+++' response.
    """

    if not text or not text.strip():
        return True

    compact = "".join(text.split())

    return bool(compact) and set(compact) <= {"+"}


# ============================================================
# 5. JSON PARSING
# ============================================================

def parse_json_response(text):
    """
    Parse the JSON object returned by Qwen.
    Handles code fences and brief surrounding text.
    """

    text = text.strip()

    if text.startswith("```"):
        lines = [
            line
            for line in text.splitlines()
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
                "No JSON object found in Qwen's response. "
                "Inspect the corresponding raw TXT output."
            )

        result = json.loads(text[start:end + 1])

    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object.")

    return result


def flatten_json(data, prefix=""):
    """
    Convert nested JSON into flat key-value pairs.
    Final values are stored as strings.
    """

    flattened = {}

    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)

        if isinstance(value, dict):
            flattened.update(
                flatten_json(value, full_key)
            )

        elif isinstance(value, list):
            flattened[full_key] = json.dumps(
                value,
                ensure_ascii=False
            )

        elif value is None:
            flattened[full_key] = ""

        elif isinstance(value, bool):
            flattened[full_key] = str(value).lower()

        else:
            flattened[full_key] = str(value).strip()

    return flattened


# ============================================================
# 6. APPROACH 2: QWEN ONLY
# Image -> Qwen endpoint -> key-value JSON
# ============================================================

def run_qwen_only(image_data):

    prompt = """
    You are a document information extraction system.

    Extract meaningful fields and their corresponding values
    directly from the original document image.

    The image may contain printed or handwritten information.
    Do not assume a particular document type.

    Instructions:
    1. Read the image directly.
    2. Use descriptive snake_case keys.
    3. Return one flat JSON object.
    4. Return every value as a string.
    5. Preserve names, dates, amounts and identifiers as written.
    6. Do not invent or guess missing information.
    7. Use an empty string for a field that is clearly present
       but whose value is unreadable.
    8. Include only fields supported by the image.
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
#
# Original image + PaddleOCR text + independent Qwen extraction
# -> Qwen checks disagreements against the original image.
# ============================================================

def run_combined_pipeline(
    image_data,
    paddle_text,
    qwen_candidate
):
    """
    Performs the final extraction and reconciliation step.

    PaddleOCR and the first Qwen output are both candidates.
    The image is the primary evidence for the final decision.
    """

    qwen_candidate_text = json.dumps(
        qwen_candidate or {},
        indent=2,
        ensure_ascii=False
    )

    prompt = f"""
    You are a careful document extraction and reconciliation system.

    You have three sources of information:
    1. The original document image attached to this request.
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

    IMPORTANT DECISION RULES:

    1. Re-examine the original image directly for every field.
    2. The original image is the primary evidence.
    3. Do not automatically prefer PaddleOCR.
    4. Do not automatically prefer the previous Qwen output.
    5. If both agree and the image supports the value, retain it.
    6. If they disagree, inspect the visible characters and
       choose the value best supported by the image.
    7. Do not copy PaddleOCR's spelling or value without checking.
    8. Retain valid information from either candidate when the
       image supports it.
    9. Do not invent, complete or infer unreadable values.
    10. If a value cannot be resolved from the image, use "".
    11. Preserve names, identifiers, dates and amounts carefully.
    12. Use descriptive snake_case keys.
    13. Return one flat JSON object with string values only.
    14. Do not return explanations, confidence scores or Markdown.
    15. Return valid JSON only.

    Independently verify disagreements using the image and return
    the most accurate key-value extraction possible.
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
# 9. MAIN PROGRAM
# ============================================================

def main():

    # Ask the user for paths.
    image_input = input(
        "Enter input image path: "
    ).strip().strip('"').strip("'")

    folder_input = input(
        "Enter result folder path: "
    ).strip().strip('"').strip("'")

    image_path = Path(image_input).expanduser()
    output_dir = Path(folder_input).expanduser()

    if not image_path.is_file():
        print(f"ERROR: Image not found: {image_path}")
        return

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        print(f"Cannot create result folder: {error}")
        return

    name = image_path.stem

    # Output paths.
    paddle_txt = output_dir / f"{name}_paddle_only.txt"

    qwen_json_path = output_dir / f"{name}_qwen_only.json"
    qwen_raw_path = output_dir / f"{name}_qwen_only_raw.txt"

    pipeline_json_path = output_dir / f"{name}_pipeline.json"
    pipeline_raw_path = output_dir / f"{name}_pipeline_raw.txt"

    print("\n" + "=" * 65)
    print("IMAGE EXTRACTION COMPARISON")
    print("=" * 65)
    print("Input image:", image_path)
    print("Result folder:", output_dir.resolve())
    print("PaddleOCR endpoint:", PADDLE_URL)
    print("PaddleOCR model:", PADDLE_MODEL)
    print("Qwen endpoint:", QWEN_URL)
    print("Qwen model:", QWEN_MODEL)
    print("Temperature:", TEMPERATURE)
    print("Top-p:", TOP_P)
    print("Local PaddleOCR installation: NOT REQUIRED")

    # Convert the image once for all approaches.
    try:
        image_data = image_to_base64(image_path)
    except Exception as error:
        print(f"Image preprocessing failed: {error}")
        return

    paddle_text = None
    qwen_fields = None

    # --------------------------------------------------------
    # APPROACH 1: PADDLEOCR ONLY
    # --------------------------------------------------------

    print("\n\n" + "=" * 65)
    print("APPROACH 1: PADDLEOCR ONLY")
    print("=" * 65)

    try:
        paddle_text = run_paddle_only(image_data)

        # Save the raw returned output for diagnosis.
        save_text(paddle_txt, paddle_text)

        print("\nPADDLEOCR OUTPUT:")
        print(paddle_text)
        print("\nSaved to:", paddle_txt)

        if is_unusable_ocr(paddle_text):
            print(
                "\nWARNING: PaddleOCR returned empty text or only "
                "'+' characters. Check the saved response and "
                "server-side model configuration."
            )
            paddle_text = None

    except Exception as error:
        paddle_text = None
        print("PaddleOCR request failed:", error)
        save_error(output_dir, name, "paddle", error)

    # --------------------------------------------------------
    # APPROACH 2: QWEN ONLY
    # --------------------------------------------------------

    print("\n\n" + "=" * 65)
    print("APPROACH 2: QWEN ONLY")
    print("=" * 65)

    try:
        qwen_fields, qwen_raw = run_qwen_only(image_data)

        save_text(qwen_raw_path, qwen_raw)
        save_json(qwen_json_path, qwen_fields)

        print("\nQWEN-ONLY KEY-VALUE JSON:")
        print(json.dumps(
            qwen_fields,
            indent=4,
            ensure_ascii=False
        ))

        print("\nRaw response saved to:", qwen_raw_path)
        print("JSON output saved to:", qwen_json_path)

    except Exception as error:
        qwen_fields = None
        print("Qwen-only extraction failed:", error)
        save_error(output_dir, name, "qwen_only", error)

    # --------------------------------------------------------
    # APPROACH 3: PADDLEOCR + QWEN RECONCILIATION
    # --------------------------------------------------------

    print("\n\n" + "=" * 65)
    print("APPROACH 3: PADDLEOCR + QWEN RECONCILIATION")
    print("=" * 65)

    if paddle_text is None:
        print(
            "Combined pipeline skipped because PaddleOCR did not "
            "return usable text. Qwen-only processing was attempted "
            "independently."
        )

    else:
        try:
            combined_fields, combined_raw = run_combined_pipeline(
                image_data=image_data,
                paddle_text=paddle_text,
                qwen_candidate=qwen_fields or {}
            )

            save_text(pipeline_raw_path, combined_raw)
            save_json(pipeline_json_path, combined_fields)

            print("\nFINAL RECONCILED KEY-VALUE JSON:")
            print(json.dumps(
                combined_fields,
                indent=4,
                ensure_ascii=False
            ))

            print("\nRaw response saved to:", pipeline_raw_path)
            print("Final JSON saved to:", pipeline_json_path)

        except Exception as error:
            print("Combined pipeline failed:", error)
            save_error(output_dir, name, "pipeline", error)

    # --------------------------------------------------------
    # FINAL OUTPUT SUMMARY
    # --------------------------------------------------------

    print("\n\n" + "=" * 65)
    print("OUTPUT SUMMARY")
    print("=" * 65)

    print("PaddleOCR only TXT :", paddle_txt)
    print("Qwen-only JSON     :", qwen_json_path)
    print("Qwen-only raw TXT  :", qwen_raw_path)
    print("Final pipeline JSON:", pipeline_json_path)
    print("Pipeline raw TXT   :", pipeline_raw_path)
    print("Result folder      :", output_dir.resolve())


if __name__ == "__main__":
    main()

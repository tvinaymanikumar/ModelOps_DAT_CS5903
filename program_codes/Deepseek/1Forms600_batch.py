#!/usr/bin/env python3
"""Batch-extract handwritten form data with DeepSeek-VL2 into flat JSON.

Output rules:
- JSON key: the printed field label or question on the form.
- JSON value: the handwritten/fill-in answer, transcribed exactly.
- No nested objects or arrays in the saved JSON.
- Include ordinary labeled fields even when blank (blank value is "").
- Checkbox/radio selections: include only clearly selected choices. Do not
  emit unchecked or ambiguous options. For a question with choices, use the
  printed question as the key and the selected printed answer as the value.
- For standalone tick boxes, use the printed label as key and "Yes" as value
  only when clearly marked; omit empty/uncertain boxes.

Examples:
  python extract_forms.py --input "C:\\forms\\images" --output "C:\\forms\\results"
  python extract_forms.py --input ./forms --output ./results --overwrite
  python extract_forms.py --input ./forms --output ./results --recursive

Dependencies:
  pip install requests pillow
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests
from PIL import Image

BASE_URL = "http://localhost:9000/v1"
MODEL_FALLBACK = "DeepSeek-VL2"
MAX_CONTEXT_TOKENS = 4096
DEFAULT_MAX_OUTPUT_TOKENS = 1800
DEFAULT_MAX_IMAGE_SIDE = 1280
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

SYSTEM_PROMPT = (
    "You are a precise handwritten-form transcription and field-extraction system. "
    "Transcribe only information that is visibly present in the image. "
    "NEVER hallucinate, fabricate, guess, infer missing information, autocomplete "
    "partial words or numbers, or autocorrect spelling or grammar. "
    "If a character or value is unclear, do not make up a likely answer. "
    "Return only one valid JSON object, with no explanation, Markdown, or code fences."
)

USER_PROMPT = r"""
Inspect only the provided form image. Extract all form information faithfully.

MAIN RULE: use the printed field label or printed question as the JSON key, and
use the handwritten or filled-in response as its value. For example, if the
printed label is "Quantity Required" and the handwritten entry is "5", return:
{"Quantity Required": "5"}.

STRICT ANTI-HALLUCINATION AND NO-AUTOCOMPLETION RULES — FOLLOW THESE FIRST:
- NEVER hallucinate, fabricate, guess, assume, infer, or invent any information.
- NEVER autocomplete a partially written word, name, sentence, number, date,
  identifier, or other handwritten entry.
- NEVER fill in missing characters based on context, familiar names, grammar,
  spelling, common patterns, or what you think the writer intended.
- NEVER autocorrect spelling, grammar, capitalization, abbreviations, or format.
- Copy only the characters and content that are actually visible in the image.
- If part of an entry is readable and part is not, transcribe only what is
  visibly readable; do not guess the unreadable portion.
- If the entire handwritten value cannot be read reliably, use "" for that field.
- Do not infer a value from another field, from the form type, or from general
  knowledge. The image is the only source of truth.
- When uncertain, leave the value blank rather than generating a likely answer.

OUTPUT FORMAT:
- Return one flat JSON object only.
- Every key must be the printed field label/question, copied as accurately as
  possible from the form. Do not use generic keys such as "key", "value", or
  "field_1" when a printed label is available.
- Every value must be a string.
- No nested objects and no arrays.
- Keep the field labels/questions as keys and handwritten/fill-in contents as values.
- Include all ordinary labeled fields visible on the form, including blank fields;
  use "" for a blank or unreadable handwritten response.
- Do not copy printed instructions/examples into the values.
- Keep spelling, capitalization, punctuation, numbers, IDs and date formats as written.
- Never guess, autocomplete, infer missing characters, or autocorrect.

FIELD LABELS AND TABLES:
- Match each handwritten entry to its correct printed label using alignment,
  boxes, proximity, row/column position, and the page layout.
- For repeating rows where the same printed label occurs more than once, make
  each key unique while preserving the printed label, such as "Quantity (Row 1)"
  and "Quantity (Row 2)". Do not merge values from different rows.
- If a visible handwritten entry has no printed label, use a short descriptive
  key based only on its visible context. Do not invent its meaning.
- If a field is multiline, combine its visible lines into one string separated
  by spaces.
- Dates and numeric values must remain exactly as visible; do not recalculate
  or reformat them.

CHECKBOXES, TICK BOXES AND RADIO BUTTONS — IMPORTANT:
- Include ONLY clearly selected/ticked/marked choices. Do not include unchecked
  choices. Do not include boxes whose selection status is unclear.
- For a printed question with mutually exclusive choices (for example, "Gender:
  Male / Female"), use the printed question as the key and the selected printed
  choice as the value. Example: {"Gender": "Female"} if Female is clearly marked.
- For a printed yes/no question, use the question as the key and the selected
  answer as the value, e.g. {"Is the applicant employed?": "Yes"} or
  {"Is the applicant employed?": "No"}, depending on which answer is ticked.
- For a standalone checkbox with its own label and no paired choice, include
  {"Printed checkbox label": "Yes"} only if it is clearly checked.
- For a multiple-selection question, use the printed question as the key and
  put the clearly selected printed option labels in a single string separated
  by ", ". Do not include unselected or ambiguous options.
- Never output "No" just to represent an unchecked option. A value "No" is
  permitted only when "No" itself is the clearly selected answer to a printed
  yes/no question.
- A tick/cross counts as selected only when the mark clearly indicates selection.
  Do not infer selection from nearby handwriting.

SIGNATURES:
- Transcribe a signature only if there is a clearly labeled signature field and
  the writing can be read reliably. Otherwise use "" for that labeled field.
- Never guess a person's name from a signature.

JSON VALIDATION:
- Exactly one JSON object, all keys and values double-quoted strings.
- No comments, no Markdown, no text outside JSON.
- Keys must be unique. If a label repeats, add a row number or suffix such as
  " (2)" while keeping the original field label recognizable.

Return ONLY the JSON object.
"""


class ModelRequestError(RuntimeError):
    """An HTTP error from the OpenAI-compatible endpoint."""

    def __init__(self, status_code: int, response_text: str):
        self.status_code = status_code
        self.response_text = response_text
        super().__init__(f"HTTP {status_code}: {response_text[:1200]}")


def image_to_data_url(path: Path, max_side: int) -> str:
    """Read an image, scale its longest side if needed, and return a JPEG data URL."""
    with Image.open(path) as original:
        image = original.convert("RGB")
        width, height = image.size
        longest = max(width, height)
        if max_side > 0 and longest > max_side:
            scale = max_side / float(longest)
            new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
            resampling = getattr(Image, "Resampling", Image).LANCZOS
            image = image.resize(new_size, resampling)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def get_model_name(base_url: str, model_override: str | None, timeout: int) -> str:
    """Use the explicitly supplied model, otherwise try the server model list."""
    if model_override:
        return model_override
    try:
        response = requests.get(f"{base_url.rstrip('/')}/models", timeout=min(timeout, 15))
        response.raise_for_status()
        models = response.json().get("data", [])
        if models and models[0].get("id"):
            return str(models[0]["id"])
    except Exception as error:
        print(f"[warning] Could not detect model name ({error}); using {MODEL_FALLBACK!r}.")
    return MODEL_FALLBACK


def call_model(
    base_url: str,
    model: str,
    data_url: str,
    max_tokens: int,
    timeout: int,
    prompt: str = USER_PROMPT,
) -> str:
    """Call the local OpenAI-compatible chat completions API."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
        "temperature": 0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
    }
    response = requests.post(
        f"{base_url.rstrip('/')}/chat/completions", json=payload, timeout=timeout
    )
    if not response.ok:
        raise ModelRequestError(response.status_code, response.text)
    body = response.json()
    try:
        return body["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError(f"Unexpected response schema: {body}") from error


def extract_json_object(raw: str) -> dict[str, Any]:
    """Parse a JSON object even if the model accidentally adds a code fence or prose."""
    text = raw.strip()
    text = re.sub(r"^```\s*(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text).strip()
    decoder = json.JSONDecoder()
    # First try the whole response.
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
        raise ValueError("Model output was valid JSON but not a JSON object.")
    except json.JSONDecodeError:
        pass
    # Then try each opening brace, which also avoids brittle first/last-brace slicing.
    for start, character in enumerate(text):
        if character != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise json.JSONDecodeError("No complete JSON object found in model response", text, 0)


def to_string(value: Any) -> str:
    """Convert an extracted scalar to a JSON string without changing its content."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def flatten_to_string_pairs(obj: Any) -> dict[str, str]:
    """Flatten nested model output to unique single-level keys with string values."""
    result: dict[str, str] = {}

    def unique_key(key: str) -> str:
        candidate = key or "Value"
        if candidate not in result:
            return candidate
        number = 2
        while f"{candidate} ({number})" in result:
            number += 1
        return f"{candidate} ({number})"

    def visit(value: Any, parts: list[str]) -> None:
        if isinstance(value, dict):
            if not value:
                result[unique_key(" - ".join(parts) or "Value")] = ""
                return
            for child_key, child_value in value.items():
                visit(child_value, parts + [str(child_key)])
        elif isinstance(value, list):
            base = " - ".join(parts) or "Value"
            if not value:
                result[unique_key(base)] = ""
                return
            for index, child in enumerate(value, start=1):
                if isinstance(child, (dict, list)):
                    visit(child, parts + [f"Row {index}"])
                else:
                    # Scalar arrays (e.g. selected options) become one string value.
                    result[unique_key(base)] = ", ".join(to_string(item) for item in value)
                    break
        else:
            result[unique_key(" - ".join(parts) or "Value")] = to_string(value)

    visit(obj, [])
    # Ensure every value in the final JSON is a string.
    return {str(key): to_string(value) for key, value in result.items()}


def get_context_error_limits(error_text: str) -> tuple[int | None, int | None]:
    """Return (context_limit, input_tokens) if the server reports those figures."""
    context_match = re.search(r"maximum context length is\s+(\d+)\s+tokens", error_text, re.I)
    input_match = re.search(r"request has\s+(\d+)\s+input tokens", error_text, re.I)
    context_limit = int(context_match.group(1)) if context_match else None
    input_tokens = int(input_match.group(1)) if input_match else None
    return context_limit, input_tokens


def is_context_length_error(error: Exception) -> bool:
    message = str(error).lower()
    return any(term in message for term in ("maximum context length", "context_length_exceeded", "max_tokens", "max_completion_tokens"))


def process_image(
    image_path: Path,
    output_path: Path,
    args: argparse.Namespace,
    model: str,
) -> bool:
    """Extract one image; retry recoverable endpoint/JSON/token errors."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = output_path.with_suffix(".raw.txt")
    last_error: Exception | None = None
    current_max_tokens = args.max_tokens
    current_max_side = args.max_side
    prompt = USER_PROMPT
    last_raw = ""

    for attempt in range(1, args.retries + 1):
        try:
            data_url = image_to_data_url(image_path, current_max_side)
            last_raw = call_model(
                args.base_url, model, data_url, current_max_tokens, args.timeout, prompt
            )
            extracted = extract_json_object(last_raw)
            flat = flatten_to_string_pairs(extracted)
            output_path.write_text(
                json.dumps(flat, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            if raw_path.exists():
                raw_path.unlink()
            return True
        except Exception as error:
            last_error = error
            print(f"   attempt {attempt}/{args.retries} failed: {error}")

            # If the model's context window was exceeded, calculate a safer output
            # budget from the server's own error message; if needed, reduce image size.
            details = error.response_text if isinstance(error, ModelRequestError) else str(error)
            if is_context_length_error(error):
                context_limit, input_tokens = get_context_error_limits(details)
                new_budget: int | None = None
                if context_limit and input_tokens is not None:
                    new_budget = context_limit - input_tokens - 64
                if new_budget is not None and new_budget >= 256 and new_budget < current_max_tokens:
                    current_max_tokens = new_budget
                    print(f"   reducing max output tokens to {current_max_tokens} and retrying")
                else:
                    reduced_side = max(640, int(current_max_side * 0.75))
                    if reduced_side < current_max_side:
                        current_max_side = reduced_side
                        current_max_tokens = min(current_max_tokens, 1200)
                        print(
                            f"   reducing image longest side to {current_max_side}px "
                            f"and output budget to {current_max_tokens}; retrying"
                        )
                    else:
                        current_max_tokens = max(256, min(current_max_tokens - 256, 1000))
                # Keep the correction prompt short: extra prompt tokens worsen this error.
            elif isinstance(error, (json.JSONDecodeError, ValueError)):
                prompt = USER_PROMPT + "\n\nRETRY: Return a complete valid JSON object only. " \
                         "Use only flat string key-value pairs. Do not use Markdown. " \
                         "Do NOT hallucinate, guess, infer, autocomplete, or autocorrect. " \
                         "Copy only visible content; use an empty string instead of guessing."
                current_max_tokens = max(700, min(current_max_tokens, 1500))
            time.sleep(args.retry_delay)

    try:
        raw_path.write_text(last_raw or str(last_error), encoding="utf-8")
        print(f"   saved diagnostic output: {raw_path}")
    except OSError:
        pass
    print(f"   [FAILED] {image_path.name}: {last_error}")
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch transcribe filled-in form images into flat JSON using DeepSeek-VL2."
    )
    parser.add_argument(
        "--input", default=None,
        help="Folder containing input form images (if omitted, you will be prompted)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Folder in which JSON results are saved (if omitted, you will be prompted)",
    )
    parser.add_argument("--base-url", default=BASE_URL, help=f"OpenAI-compatible endpoint (default: {BASE_URL})")
    parser.add_argument("--model", default=None, help="Model ID; auto-detected from /models if omitted")
    parser.add_argument(
        "--max-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS,
        help=f"Maximum output tokens (default: {DEFAULT_MAX_OUTPUT_TOKENS}; must fit within 4096-token context)",
    )
    parser.add_argument("--max-side", type=int, default=DEFAULT_MAX_IMAGE_SIDE,
                        help=f"Resize longest image side to this many pixels (default: {DEFAULT_MAX_IMAGE_SIDE})")
    parser.add_argument("--retries", type=int, default=3, help="Attempts per image (default: 3)")
    parser.add_argument("--timeout", type=int, default=300, help="Request timeout in seconds (default: 300)")
    parser.add_argument("--retry-delay", type=float, default=1.0, help="Delay between retries in seconds")
    parser.add_argument("--recursive", action="store_true", help="Also process images in subfolders")
    parser.add_argument("--overwrite", action="store_true", help="Reprocess images whose JSON already exists")
    return parser.parse_args()


def clean_path_input(value: str) -> str:
    """Trim whitespace and matching quotes from a path pasted into the prompt."""
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"\"", "'"}:
        value = value[1:-1].strip()
    return value


def main() -> int:
    args = parse_args()

    print("DEEPSEEK-VL2 BATCH FORM EXTRACTION - FOLDER SETUP")
    input_value = args.input
    if not input_value:
        input_value = input("Enter input image folder path: ")
    input_value = clean_path_input(input_value)
    if not input_value:
        print("Input folder path cannot be empty.", file=sys.stderr)
        return 2

    output_value = args.output
    if not output_value:
        output_value = input("Enter output/results folder path: ")
    output_value = clean_path_input(output_value)
    if not output_value:
        print("Output folder path cannot be empty.", file=sys.stderr)
        return 2

    input_dir = Path(input_value).expanduser()
    output_dir = Path(output_value).expanduser()

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Input folder not found or is not a folder: {input_dir}", file=sys.stderr)
        return 2
    if args.max_tokens < 256:
        print("--max-tokens must be at least 256", file=sys.stderr)
        return 2
    if args.max_side < 640:
        print("--max-side should be 640 or greater to preserve handwriting readability", file=sys.stderr)
        return 2
    output_dir.mkdir(parents=True, exist_ok=True)

    finder = input_dir.rglob("*") if args.recursive else input_dir.iterdir()
    images = sorted(
        (path for path in finder if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: str(path).lower(),
    )
    if not images:
        print(f"No supported images found in: {input_dir}", file=sys.stderr)
        return 1

    model = get_model_name(args.base_url, args.model, args.timeout)
    print("=" * 76)
    print("DEEPSEEK-VL2 BATCH FORM EXTRACTION")
    print(f"Model        : {model}")
    print(f"Endpoint     : {args.base_url}")
    print(f"Input folder : {input_dir}")
    print(f"Output folder: {output_dir}")
    print(f"Images found : {len(images)}")
    print(f"Recursive    : {'Yes' if args.recursive else 'No'}")
    print(f"Max tokens   : {args.max_tokens}")
    print("=" * 76)

    success = failed = skipped = 0
    used_outputs: set[Path] = set()
    for index, image_path in enumerate(images, start=1):
        relative_parent = image_path.parent.relative_to(input_dir) if args.recursive else Path()
        target_dir = output_dir / relative_parent
        output_path = target_dir / f"{image_path.stem}.json"
        # Prevent overwrite if same basename exists in the input folder with another extension.
        if output_path in used_outputs:
            output_path = target_dir / f"{image_path.stem}_{image_path.suffix.lstrip('.').lower()}.json"
        used_outputs.add(output_path)

        print(f"\n[{index}/{len(images)}] {image_path.name}")
        if output_path.exists() and not args.overwrite:
            print(f"   skipped (already exists): {output_path}")
            skipped += 1
            continue
        started = time.time()
        if process_image(image_path, output_path, args, model):
            print(f"   saved: {output_path} ({time.time() - started:.1f}s)")
            success += 1
        else:
            failed += 1

    print("\n" + "=" * 76)
    print("BATCH PROCESSING SUMMARY")
    print(f"Total images: {len(images)}")
    print(f"Succeeded   : {success}")
    print(f"Failed      : {failed}")
    print(f"Skipped     : {skipped}")
    print(f"Output      : {output_dir}")
    print("=" * 76)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nStopped by user.", file=sys.stderr)
        raise SystemExit(130)

"""
MinerU2.5 endpoint test: IAM handwriting dataset (schema-free).

Endpoint: http://10.68.0.39:9008/v1
Model: mineru2.5

Install:
    python -m pip install -U openai pillow

Run:
    python mineru_iam_endpoint.py

The script asks for the input image folder and output folder, then writes one
CSV called iam_mineru_predictions.csv with columns filename, transcription.
"""

import base64
import csv
import time
from io import BytesIO
from pathlib import Path

from openai import OpenAI
from PIL import Image, ImageOps, UnidentifiedImageError

BASE_URL = "http://10.68.0.39:9008/v1"
MODEL_NAME = "mineru2.5"
API_KEY = "EMPTY"  # Replace if your endpoint requires authentication.
TEMPERATURE = 0.0
MAX_TOKENS = 1500
REQUEST_DELAY_SECONDS = 2

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

TRANSCRIPTION_PROMPT = """
Transcribe the handwritten English text visible in this image.
Return only the transcription, without a heading, explanation, JSON, or markdown.
Preserve original words, spelling, capitalization, punctuation, and meaningful
line breaks. Do not paraphrase, correct grammar, or invent words using context.
If a short part is genuinely unreadable, write [UNCLEAR].
"""


def get_image_paths(input_dir: Path) -> list[Path]:
    return sorted(
        [
            p for p in input_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda p: p.relative_to(input_dir).as_posix().casefold(),
    )


def image_to_data_url(image_path: Path) -> str:
    with Image.open(image_path) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")
    image.thumbnail((2400, 2400))
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def transcribe_image(client: OpenAI, image_path: Path) -> str:
    image_url = image_to_data_url(image_path)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": TRANSCRIPTION_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url, "detail": "high"},
                    },
                ],
            }
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )

    if not response.choices:
        raise RuntimeError("Endpoint returned no choices.")

    content = response.choices[0].message.content
    if isinstance(content, str) and content.strip():
        return content.strip()

    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("text"):
                parts.append(str(part["text"]))
            elif getattr(part, "text", None):
                parts.append(str(part.text))
        joined = "\n".join(parts).strip()
        if joined:
            return joined

    raise RuntimeError("Endpoint returned an empty or unsupported response.")


def main() -> int:
    print("=" * 72)
    print("MINERU2.5 ENDPOINT - IAM HANDWRITING TRANSCRIPTION")
    print("=" * 72)
    print("Endpoint:", BASE_URL)
    print("Model:", MODEL_NAME)

    input_value = input("\nEnter the IAM INPUT image folder path: ").strip().strip('"').strip("'")
    output_value = input("Enter the OUTPUT folder path for the CSV: ").strip().strip('"').strip("'")

    if not input_value or not output_value:
        print("ERROR: Both input and output folder paths are required.")
        return 1

    try:
        input_dir = Path(input_value).expanduser().resolve(strict=True)
    except (OSError, RuntimeError):
        print(f"ERROR: Input folder not found: {input_value}")
        return 1

    if not input_dir.is_dir():
        print(f"ERROR: Input path is not a folder: {input_dir}")
        return 1

    output_dir = Path(output_value).expanduser().resolve()
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        print(f"ERROR: Could not create output folder: {error}")
        return 1

    image_paths = get_image_paths(input_dir)
    if not image_paths:
        print(f"ERROR: No supported images found in {input_dir}")
        return 1

    output_csv = output_dir / "iam_mineru_predictions.csv"
    print(f"\nImages found: {len(image_paths)}")
    print("CSV output:", output_csv)

    try:
        client = OpenAI(
            base_url=BASE_URL,
            api_key=API_KEY,
            timeout=180.0,
            max_retries=0,
        )
    except Exception as error:
        print(f"ERROR: Could not initialize API client: {error}")
        return 1

    success_count = 0
    error_count = 0

    try:
        with output_csv.open("w", newline="", encoding="utf-8-sig") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=["filename", "transcription"])
            writer.writeheader()

            for index, image_path in enumerate(image_paths, start=1):
                filename = image_path.relative_to(input_dir).as_posix()
                print(f"\n[{index}/{len(image_paths)}] Processing: {filename}")

                if REQUEST_DELAY_SECONDS > 0:
                    time.sleep(REQUEST_DELAY_SECONDS)

                try:
                    transcription = transcribe_image(client, image_path)
                    success_count += 1
                    print(transcription)
                except (UnidentifiedImageError, OSError) as error:
                    error_count += 1
                    transcription = ""
                    print(f"IMAGE ERROR: {error}")
                except Exception as error:
                    error_count += 1
                    transcription = ""
                    print(f"ENDPOINT ERROR: {type(error).__name__}: {error}")
                    print(
                        "Check that the endpoint is reachable, model name is correct, "
                        "and the server accepts OpenAI-compatible image_url inputs."
                    )

                writer.writerow({"filename": filename, "transcription": transcription})
                csv_file.flush()

    except PermissionError:
        print(f"ERROR: Cannot write {output_csv}. Close it in Excel and try again.")
        return 1
    finally:
        client.close()

    print("\n" + "=" * 72)
    print("IAM DATASET FINISHED")
    print("=" * 72)
    print("Total images:", len(image_paths))
    print("Successful:", success_count)
    print("Failed:", error_count)
    print("CSV saved to:", output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

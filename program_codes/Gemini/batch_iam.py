"""
Gemini-only IAM handwriting transcription batch script.

What it does:
- Prompts for the input image folder and output folder.
- Reads images recursively, including images in subfolders.
- Sends one handwritten image at a time to Gemini.
- Writes ONE combined CSV named iam_transcriptions.csv.
- CSV columns: filename, transcription.
- Waits between Gemini requests and retries temporary failures.

Install:
    python -m pip install -U google-genai pillow python-dotenv

Optional .env file next to this script:
    GEMINI_API_KEY=your_api_key_here
    GEMINI_MODEL=gemini-3.5-flash-lite

Run:
    python gemini_iam_transcription_csv.py

Security:
    You can paste your key into API_KEY below. Do not share the script or commit
    it to GitHub while it contains your real API key.
"""

import csv
import os
import time
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image, ImageOps, UnidentifiedImageError


# ============================================================
# 1. GEMINI CONFIGURATION
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
load_dotenv(SCRIPT_DIR / ".env")

# Load Gemini API key from environment or local .env file
API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip()

TEMPERATURE = 0.0
TOP_P = 1.0
MAX_OUTPUT_TOKENS = 1500
MAX_IMAGE_SIDE = 2200

# Wait before every Gemini API request.
REQUEST_DELAY_SECONDS = 6
MAX_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 10

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"
}

if not API_KEY:
    raise SystemExit(
        "ERROR: Gemini API key is missing. Paste it into API_KEY in this script, "
        "or set GEMINI_API_KEY in the .env file."
    )


# ============================================================
# 2. IAM DATASET SETTINGS AND TRANSCRIPTION PROMPT
# ============================================================

PROMPT = r"""
You are a careful handwriting transcription system working with the IAM
handwriting dataset. The image contains a short handwritten English sentence,
often written across approximately three lines.

Transcribe the text visible in the image.

Rules:
1. Return only the transcription, without a heading, explanation, markdown,
   JSON wrapper, or quotation marks.
2. Preserve the writer's original words, spelling, capitalization, and
   punctuation as accurately as possible.
3. Do not paraphrase, summarize, or correct grammar.
4. Preserve meaningful line breaks using newline characters.
5. Do not add words based on context or guess text that is not visible.
6. If a small part is genuinely unreadable, write [UNCLEAR] for that part.
7. Ignore blank margins, borders, and scanning artifacts.
8. The image is the source of truth; do not use a familiar sentence to fill
   in words that cannot be read.
"""


# ============================================================
# 3. IMAGE PREPROCESSING
# ============================================================

def image_to_jpeg_bytes(image_path: Path) -> bytes:
    """Correct image orientation, resize if needed, and encode as JPEG."""
    with Image.open(image_path) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")

    image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))

    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


# ============================================================
# 4. GEMINI TRANSCRIPTION
# ============================================================

def transcribe_with_gemini(client: genai.Client, image_bytes: bytes) -> str:
    """Call Gemini and return plain-text transcription."""
    if REQUEST_DELAY_SECONDS > 0:
        print(f"Waiting {REQUEST_DELAY_SECONDS} seconds before Gemini request...")
        time.sleep(REQUEST_DELAY_SECONDS)

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            PROMPT,
        ],
        config=types.GenerateContentConfig(
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            response_mime_type="text/plain",
        ),
    )

    text = (getattr(response, "text", None) or "").strip()
    if not text:
        raise ValueError(
            "Gemini returned no transcription. Check the model name, API key, "
            "quota, or response finish reason."
        )

    # Remove wrapping quotation marks if Gemini puts quotes around the whole text.
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1].strip()

    return text


# ============================================================
# 5. IMAGE DISCOVERY
# ============================================================

def get_image_files(input_dir: Path) -> list[Path]:
    """Find supported images in the input folder and its subfolders."""
    return sorted(
        (
            path for path in input_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: path.relative_to(input_dir).as_posix().casefold(),
    )


# ============================================================
# 6. PROCESS ONE IMAGE WITH RETRIES
# ============================================================

def process_one_image(
    client: genai.Client,
    image_path: Path,
    input_dir: Path,
    index: int,
    total: int,
) -> tuple[str, str]:
    """Return the relative filename and transcription for one image."""
    relative_filename = image_path.relative_to(input_dir).as_posix()

    print("\n" + "=" * 72)
    print(f"IMAGE {index}/{total}: {relative_filename}")
    print("=" * 72)

    try:
        image_bytes = image_to_jpeg_bytes(image_path)
    except (OSError, UnidentifiedImageError) as error:
        print(f"FAILED to open image: {error}")
        return relative_filename, ""

    last_error = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            transcription = transcribe_with_gemini(client, image_bytes)
            print("\nTRANSCRIPTION:")
            print(transcription)
            return relative_filename, transcription
        except Exception as error:
            last_error = error
            print(f"Attempt {attempt}/{MAX_ATTEMPTS} failed: {error}")
            if attempt < MAX_ATTEMPTS:
                print(f"Retrying in {RETRY_DELAY_SECONDS} seconds...")
                time.sleep(RETRY_DELAY_SECONDS)

    # Keep the CSV format stable. Failure details are printed in the terminal;
    # the transcription cell remains blank when all attempts fail.
    print(f"FAILED after {MAX_ATTEMPTS} attempts: {last_error}")
    return relative_filename, ""


# ============================================================
# 7. SEQUENTIAL CSV BATCH PROCESSING
# ============================================================

def main() -> None:
    print("=" * 72)
    print("GEMINI IAM HANDWRITING TRANSCRIPTION TO CSV")
    print("=" * 72)
    print("Gemini model:", GEMINI_MODEL)
    print("Delay between requests:", REQUEST_DELAY_SECONDS, "seconds")

    input_value = input("\nEnter input image folder path: ").strip().strip('"').strip("'")
    output_value = input("Enter output folder path for the CSV: ").strip().strip('"').strip("'")

    if not input_value or not output_value:
        raise SystemExit("ERROR: Both input and output folder paths are required.")

    input_dir = Path(input_value).expanduser()
    output_dir = Path(output_value).expanduser()

    try:
        input_dir = input_dir.resolve(strict=True)
    except (OSError, RuntimeError):
        raise SystemExit(f"ERROR: Input folder not found: {input_value}")

    if not input_dir.is_dir():
        raise SystemExit(f"ERROR: Input path is not a folder: {input_dir}")

    try:
        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
    except (OSError, RuntimeError) as error:
        raise SystemExit(f"ERROR: Cannot create output folder: {error}")

    output_csv = output_dir / "iam_transcriptions.csv"
    image_files = get_image_files(input_dir)

    if not image_files:
        raise SystemExit(f"ERROR: No supported images found in: {input_dir}")

    print("\nInput folder:", input_dir)
    print("Output CSV:", output_csv)
    print("Images found:", len(image_files))
    print("=" * 72)

    try:
        client = genai.Client(api_key=API_KEY)
    except Exception as error:
        raise SystemExit(f"ERROR: Could not initialize Gemini client: {error}")

    successes = 0
    failures = 0

    try:
        # utf-8-sig improves compatibility when opening the CSV in Excel.
        with output_csv.open("w", newline="", encoding="utf-8-sig") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=["filename", "transcription"],
            )
            writer.writeheader()

            for index, image_path in enumerate(image_files, start=1):
                filename, transcription = process_one_image(
                    client=client,
                    image_path=image_path,
                    input_dir=input_dir,
                    index=index,
                    total=len(image_files),
                )

                writer.writerow({
                    "filename": filename,
                    "transcription": transcription,
                })
                csv_file.flush()

                if transcription:
                    successes += 1
                else:
                    failures += 1
    except PermissionError as error:
        raise SystemExit(
            f"ERROR: Cannot write {output_csv}. If it is open in Excel, close it "
            f"and rerun the script. Details: {error}"
        )
    finally:
        try:
            client.close()
        except Exception:
            pass

    print("\n" + "=" * 72)
    print("BATCH PROCESSING FINISHED")
    print("=" * 72)
    print("Total images:", len(image_files))
    print("Successful:", successes)
    print("Failed or blank:", failures)
    print("CSV saved to:", output_csv)


if __name__ == "__main__":
    main()

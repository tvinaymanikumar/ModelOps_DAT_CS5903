"""
MinerU2.5 endpoint - IAM handwriting batch transcription to one CSV.

Install:
    python -m pip install -U openai

Run:
    python mineru2_5_iam_csv.py

Output:
    iam_mineru_predictions.csv
    Columns: filename, transcription

Successful rows already present in the CSV are skipped on reruns.
"""

import base64
import csv
import mimetypes
import re
import tempfile
import time
from pathlib import Path

from openai import OpenAI


# ============================================================
# 1. MINERU2.5 ENDPOINT
# ============================================================

BASE_URL = "http://10.68.0.39:9008/v1"
MODEL_NAME = "mineru2.5"
API_KEY = "dummy"  # Change this only if your endpoint requires authentication.

TEMPERATURE = 0.0
TOP_P = 1.0
MAX_TOKENS = 1000
REQUEST_DELAY_SECONDS = 2
REQUEST_TIMEOUT_SECONDS = 180
MAX_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 5

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
CSV_FILENAME = "iam_mineru_predictions.csv"


# ============================================================
# 2. STRICT IAM HANDWRITING TRANSCRIPTION PROMPT
# ============================================================

PROMPT = r"""
You are a high-accuracy handwriting transcription system.
Analyze ONLY the provided image and transcribe handwritten text actually visible.

MOST IMPORTANT RULE: WRITE ONLY WHAT IS ACTUALLY VISIBLE IN THE IMAGE.
The image is the only source of truth.

STRICT RULES:
1. Never hallucinate, guess, autocomplete, or infer missing text.
2. Never complete a partially written word or character.
3. Never correct spelling or grammar.
4. Never replace an unclear word with a likely word.
5. Never use external knowledge or assume what the writer intended.
6. Never add words that are not visible or remove visible handwriting.
7. Never summarize, paraphrase, or explain the handwriting.
8. Preserve spelling, letters, capitalization, numbers, punctuation,
   abbreviations, symbols, and word order.
9. If a character or word is genuinely impossible to read, write [UNCLEAR]
   at that position. Do not guess it.
10. If only part of a word is visible, transcribe only the visible part.
11. Preserve meaningful line breaks.
12. Read all visible handwritten regions in natural reading order.
13. Ignore paper texture, background lines, shadows, noise, borders, and blank areas.
14. If printed text appears with handwriting, do not transcribe printed text
    unless it is clearly part of the handwritten content.

OUTPUT FORMAT:
Return ONLY the transcription as plain text.
Do not return JSON, XML, Markdown, code fences, quotation marks, explanations,
comments, or confidence scores.
Before responding, verify that nothing was guessed, corrected, or invented.
Return ONLY the handwritten transcription.
"""


# ============================================================
# 3. INPUT / OUTPUT FOLDERS
# ============================================================

def ask_for_folders() -> tuple[Path, Path]:
    print("=" * 75)
    print("IAM HANDWRITING - MINERU2.5 BATCH TRANSCRIPTION TO CSV")
    print("=" * 75)
    print("Endpoint:", BASE_URL)
    print("Model   :", MODEL_NAME)

    input_value = input("\nEnter input image folder: ").strip().strip('"').strip("'")
    output_value = input("Enter output folder for the CSV: ").strip().strip('"').strip("'")

    if not input_value or not output_value:
        raise SystemExit("ERROR: Both input and output folder paths are required.")

    try:
        input_folder = Path(input_value).expanduser().resolve(strict=True)
    except (OSError, RuntimeError):
        raise SystemExit(f"ERROR: Input folder does not exist: {input_value}")

    if not input_folder.is_dir():
        raise SystemExit(f"ERROR: Input path is not a folder: {input_folder}")

    output_folder = Path(output_value).expanduser().resolve()
    try:
        output_folder.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise SystemExit(f"ERROR: Could not create output folder: {error}")

    return input_folder, output_folder


# ============================================================
# 4. DISCOVER IMAGES
# ============================================================

def get_image_files(input_folder: Path, output_folder: Path) -> list[Path]:
    """Find images recursively, excluding images contained in the output folder."""
    output_resolved = output_folder.resolve()
    images = []

    for path in input_folder.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        try:
            path.resolve().relative_to(output_resolved)
            continue
        except ValueError:
            pass
        images.append(path)

    return sorted(
        images,
        key=lambda p: p.relative_to(input_folder).as_posix().casefold(),
    )


# ============================================================
# 5. IMAGE BASE64 AND MIME TYPE
# ============================================================

def image_to_base64(image_path: Path) -> str:
    with image_path.open("rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")


def get_mime_type(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(image_path))
    return mime_type or "image/png"


# ============================================================
# 6. CLEAN RESPONSE
# ============================================================

def clean_response(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^\s*```(?:text|plaintext)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)

    prefixes = (
        "Here is the transcription:",
        "Here is the transcription",
        "Transcription:",
        "The transcription is:",
        "The text is:",
    )
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()
                changed = True
                break
    return text.strip()


# ============================================================
# 7. CALL MINERU2.5
# ============================================================

def call_model(client: OpenAI, image_path: Path) -> str:
    encoded_image = image_to_base64(image_path)
    mime_type = get_mime_type(image_path)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{encoded_image}"
                        },
                    },
                ],
            }
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
    )

    if not response.choices:
        raise RuntimeError("Endpoint returned no choices.")

    content = response.choices[0].message.content
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Endpoint returned an empty or non-text response.")
    return clean_response(content)


# ============================================================
# 8. CSV LOAD / SAVE (RESUME SUPPORT)
# ============================================================

def load_existing_results(csv_path: Path) -> dict[str, str]:
    """Load existing successful rows so the run can resume."""
    results = {}
    if not csv_path.is_file():
        return results

    try:
        with csv_path.open("r", newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            if not reader.fieldnames or not {"filename", "transcription"}.issubset(reader.fieldnames):
                print("WARNING: Existing CSV has a different format; starting fresh.")
                return {}
            for row in reader:
                filename = (row.get("filename") or "").strip()
                if filename:
                    results[filename] = row.get("transcription") or ""
    except (OSError, csv.Error) as error:
        print(f"WARNING: Could not read old CSV ({error}); starting fresh.")
        return {}
    return results


def save_results(csv_path: Path, filenames: list[str], results: dict[str, str]) -> None:
    """Write a complete CSV via a temporary file to reduce corruption risk."""
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8-sig",
            dir=csv_path.parent,
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            writer = csv.DictWriter(temp_file, fieldnames=["filename", "transcription"])
            writer.writeheader()
            for filename in filenames:
                writer.writerow({
                    "filename": filename,
                    "transcription": results.get(filename, ""),
                })
        temp_path.replace(csv_path)
    finally:
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


# ============================================================
# 9. BATCH PROCESSING
# ============================================================

def main() -> int:
    input_folder, output_folder = ask_for_folders()
    image_files = get_image_files(input_folder, output_folder)
    output_csv = output_folder / CSV_FILENAME

    if not image_files:
        print("No supported images found in:", input_folder)
        return 1

    filenames = [
        image.relative_to(input_folder).as_posix()
        for image in image_files
    ]
    results = load_existing_results(output_csv)

    print("\nInput folder :", input_folder)
    print("Output CSV   :", output_csv)
    print("Total images :", len(image_files))
    print("=" * 75)

    try:
        client = OpenAI(
            base_url=BASE_URL,
            api_key=API_KEY,
            timeout=REQUEST_TIMEOUT_SECONDS,
            max_retries=0,
        )
    except Exception as error:
        print("ERROR initializing OpenAI-compatible client:", error)
        return 1

    successful = failed = skipped = 0

    try:
        for index, (image_path, filename) in enumerate(zip(image_files, filenames), start=1):
            print("\n" + "-" * 75)
            print(f"[{index}/{len(image_files)}] Processing: {filename}")

            # Skip only successful existing rows. Blank/error rows are retried.
            if results.get(filename, "").strip():
                print("Already transcribed in CSV -> SKIPPED")
                skipped += 1
                continue

            transcription = ""
            last_error = None

            for attempt in range(1, MAX_ATTEMPTS + 1):
                try:
                    if REQUEST_DELAY_SECONDS > 0:
                        print(f"Waiting {REQUEST_DELAY_SECONDS} seconds...")
                        time.sleep(REQUEST_DELAY_SECONDS)

                    print("Sending image to MinerU2.5...")
                    transcription = call_model(client, image_path)
                    if not transcription:
                        raise RuntimeError("Model returned an empty transcription.")
                    last_error = None
                    break
                except Exception as error:
                    last_error = error
                    print(f"Attempt {attempt}/{MAX_ATTEMPTS} failed: {type(error).__name__}: {error}")
                    if attempt < MAX_ATTEMPTS:
                        print(f"Retrying in {RETRY_DELAY_SECONDS} seconds...")
                        time.sleep(RETRY_DELAY_SECONDS)

            if last_error is None and transcription:
                results[filename] = transcription
                successful += 1
                print("\nTRANSCRIPTION:\n" + "-" * 75)
                print(transcription)
                print("-" * 75)
            else:
                results[filename] = ""
                failed += 1
                print("FAILED:", last_error)

            try:
                save_results(output_csv, filenames, results)
                print("CSV updated:", output_csv)
            except PermissionError:
                print(
                    f"ERROR: Cannot update {output_csv}. Close it in Excel and rerun."
                )
                return 1
            except OSError as error:
                print("ERROR saving CSV:", error)
                return 1
    finally:
        client.close()

    print("\n" + "=" * 75)
    print("PROCESSING COMPLETE")
    print("=" * 75)
    print("Total images :", len(image_files))
    print("Successful   :", successful)
    print("Skipped      :", skipped)
    print("Failed       :", failed)
    print("\nCSV results saved in:")
    print(output_csv)
    print("=" * 75)

    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

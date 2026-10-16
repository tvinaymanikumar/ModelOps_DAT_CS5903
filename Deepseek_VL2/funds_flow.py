from openai import OpenAI
import base64
import json
import os
import re
import mimetypes


# ============================================================
# 1. DEEPSEEK-VL2 ENDPOINT
# ============================================================

client = OpenAI(
    base_url="http://localhost:9000/v1",
    api_key="dummy"
)

MODEL_NAME = "DeepSeek-VL2"


# ============================================================
# 2. INPUT / OUTPUT FOLDERS
# ============================================================

input_folder = r"C:\Users\telug\Downloads\DAT_Datasets\funds_handwritten_dataset\funds_handwritten_dataset\images"

output_folder = r"C:\Users\telug\Downloads\DAT_Datasets\Results\funds_handwritten"

os.makedirs(
    output_folder,
    exist_ok=True
)


# ============================================================
# 3. SUPPORTED IMAGE FORMATS
# ============================================================

image_extensions = (
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".bmp",
    ".webp"
)


# ============================================================
# 4. GET ALL IMAGES
# ============================================================

image_files = [
    os.path.join(
        input_folder,
        filename
    )
    for filename in os.listdir(input_folder)
    if filename.lower().endswith(image_extensions)
]

image_files.sort()


print("=" * 75)
print("BATCH DOCUMENT INFORMATION EXTRACTION")
print("=" * 75)

print(
    f"Input folder : {input_folder}"
)

print(
    f"Output folder: {output_folder}"
)

print(
    f"Total images : {len(image_files)}"
)

print("=" * 75)


# ============================================================
# 5. GENERIC PROMPT
# ============================================================

prompt = r"""
You are a high-accuracy document information extraction system.

Analyze ONLY the provided image.

The document can be of ANY type.

Your task is to extract the information contained in the image
as structured JSON key-value pairs.

The document may contain printed text, typed text, handwritten
text, tables, headings, sections, labels, numbers, dates,
addresses, names, phone numbers, checkboxes, and other data.

============================================================
MOST IMPORTANT RULE
============================================================

Extract ONLY information that is visibly present in the image.

Nothing more.
Nothing less.

============================================================
ANTI-HALLUCINATION RULES
============================================================

1. NEVER hallucinate.

2. NEVER guess.

3. NEVER autocomplete text.

4. NEVER complete partially visible words.

5. NEVER infer missing characters.

6. NEVER infer missing digits.

7. NEVER correct spelling.

8. NEVER correct grammar.

9. NEVER rewrite extracted text.

10. NEVER normalize names, addresses, dates, or numbers.

11. NEVER use external knowledge.

12. NEVER use common document patterns to invent information.

13. NEVER calculate missing values.

14. NEVER derive one value from another value.

15. If a value is unclear or unreadable, use "".

16. The image is the ONLY source of information.

============================================================
FIELD EXTRACTION
============================================================

Identify the field labels visibly present in the document.

Use the field label as the JSON key.

Use the corresponding visible value as the JSON value.

Do not invent field names.

Do not rename field labels.

============================================================
TEXT TRANSCRIPTION
============================================================

Copy the visible value as accurately as possible.

Preserve:

- spelling
- capitalization
- numbers
- punctuation
- abbreviations
- symbols
- date format
- phone number format

Do not correct or rewrite the text.

============================================================
NUMBERS
============================================================

Copy numbers exactly as visible.

Do NOT:

- add digits
- remove digits
- change digits
- infer missing digits
- add leading zeros
- remove leading zeros
- calculate values

============================================================
DATES AND TIMES
============================================================

Copy dates and times exactly as visible.

Do not change their format.

============================================================
CHECKBOXES / SELECTIONS
============================================================

If the document contains checkboxes or selectable options:

Clearly selected -> "Yes"

Clearly unselected -> "No"

Unclear -> ""

Do not return checkbox symbols.

Do not return "checked" or "unchecked".

============================================================
SIGNATURES
============================================================

Ignore signatures unless there is a clearly labeled signature
field.

If there is a clearly labeled signature field and the signature
can reasonably be read as text, transcribe the visible signature.

If it cannot be read reliably, use "".

Never guess a person's name from a signature.

============================================================
TABLES
============================================================

If the document contains tables:

- associate each value with the correct row and column
- do not move values between rows
- do not move values between columns
- preserve repeated records separately

============================================================
SECTIONS
============================================================

If the document has clearly defined sections, preserve their
hierarchy using nested JSON objects.

Use the visible section heading as the section key.

============================================================
REPEATED RECORDS
============================================================

If the document contains repeated records, keep each record
separate.

Use arrays or nested objects when appropriate.

Do not merge unrelated records.

============================================================
BLANK FIELDS
============================================================

If a clearly labeled field has no visible value, use:

""

Do not invent a value.

============================================================
FIELD ASSOCIATION
============================================================

Associate each value with the correct field label using the
visual structure of the document.

Consider:

- proximity
- alignment
- lines
- boxes
- rows
- columns
- section boundaries

============================================================
OUTPUT FORMAT
============================================================

Return DIRECT JSON key-value pairs.

Do NOT create a "fields" array.

Do NOT create separate "key" and "value" properties.

Use:

{
    "Field Label": "Value"
}

For sections:

{
    "Section Name": {
        "Field Label": "Value"
    }
}

For repeated records, use arrays or nested objects when
appropriate.

============================================================
JSON REQUIREMENTS
============================================================

Return ONLY valid JSON.

Every key must use double quotes.

Every string value must use double quotes.

Separate key-value pairs with commas.

Do not include comments.

Do not include explanations.

Do not include text before or after the JSON.

Do not use Markdown.

Do not use ```json.

============================================================
FINAL CHECK
============================================================

Before returning the JSON, verify:

- Every extracted value is visible in the image.
- Nothing was guessed.
- Nothing was auto-completed.
- No spelling was corrected.
- No characters were invented.
- No digits were invented.
- No values were calculated.
- The correct label was associated with each value.
- Checkbox values are Yes, No, or "".
- Signatures are considered only when there is a signature field.
- The response is valid JSON.

If a value cannot be read directly from the image, use "".

Return ONLY the JSON object.
"""


# ============================================================
# 6. IMAGE -> BASE64
# ============================================================

def image_to_base64(image_path):

    with open(
        image_path,
        "rb"
    ) as image_file:

        image_data = image_file.read()

    return base64.b64encode(
        image_data
    ).decode("utf-8")


# ============================================================
# 7. MIME TYPE
# ============================================================

def get_mime_type(image_path):

    mime_type, _ = mimetypes.guess_type(
        image_path
    )

    if mime_type is None:
        mime_type = "image/png"

    return mime_type


# ============================================================
# 8. CLEAN MODEL RESPONSE
# ============================================================

def clean_response(response_text):

    response_text = response_text.strip()

    # Remove markdown fences
    response_text = re.sub(
        r"^```json\s*",
        "",
        response_text,
        flags=re.IGNORECASE
    )

    response_text = re.sub(
        r"^```\s*",
        "",
        response_text
    )

    response_text = re.sub(
        r"\s*```$",
        "",
        response_text
    )

    response_text = response_text.strip()

    # Extract JSON object if extra text was added
    first_brace = response_text.find("{")
    last_brace = response_text.rfind("}")

    if (
        first_brace != -1
        and last_brace != -1
        and last_brace > first_brace
    ):

        response_text = response_text[
            first_brace:last_brace + 1
        ]

    return response_text.strip()


# ============================================================
# 9. PARSE JSON
# ============================================================

def parse_json_response(response_text):

    cleaned = clean_response(
        response_text
    )

    return json.loads(
        cleaned
    )


# ============================================================
# 10. CALL DEEPSEEK-VL2
# ============================================================

def call_model(
    image_base64,
    mime_type,
    current_prompt
):

    response = client.chat.completions.create(

        model=MODEL_NAME,

        messages=[
            {
                "role": "user",

                "content": [

                    {
                        "type": "text",
                        "text": current_prompt
                    },

                    {
                        "type": "image_url",

                        "image_url": {
                            "url":
                            f"data:{mime_type};base64,"
                            + image_base64
                        }
                    }

                ]
            }
        ],

        temperature=0,

        top_p=1.0,

        max_tokens=2000
    )

    return response.choices[0].message.content


# ============================================================
# 11. PROCESS ONE IMAGE
# ============================================================

def process_image(
    image_path,
    index,
    total_images
):

    filename = os.path.basename(
        image_path
    )

    image_name = os.path.splitext(
        filename
    )[0]

    output_path = os.path.join(
        output_folder,
        image_name + ".json"
    )


    print()
    print("-" * 75)

    print(
        f"[{index}/{total_images}] {filename}"
    )


    # --------------------------------------------------------
    # Skip existing JSON
    # --------------------------------------------------------

    if os.path.exists(
        output_path
    ):

        print(
            "Already processed -> SKIPPED"
        )

        return "skipped"


    try:

        # ----------------------------------------------------
        # Convert image to Base64
        # ----------------------------------------------------

        image_base64 = image_to_base64(
            image_path
        )

        mime_type = get_mime_type(
            image_path
        )


        # ====================================================
        # FIRST MODEL CALL
        # ====================================================

        print(
            "Sending image to DeepSeek-VL2..."
        )

        raw_output = call_model(
            image_base64,
            mime_type,
            prompt
        )


        print()
        print("MODEL OUTPUT:")
        print(raw_output)


        # ----------------------------------------------------
        # Try parsing JSON
        # ----------------------------------------------------

        try:

            result = parse_json_response(
                raw_output
            )

            print(
                "Valid JSON received."
            )


        except json.JSONDecodeError as e:

            print()
            print(
                "Invalid JSON received."
            )

            print(
                f"Line   : {e.lineno}"
            )

            print(
                f"Column : {e.colno}"
            )


            # =================================================
            # RETRY ONCE
            # =================================================

            retry_prompt = prompt + r"""

IMPORTANT RETRY:

Your previous response was not valid JSON.

Generate the extraction again.

Make absolutely sure that:

- every key uses double quotes
- every string value uses double quotes
- every key-value pair is separated by a comma
- nested objects use valid JSON syntax
- arrays use valid JSON syntax
- there are no comments
- there is no text before the JSON
- there is no text after the JSON

Return ONLY ONE valid JSON object.
"""


            print()
            print(
                "Retrying once..."
            )


            raw_output_retry = call_model(
                image_base64,
                mime_type,
                retry_prompt
            )


            print()
            print(
                "RETRY MODEL OUTPUT:"
            )

            print(
                raw_output_retry
            )


            try:

                result = parse_json_response(
                    raw_output_retry
                )

                raw_output = raw_output_retry

                print(
                    "Retry produced valid JSON."
                )


            except json.JSONDecodeError as e2:

                print()
                print(
                    "Retry also produced invalid JSON."
                )

                print(
                    f"Line   : {e2.lineno}"
                )

                print(
                    f"Column : {e2.colno}"
                )


                # --------------------------------------------
                # Save raw outputs
                # --------------------------------------------

                raw_path_1 = output_path.replace(
                    ".json",
                    "_raw_attempt1.txt"
                )

                raw_path_2 = output_path.replace(
                    ".json",
                    "_raw_attempt2.txt"
                )


                with open(
                    raw_path_1,
                    "w",
                    encoding="utf-8"
                ) as f:

                    f.write(
                        raw_output
                    )


                with open(
                    raw_path_2,
                    "w",
                    encoding="utf-8"
                ) as f:

                    f.write(
                        raw_output_retry
                    )


                print(
                    f"Raw output 1 saved to:\n{raw_path_1}"
                )

                print(
                    f"Raw output 2 saved to:\n{raw_path_2}"
                )

                return "failed"


        # ====================================================
        # VALIDATE TOP-LEVEL JSON
        # ====================================================

        if not isinstance(
            result,
            dict
        ):

            print(
                "ERROR: JSON output is not an object."
            )

            return "failed"


        # ====================================================
        # SAVE JSON
        # ====================================================

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                result,
                f,
                indent=4,
                ensure_ascii=False
            )


        print()
        print(
            f"Saved -> {output_path}"
        )

        return "success"


    except Exception as e:

        print()
        print(
            f"ERROR processing {filename}:"
        )

        print(e)

        return "failed"


# ============================================================
# 12. PROCESS ALL IMAGES ONE BY ONE
# ============================================================

successful = 0
failed = 0
skipped = 0


for index, image_path in enumerate(
    image_files,
    start=1
):

    result = process_image(
        image_path,
        index,
        len(image_files)
    )


    if result == "success":

        successful += 1


    elif result == "failed":

        failed += 1


    elif result == "skipped":

        skipped += 1


# ============================================================
# 13. FINAL SUMMARY
# ============================================================

print()
print("=" * 75)
print("PROCESSING COMPLETE")
print("=" * 75)

print(
    f"Total images : {len(image_files)}"
)

print(
    f"Successful   : {successful}"
)

print(
    f"Skipped      : {skipped}"
)

print(
    f"Failed       : {failed}"
)

print()
print(
    "Results saved in:"
)

print(
    output_folder
)

print("=" * 75)
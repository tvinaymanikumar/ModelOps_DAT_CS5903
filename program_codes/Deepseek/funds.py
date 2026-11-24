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
# 2. INPUT IMAGE / OUTPUT JSON
# ============================================================

image_path = r"C:\Users\telug\Downloads\DAT_Datasets\NewDataset\Images\image1.png"

output_path = r"C:\Users\telug\Downloads\DAT_Datasets\Results\NewDataset\image1.json"

os.makedirs(
    os.path.dirname(output_path),
    exist_ok=True
)


# ============================================================
# 3. PROMPT
# ============================================================

prompt = r"""
You are a high-accuracy document information extraction system.

Analyze ONLY the provided image.

Your task is to extract the information contained in the document
as structured JSON key-value pairs.

The document may contain printed text, typed text, handwritten
text, tables, headings, sections, labels, forms, checkboxes,
numbers, dates, addresses, names, phone numbers, and other data.

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

9. NEVER rewrite the extracted text.

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

Identify the field labels that are visibly present in the document.

Use the field label as the JSON key.

Use the corresponding visible value as the JSON value.

Example concept:

Printed label -> value

The label becomes the key.
The visible content becomes the value.

Do not invent field names.

Do not replace a visible field label with a different name.

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
- perform calculations

============================================================
DATES AND TIMES
============================================================

Copy dates and times exactly as visible.

Do not change their format.

============================================================
TABLES
============================================================

If the document contains a table, associate each value with the
correct row and column.

Do not move values between rows or columns.

Preserve repeated records separately.

============================================================
SECTIONS
============================================================

If the document has clearly defined sections, preserve the
section hierarchy using nested JSON objects.

Use the visible section heading as the section key.

For example:

{
    "Section Name": {
        "Field": "Value"
    }
}

Do not create sections when there is no clear visual section.

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
can be read as text, transcribe the visible signature.

If it cannot be read reliably, return "".

Never guess a person's name from a signature.

============================================================
BLANK FIELDS
============================================================

If a clearly labeled field has no visible value, use:

""

Do not invent a value.

============================================================
FIELD ASSOCIATION
============================================================

Associate each value with the correct label based on the visual
layout of the document.

Use:

- proximity
- alignment
- lines
- boxes
- rows
- columns
- section boundaries

Do not associate a value with the wrong nearby label.

============================================================
OUTPUT FORMAT
============================================================

Return DIRECT JSON key-value pairs.

Do NOT create a "fields" array.

Do NOT create separate "key" and "value" properties.

Do NOT add location coordinates.

Do NOT add confidence scores.

Do NOT add explanations.

Do NOT add metadata.

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

For repeated records, use arrays or nested objects when the
document clearly contains repeated records.

============================================================
JSON REQUIREMENTS
============================================================

Return ONLY valid JSON.

Every key must use double quotes.

Every string value must use double quotes.

Separate key-value pairs using commas.

Do not use comments.

Do not use Markdown.

Do not use ```json.

Do not write anything before or after the JSON.

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
- The JSON is syntactically valid.

If a value cannot be read directly from the image, use "".

Return ONLY the JSON object.
"""


# ============================================================
# 4. IMAGE -> BASE64
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
# 5. GET MIME TYPE
# ============================================================

def get_mime_type(image_path):

    mime_type, _ = mimetypes.guess_type(
        image_path
    )

    if mime_type is None:
        mime_type = "image/png"

    return mime_type


# ============================================================
# 6. CLEAN MODEL RESPONSE
# ============================================================

def clean_response(response_text):

    response_text = response_text.strip()

    # Remove Markdown code fences
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

    # Find JSON object if model added extra text
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
# 7. PARSE JSON
# ============================================================

def parse_json_response(response_text):

    cleaned = clean_response(
        response_text
    )

    return json.loads(
        cleaned
    )


# ============================================================
# 8. CALL DEEPSEEK-VL2
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
# 9. MAIN
# ============================================================

print("=" * 70)
print("DOCUMENT EXTRACTION - SINGLE IMAGE")
print("=" * 70)

print(
    f"Input image:\n{image_path}"
)

print()

print(
    f"Output JSON:\n{output_path}"
)

print("=" * 70)


try:

    # --------------------------------------------------------
    # Check image exists
    # --------------------------------------------------------

    if not os.path.exists(image_path):

        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )


    # --------------------------------------------------------
    # Convert image to Base64
    # --------------------------------------------------------

    image_base64 = image_to_base64(
        image_path
    )

    mime_type = get_mime_type(
        image_path
    )


    # --------------------------------------------------------
    # First attempt
    # --------------------------------------------------------

    print()
    print("Sending image to DeepSeek-VL2...")

    raw_output = call_model(
        image_base64,
        mime_type,
        prompt
    )


    print()
    print("MODEL OUTPUT")
    print("=" * 70)
    print(raw_output)
    print("=" * 70)


    # --------------------------------------------------------
    # Parse JSON
    # --------------------------------------------------------

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
            "First response was not valid JSON."
        )

        print(
            f"Line   : {e.lineno}"
        )

        print(
            f"Column : {e.colno}"
        )

        # ----------------------------------------------------
        # Retry once
        # ----------------------------------------------------

        retry_prompt = prompt + r"""

IMPORTANT RETRY:

Your previous response was not valid JSON.

Generate the extraction again.

Make sure:

- every key is enclosed in double quotes
- every string value is enclosed in double quotes
- every key-value pair is separated by a comma
- nested objects are valid JSON
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
        print("RETRY MODEL OUTPUT")
        print("=" * 70)
        print(raw_output_retry)
        print("=" * 70)


        try:

            result = parse_json_response(
                raw_output_retry
            )

            print(
                "Retry produced valid JSON."
            )

        except json.JSONDecodeError as e2:

            print()
            print(
                "ERROR: MODEL DID NOT RETURN VALID JSON"
            )

            print(
                f"Line   : {e2.lineno}"
            )

            print(
                f"Column : {e2.colno}"
            )


            # Save raw responses
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

            raise


    # --------------------------------------------------------
    # Validate top-level object
    # --------------------------------------------------------

    if not isinstance(
        result,
        dict
    ):

        raise ValueError(
            "Model output is not a JSON object."
        )


    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Print final JSON
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL JSON")
    print("=" * 70)

    print(
        json.dumps(
            result,
            indent=4,
            ensure_ascii=False
        )
    )

    print("=" * 70)

    print()
    print("SUCCESS")

    print(
        f"JSON saved to:\n{output_path}"
    )


except FileNotFoundError as e:

    print()
    print("FILE ERROR")
    print(e)


except json.JSONDecodeError as e:

    print()
    print("JSON ERROR")
    print(e)


except Exception as e:

    print()
    print("ERROR")
    print(e)
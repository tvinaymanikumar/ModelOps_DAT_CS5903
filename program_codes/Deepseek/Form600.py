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

image_path = r"C:\Users\telug\Downloads\DAT_Datasets\Forms600\Images\13.png"

output_path = r"C:\Users\telug\Downloads\DAT_Datasets\Results\Forms600\13.json"

os.makedirs(
    os.path.dirname(output_path),
    exist_ok=True
)


# ============================================================
# 3. PROMPT
# ============================================================

prompt = r"""
You are a high-accuracy handwritten form transcription system.

Analyze ONLY the provided image.

The form can be of ANY type.

Your task is to TRANSCRIBE information that is visibly handwritten
or manually filled in the image.

This is a transcription task, NOT an interpretation task.

============================================================
MOST IMPORTANT RULE
============================================================

TRANSCRIBE ONLY WHAT IS ACTUALLY VISIBLE IN THE IMAGE.

Nothing more.
Nothing less.

============================================================
STRICT ANTI-HALLUCINATION RULES
============================================================

1. NEVER hallucinate.

2. NEVER guess.

3. NEVER autocomplete text.

4. NEVER complete a partially written word.

5. NEVER complete a partially visible name.

6. NEVER infer missing characters.

7. NEVER infer missing digits.

8. NEVER correct spelling.

9. NEVER correct grammar.

10. NEVER rewrite handwritten text.

11. NEVER normalize handwritten text.

12. NEVER replace unclear handwriting with a likely word.

13. NEVER use external knowledge.

14. NEVER use common sense to fill missing information.

15. NEVER derive one value from another value.

16. NEVER create information that is not visible in the image.

17. If handwriting is unclear or unreadable, use "".

18. The image is the ONLY source of information.

============================================================
PRINTED TEXT VS HANDWRITTEN TEXT
============================================================

Use the corresponding PRINTED FIELD LABEL as the JSON key.

Use the visible HANDWRITTEN or manually filled content as
the JSON value.

Do NOT copy printed instructions as values.

Do NOT copy printed examples as values.

Do NOT treat ordinary printed text as handwritten data.

Do NOT invent a field label.

============================================================
FORM TYPE
============================================================

The form type is UNKNOWN.

Do NOT assume a particular form type.

Do NOT assume a predefined schema.

Do NOT assume specific field names.

Determine the fields directly from the current image.

Every image must be analyzed independently.

============================================================
HANDWRITTEN TEXT
============================================================

Copy the visible handwriting exactly.

Preserve:

- spelling
- letters
- numbers
- punctuation
- abbreviations
- capitalization where visible
- symbols

Do NOT:

- correct spelling
- expand abbreviations
- complete words
- rewrite text
- infer characters

============================================================
NUMBERS
============================================================

Copy handwritten numbers exactly as visible.

Do NOT:

- add digits
- remove digits
- change digits
- infer missing digits
- add leading zeros
- remove leading zeros
- calculate values

============================================================
DATES
============================================================

Copy handwritten dates exactly as visible.

Do NOT change the visible date format.

If day, month, and year are separate printed fields,
extract them separately.

Do not infer missing date components.

============================================================
CHECKBOXES / RADIO BUTTONS / SELECTIONS
============================================================

For a checkbox, radio button, tick, cross, or selection:

Clearly selected/marked:
"Yes"

Clearly empty/unselected:
"No"

Cannot determine:
""

Use ONLY:

"Yes"
"No"
""

Do NOT return:

"checked"
"unchecked"
"selected"
"not selected"
"✓"
"X"

For multiple options belonging to the same selection group,
preserve the printed option labels as keys when appropriate.

============================================================
SIGNATURES
============================================================

Only consider a signature when the form has a clearly
identified signature field or signature label.

If a signature field exists and the signature can reasonably
be read as text, transcribe the visible signature.

If the signature cannot be reliably read as text, use "".

If there is a signature without a corresponding signature field,
ignore it.

NEVER guess a person's name from a signature.

============================================================
SECTIONS
============================================================

If the form contains clearly defined sections, preserve
the section structure using nested JSON objects.

Use the visible section heading as the object key.

Do not merge unrelated sections.

============================================================
REPEATED ROWS / RECORDS
============================================================

If the form contains repeated rows or records, keep them
separate.

Use arrays of objects when several records have the same
structure.

Do NOT merge data from different rows.

Do NOT move a value from one row to another.

============================================================
TABLES
============================================================

For tables, associate handwritten values with the correct
row and column using the visual layout.

Consider:

- row alignment
- column alignment
- field boundaries
- boxes
- proximity

Do not associate a value with the wrong field.

============================================================
BLANK FIELDS
============================================================

If a clearly labeled field is present but has no handwritten
or manually entered value, use:

""

Do NOT invent a value.

============================================================
MULTI-LINE VALUES
============================================================

If one field contains handwriting over multiple lines,
combine the visible lines into one JSON string.

Do not omit visible content.

Do not add missing content.

============================================================
OUTPUT FORMAT
============================================================

Return DIRECT JSON KEY-VALUE PAIRS.

Do NOT create a "fields" array.

Do NOT create:

{
    "key": "...",
    "value": "..."
}

Instead use:

{
    "Field Label": "Handwritten Value"
}

For sections:

{
    "Section Name": {
        "Field Label": "Handwritten Value"
    }
}

For repeated records, use arrays or nested objects when
the visual structure clearly contains repeated records.

============================================================
JSON RULES
============================================================

The response MUST be valid JSON.

Every key and string value MUST use double quotes.

Separate every key-value pair with a comma.

Do NOT include comments.

Do NOT include explanations.

Do NOT include text before the JSON.

Do NOT include text after the JSON.

Do NOT use Markdown.

Do NOT use ```json.

============================================================
FINAL VALIDATION
============================================================

Before generating the final answer, check:

- Did I extract only visible handwritten/fill-in information?
- Did I avoid guessing?
- Did I avoid hallucination?
- Did I avoid autocomplete?
- Did I avoid correcting spelling?
- Did I avoid inventing characters?
- Did I avoid inventing digits?
- Did I ignore printed instructions?
- Did I handle checkboxes as Yes/No?
- Did I consider signatures only when there is a signature field?
- Is the JSON syntactically valid?

If a value cannot be read directly from the image,
use "" rather than guessing.

============================================================
FINAL RESPONSE
============================================================

Return ONLY the valid JSON object.
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
# 5. GET IMAGE MIME TYPE
# ============================================================

def get_mime_type(image_path):

    mime_type, _ = mimetypes.guess_type(
        image_path
    )

    if mime_type is None:

        mime_type = "image/png"

    return mime_type


# ============================================================
# 6. CLEAN MODEL OUTPUT
# ============================================================

def clean_response(response_text):

    response_text = response_text.strip()

    # Remove Markdown fences
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

    # --------------------------------------------------------
    # Find JSON object inside any accidental extra text
    # --------------------------------------------------------

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

    try:

        return json.loads(
            cleaned
        )

    except json.JSONDecodeError:

        # ----------------------------------------------------
        # Try JSONDecoder raw parsing
        # ----------------------------------------------------

        decoder = json.JSONDecoder()

        try:

            result, _ = decoder.raw_decode(
                cleaned
            )

            return result

        except Exception:

            raise


# ============================================================
# 8. SEND IMAGE TO MODEL
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
print("FORMS600 - SINGLE IMAGE EXTRACTION")
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
    # Convert image
    # --------------------------------------------------------

    image_base64 = image_to_base64(
        image_path
    )

    mime_type = get_mime_type(
        image_path
    )


    # ========================================================
    # FIRST ATTEMPT
    # ========================================================

    print()
    print("Sending image to DeepSeek-VL2...")

    raw_output = call_model(
        image_base64,
        mime_type,
        prompt
    )


    print()
    print("FIRST MODEL OUTPUT")
    print("=" * 70)
    print(raw_output)
    print("=" * 70)


    # --------------------------------------------------------
    # Try parsing JSON
    # --------------------------------------------------------

    try:

        result = parse_json_response(
            raw_output
        )

        print()
        print("First response contains valid JSON.")


    except json.JSONDecodeError as e:

        print()
        print("=" * 70)
        print("FIRST RESPONSE WAS NOT VALID JSON")
        print("=" * 70)

        print(
            f"Line   : {e.lineno}"
        )

        print(
            f"Column : {e.colno}"
        )

        print(
            f"Position: {e.pos}"
        )


        # ====================================================
        # SECOND ATTEMPT
        # ====================================================

        retry_prompt = prompt + r"""

IMPORTANT RETRY INSTRUCTION:

Your previous response was not valid JSON.

Generate the answer again.

This time, pay special attention to JSON syntax.

Every object must use:

"key": "value"

Separate every key-value pair with a comma.

Use double quotes around every key and string value.

Do not put unescaped double quotes inside a string.

Do not add explanations.

Return ONLY ONE valid JSON object.
"""

        print()
        print(
            "Retrying once with stricter JSON instructions..."
        )


        raw_output_retry = call_model(
            image_base64,
            mime_type,
            retry_prompt
        )


        print()
        print("SECOND MODEL OUTPUT")
        print("=" * 70)
        print(raw_output_retry)
        print("=" * 70)


        try:

            result = parse_json_response(
                raw_output_retry
            )

            print()
            print(
                "Retry produced valid JSON."
            )

            # Use successful retry output
            raw_output = raw_output_retry


        except json.JSONDecodeError as e2:

            print()
            print("=" * 70)
            print("ERROR: MODEL DID NOT RETURN VALID JSON")
            print("=" * 70)

            print(
                f"Line   : {e2.lineno}"
            )

            print(
                f"Column : {e2.colno}"
            )

            print(
                f"Position: {e2.pos}"
            )


            # ------------------------------------------------
            # Save first raw response
            # ------------------------------------------------

            raw_path_1 = output_path.replace(
                ".json",
                "_raw_attempt1.txt"
            )

            with open(
                raw_path_1,
                "w",
                encoding="utf-8"
            ) as f:

                f.write(
                    raw_output
                )


            # ------------------------------------------------
            # Save second raw response
            # ------------------------------------------------

            raw_path_2 = output_path.replace(
                ".json",
                "_raw_attempt2.txt"
            )

            with open(
                raw_path_2,
                "w",
                encoding="utf-8"
            ) as f:

                f.write(
                    raw_output_retry
                )


            print()
            print(
                f"First raw response saved to:\n{raw_path_1}"
            )

            print(
                f"Second raw response saved to:\n{raw_path_2}"
            )

            raise


    # ========================================================
    # VALIDATE TOP-LEVEL JSON
    # ========================================================

    if not isinstance(
        result,
        dict
    ):

        raise ValueError(
            "Model output is valid JSON but is not a JSON object."
        )


    # ========================================================
    # SAVE FINAL JSON
    # ========================================================

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


    # ========================================================
    # PRINT FINAL RESULT
    # ========================================================

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
    print("=" * 70)
    print("FILE ERROR")
    print("=" * 70)
    print(e)


except json.JSONDecodeError as e:

    print()
    print("=" * 70)
    print("JSON ERROR")
    print("=" * 70)
    print(e)


except Exception as e:

    print()
    print("=" * 70)
    print("ERROR")
    print("=" * 70)
    print(e)
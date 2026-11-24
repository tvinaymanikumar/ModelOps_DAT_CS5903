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

image_path = r"C:\Users\telug\Downloads\DAT_Datasets\Nigeria_medical_records\Nigeria_medical_records\Images\AMR_001_HTR.png"

output_path = r"C:\Users\telug\Downloads\DAT_Datasets\Results\Nigeria\AMR_001_HTR_test.json"

os.makedirs(
    os.path.dirname(output_path),
    exist_ok=True
)


# ============================================================
# 3. PROMPT
# ============================================================

prompt = r"""
You are a highly accurate handwritten text extraction system.

Analyze ONLY the provided image.

Extract ONLY the following three fields:

{
    "name": "",
    "age": "",
    "sex": ""
}

Ignore all other information in the image.

============================================================
STRICT ANTI-HALLUCINATION RULES
============================================================

1. NEVER hallucinate.

2. NEVER guess.

3. NEVER autocomplete text.

4. NEVER complete partially visible handwriting.

5. NEVER infer missing characters.

6. NEVER infer missing digits.

7. NEVER correct spelling.

8. NEVER replace an unclear handwritten value with a likely value.

9. NEVER use external knowledge.

10. NEVER use information from any other part of the document
    to guess a missing value.

11. Extract ONLY what is visibly present in the image.

12. If a value cannot be read reliably, return "".

13. Do not invent any information.

============================================================
NAME
============================================================

Extract the person's name exactly as it is visibly handwritten.

Do not correct spelling.

Do not autocomplete the name.

Do not add missing words.

If the name is unclear, return:

"name": ""

============================================================
AGE
============================================================

Extract the age exactly from the image.

Convert the age to a clear textual representation using
"years" and/or "months".

RULES:

1. If the image contains ONLY a number and no visible age unit,
   interpret the number as YEARS.

   Example:
   41
   -> "41 years"

2. If the image contains a number followed by:
   y, yr, yrs, year, or years,
   output the number using "years".

   Example:
   41 y
   -> "41 years"

   Example:
   41 years
   -> "41 years"

3. If the image contains a number followed by:
   m, mo, month, or months,
   output the number using "months".

   Example:
   6 m
   -> "6 months"

   Example:
   6 months
   -> "6 months"

4. If the image contains both years and months,
   preserve both.

   Example:
   2 years 6 months
   -> "2 years 6 months"

5. If the image contains:
   2y 6m
   output:
   "2 years 6 months"

6. Do NOT calculate or convert years into months.

7. Do NOT convert months into years.

8. Do NOT invent months if only years are visible.

9. Do NOT invent years if only months are visible.

10. If the age is unclear or unreadable, return:
    ""

11. If only a number is visible and it is clearly the age field,
    treat that number as years.

The image is the only source of truth.

============================================================
SEX
============================================================

Read the sex/gender information directly from the image.

If the image clearly indicates:

M -> return "Male"

F -> return "Female"

Male -> return "Male"

Female -> return "Female"

Do NOT return:

"M"
"F"

Do NOT guess the sex from the person's name.

If the sex cannot be determined clearly from the image,
return:

"sex": ""

============================================================
IMPORTANT
============================================================

The image is the ONLY source of truth.

Do not use common knowledge.

Do not infer missing information.

Do not autocomplete handwriting.

Do not correct handwriting.

Transcribe only what is actually visible.

============================================================
OUTPUT
============================================================

Return ONLY valid JSON.

Do not use Markdown.

Do not use ```json.

Do not provide explanations.

Do not add extra fields.

The output MUST contain exactly these three keys:

{
    "name": "",
    "age": "",
    "sex": ""
}
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
# 5. MIME TYPE
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

    # Remove markdown code fences
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

    # Extract JSON object if extra text was returned
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
# 8. CALL MODEL
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

        max_tokens=200
    )

    return response.choices[0].message.content


# ============================================================
# 9. VALIDATE FINAL JSON
# ============================================================

def validate_result(result):

    if not isinstance(result, dict):
        raise ValueError(
            "Model output is not a JSON object."
        )

    final_result = {
        "name": "",
        "age": "",
        "sex": ""
    }

    # --------------------------------------------------------
    # Name
    # --------------------------------------------------------

    name = result.get(
        "name",
        ""
    )

    if name is None:
        name = ""

    final_result["name"] = str(
        name
    ).strip()


    # --------------------------------------------------------
    # Age
    # --------------------------------------------------------

    age = result.get(
        "age",
        ""
    )

    if age is None:
        age = ""

    final_result["age"] = str(
        age
    ).strip()


    # --------------------------------------------------------
    # Sex
    # --------------------------------------------------------

    sex = result.get(
        "sex",
        ""
    )

    if sex is None:
        sex = ""

    sex = str(
        sex
    ).strip()


    # Normalize only allowed sex outputs
    if sex.lower() == "m":
        sex = "Male"

    elif sex.lower() == "f":
        sex = "Female"

    elif sex.lower() == "male":
        sex = "Male"

    elif sex.lower() == "female":
        sex = "Female"

    elif sex != "":
        # Do not allow arbitrary output
        sex = ""


    final_result["sex"] = sex

    return final_result


# ============================================================
# 10. MAIN
# ============================================================

print("=" * 70)
print("HANDWRITTEN AGE / NAME / SEX EXTRACTION")
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
    # Check image
    # --------------------------------------------------------

    if not os.path.exists(
        image_path
    ):
        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )


    # --------------------------------------------------------
    # Encode image
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
    print(
        "Sending image to DeepSeek-VL2..."
    )

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


        # ====================================================
        # RETRY
        # ====================================================

        retry_prompt = prompt + r"""

IMPORTANT:

Your previous response was not valid JSON.

Generate the answer again.

Return ONLY this JSON structure:

{
    "name": "",
    "age": "",
    "sex": ""
}

Make sure:
- all keys use double quotes
- all string values use double quotes
- key-value pairs are separated by commas
- no extra text is returned
- no Markdown is returned
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

            raw_output = raw_output_retry

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


            # Save raw outputs
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


    # ========================================================
    # VALIDATE
    # ========================================================

    final_result = validate_result(
        result
    )


    # ========================================================
    # SAVE JSON
    # ========================================================

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            final_result,
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
            final_result,
            indent=4,
            ensure_ascii=False
        )
    )

    print("=" * 70)

    print()
    print(
        "SUCCESS"
    )

    print(
        f"JSON saved to:\n{output_path}"
    )


except FileNotFoundError as e:

    print()
    print(
        "FILE ERROR:"
    )

    print(e)


except Exception as e:

    print()
    print(
        "ERROR:"
    )

    print(e)
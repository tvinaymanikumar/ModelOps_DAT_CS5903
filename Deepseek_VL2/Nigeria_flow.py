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

input_folder = r"C:\Users\telug\Downloads\DAT_Datasets\Nigeria_medical_records\Nigeria_medical_records\Images"

output_folder = r"C:\Users\telug\Downloads\DAT_Datasets\Results\Nigeria"

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


print("=" * 70)
print("BATCH NAME / AGE / SEX EXTRACTION")
print("=" * 70)

print(
    f"Input folder : {input_folder}"
)

print(
    f"Output folder: {output_folder}"
)

print(
    f"Total images : {len(image_files)}"
)

print("=" * 70)


# ============================================================
# 5. PROMPT
# ============================================================

prompt = r"""
You are a highly accurate handwritten medical information
extraction system.

Analyze ONLY the provided image.

Extract ONLY these three fields:

{
    "name": "",
    "age": "",
    "sex": ""
}

Ignore all other information in the image.

This is an extraction/transcription task, NOT an interpretation
or reasoning task.

============================================================
MOST IMPORTANT RULE
============================================================

TRANSCRIBE ONLY INFORMATION THAT IS ACTUALLY VISIBLE IN THE IMAGE.

Nothing more.
Nothing less.

The image is the ONLY source of truth.

============================================================
STRICT ANTI-HALLUCINATION RULES
============================================================

1. NEVER hallucinate.

2. NEVER guess.

3. NEVER autocomplete text.

4. NEVER complete partially visible handwriting.

5. NEVER complete a partially written name.

6. NEVER infer missing characters.

7. NEVER infer missing digits.

8. NEVER correct spelling.

9. NEVER replace an unclear handwritten value with a likely value.

10. NEVER use external knowledge.

11. NEVER infer age from the person's name.

12. NEVER infer sex from the person's name.

13. NEVER infer age from unrelated numbers.

14. NEVER derive missing information from another field.

15. NEVER invent information.

16. Extract ONLY information visibly present in the image.

17. If a value cannot be read reliably, return "".

18. Do not use common sense to fill missing information.

============================================================
NAME
============================================================

Extract the person's name exactly as visibly written.

The name may appear:

- in a labeled field
- inside a sentence
- inside a paragraph
- near other patient information

Copy the visible name exactly.

Do NOT:

- correct spelling
- autocomplete the name
- add missing words
- invent a surname
- expand initials into a full name

If the name cannot be read reliably:

"name": ""

============================================================
AGE
============================================================

Extract the person's age directly from the image.

The age may appear:

- as a separate labeled field
- inside a paragraph
- inside a sentence
- beside the name
- beside the sex
- in compact notation
- using abbreviations

Use ONLY a number that is clearly associated with the
person's age.

Do NOT use:

- dates
- years from dates
- phone numbers
- patient IDs
- registration numbers
- hospital numbers
- medical record numbers
- room numbers
- unrelated numerical values

------------------------------------------------------------
AGE WITH NO UNIT
------------------------------------------------------------

If ONLY a number is visible and it is clearly identified
as the person's age, treat it as YEARS.

Examples:

41
-> "41 years"

27
-> "27 years"

------------------------------------------------------------
AGE IN YEARS
------------------------------------------------------------

The following indicate YEARS:

y
yr
yrs
year
years

Always output them as:

"<number> years"

Examples:

41y
-> "41 years"

41yr
-> "41 years"

41yrs
-> "41 years"

41 year
-> "41 years"

41 years
-> "41 years"

------------------------------------------------------------
AGE IN MONTHS
------------------------------------------------------------

The following indicate MONTHS:

m
mo
mos
month
months

Always output them as:

"<number> months"

Examples:

6m
-> "6 months"

6mo
-> "6 months"

6mos
-> "6 months"

6month
-> "6 months"

6months
-> "6 months"

------------------------------------------------------------
YEARS AND MONTHS TOGETHER
------------------------------------------------------------

If both years and months are explicitly visible, preserve both.

Examples:

2y 6m
-> "2 years 6 months"

2yrs 6months
-> "2 years 6 months"

2 years 6 months
-> "2 years 6 months"

Do NOT convert years into months.

Do NOT convert months into years.

Do NOT invent a missing unit.

------------------------------------------------------------
PARAGRAPH / SENTENCE FORMAT
------------------------------------------------------------

Age information may appear inside a paragraph or sentence.

Identify the age ONLY when the surrounding text clearly
shows that the number refers to the person's age.

Do not select an unrelated number simply because it appears
near the patient information.

If the context does not clearly identify the number as age,
return:

"age": ""

------------------------------------------------------------
COMPACT AGE + SEX FORMAT
------------------------------------------------------------

Age and sex may appear together in compact notation.

Possible formats include:

41/F
41/M
41y/F
41y/M
41yr/F
41yr/M
41yrs/F
41yrs/M
41years/F
41years/M
6m/F
6m/M
6mo/F
6mo/M
6months/F
6months/M

Interpret only what is visibly written.

For example:

41yrs/F

means:

age = "41 years"
sex = "Female"

41/M

means:

age = "41 years"
sex = "Male"

6m/F

means:

age = "6 months"
sex = "Female"

Do NOT infer anything beyond the visible notation.

============================================================
SEX
============================================================

Extract the person's sex directly from the image.

Possible visible forms include:

M
F
Male
Female

Convert them to exactly:

M -> "Male"

F -> "Female"

Male -> "Male"

Female -> "Female"

Do NOT return "M".

Do NOT return "F".

Do NOT infer sex from:

- name
- appearance
- diagnosis
- occupation
- context
- other information

If sex cannot be determined clearly:

"sex": ""

============================================================
MISSING OR UNCLEAR VALUES
============================================================

If name is unclear:

"name": ""

If age is unclear:

"age": ""

If sex is unclear:

"sex": ""

NEVER guess.

============================================================
HANDWRITING
============================================================

Copy handwriting exactly as visible.

Preserve:

- spelling
- letters
- digits
- punctuation
- abbreviations

Do NOT:

- correct spelling
- autocomplete
- complete initials
- infer hidden characters
- rewrite unclear handwriting

============================================================
OUTPUT FORMAT
============================================================

Return ONLY valid JSON.

The output MUST contain exactly these three keys:

{
    "name": "",
    "age": "",
    "sex": ""
}

Do NOT add any other fields.

Do NOT add explanations.

Do NOT add comments.

Do NOT add confidence scores.

Do NOT add location information.

Do NOT add metadata.

Do NOT use Markdown.

Do NOT use ```json.

Do NOT write anything before or after the JSON.

============================================================
FINAL VALIDATION
============================================================

Before returning the JSON, verify:

1. The name is directly visible.

2. The age is directly visible or clearly identified
   in the surrounding text.

3. A bare age number is treated as years ONLY when it is
   clearly identified as the person's age.

4. y, yr, yrs, year, years -> "years".

5. m, mo, mos, month, months -> "months".

6. If both years and months are explicitly present,
   preserve both.

7. M -> "Male".

8. F -> "Female".

9. Do not infer sex from the name.

10. Do not use unrelated numbers as age.

11. Do not hallucinate.

12. Do not autocomplete.

13. Do not correct spelling.

14. Do not invent information.

15. The output is valid JSON.

If any information cannot be directly determined from
the image, use "".

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

    # Remove Markdown JSON fences
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

    # Find JSON object
    start = response_text.find("{")
    end = response_text.rfind("}")

    if (
        start != -1
        and end != -1
        and end > start
    ):

        response_text = response_text[
            start:end + 1
        ]

    return response_text.strip()


# ============================================================
# 9. PARSE JSON
# ============================================================

def parse_json(response_text):

    cleaned = clean_response(
        response_text
    )

    return json.loads(
        cleaned
    )


# ============================================================
# 10. NORMALIZE AGE
# ============================================================

def normalize_age(age):

    if age is None:
        return ""

    age = str(age).strip()

    if age == "":
        return ""


    # Convert multiple spaces
    age = re.sub(
        r"\s+",
        " ",
        age
    ).strip()


    # --------------------------------------------------------
    # Years + months
    # --------------------------------------------------------

    pattern_both = re.match(
        r"^\s*(\d+)\s*(?:y|yr|yrs|year|years)\s*"
        r"(?:and\s*)?"
        r"(\d+)\s*(?:m|mo|mos|month|months)\s*$",
        age,
        flags=re.IGNORECASE
    )

    if pattern_both:

        years = pattern_both.group(1)
        months = pattern_both.group(2)

        return (
            f"{years} years "
            f"{months} months"
        )


    # --------------------------------------------------------
    # Months only
    # --------------------------------------------------------

    pattern_months = re.match(
        r"^\s*(\d+)\s*"
        r"(?:m|mo|mos|month|months)\s*$",
        age,
        flags=re.IGNORECASE
    )

    if pattern_months:

        months = pattern_months.group(1)

        return f"{months} months"


    # --------------------------------------------------------
    # Years only
    # --------------------------------------------------------

    pattern_years = re.match(
        r"^\s*(\d+)\s*"
        r"(?:y|yr|yrs|year|years)\s*$",
        age,
        flags=re.IGNORECASE
    )

    if pattern_years:

        years = pattern_years.group(1)

        return f"{years} years"


    # --------------------------------------------------------
    # Number only
    # --------------------------------------------------------

    pattern_number = re.match(
        r"^\s*(\d+)\s*$",
        age
    )

    if pattern_number:

        years = pattern_number.group(1)

        return f"{years} years"


    # --------------------------------------------------------
    # Compact age + sex accidentally returned in age
    # Example: 41yrs/F
    # --------------------------------------------------------

    pattern_compact = re.match(
        r"^\s*(\d+)\s*"
        r"(y|yr|yrs|year|years|m|mo|mos|month|months)?"
        r"\s*/\s*[MF]\s*$",
        age,
        flags=re.IGNORECASE
    )

    if pattern_compact:

        number = pattern_compact.group(1)

        unit = pattern_compact.group(2)

        if unit is None:
            return f"{number} years"

        if unit.lower() in (
            "m",
            "mo",
            "mos",
            "month",
            "months"
        ):
            return f"{number} months"

        return f"{number} years"


    # --------------------------------------------------------
    # If model returned a more complex expression,
    # preserve it rather than inventing a transformation.
    # --------------------------------------------------------

    return age


# ============================================================
# 11. NORMALIZE SEX
# ============================================================

def normalize_sex(sex):

    if sex is None:
        return ""

    sex = str(sex).strip()

    if sex.lower() in (
        "m",
        "male"
    ):

        return "Male"

    if sex.lower() in (
        "f",
        "female"
    ):

        return "Female"

    return ""


# ============================================================
# 12. VALIDATE RESULT
# ============================================================

def validate_result(result):

    if not isinstance(
        result,
        dict
    ):

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

    final_result["age"] = normalize_age(
        age
    )


    # --------------------------------------------------------
    # Sex
    # --------------------------------------------------------

    sex = result.get(
        "sex",
        ""
    )

    final_result["sex"] = normalize_sex(
        sex
    )


    return final_result


# ============================================================
# 13. CALL DEEPSEEK-VL2
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
# 14. PROCESS ONE IMAGE
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
    print("-" * 70)

    print(
        f"[{index}/{total_images}] "
        f"Processing: {filename}"
    )


    # --------------------------------------------------------
    # Skip existing result
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
        # Encode image
        # ----------------------------------------------------

        image_base64 = image_to_base64(
            image_path
        )

        mime_type = get_mime_type(
            image_path
        )


        # ====================================================
        # FIRST MODEL ATTEMPT
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


        # ====================================================
        # PARSE FIRST RESPONSE
        # ====================================================

        try:

            result = parse_json(
                raw_output
            )


        except json.JSONDecodeError:

            print(
                "Invalid JSON. Retrying once..."
            )


            # =================================================
            # RETRY PROMPT
            # =================================================

            retry_prompt = prompt + r"""

IMPORTANT RETRY:

Your previous response was not valid JSON.

Return ONLY:

{
    "name": "",
    "age": "",
    "sex": ""
}

Make sure:

- JSON syntax is valid
- all keys use double quotes
- all string values use double quotes
- commas are present between key-value pairs
- there is no text before the JSON
- there is no text after the JSON

Do not change the extraction rules.
Return ONLY the JSON object.
"""


            raw_output_retry = call_model(
                image_base64,
                mime_type,
                retry_prompt
            )


            print()
            print("RETRY OUTPUT:")
            print(raw_output_retry)


            try:

                result = parse_json(
                    raw_output_retry
                )

                raw_output = raw_output_retry


            except json.JSONDecodeError:

                print(
                    "Retry also returned invalid JSON."
                )


                # Save raw responses
                raw1 = output_path.replace(
                    ".json",
                    "_raw_attempt1.txt"
                )

                raw2 = output_path.replace(
                    ".json",
                    "_raw_attempt2.txt"
                )


                with open(
                    raw1,
                    "w",
                    encoding="utf-8"
                ) as f:

                    f.write(
                        raw_output
                    )


                with open(
                    raw2,
                    "w",
                    encoding="utf-8"
                ) as f:

                    f.write(
                        raw_output_retry
                    )


                print(
                    f"Raw output saved: {raw1}"
                )

                print(
                    f"Raw retry saved: {raw2}"
                )

                return "failed"


        # ====================================================
        # VALIDATE / NORMALIZE
        # ====================================================

        final_result = validate_result(
            result
        )


        # ====================================================
        # SAVE JSON
        # ====================================================

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


        # ====================================================
        # PRINT RESULT
        # ====================================================

        print()
        print("FINAL JSON:")
        print(
            json.dumps(
                final_result,
                indent=4,
                ensure_ascii=False
            )
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
# 15. PROCESS ALL IMAGES ONE BY ONE
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
# 16. FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("PROCESSING COMPLETE")
print("=" * 70)

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

print("=" * 70)
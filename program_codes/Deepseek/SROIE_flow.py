from openai import OpenAI
import base64
import json
import os
import re


# ============================================================
# 1. DEEPSEEK-VL2 CONNECTION
# ============================================================

client = OpenAI(
    base_url="http://localhost:9000/v1",
    api_key="dummy"
)

MODEL_NAME = "DeepSeek-VL2"


# ============================================================
# 2. INPUT / OUTPUT FOLDERS
# ============================================================

input_folder = r"C:\Users\telug\Downloads\DAT_Datasets\SROIE_Invoice Dataset\Images"

output_folder = r"C:\Users\telug\Downloads\DAT_Datasets\Results\SROIE_Invoice"

os.makedirs(output_folder, exist_ok=True)


# ============================================================
# 3. GET JPG IMAGES
# ============================================================

image_files = [
    os.path.join(input_folder, file)
    for file in os.listdir(input_folder)
    if file.lower().endswith(".jpg")
]

image_files.sort()


print("=" * 60)
print("SROIE INVOICE EXTRACTION")
print("=" * 60)
print(f"Input folder : {input_folder}")
print(f"Output folder: {output_folder}")
print(f"Total images : {len(image_files)}")
print("=" * 60)


# ============================================================
# 4. PROMPT
# ============================================================

prompt = """
You are an invoice information extraction system.

Analyze ONLY the provided invoice image.

Extract exactly these four fields:

{
    "company": "",
    "date": "",
    "address": "",
    "total": ""
}

STRICT RULES:

1. Extract ONLY information that is visibly present in the image.
2. NEVER guess.
3. NEVER infer missing information.
4. NEVER autocomplete text.
5. NEVER correct spelling.
6. NEVER invent characters, digits, words, or addresses.
7. If a field is unclear, return "".
8. Do not use information from outside the image.
9. Do not derive one field from another.

COMPANY:
Extract the seller/company/business name.

DATE:
Extract the invoice date exactly as visible.
Do not infer or calculate the date.

ADDRESS:
Extract the seller/company address exactly as visible.
Do not construct or complete the address.

TOTAL:
Extract the FINAL TOTAL / GRAND TOTAL / TOTAL AMOUNT.

Read the total directly from the invoice.

DO NOT calculate the total.
DO NOT add numbers together.
DO NOT derive the total from individual items.
DO NOT confuse subtotal, tax, invoice number, phone number, or other numbers with the final total.

For total:
- Preserve the visible digits.
- Remove commas and currency symbols only.
- Do not add digits.
- Do not remove digits.
- Do not add zeros.
- Do not remove zeros.

Examples:
₹1,250.00 -> "1250.00"
1,250.00 -> "1250.00"
$250.00 -> "250.00"

If the final total cannot be identified clearly, return "".

IMPORTANT:
Return ONLY valid JSON.
Do NOT provide explanations.
Do NOT use Markdown.
Do NOT use ```json.
Do NOT add additional fields.

Required output:

{
    "company": "",
    "date": "",
    "address": "",
    "total": ""
}
"""


# ============================================================
# 5. CONVERT JPG TO BASE64
# ============================================================

def image_to_base64(image_path):

    with open(image_path, "rb") as image_file:

        image_data = image_file.read()

    return base64.b64encode(
        image_data
    ).decode("utf-8")


# ============================================================
# 6. CLEAN MODEL RESPONSE
# ============================================================

def clean_response(response_text):

    response_text = response_text.strip()

    # Remove ```json if model accidentally returns it
    if response_text.startswith("```json"):
        response_text = response_text[7:]

    elif response_text.startswith("```"):
        response_text = response_text[3:]

    if response_text.endswith("```"):
        response_text = response_text[:-3]

    return response_text.strip()


# ============================================================
# 7. CLEAN TOTAL
# ============================================================

def clean_total(total):

    if total is None:
        return ""

    total = str(total).strip()

    # Remove currency symbols
    total = total.replace("₹", "")
    total = total.replace("$", "")
    total = total.replace("€", "")
    total = total.replace("£", "")

    # Remove currency abbreviations
    total = total.replace("Rs.", "")
    total = total.replace("Rs", "")

    # Remove commas
    total = total.replace(",", "")

    return total.strip()


# ============================================================
# 8. PROCESS ONE IMAGE
# ============================================================

def process_image(image_path, index, total_images):

    filename = os.path.basename(image_path)

    image_name = os.path.splitext(filename)[0]

    output_path = os.path.join(
        output_folder,
        image_name + ".json"
    )

    print()
    print("-" * 60)
    print(f"[{index}/{total_images}] {filename}")


    # --------------------------------------------------------
    # Skip already processed images
    # --------------------------------------------------------

    if os.path.exists(output_path):

        print("Already processed -> SKIPPED")

        return "skipped"


    try:

        # ----------------------------------------------------
        # Convert image to Base64
        # ----------------------------------------------------

        image_base64 = image_to_base64(
            image_path
        )


        # ----------------------------------------------------
        # Send image to DeepSeek-VL2
        # ----------------------------------------------------

        response = client.chat.completions.create(

            model=MODEL_NAME,

            messages=[
                {
                    "role": "user",

                    "content": [

                        {
                            "type": "text",
                            "text": prompt
                        },

                        {
                            "type": "image_url",

                            "image_url": {
                                "url":
                                "data:image/jpeg;base64,"
                                + image_base64
                            }
                        }

                    ]
                }
            ],

            # Deterministic output
            temperature=0,

            top_p=1.0,

            max_tokens=300
        )


        # ----------------------------------------------------
        # Get model response
        # ----------------------------------------------------

        raw_output = response.choices[0].message.content

        print("Model response:")
        print(raw_output)


        # ----------------------------------------------------
        # Clean response
        # ----------------------------------------------------

        cleaned_output = clean_response(
            raw_output
        )


        # ----------------------------------------------------
        # Parse JSON
        # ----------------------------------------------------

        result = json.loads(
            cleaned_output
        )


        # ----------------------------------------------------
        # Extract fields
        # ----------------------------------------------------

        company = result.get(
            "company",
            ""
        )

        date = result.get(
            "date",
            ""
        )

        address = result.get(
            "address",
            ""
        )

        total = result.get(
            "total",
            ""
        )


        # ----------------------------------------------------
        # Clean total
        # ----------------------------------------------------

        total = clean_total(total)


        # ----------------------------------------------------
        # Final JSON
        # ----------------------------------------------------

        final_result = {

            "company":
                str(company).strip(),

            "date":
                str(date).strip(),

            "address":
                str(address).strip(),

            "total":
                total
        }


        # ----------------------------------------------------
        # Save JSON
        # ----------------------------------------------------

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


        print(
            f"Saved: {output_path}"
        )

        return "success"


    except json.JSONDecodeError as e:

        print(
            "ERROR: Invalid JSON returned by model."
        )

        print(
            f"JSON error: {e}"
        )

        return "failed"


    except Exception as e:

        print(
            f"ERROR: {e}"
        )

        return "failed"


# ============================================================
# 9. PROCESS ALL JPG IMAGES
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
# 10. FINAL SUMMARY
# ============================================================

print()
print("=" * 60)
print("PROCESSING COMPLETE")
print("=" * 60)

print(f"Total images : {len(image_files)}")
print(f"Successful   : {successful}")
print(f"Skipped      : {skipped}")
print(f"Failed       : {failed}")

print()
print("Results:")
print(output_folder)

print("=" * 60)
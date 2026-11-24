from openai import OpenAI
from PIL import Image
import base64
import io
import json
import os
import re


# ============================================================
# 1. DeepSeek-VL2 local model endpoint
# ============================================================

client = OpenAI(
    base_url="http://localhost:9000/v1",
    api_key="dummy"
)

MODEL_NAME = "DeepSeek-VL2"


# ============================================================
# 2. Input image
# ============================================================

image_path = r"C:\Users\telug\Downloads\DAT_Datasets\IDRBT_Cheque_Dataset\IDRBT_Cheque_Dataset\Images\Cheque 120611.tif"


# ============================================================
# 3. Output JSON file
# ============================================================

output_path = r"C:\Users\telug\Downloads\DAT_Datasets\Results\IDRBT\Cheque test.json"


# ============================================================
# 4. Check input image
# ============================================================

if not os.path.exists(image_path):

    print("ERROR: Image file not found!")
    print(image_path)
    exit()


# ============================================================
# 5. Create output directory if necessary
# ============================================================

output_directory = os.path.dirname(output_path)

if not os.path.exists(output_directory):
    os.makedirs(output_directory)


# ============================================================
# 6. Load TIFF image
# ============================================================

print("Loading image...")

image = Image.open(image_path)

print("Image format :", image.format)
print("Image size   :", image.size)
print("Image mode   :", image.mode)


# ============================================================
# 7. Convert TIFF to PNG in memory
# ============================================================

if image.mode != "RGB":
    image = image.convert("RGB")

buffer = io.BytesIO()

image.save(
    buffer,
    format="PNG"
)

image_base64 = base64.b64encode(
    buffer.getvalue()
).decode("utf-8")


# ============================================================
# 8. JSON schema
#
# IMPORTANT:
# amount is temporarily a STRING so that the model only
# reads the digits. Python converts it to a number later.
# ============================================================

schema = {
    "payee_name": "",
    "amount_in_words": "",
    "amount": "",
    "year": "",
    "account_no": ""
}


# ============================================================
# 9. Prompt
# ============================================================

prompt = f"""
You are a highly accurate bank cheque information extraction system.

Analyze ONLY the provided cheque image.

Extract exactly these five fields:

{json.dumps(schema, indent=4)}


============================================================
FIELD 1: payee_name
============================================================

Find the name of the person or organization to whom the
cheque is payable.

Read it directly from the cheque.

Do NOT guess.

If it cannot be read clearly, return an empty string.


============================================================
FIELD 2: amount_in_words
============================================================

Read the amount written in words.

Copy the words exactly as they appear on the cheque.

Do NOT calculate this value.

Do NOT convert the words into another representation.

If it cannot be read clearly, return an empty string.


============================================================
FIELD 3: amount
============================================================

THIS IS THE MOST IMPORTANT FIELD.

Read the NUMERICAL AMOUNT directly from the numerical
amount field on the cheque.

DO NOT calculate the amount from amount_in_words.

DO NOT infer the amount.

DO NOT add digits.

DO NOT remove digits.

DO NOT change digits.

DO NOT add zeros.

DO NOT remove zeros.

Preserve the exact digits visible in the image.

Examples:

If image shows:

5,000

return:

"5000"


If image shows:

10,000

return:

"10000"


If image shows:

25,500

return:

"25500"


If image shows:

1,25,000

return:

"125000"


Remove ONLY:

- commas
- currency symbols
- Rs
- Rs.

The digits themselves MUST NOT be changed.

If the numerical amount cannot be read clearly,
return an empty string.


============================================================
FIELD 4: year
============================================================

Read the cheque date.

Extract ONLY the year.

Example:

15/08/2024

return:

"2024"

If the year cannot be read clearly, return an empty string.


============================================================
FIELD 5: account_no
============================================================

Extract the account number only if it is clearly visible.

Do NOT confuse the account number with:

- cheque number
- MICR number
- IFSC code
- branch code
- other numbers

If the account number cannot be identified clearly,
return an empty string.


============================================================
IMPORTANT RULES
============================================================

The numerical amount and amount_in_words are independent.

Read the numerical amount DIRECTLY from the numerical
amount printed on the cheque.

DO NOT derive the numerical amount from amount_in_words.

For example, if the image contains:

Amount: 5000
Amount in words: Five Thousand Only

return:

"amount": "5000"

NOT:

"amount": "50000"

NOT:

"amount": "500"

NOT:

"amount": "50000.0"


============================================================
OUTPUT REQUIREMENTS
============================================================

Return ONLY valid JSON.

Do NOT provide explanations.

Do NOT use Markdown.

Do NOT use ```json.

Do NOT add additional fields.

The output MUST contain exactly these five fields:

{{
    "payee_name": "",
    "amount_in_words": "",
    "amount": "",
    "year": "",
    "account_no": ""
}}
"""


# ============================================================
# 10. Send image to DeepSeek-VL2
# ============================================================

print("\nSending image to DeepSeek-VL2...")

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
                        "url": f"data:image/png;base64,{image_base64}"
                    }
                }

            ]
        }
    ],

    # Deterministic generation
    temperature=0,

    # Don't sample from a restricted probability mass
    top_p=1.0,

    # Small output because we only need 5 fields
    max_tokens=300
)


# ============================================================
# 11. Get model response
# ============================================================

output = response.choices[0].message.content

print("\n========================================")
print("RAW MODEL OUTPUT")
print("========================================\n")

print(output)


# ============================================================
# 12. Clean response
# ============================================================

output_clean = output.strip()


# Remove Markdown code fences if present

if output_clean.startswith("```json"):

    output_clean = output_clean[
        len("```json"):
    ]

elif output_clean.startswith("```"):

    output_clean = output_clean[
        len("```"):
    ]


if output_clean.endswith("```"):

    output_clean = output_clean[
        :-3
    ]


output_clean = output_clean.strip()


# ============================================================
# 13. Parse JSON
# ============================================================

try:

    result = json.loads(output_clean)

except json.JSONDecodeError as e:

    print("\n========================================")
    print("ERROR: INVALID JSON")
    print("========================================")

    print(e)

    raw_output_path = output_path.replace(
        ".json",
        "_raw.txt"
    )

    with open(
        raw_output_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(output)

    print("\nRaw model output saved to:")
    print(raw_output_path)

    exit()


# ============================================================
# 14. Extract fields
# ============================================================

payee_name = result.get(
    "payee_name",
    ""
)

amount_in_words = result.get(
    "amount_in_words",
    ""
)

amount = result.get(
    "amount",
    ""
)

year = result.get(
    "year",
    ""
)

account_no = result.get(
    "account_no",
    ""
)


# ============================================================
# 15. Clean numerical amount
# ============================================================

if amount is None:

    amount = ""


amount = str(amount).strip()


# Remove only formatting characters

amount = amount.replace(",", "")
amount = amount.replace("₹", "")
amount = amount.replace("Rs.", "")
amount = amount.replace("Rs", "")
amount = amount.strip()


# Keep digits only

amount_digits = re.sub(
    r"[^0-9]",
    "",
    amount
)


# Convert to integer

if amount_digits:

    amount = int(amount_digits)

else:

    amount = 0


# ============================================================
# 16. Construct final JSON
# ============================================================

final_result = {

    "payee_name": str(
        payee_name
    ).strip(),

    "amount_in_words": str(
        amount_in_words
    ).strip(),

    "amount": amount,

    "year": str(
        year
    ).strip(),

    "account_no": str(
        account_no
    ).strip()
}


# ============================================================
# 17. Save JSON
# ============================================================

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


# ============================================================
# 18. Display final result
# ============================================================

print("\n========================================")
print("FINAL JSON")
print("========================================\n")

print(
    json.dumps(
        final_result,
        indent=4,
        ensure_ascii=False
    )
)


print("\n========================================")
print("SUCCESS")
print("========================================")

print("\nJSON file saved at:")

print(output_path)
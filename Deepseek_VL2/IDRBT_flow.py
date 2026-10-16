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
# 2. INPUT FOLDER
# ============================================================

input_folder = r"C:\Users\telug\Downloads\DAT_Datasets\IDRBT_Cheque_Dataset\Images"


# ============================================================
# 3. OUTPUT FOLDER
# ============================================================

output_folder = r"C:\Users\telug\Downloads\DAT_Datasets\Results\IDRBT"


# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)


# ============================================================
# 4. Supported image extensions
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
# 5. Get all images
# ============================================================

image_files = [
    os.path.join(input_folder, file)
    for file in os.listdir(input_folder)
    if file.lower().endswith(image_extensions)
]


# Sort images
image_files.sort()


# ============================================================
# 6. Check images
# ============================================================

if len(image_files) == 0:

    print("No images found in:")
    print(input_folder)
    exit()


print("========================================")
print("BATCH PROCESSING")
print("========================================")

print(f"Input folder : {input_folder}")
print(f"Output folder: {output_folder}")
print(f"Images found : {len(image_files)}")

print()


# ============================================================
# 7. Prompt
# ============================================================

prompt = """
You are a highly accurate bank cheque information extraction system.

Analyze ONLY the provided cheque image.

Extract exactly these five fields:

{
    "payee_name": "",
    "amount_in_words": "",
    "amount": "",
    "year": "",
    "account_no": ""
}


============================================================
1. payee_name
============================================================

Extract the name of the person or organization to whom
the cheque is payable.

Read the name directly from the cheque.

Do NOT guess.

Do NOT autocomplete.

Preserve the spelling visible in the image.

If it cannot be read clearly, return "".


============================================================
2. amount_in_words
============================================================

Extract the amount written in words.

Copy the words exactly as they appear.

Do NOT calculate this value.

Do NOT convert it into another representation.

If it cannot be read clearly, return "".


============================================================
3. amount
============================================================

Read the NUMERICAL AMOUNT directly from the numerical
amount field.

Do NOT calculate the amount from amount_in_words.

Do NOT infer the amount.

Do NOT add digits.

Do NOT remove digits.

Do NOT add zeros.

Do NOT remove zeros.

Preserve the exact digits visible in the image.

Remove ONLY commas and currency symbols.

Examples:

5,000 -> "5000"

10,000 -> "10000"

25,500 -> "25500"

1,25,000 -> "125000"

Return the amount as a STRING.

If the amount cannot be read clearly, return "".


============================================================
4. year
============================================================

Extract ONLY the year from the cheque date.

Example:

15/03/2016 -> "2016"

Return the year as a string.

If it cannot be read clearly, return "".


============================================================
5. account_no
============================================================

Extract the account number if it is clearly visible.

Do NOT confuse the account number with:

- cheque number
- MICR number
- IFSC code
- branch code
- other numbers

If it cannot be identified clearly, return "".


============================================================
IMPORTANT
============================================================

The numerical amount and amount_in_words are independent.

Read the numerical amount directly from the numerical
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


============================================================
OUTPUT
============================================================

Return ONLY valid JSON.

Do NOT provide explanations.

Do NOT use Markdown.

Do NOT use ```json.

Do NOT add additional fields.

The output MUST contain exactly:

{
    "payee_name": "",
    "amount_in_words": "",
    "amount": "",
    "year": "",
    "account_no": ""
}
"""


# ============================================================
# 8. Function to process ONE image
# ============================================================

def process_image(image_path, image_number, total_images):

    print("\n")
    print("========================================")
    print(f"Processing image {image_number}/{total_images}")
    print("========================================")

    print("Image:")
    print(image_path)


    # --------------------------------------------------------
    # Output JSON path
    # --------------------------------------------------------

    image_filename = os.path.basename(image_path)

    image_name_without_extension = os.path.splitext(
        image_filename
    )[0]

    output_path = os.path.join(
        output_folder,
        image_name_without_extension + ".json"
    )


    # --------------------------------------------------------
    # Skip if JSON already exists
    # --------------------------------------------------------

    if os.path.exists(output_path):

        print("\nJSON already exists.")

        print("Skipping:")
        print(output_path)

        return True


    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    try:

        image = Image.open(image_path)

        print("\nImage format:", image.format)
        print("Image size  :", image.size)
        print("Image mode  :", image.mode)

    except Exception as e:

        print("\nERROR loading image:")
        print(e)

        return False


    # --------------------------------------------------------
    # Convert image to RGB
    # --------------------------------------------------------

    try:

        if image.mode != "RGB":
            image = image.convert("RGB")

    except Exception as e:

        print("\nERROR converting image:")
        print(e)

        return False


    # --------------------------------------------------------
    # Convert image to PNG in memory
    # --------------------------------------------------------

    try:

        buffer = io.BytesIO()

        image.save(
            buffer,
            format="PNG"
        )

        image_base64 = base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")

    except Exception as e:

        print("\nERROR converting image to PNG:")
        print(e)

        return False


    # --------------------------------------------------------
    # Send image to DeepSeek-VL2
    # --------------------------------------------------------

    try:

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
                                "url": (
                                    "data:image/png;base64,"
                                    + image_base64
                                )
                            }
                        }

                    ]
                }
            ],

            # Deterministic generation
            temperature=0,

            # Use full probability distribution
            top_p=1.0,

            # Small output is enough for 5 fields
            max_tokens=300
        )

    except Exception as e:

        print("\nERROR calling DeepSeek-VL2:")
        print(e)

        return False


    # --------------------------------------------------------
    # Get model output
    # --------------------------------------------------------

    try:

        output = response.choices[0].message.content

    except Exception as e:

        print("\nERROR reading model response:")
        print(e)

        return False


    print("\n----------------------------------------")
    print("RAW MODEL OUTPUT")
    print("----------------------------------------")

    print(output)


    # --------------------------------------------------------
    # Clean model output
    # --------------------------------------------------------

    output_clean = output.strip()


    if output_clean.startswith("```json"):

        output_clean = output_clean[
            len("```json"):
        ]


    elif output_clean.startswith("```"):

        output_clean = output_clean[
            len("```"):
        ]


    if output_clean.endswith("```"):

        output_clean = output_clean[:-3]


    output_clean = output_clean.strip()


    # --------------------------------------------------------
    # Convert to JSON
    # --------------------------------------------------------

    try:

        result = json.loads(output_clean)

    except json.JSONDecodeError as e:

        print("\nERROR: Model returned invalid JSON.")

        print(e)


        # Save raw output
        raw_output_path = os.path.join(
            output_folder,
            image_name_without_extension + "_raw.txt"
        )


        with open(
            raw_output_path,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(output)


        print("\nRaw output saved:")
        print(raw_output_path)

        return False


    # --------------------------------------------------------
    # Extract fields
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Clean amount
    # --------------------------------------------------------

    if amount is None:

        amount = ""


    amount = str(amount).strip()


    amount = amount.replace(
        ",",
        ""
    )

    amount = amount.replace(
        "₹",
        ""
    )

    amount = amount.replace(
        "Rs.",
        ""
    )

    amount = amount.replace(
        "Rs",
        ""
    )


    # Keep only digits
    amount_digits = re.sub(
        r"[^0-9]",
        "",
        amount
    )


    if amount_digits:

        amount = int(
            amount_digits
        )

    else:

        amount = 0


    # --------------------------------------------------------
    # Final JSON
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    try:

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

    except Exception as e:

        print("\nERROR saving JSON:")
        print(e)

        return False


    # --------------------------------------------------------
    # Print result
    # --------------------------------------------------------

    print("\n----------------------------------------")
    print("FINAL JSON")
    print("----------------------------------------")

    print(
        json.dumps(
            final_result,
            indent=4,
            ensure_ascii=False
        )
    )


    print("\nJSON saved:")
    print(output_path)


    print("\nCompleted image:")
    print(image_number, "/", total_images)


    return True


# ============================================================
# 9. PROCESS IMAGES ONE BY ONE
# ============================================================

successful = 0
failed = 0
skipped = 0


for index, image_path in enumerate(
    image_files,
    start=1
):

    image_filename = os.path.basename(
        image_path
    )

    image_name_without_extension = os.path.splitext(
        image_filename
    )[0]

    expected_json = os.path.join(
        output_folder,
        image_name_without_extension + ".json"
    )


    # --------------------------------------------------------
    # Already processed?
    # --------------------------------------------------------

    if os.path.exists(expected_json):

        print("\n")
        print("========================================")
        print(f"Image {index}/{len(image_files)}")
        print("SKIPPED - JSON already exists")
        print("========================================")

        print(image_filename)

        skipped += 1

        continue


    # --------------------------------------------------------
    # Process current image
    # --------------------------------------------------------

    success = process_image(
        image_path,
        index,
        len(image_files)
    )


    if success:

        successful += 1

    else:

        failed += 1


# ============================================================
# 10. FINAL SUMMARY
# ============================================================

print("\n\n")
print("========================================")
print("BATCH PROCESSING COMPLETE")
print("========================================")

print(
    f"Total images : {len(image_files)}"
)

print(
    f"Successful   : {successful}"
)

print(
    f"Failed       : {failed}"
)

print(
    f"Skipped      : {skipped}"
)

print(
    f"Output folder: {output_folder}"
)

print("========================================")
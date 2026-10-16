from openai import OpenAI
import base64
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

input_folder = r"C:\Users\telug\Downloads\DAT_Datasets\iam_handwriting_finevision\images"

output_folder = r"C:\Users\telug\Downloads\DAT_Datasets\Results\iam_handwritten"

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
print("IAM HANDWRITING - DEEPSEEK-VL2 BATCH TRANSCRIPTION")
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
# 5. PROMPT
# ============================================================

prompt = r"""
You are a high-accuracy handwriting transcription system.

Analyze ONLY the provided image.

Your task is to TRANSCRIBE the handwritten text that is actually
visible in the image.

This is a transcription task only.

============================================================
MOST IMPORTANT RULE
============================================================

WRITE ONLY WHAT IS ACTUALLY VISIBLE IN THE IMAGE.

Nothing more.
Nothing less.

The image is the ONLY source of truth.

============================================================
STRICT ANTI-HALLUCINATION RULES
============================================================

1. NEVER hallucinate.

2. NEVER guess.

3. NEVER autocomplete text.

4. NEVER complete a partially written word.

5. NEVER complete a partially visible character.

6. NEVER infer missing characters.

7. NEVER infer missing words.

8. NEVER correct spelling.

9. NEVER correct grammar.

10. NEVER rewrite the handwriting into standard or proper text.

11. NEVER replace an unclear word with a likely word.

12. NEVER use external knowledge.

13. NEVER use language-model knowledge to fill missing text.

14. NEVER assume what the writer intended to write.

15. NEVER change an unclear character into a similar-looking
    character based on context.

16. NEVER add words that are not visible.

17. NEVER remove visible handwritten text.

18. NEVER summarize the handwriting.

19. NEVER paraphrase the handwriting.

20. NEVER explain the handwriting.

============================================================
TRANSCRIPTION
============================================================

Transcribe the handwritten text exactly as it appears.

Preserve:

- spelling
- letters
- capitalization where visible
- numbers
- punctuation
- abbreviations
- symbols
- word order

Do not correct mistakes made by the writer.

If the writer appears to spell a word incorrectly, preserve
the visible spelling.

============================================================
UNCLEAR HANDWRITING
============================================================

If a character or word is genuinely impossible to read:

DO NOT GUESS IT.

Do NOT replace it with what you think it should be.

Use:

[UNCLEAR]

at the corresponding position.

For example:

Visible:
John Sm?th

If the uncertain character cannot be determined, return:

John Sm[UNCLEAR]th

If an entire word is unreadable:

[UNCLEAR]

============================================================
PARTIALLY VISIBLE TEXT
============================================================

If only part of a word is actually visible, transcribe only
the visible part.

Do NOT complete the missing part.

============================================================
LINE BREAKS
============================================================

Preserve line breaks when they are clearly present in the image.

Do not unnecessarily create new lines when the handwriting
continues on the same line.

============================================================
MULTIPLE TEXT REGIONS
============================================================

If there are multiple handwritten text regions in the image,
transcribe all visible handwritten text in the natural
reading order.

Normally read:

top to bottom
and
left to right

Do not merge unrelated text in a way that changes its meaning.

============================================================
NON-TEXT CONTENT
============================================================

Ignore:

- background lines
- paper texture
- shadows
- noise
- decorative marks
- boxes
- borders
- blank areas

Do not describe the image.

Do not describe handwriting style.

Do not provide OCR confidence.

============================================================
PRINTED TEXT
============================================================

If printed text is present together with handwriting, do NOT
transcribe the printed text unless it is clearly part of the
actual handwritten content.

The objective is handwritten transcription.

============================================================
OUTPUT FORMAT
============================================================

Return ONLY the transcription as plain text.

Do NOT return JSON.

Do NOT return XML.

Do NOT return Markdown.

Do NOT use code blocks.

Do NOT add quotation marks around the transcription.

Do NOT write:

"Here is the transcription"

Do NOT write:

"The text is"

Do NOT provide explanations.

Do NOT provide comments.

Do NOT provide confidence scores.

Do NOT add any text before or after the transcription.

============================================================
FINAL CHECK
============================================================

Before responding, verify:

- Every character comes from the visible image.
- Nothing was guessed.
- Nothing was auto-completed.
- Nothing was corrected.
- No spelling was changed.
- No missing text was invented.
- No visible handwritten text was omitted.
- No summary was generated.
- No explanation was added.

When a character cannot be determined, use [UNCLEAR]
instead of guessing.

Return ONLY the handwritten transcription.
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
# 8. CLEAN MODEL OUTPUT
# ============================================================

def clean_response(response_text):

    response_text = response_text.strip()

    # Remove accidental Markdown code fences
    response_text = re.sub(
        r"^```text\s*",
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

    # Remove a few common accidental prefixes
    prefixes = [
        "Here is the transcription:",
        "Here is the transcription",
        "Transcription:",
        "The transcription is:"
    ]

    for prefix in prefixes:

        if response_text.startswith(prefix):

            response_text = response_text[
                len(prefix):
            ].strip()

    return response_text.strip()


# ============================================================
# 9. CALL DEEPSEEK-VL2
# ============================================================

def call_model(
    image_base64,
    mime_type
):

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
                            f"data:{mime_type};base64,"
                            + image_base64

                        }
                    }

                ]
            }
        ],

        temperature=0,

        top_p=1.0,

        max_tokens=1000
    )

    return response.choices[0].message.content


# ============================================================
# 10. PROCESS ONE IMAGE
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
        image_name + ".txt"
    )


    print()
    print("-" * 75)

    print(
        f"[{index}/{total_images}] "
        f"Processing: {filename}"
    )


    # --------------------------------------------------------
    # Skip if already processed
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
        # Check image
        # ----------------------------------------------------

        if not os.path.exists(
            image_path
        ):

            print(
                "Image not found."
            )

            return "failed"


        # ----------------------------------------------------
        # Convert image to Base64
        # ----------------------------------------------------

        image_base64 = image_to_base64(
            image_path
        )

        mime_type = get_mime_type(
            image_path
        )


        # ----------------------------------------------------
        # Call DeepSeek-VL2
        # ----------------------------------------------------

        print(
            "Sending image to DeepSeek-VL2..."
        )

        raw_output = call_model(
            image_base64,
            mime_type
        )


        # ----------------------------------------------------
        # Clean output
        # ----------------------------------------------------

        transcription = clean_response(
            raw_output
        )


        # ----------------------------------------------------
        # Print result
        # ----------------------------------------------------

        print()
        print("TRANSCRIPTION:")
        print("-" * 75)
        print(transcription)
        print("-" * 75)


        # ----------------------------------------------------
        # Save TXT
        # ----------------------------------------------------

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(
                transcription
            )


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
# 11. PROCESS ALL IMAGES ONE BY ONE
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
# 12. FINAL SUMMARY
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
    "TXT results saved in:"
)

print(
    output_folder
)

print("=" * 75)
"""
Basic connectivity test for the MinerU2.5 OpenAI-compatible endpoint.

Checks:
1. Can Python connect to the endpoint?
2. Does /v1/models respond, and is "mineru2.5" listed?
3. Does a simple chat-completions request return a response?

Install:
    python -m pip install -U openai

Run:
    python test_mineru_endpoint.py
"""

from openai import OpenAI

BASE_URL = "http://10.68.0.39:9008/v1"
MODEL_NAME = "mineru2.5"
API_KEY = "dummy"  # Change only if your endpoint requires an API key.


def main():
    print("=" * 70)
    print("MINERU2.5 ENDPOINT CONNECTION TEST")
    print("=" * 70)
    print("Endpoint:", BASE_URL)
    print("Model:   ", MODEL_NAME)

    client = OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
        timeout=30.0,
        max_retries=0,
    )

    # Test 1: Endpoint connectivity and model list.
    print("\n[TEST 1] Checking /models ...")
    try:
        models_response = client.models.list()
        model_ids = [
            getattr(model, "id", str(model))
            for model in models_response.data
        ]
        print("SUCCESS: Endpoint responded to the models request.")
        print("Models returned by server:")
        for model_id in model_ids:
            print("  -", model_id)

        if any(MODEL_NAME.lower() == model_id.lower() for model_id in model_ids):
            print(f"SUCCESS: Exact model name '{MODEL_NAME}' is listed.")
        else:
            print(
                f"NOTE: '{MODEL_NAME}' was not an exact match in the model list. "
                "The server may use a different model identifier; check the IDs above."
            )
    except Exception as error:
        print(f"FAILED: Could not list models: {type(error).__name__}: {error}")
        print(
            "\nThis may be a network/VPN/firewall issue, wrong endpoint URL, "
            "authentication requirement, or an unsupported /models route."
        )
        client.close()
        return 1

    # Test 2: Simple text-only chat completion.
    print("\n[TEST 2] Sending a simple text-only chat request ...")
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": "Reply with exactly this text: MINERU_ENDPOINT_OK",
                }
            ],
            temperature=0,
            max_tokens=30,
        )

        if not response.choices:
            print("FAILED: The endpoint returned no choices.")
            return 1

        content = response.choices[0].message.content
        print("SUCCESS: Chat completion request returned.")
        print("Model response:", repr(content))
        print("\nRESULT: The endpoint is reachable and the chat API returned a response.")
        print("Next, test image input separately before running the full IAM batch.")
        return 0

    except Exception as error:
        print(f"FAILED: Chat completion request: {type(error).__name__}: {error}")
        print(
            "\nThe endpoint may be reachable but reject this model name, this "
            "request format, or text-only requests. Share the complete error "
            "message so the next test can target the failure."
        )
        return 1
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())

"""Remote endpoint calls shared by the self-hosted models.

All are served as OpenAI-compatible endpoints (vLLM):

    Qwen2.5-VL      http://10.68.0.39:9005/v1  model "qwen2.5-vl-7b"   prompt FIRST, then image
    NuExtract3      http://10.68.0.39:9003/v1  model "nuextract3"      image only + chat_template_kwargs
    GLM-4.6V-Flash  http://10.68.0.39:9007/v1  model "glm-4.6v-flash"  image FIRST, then prompt

Images are sent as JPEG (quality 95), resized so the longest side is at most
MAX_IMAGE_SIDE (1600), with temperature 0.0, top_p 1.0 and max_tokens 3000,
exactly as in the scripts. Endpoints and parameters come from .env.
"""
from __future__ import annotations

import base64
from io import BytesIO
from typing import Any

from PIL import Image, ImageOps

from config.settings import env_float, env_int
from models.base_adapter import AdapterError, TransientAdapterError

MAX_IMAGE_SIDE = env_int("MAX_IMAGE_SIDE", 1600)
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_MAX_TOKENS = 3000
DEFAULT_TIMEOUT = env_float("REQUEST_TIMEOUT_S", 180.0)


# ---- image preprocessing (image_to_base64 in the scripts) ------------------

def image_to_base64(image: Image.Image, max_side: int = MAX_IMAGE_SIDE) -> str:
    """Correct orientation, convert to RGB, shrink to max_side (never enlarge),
    and encode as JPEG Base64, as in the scripts' image_to_base64()."""
    img = ImageOps.exif_transpose(image).convert("RGB")
    if max_side:
        img.thumbnail((max_side, max_side))
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=95)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ---- common remote vision call (call_vision_model in the scripts) ----------

def get_response_text(content: Any) -> str:
    """Convert OpenAI-compatible response content to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            text = item.get("text") if isinstance(item, dict) else getattr(item, "text", None)
            if text is not None:
                parts.append(str(text))
        return "\n".join(parts)
    return "" if content is None else str(content)


def make_client(base_url: str, timeout: float = DEFAULT_TIMEOUT, api_key: str = ""):
    from openai import OpenAI
    return OpenAI(base_url=base_url, api_key=api_key or "not-needed", timeout=timeout, max_retries=0)


def call_vision_model(client, model: str, prompt: str | None, image_data: str, image_first: bool = False,
                      temperature: float = DEFAULT_TEMPERATURE, top_p: float = DEFAULT_TOP_P,
                      max_tokens: int = DEFAULT_MAX_TOKENS, extra_body: dict | None = None) -> str:
    """Send one image + prompt to an OpenAI-compatible vision endpoint.
    Qwen gets the instruction first, GLM the image first. With prompt=None only
    the image is sent (NuExtract3 takes its template via chat_template_kwargs).
    Network errors are mapped to the app's retryable / non-retryable errors."""
    import openai

    image_block = {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + image_data}}
    if not prompt:
        content = [image_block]
    else:
        text_block = {"type": "text", "text": prompt}
        content = [image_block, text_block] if image_first else [text_block, image_block]
    kwargs: dict[str, Any] = dict(model=model, messages=[{"role": "user", "content": content}],
                                  temperature=temperature, top_p=top_p, max_tokens=max_tokens)
    if extra_body:
        kwargs["extra_body"] = extra_body
    try:
        response = client.chat.completions.create(**kwargs)
    except (openai.APITimeoutError, openai.APIConnectionError, openai.RateLimitError,
            openai.InternalServerError) as exc:
        raise TransientAdapterError(f"{model}: {type(exc).__name__}") from exc
    except openai.APIStatusError as exc:
        raise AdapterError(f"{model}: HTTP {exc.status_code} {str(exc)[:200]}") from exc
    return get_response_text(response.choices[0].message.content)


def list_models(base_url: str, timeout: float = 10.0) -> tuple[bool, str, list[str]]:
    """GET <base_url>/models, as in 1test.py. Returns (reachable, message, model ids)."""
    import requests
    try:
        r = requests.get(base_url.rstrip("/") + "/models", timeout=timeout)
    except requests.RequestException as exc:
        return False, f"unreachable ({type(exc).__name__})", []
    if r.status_code != 200:
        return False, f"HTTP {r.status_code}", []
    try:
        ids = [m.get("id", "") for m in r.json().get("data", [])]
    except ValueError:
        ids = []
    return True, "reachable", ids

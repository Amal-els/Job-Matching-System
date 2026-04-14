import json
import os
import re
import time
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def is_enabled(flag_name, default="false"):
    return os.getenv(flag_name, default).strip().lower() in {"1", "true", "yes", "on"}


def get_llm_client():
    # Priority: Groq -> OpenRouter
    groq_api_key = os.getenv("GROQ_API_KEY")
    if groq_api_key and groq_api_key != "your_groq_api_key_here":
        return OpenAI(
            api_key=groq_api_key,
            base_url="https://api.groq.com/openai/v1",
        )

    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_api_key:
        return OpenAI(
            api_key=openrouter_api_key,
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        )

    return None


def _coerce_content_to_text(content):
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "\n".join(part for part in parts if part)

    return str(content or "")


def extract_json_payload(text):
    cleaned = (text or "").strip()
    if not cleaned:
        raise ValueError("LLM returned empty content.")

    fenced_match = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fenced_match:
        cleaned = fenced_match.group(1).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    object_match = re.search(r"(\{.*\}|\[.*\])", cleaned, flags=re.DOTALL)
    if object_match:
        return json.loads(object_match.group(1))

    raise ValueError("Could not parse JSON payload from LLM response.")


def chat_json(model, system_prompt, user_prompt, temperature=0, max_retries=3):
    client = get_llm_client()
    if client is None:
        raise RuntimeError("LLM client (Groq or OpenRouter) is not configured.")

    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = _coerce_content_to_text(response.choices[0].message.content)
            return extract_json_payload(content)
        except Exception as exc:
            last_exc = exc
            # Retry on rate limit errors with backoff
            if getattr(exc, "status_code", None) == 429 or "429" in str(exc):
                if attempt < max_retries:
                    time.sleep(2 ** attempt)  # 2s, 4s, 8s ...
                    continue
            raise

    raise last_exc

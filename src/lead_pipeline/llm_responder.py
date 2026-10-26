import json
import logging
import os
from urllib.request import Request, urlopen
from typing import Any


class LLMResponder:
    def respond(
        self,
        meaning: dict[str, Any],
        fallback: str,
        *,
        allowed: bool = True,
    ) -> str:
        api_key = os.environ.get("ISL_LLM_API_KEY")
        if not api_key:
            logging.warning("LLM fallback: missing_api_key")
            return fallback
        if not allowed:
            logging.warning("LLM fallback: rate_limited")
            return fallback

        base_url = os.environ.get(
            "ISL_LLM_BASE_URL", "https://api.openai.com/v1"
        ).rstrip("/")
        endpoint = (
            base_url
            if base_url.endswith("/chat/completions")
            else f"{base_url}/chat/completions"
        )
        payload = {
            "model": os.environ.get("ISL_LLM_MODEL", "gpt-4o-mini"),
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Respond only within the supplied semantic frame. Return one "
                        "short sentence, with no invented entities or facts. Use only "
                        "supported gloss words where possible."
                    ),
                },
                {"role": "user", "content": json.dumps(meaning, sort_keys=True)},
            ],
            "temperature": 0,
        }
        request = Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=5) as response:
                result = json.loads(response.read().decode("utf-8"))
            content = result["choices"][0]["message"]["content"]
            if (
                not isinstance(content, str)
                or not content.strip()
                or len(content.split()) > 24
                or "\n" in content
            ):
                raise ValueError("invalid response content")
            return content.strip()
        except Exception as error:
            logging.warning(
                "LLM fallback: request_failed (%s)",
                type(error).__name__,
            )
            return fallback

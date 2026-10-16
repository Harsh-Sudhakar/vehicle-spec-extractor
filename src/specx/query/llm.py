"""Thin OpenRouter client (OpenAI-compatible) that returns parsed JSON."""

from __future__ import annotations

import json
import logging
import re
from functools import lru_cache

from openai import BadRequestError, OpenAI

from ..config import get_settings

log = logging.getLogger(__name__)


class LLMUnavailable(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    s = get_settings()
    if not s.openrouter_api_key:
        raise LLMUnavailable("OPENROUTER_API_KEY is not set")
    return OpenAI(
        api_key=s.openrouter_api_key,
        base_url=s.openrouter_base_url,
        timeout=s.llm_timeout_s,
        max_retries=2,
        default_headers={"X-Title": "SpecX vehicle spec extractor"},
    )


def llm_available() -> bool:
    return bool(get_settings().openrouter_api_key)


def _parse_json(text: str) -> dict:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in model output")
    return json.loads(text[start : end + 1])


def chat_json(system: str, user: str, max_tokens: int = 2500) -> dict:
    """Call the model in JSON mode with temperature 0; retry once if the output is not valid JSON."""
    s = get_settings()
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    kwargs = dict(
        model=s.openrouter_model,
        messages=messages,
        temperature=0,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    # Reasoning traces add latency and cost without helping a copy-out task.
    extra = {"reasoning": {"enabled": False}}
    last_err: Exception | None = None
    for attempt in range(2):
        try:
            resp = _client().chat.completions.create(**kwargs, extra_body=extra)
        except BadRequestError as e:  # provider rejected the reasoning flag -> plain call
            if extra:
                log.info("retrying without reasoning flag: %s", e)
                extra = {}
                resp = _client().chat.completions.create(**kwargs)
            else:
                raise
        content = resp.choices[0].message.content or ""
        try:
            return _parse_json(content)
        except (ValueError, json.JSONDecodeError) as e:
            last_err = e
            messages = messages + [
                {"role": "assistant", "content": content[:4000]},
                {"role": "user", "content": f"That was not valid JSON ({e}). Reply with the JSON object only."},
            ]
            kwargs["messages"] = messages
    raise ValueError(f"model did not return valid JSON: {last_err}")

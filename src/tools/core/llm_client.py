"""
core/llm_client.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import openai
import litellm
# litellm._turn_on_debug()
litellm.suppress_debug_info = True
litellm.set_verbose = False
# 查證後確認這行 Provider List: https://docs.litellm.ai/docs/providers 
# 是 litellm 內部的 verbose_logger（LiteLLM:ERROR 開頭那種訊息）自己印到 console 的 log。
# 常見觸發點是 litellm 認不出某個 model（例如較新/較冷門的 OpenRouter model id）
# 該支援哪些參數時，就會連帶印出這行 footer。

from core.config import get_llm_config


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    finish_reason: Optional[str] = None


class LLMCallError(RuntimeError):
    """Raised when an LLM call fails for any reason (bad model string,
    missing/expired API key, usage limit reached, model unavailable for
    the requested variant, network error, etc.). Always carries enough
    context in its message to debug WITHOUT needing to re-read the raw
    litellm traceback: which model/provider was being called, and the
    original exception's message. Deliberately left uncaught inside this
    module -- raising it is the intended way to fail loudly and stop the
    run rather than silently returning an empty/partial LLMResponse."""


# litellm's own exceptions (BadRequestError, AuthenticationError,
# RateLimitError, NotFoundError, PermissionDeniedError -- e.g. the
# "agentic harnesses only" 403 case -- Timeout, APIConnectionError,
# ServiceUnavailableError, ContextWindowExceededError,
# ContentPolicyViolationError, etc.) all inherit from openai.APIError,
# NOT from litellm.exceptions.APIError (verified against the installed
# litellm package -- litellm.exceptions.APIError is a sibling, not a
# common base). Catching openai.APIError is what actually covers the
# whole family in one except clause.
_LITELLM_EXCEPTIONS = (openai.APIError,)


class LLMClient:
    """
    Provider-agnostic LLM client backed by litellm. Call sites should
    depend on this class (not on litellm or any provider SDK directly)
    so the underlying provider/model can be swapped via .env alone.
    """

    def __init__(self) -> None:
        config = get_llm_config()
        self._models = config.models

    def generate_text(self, system_prompt, user_prompt, temperature=0.3, max_tokens=800) -> LLMResponse:
        errors = []
        for model in self._models:
            try:
                response = litellm.completion(
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                choice = response.choices[0]
                return LLMResponse(
                    text=choice.message.content or "",
                    model=response.model,
                    finish_reason=choice.finish_reason,
                )
            except openai.APIError as exc:
                errors.append(f"{model}: {exc}")
                continue

        raise LLMCallError("All configured models failed:\n\n" + "\n\n".join(errors))


_client_singleton: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """
    Returns a process-wide singleton so we don't reinitialize config/env
    reads on every call.
    """
    global _client_singleton
    if _client_singleton is None:
        _client_singleton = LLMClient()
    return _client_singleton
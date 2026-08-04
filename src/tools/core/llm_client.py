"""
core/llm_client.py
Thin wrapper around litellm, giving a single provider-agnostic interface
for all feature modules (onboard, cv_gen, jd_collector, ...).

litellm supports 100+ providers (OpenAI, Anthropic, Gemini, etc.) through
one unified `completion()` call -- switching providers is just a matter of
changing LLM_MODEL (and its matching API key) in .env, no code changes.

Right now this only needs to support plain text generation (e.g. drafting
position.responsibility or project.project_context from raw source text).
Structured extraction (problem/action/claim fields) will be added
once those tables are finalized -- this client is intentionally kept
generic so it can support both call styles later without changing its
public interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import litellm

from core.config import get_llm_config


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    finish_reason: Optional[str] = None


class LLMClient:
    """
    Provider-agnostic LLM client backed by litellm. Call sites should
    depend on this class (not on litellm or any provider SDK directly)
    so the underlying provider/model can be swapped via .env alone.
    """

    def __init__(self) -> None:
        config = get_llm_config()
        self._model = config.model
        # api key already mirrored into the correct provider env var
        # by get_llm_config(); litellm reads it from there automatically.

    def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 800,
    ) -> LLMResponse:
        """
        Plain text generation -- used for drafting responsibility /
        project_context / result_statement summaries. Low default
        temperature to keep drafts close to the source material rather
        than creatively embellished (aligned with the "no overclaiming"
        governance goal).
        """
        response = litellm.completion(
            model=self._model,
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

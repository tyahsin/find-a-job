"""
core/config.py
Version: v3 (2026-08-19)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
if _ENV_PATH.exists():
    load_dotenv(dotenv_path=_ENV_PATH)
else:
    load_dotenv()


def _require(name: str) -> str:
    value = os.getenv(name)
    if value is None or value == "":
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            f"Check your .env file (expected at {_ENV_PATH})."
        )
    return value


@dataclass(frozen=True)
class DBConfig:
    user: str
    password: str
    db_name: str
    host: str
    port: str

    @property
    def sqlalchemy_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.db_name}"
        )


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    model: str
    provider_env_var: str


@dataclass(frozen=True)
class CvConfig:
    output_dir: Path


_PROVIDER_ENV_VAR_MAP = {
    "claude": "ANTHROPIC_API_KEY",
    "gpt": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
}
_DEFAULT_PROVIDER_ENV_VAR = "OPENAI_API_KEY"


def _infer_provider_env_var(model: str) -> str:
    model_lower = model.lower()
    for prefix, env_var in _PROVIDER_ENV_VAR_MAP.items():
        if prefix in model_lower:
            return env_var
    return _DEFAULT_PROVIDER_ENV_VAR


def get_db_config() -> DBConfig:
    return DBConfig(
        user=_require("DB_USER"),
        password=_require("DB_PASSWORD"),
        db_name=_require("DB_NAME"),
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
    )


def get_llm_config() -> LLMConfig:
    model = os.getenv("LLM_MODEL", "claude-sonnet-4-5")
    api_key = _require("LLM_API_KEY")
    provider_env_var = _infer_provider_env_var(model)
    os.environ.setdefault(provider_env_var, api_key)
    return LLMConfig(api_key=api_key, model=model, provider_env_var=provider_env_var)


def get_cv_config() -> CvConfig:
    raw = os.getenv("OUTPUT_DIR")
    output_dir = Path(raw).expanduser() if raw else Path.home() / "applications"
    return CvConfig(output_dir=output_dir)

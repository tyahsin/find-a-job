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
    models: list[str]

@dataclass(frozen=True)
class CvConfig:
    output_dir: Path


_PROVIDER_ENV_VAR_MAP = {
    "claude": "ANTHROPIC_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "gpt": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
}
_DEFAULT_PROVIDER_ENV_VAR = "ANTHROPIC_API_KEY"


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
    models = os.getenv("LLM_MODELS")
    models = [m for m in models.split(",")]
    if not models:
        raise ValueError("LLM_MODELS is empty after parsing.")

    missing = []
    for model in models:
        env_var = _infer_provider_env_var(model)
        if not os.getenv(env_var):
            missing.append(f"{model} -> needs {env_var}")
    if missing:
        raise ValueError(
            "Missing API key(s) for the following configured model(s):\n"
            + "\n".join(f"  - {m}" for m in missing)
        )

    return LLMConfig(models=models)


def get_cv_config() -> CvConfig:
    raw = os.getenv("OUTPUT_DIR")
    output_dir = Path(raw).expanduser() if raw else Path.home() / "applications"
    return CvConfig(output_dir=output_dir)

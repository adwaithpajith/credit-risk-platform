from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env once, without overriding variables already supplied by
# the environment or Docker.
load_dotenv(override=False)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _bool(name: str, default: bool) -> bool:
    val = os.getenv(name)

    if val is None:
        return default

    return val.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _float(name: str, default: float) -> float:
    val = os.getenv(name)

    return float(val) if val not in (None, "") else default


def _int(name: str, default: int) -> int:
    val = os.getenv(name)

    return int(val) if val not in (None, "") else default


@dataclass(frozen=True)
class PathConfig:
    root: Path = PROJECT_ROOT

    data_dir: Path = PROJECT_ROOT / "data"
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"

    models_dir: Path = PROJECT_ROOT / "models"

    sql_dir: Path = PROJECT_ROOT / "sql"

    logs_dir: Path = PROJECT_ROOT / "logs"

    def ensure(self) -> None:
        for path in (
            self.data_dir,
            self.raw_dir,
            self.processed_dir,
            self.models_dir,
            self.logs_dir,
        ):
            path.mkdir(
                parents=True,
                exist_ok=True,
            )


@dataclass(frozen=True)
class DataConfig:
    # The assignment requires the real Home Credit Default Risk dataset.
    # Synthetic data is therefore OFF by default and must be explicitly
    # enabled for development/testing.

    use_synthetic_fallback: bool = field(
        default_factory=lambda: _bool(
            "USE_SYNTHETIC_FALLBACK",
            False,
        )
    )

    synthetic_rows: int = field(
        default_factory=lambda: _int(
            "SYNTHETIC_ROWS",
            15000,
        )
    )

    synthetic_seed: int = field(
        default_factory=lambda: _int(
            "SYNTHETIC_SEED",
            42,
        )
    )

    test_size: float = field(
        default_factory=lambda: _float(
            "TEST_SIZE",
            0.2,
        )
    )

    val_size: float = field(
        default_factory=lambda: _float(
            "VAL_SIZE",
            0.15,
        )
    )

    random_state: int = field(
        default_factory=lambda: _int(
            "RANDOM_STATE",
            42,
        )
    )


@dataclass(frozen=True)
class ModelConfig:
    n_folds: int = field(
        default_factory=lambda: _int(
            "CV_FOLDS",
            5,
        )
    )

    early_stopping_rounds: int = field(
        default_factory=lambda: _int(
            "EARLY_STOPPING_ROUNDS",
            100,
        )
    )

    num_boost_round: int = field(
        default_factory=lambda: _int(
            "NUM_BOOST_ROUND",
            2000,
        )
    )

    learning_rate: float = field(
        default_factory=lambda: _float(
            "LEARNING_RATE",
            0.03,
        )
    )

    # Business cost ratio:
    # how many times worse a missed defaulter (false negative)
    # is compared with unnecessarily declining a good applicant
    # (false positive).

    fn_cost_ratio: float = field(
        default_factory=lambda: _float(
            "FN_COST_RATIO",
            8.0,
        )
    )

    low_risk_band_max: float = field(
        default_factory=lambda: _float(
            "LOW_RISK_BAND_MAX",
            0.30,
        )
    )

    medium_risk_band_max: float = field(
        default_factory=lambda: _float(
            "MEDIUM_RISK_BAND_MAX",
            0.65,
        )
    )


@dataclass(frozen=True)
class DBConfig:
    # SQLite is the default so the application can run locally or
    # in Streamlit without requiring an external database.
    #
    # docker-compose can override DATABASE_URL when PostgreSQL
    # is used.

    database_url: str = field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL",
            f"sqlite:///{PROJECT_ROOT / 'data' / 'credit_risk.db'}",
        )
    )

    query_row_limit: int = field(
        default_factory=lambda: _int(
            "QUERY_ROW_LIMIT",
            500,
        )
    )


@dataclass(frozen=True)
class LLMConfig:
    """
    Gemini-only configuration for the Talk-to-Data system.
    """

    provider: str = field(
        default_factory=lambda: os.getenv(
            "LLM_PROVIDER",
            "gemini",
        )
    )

    gemini_api_key: str = field(
        default_factory=lambda: os.getenv(
            "GEMINI_API_KEY",
            "",
        )
    )

    gemini_model: str = field(
        default_factory=lambda: os.getenv(
            "GEMINI_MODEL",
            "gemini-3.6-flash",
        )
    )

    max_tokens: int = field(
        default_factory=lambda: _int(
            "LLM_MAX_TOKENS",
            500,
        )
    )

    temperature: float = field(
        default_factory=lambda: _float(
            "LLM_TEMPERATURE",
            0.0,
        )
    )

    # Cache identical questions to reduce unnecessary Gemini API calls
    # and improve response speed.
    enable_cache: bool = field(
        default_factory=lambda: _bool(
            "LLM_ENABLE_CACHE",
            True,
        )
    )

    @property
    def has_api_key(self) -> bool:
        """
        Return True only when Gemini is configured with an API key.
        """

        return (
            self.provider.lower() == "gemini"
            and bool(self.gemini_api_key)
        )

    @property
    def active_model_name(self) -> str:
        """
        Return the configured Gemini model name.
        """

        if self.provider.lower() == "gemini":
            return self.gemini_model

        return "unknown"


@dataclass(frozen=True)
class AppConfig:
    paths: PathConfig = field(
        default_factory=PathConfig
    )

    data: DataConfig = field(
        default_factory=DataConfig
    )

    model: ModelConfig = field(
        default_factory=ModelConfig
    )

    db: DBConfig = field(
        default_factory=DBConfig
    )

    llm: LLMConfig = field(
        default_factory=LLMConfig
    )

    log_level: str = field(
        default_factory=lambda: os.getenv(
            "LOG_LEVEL",
            "INFO",
        )
    )


config = AppConfig()

# Ensure required directories exist when configuration is loaded.
config.paths.ensure()
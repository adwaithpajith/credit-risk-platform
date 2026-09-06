"""Shared logging setup. Import get_logger(__name__) everywhere instead of
print() so behaviour is consistent across the CLI pipeline, the Streamlit
app, and Docker (where stdout is what `docker logs` shows)."""
from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler

from src.utils.config import config

_CONFIGURED = False


def _configure_root() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    root = logging.getLogger("credit_risk")
    root.setLevel(config.log_level)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    root.addHandler(stream_handler)

    try:
        config.paths.logs_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            config.paths.logs_dir / "platform.log",
            maxBytes=2_000_000,
            backupCount=3,
        )
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)
    except OSError:
        # Read-only filesystem edge case (e.g. some restricted containers) --
        # stdout logging still works, so don't crash the app over this.
        pass

    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    _configure_root()
    return logging.getLogger(f"credit_risk.{name}")

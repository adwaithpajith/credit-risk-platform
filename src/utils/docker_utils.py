"""Helpers that only matter when running inside Docker Compose -- mainly
waiting for the Postgres service to accept connections before the app or
the training job tries to use it (Compose starts containers in parallel,
it does not wait for Postgres to finish booting)."""
from __future__ import annotations

import time

from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger("docker_utils")


def wait_for_database(max_attempts: int = 30, delay_seconds: float = 2.0) -> bool:
    """Poll the configured database until it responds to `SELECT 1`.

    Returns True if the DB became reachable, False if it never did within
    the attempt budget (caller decides whether that's fatal).
    """
    url = config.db.database_url
    if url.startswith("sqlite"):
        return True  # nothing to wait for

    for attempt in range(1, max_attempts + 1):
        try:
            engine = create_engine(url, pool_pre_ping=True)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Database reachable after %d attempt(s).", attempt)
            return True
        except OperationalError as exc:
            logger.info(
                "Database not ready yet (attempt %d/%d): %s",
                attempt, max_attempts, str(exc).splitlines()[0],
            )
            time.sleep(delay_seconds)
    logger.error("Database never became reachable at %s", url)
    return False


if __name__ == "__main__":
    ok = wait_for_database()
    raise SystemExit(0 if ok else 1)

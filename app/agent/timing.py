import time
from datetime import datetime, timezone


def start_timer() -> tuple[str, float]:
    """Return (ISO-8601 timestamp, monotonic start) at node entry."""
    return datetime.now(timezone.utc).isoformat(), time.monotonic()


def elapsed_ms(start: float) -> int:
    """Milliseconds elapsed since start_timer() was called."""
    return int((time.monotonic() - start) * 1000)

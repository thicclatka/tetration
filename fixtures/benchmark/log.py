"""Benchmark logging and timing helpers."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[bench {ts}] {msg}", flush=True)


def format_elapsed(seconds: float) -> str:
    return f"{seconds:.3f}s"


def time_warm_logged(label: str, compute: Callable[[], float]) -> tuple[float, float]:
    log(f"{label}: warm pass …")
    compute()
    log(f"{label}: timed pass …")
    import time

    t0 = time.perf_counter()
    value = compute()
    elapsed = time.perf_counter() - t0
    log(f"{label}: done in {format_elapsed(elapsed)}  value={value}")
    return elapsed, value

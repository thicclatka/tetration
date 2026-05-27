"""Benchmark constants and types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from generate.constants import FIXTURES_ROOT

REPO_ROOT = FIXTURES_ROOT.parent
RESULTS_DIR = FIXTURES_ROOT / "bench_results"
RUNS_DIR = RESULTS_DIR / "runs"
RESULTS_FILE = RESULTS_DIR / "latest.md"

FormatName = Literal["h5", "netcdf", "zarr"]
TierName = Literal["large", "extra"]
OpName = Literal["mean", "sum", "min", "max", "count", "std", "var"]

DEFAULT_FORMATS: tuple[FormatName, ...] = ("h5", "netcdf", "zarr")
DEFAULT_OPS: tuple[OpName, ...] = ("mean", "sum", "min", "max", "count", "std", "var")

# Passed as query `execution.device` (and `tet query --device` when set). Use `auto` for
# Phase 10 routing; `cpu` to match pre-GPU benches.
DEFAULT_TET_DEVICE = "auto"

POP_DDOF = 0

TOLERANCE: dict[OpName, float] = {
    "mean": 1e-4,
    "sum": 1e-2,
    "min": 1e-5,
    "max": 1e-5,
    "count": 0.5,
    "std": 1e-4,
    "var": 1e-4,
}

TET_EXECUTION_KEY: dict[OpName, str] = {
    "mean": "operation_mean",
    "sum": "operation_sum",
    "min": "operation_min",
    "max": "operation_max",
    "count": "operation_element_count",
    "std": "operation_std",
    "var": "operation_var",
}


@dataclass(frozen=True)
class OpCompat:
    bench_source: bool
    note: str | None = None


def op_compat(format: FormatName, op: OpName) -> OpCompat:
    if format == "zarr":
        return OpCompat(
            bench_source=True,
            note="zarr: Python dir-store raw f32 chunks (still slower than h5/netcdf C libs)",
        )
    if format == "netcdf" and op in ("std", "var"):
        return OpCompat(
            bench_source=True, note="netcdf: population var/std ddof=0 via numpy slabs"
        )
    return OpCompat(bench_source=True)

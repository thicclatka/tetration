"""Committed bench workload contract and post-generate verification."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmark.case import BenchCase
from generate.constants import CHUNK_ELEMS
from generate.util import iter_slabs

SPEC_PATH = Path(__file__).resolve().parent / "spec.json"


@dataclass(frozen=True)
class CaseSpec:
    element_count: int
    mean: float
    mean_tolerance: float


def load_spec() -> dict[str, Any]:
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


def spec_sha256() -> str:
    raw = SPEC_PATH.read_bytes()
    return hashlib.sha256(raw).hexdigest()


def case_key(case: BenchCase) -> str:
    return f"{case.format}/{case.tier}"


def case_spec(case: BenchCase) -> CaseSpec:
    data = load_spec()["cases"][case_key(case)]
    return CaseSpec(
        element_count=int(data["element_count"]),
        mean=float(data["mean"]),
        mean_tolerance=float(data["mean_tolerance"]),
    )


def fingerprint_case(case: BenchCase) -> tuple[int, float]:
    """Stream `data` once: element count + mean (format-specific reader)."""
    match case.format:
        case "h5":
            return _fingerprint_h5(case)
        case "netcdf":
            return _fingerprint_netcdf(case)
        case "zarr":
            return _fingerprint_zarr(case)
    raise ValueError(case.format)


def verify_case(case: BenchCase) -> None:
    spec = case_spec(case)
    count, mean = fingerprint_case(case)
    if count != spec.element_count:
        raise RuntimeError(
            f"{case_key(case)}: element count {count} != spec {spec.element_count}"
        )
    if abs(mean - spec.mean) > spec.mean_tolerance:
        raise RuntimeError(
            f"{case_key(case)}: mean {mean:.6g} outside "
            f"{spec.mean} ± {spec.mean_tolerance}"
        )


def _fingerprint_h5(case: BenchCase) -> tuple[int, float]:
    import h5py

    with h5py.File(case.src, "r") as f:
        d = f["data"]
        length = int(d.shape[0])
        total = 0.0
        for start, end in iter_slabs(length):
            total += float(d[start:end].sum(dtype=float))
        return length, total / length if length else 0.0


def _fingerprint_netcdf(case: BenchCase) -> tuple[int, float]:
    import netCDF4 as nc

    with nc.Dataset(case.src, "r") as ds:
        var = ds.variables["data"]
        length = int(var.shape[0])
        total = 0.0
        for start, end in iter_slabs(length):
            total += float(var[start:end].sum(dtype=float))
        return length, total / length if length else 0.0


def _fingerprint_zarr(case: BenchCase) -> tuple[int, float]:
    import zarr

    arr = zarr.open_group(str(case.src), mode="r")["data"]
    length = int(arr.shape[0])
    chunk_len = int(arr.chunks[0])
    total = 0.0
    for start in range(0, length, chunk_len):
        end = min(start + chunk_len, length)
        total += float(arr[start:end].sum(dtype=float))
    return length, total / length if length else 0.0


def spec_summary() -> dict[str, Any]:
    spec = load_spec()
    return {
        "schema_version": spec["schema_version"],
        "spec_sha256": spec_sha256(),
        "chunk_elems": CHUNK_ELEMS,
    }

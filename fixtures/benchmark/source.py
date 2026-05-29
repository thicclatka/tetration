"""Native source reads for benchmark comparisons."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from generate.constants import CHUNK_ELEMS
from generate.util import iter_slabs

from benchmark.case import BenchCase
from benchmark.constants import OpName, op_meta
from benchmark.log import time_warm_logged


@dataclass
class RunningStats:
    count: int = 0
    sum: float = 0.0
    min: float = math.inf
    max: float = -math.inf
    m2: float = 0.0
    any_inf: bool = False
    inf_count: int = 0
    nan_count: int = 0

    def push_slab(self, arr: np.ndarray) -> None:
        flat = np.asarray(arr, dtype=np.float64).ravel()
        n = int(flat.size)
        if n == 0:
            return
        if np.isinf(flat).any():
            self.any_inf = True
            self.inf_count += int(np.isinf(flat).sum())
        if np.isnan(flat).any():
            self.nan_count += int(np.isnan(flat).sum())

        finite = flat[np.isfinite(flat)]
        n_fin = int(finite.size)
        if n_fin == 0:
            return

        slab_sum = float(finite.sum())
        slab_mean = slab_sum / n_fin
        slab_m2 = float(((finite - slab_mean) ** 2).sum())
        slab_min = float(finite.min())
        slab_max = float(finite.max())

        if self.count == 0:
            self.count = n_fin
            self.sum = slab_sum
            self.min = slab_min
            self.max = slab_max
            self.m2 = slab_m2
            return

        n_a = self.count
        n_b = n_fin
        total = n_a + n_b
        mean_a = self.sum / n_a
        delta = slab_mean - mean_a
        self.m2 += slab_m2 + delta * delta * n_a * n_b / total
        self.count = total
        self.sum += slab_sum
        self.min = min(self.min, slab_min)
        self.max = max(self.max, slab_max)

    def finish_scalar(self, op: OpName) -> float:
        if self.count == 0:
            raise ValueError("empty selection")
        match op:
            case "mean" | "nan_mean":
                return self.sum / self.count
            case "sum":
                return self.sum
            case "min":
                return self.min
            case "max":
                return self.max
            case "count":
                return float(self.count)
            case "var":
                return self.m2 / self.count if self.count else 0.0
            case "std" | "nan_std":
                return math.sqrt(self.m2 / self.count) if self.count else 0.0
            case "any_inf":
                return 1.0 if self.any_inf else 0.0
            case "inf_count":
                return float(self.inf_count)
            case "nan_count":
                return float(self.nan_count)
        raise ValueError(op)


def _iter_source_slabs(case: BenchCase):
    slab_mib = CHUNK_ELEMS // (1024 * 1024)

    def run_h5():
        import h5py

        with h5py.File(case.src, "r") as f:
            d = f["data"]
            length = int(d.shape[0])
            for start, end in iter_slabs(length):
                yield np.asarray(d[start:end], dtype=np.float64)

    def run_netcdf():
        import netCDF4 as nc

        with nc.Dataset(case.src, "r") as ds:
            var = ds.variables["data"]
            length = int(var.shape[0])
            for start, end in iter_slabs(length):
                yield np.asarray(var[start:end], dtype=np.float64)

    def run_zarr():
        import zarr

        root = zarr.open_group(str(case.src), mode="r")
        arr = root["data"]
        length = int(arr.shape[0])
        chunk_elems = int(arr.chunks[0])
        for start in range(0, length, chunk_elems):
            end = min(start + chunk_elems, length)
            yield np.asarray(arr[start:end], dtype=np.float64)

    match case.format:
        case "h5":
            return run_h5(), f"source ({slab_mib} MiB slabs, h5py)"
        case "netcdf":
            return run_netcdf(), f"source ({slab_mib} MiB slabs, netCDF4)"
        case "zarr":
            chunk_mib = CHUNK_ELEMS * 4 // (1024 * 1024)
            return run_zarr(), f"source ({chunk_mib} MiB zarr chunks, raw)"


def _fold_source(case: BenchCase) -> RunningStats:
    slabs, _ = _iter_source_slabs(case)
    stats = RunningStats()
    for slab in slabs:
        stats.push_slab(slab)
    return stats


def reduce_source_op(case: BenchCase, op: OpName) -> tuple[float, float]:
    meta = op_meta(op)
    if not meta.bench_source:
        raise ValueError(meta.note or "source not supported")

    _, slab_label = _iter_source_slabs(case)

    if meta.is_transform:
        method = meta.transform_method
        assert method is not None

        def compute() -> float:
            stats = _fold_source(case)
            if method in ("zscore", "center"):
                return stats.finish_scalar("mean")
            if method in ("minmax", "log1p", "sqrt"):
                return stats.finish_scalar("min")
            if method == "softmax":
                return stats.finish_scalar("max")
            if method == "scale":
                return stats.finish_scalar("std")
            if method == "l1":
                return _l1_norm(case)
            if method == "l2":
                return _l2_norm(case)
            raise ValueError(method)

        return time_warm_logged(
            f"source transform/{method} pass-1 stat ({slab_label})",
            compute,
        )

    def compute_scalar() -> float:
        return _fold_source(case).finish_scalar(op)

    return time_warm_logged(f"source {op} ({slab_label})", compute_scalar)


def _l1_norm(case: BenchCase) -> float:
    total = 0.0
    slabs, _ = _iter_source_slabs(case)
    for slab in slabs:
        total += float(np.abs(slab).sum())
    return total


def _l2_norm(case: BenchCase) -> float:
    total = 0.0
    slabs, _ = _iter_source_slabs(case)
    for slab in slabs:
        total += float((slab * slab).sum())
    return math.sqrt(total)

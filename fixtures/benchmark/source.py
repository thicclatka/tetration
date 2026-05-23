"""Native source reads for benchmark comparisons."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from benchmark.case import BenchCase
from benchmark.constants import OpName, op_compat
from benchmark.log import time_warm_logged
from generate.constants import CHUNK_ELEMS
from generate.util import iter_slabs


@dataclass
class RunningStats:
    count: int = 0
    sum: float = 0.0
    min: float = math.inf
    max: float = -math.inf
    m2: float = 0.0

    def push_slab(self, arr: np.ndarray) -> None:
        flat = np.asarray(arr, dtype=np.float64).ravel()
        n = int(flat.size)
        if n == 0:
            return
        slab_sum = float(flat.sum())
        slab_mean = slab_sum / n
        slab_m2 = float(((flat - slab_mean) ** 2).sum())
        slab_min = float(flat.min())
        slab_max = float(flat.max())

        if self.count == 0:
            self.count = n
            self.sum = slab_sum
            self.min = slab_min
            self.max = slab_max
            self.m2 = slab_m2
            return

        n_a = self.count
        n_b = n
        total = n_a + n_b
        mean_a = self.sum / n_a
        delta = slab_mean - mean_a
        self.m2 += slab_m2 + delta * delta * n_a * n_b / total
        self.count = total
        self.sum += slab_sum
        self.min = min(self.min, slab_min)
        self.max = max(self.max, slab_max)

    def finish(self, op: OpName) -> float:
        if self.count == 0:
            raise ValueError("empty selection")
        match op:
            case "mean":
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
            case "std":
                return math.sqrt(self.m2 / self.count) if self.count else 0.0
        raise ValueError(op)


def reduce_source_op(case: BenchCase, op: OpName) -> tuple[float, float]:
    compat = op_compat(case.format, op)
    if not compat.bench_source:
        raise ValueError(compat.note or "source not supported")

    slab_mib = CHUNK_ELEMS // (1024 * 1024)

    def run_h5() -> tuple[float, float]:
        import h5py

        with h5py.File(case.src, "r") as f:
            d = f["data"]
            length = int(d.shape[0])

            def compute() -> float:
                stats = RunningStats()
                for start, end in iter_slabs(length):
                    stats.push_slab(d[start:end])
                return stats.finish(op)

            return time_warm_logged(
                f"source {op} ({slab_mib} MiB slabs, h5py)",
                compute,
            )

    def run_netcdf() -> tuple[float, float]:
        import netCDF4 as nc

        with nc.Dataset(case.src, "r") as ds:
            var = ds.variables["data"]
            length = int(var.shape[0])

            def compute() -> float:
                stats = RunningStats()
                for start, end in iter_slabs(length):
                    stats.push_slab(var[start:end])
                return stats.finish(op)

            return time_warm_logged(
                f"source {op} ({slab_mib} MiB slabs, netCDF4)",
                compute,
            )

    def run_zarr() -> tuple[float, float]:
        import zarr

        root = zarr.open_group(str(case.src), mode="r")
        arr = root["data"]
        length = int(arr.shape[0])

        def compute() -> float:
            stats = RunningStats()
            for start, end in iter_slabs(length):
                stats.push_slab(arr[start:end])
            return stats.finish(op)

        return time_warm_logged(
            f"source {op} ({slab_mib} MiB slabs, zarr)",
            compute,
        )

    match case.format:
        case "h5":
            return run_h5()
        case "netcdf":
            return run_netcdf()
        case "zarr":
            return run_zarr()

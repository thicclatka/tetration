#!/usr/bin/env python3
"""Generate small (tracked) and large (gitignored) HDF5 / NetCDF fixture tensors."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Literal, TypeVar

import h5py
import netCDF4 as nc
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
SMALL_H5 = ROOT / "small" / "h5"
SMALL_NC = ROOT / "small" / "netcdf"
LARGE_H5 = ROOT / "large" / "h5"
LARGE_NC = ROOT / "large" / "netcdf"

SMALL_SHAPES: dict[int, tuple[int, ...]] = {
    3: (32, 32, 32),
    4: (16, 16, 16, 16),
    5: (8, 8, 8, 8, 8),
}

NUMERIC_DTYPES: tuple[str, ...] = ("f32", "f64", "i32", "i64")

DtypeName = Literal["f32", "f64", "i32", "i64"]

LARGE_LOGICAL_BYTES = 20 * 1024**3  # 20 GiB
FLOAT32_BYTES = 4  # 4 bytes per f32
LARGE_ELEM_COUNT = LARGE_LOGICAL_BYTES // FLOAT32_BYTES
LARGE_SHAPE = (LARGE_ELEM_COUNT,)
# ~64 MiB of f32 per write slab (keeps peak RAM low while building 20 GiB files).
CHUNK_ELEMS = 16 * 1024 * 1024

SEED_SMALL = 0
SEED_LARGE = 1

T = TypeVar("T")


def _slab_count(total: int, slab_elems: int) -> int:
    return (total + slab_elems - 1) // slab_elems


def _iter_slabs(total: int, slab_elems: int) -> Iterator[tuple[int, int]]:
    for start in range(0, total, slab_elems):
        end = min(start + slab_elems, total)
        yield start, end


def _progress(
    iterable: Iterator[T],
    *,
    total: int | None,
    desc: str,
    quiet: bool,
    unit: str = "it",
) -> Iterator[T]:
    if quiet:
        return iterable
    return tqdm(iterable, total=total, desc=desc, unit=unit, dynamic_ncols=True)


def _seed_for(ndim: int, dtype: DtypeName) -> int:
    return SEED_SMALL + ndim * 17 + NUMERIC_DTYPES.index(dtype) * 997


def _numpy_dtype(dtype: DtypeName) -> np.dtype:
    return np.dtype({"f32": "f4", "f64": "f8", "i32": "i4", "i64": "i8"}[dtype])


def _netcdf_dtype(dtype: DtypeName) -> str:
    return {"f32": "f4", "f64": "f8", "i32": "i4", "i64": "i8"}[dtype]


def _small_array(ndim: int, dtype: DtypeName) -> np.ndarray:
    shape = SMALL_SHAPES[ndim]
    n = int(np.prod(shape))
    rng = np.random.default_rng(_seed_for(ndim, dtype))
    np_dtype = _numpy_dtype(dtype)

    if dtype in ("f32", "f64"):
        base = np.linspace(0.0, 1.0, num=n, dtype=np_dtype)
        noise = rng.standard_normal(shape, dtype=np_dtype) * np_dtype.type(0.01)
        return (base.reshape(shape) + noise).astype(np_dtype, copy=False)

    modulus = np.int32(1_000) if dtype == "i32" else np.int64(1_000_000)
    base = (np.arange(n, dtype=np_dtype) % modulus).reshape(shape)
    noise = rng.integers(-3, 4, size=shape, dtype=np_dtype)
    return base + noise


def _write_h5_small(path: Path, ndim: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["tetration_fixture"] = f"small_{ndim}d"
        f.attrs["tetration_ndim"] = ndim
        f.attrs["tetration_dtypes"] = ",".join(NUMERIC_DTYPES)
        for dtype in NUMERIC_DTYPES:
            data = _small_array(ndim, dtype)
            dset = f.create_dataset(
                dtype,
                data=data,
                compression="gzip",
                compression_opts=4,
            )
            dset.attrs["tetration_dtype"] = dtype


def _write_nc_small(path: Path, ndim: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dim_names = tuple(f"d{i}" for i in range(ndim))
    shape = SMALL_SHAPES[ndim]
    with nc.Dataset(path, "w", format="NETCDF4") as ds:
        ds.setncattr("tetration_fixture", f"small_{ndim}d")
        ds.setncattr("tetration_ndim", ndim)
        ds.setncattr("tetration_dtypes", ",".join(NUMERIC_DTYPES))
        for name, size in zip(dim_names, shape, strict=True):
            ds.createDimension(name, size)
        for dtype in NUMERIC_DTYPES:
            data = _small_array(ndim, dtype)
            var = ds.createVariable(
                dtype,
                _netcdf_dtype(dtype),
                dim_names,
                zlib=True,
                complevel=4,
            )
            var.setncattr("tetration_dtype", dtype)
            var[:] = data


def _write_h5_large(path: Path, *, quiet: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED_LARGE)
    total = LARGE_ELEM_COUNT
    n_slabs = _slab_count(total, CHUNK_ELEMS)
    giB = LARGE_LOGICAL_BYTES / (1024**3)

    if not quiet:
        print(f"HDF5: {path} ({giB:.0f} GiB logical f32, {n_slabs} slabs)")

    with h5py.File(path, "w") as f:
        f.attrs["tetration_fixture"] = "large_20gb"
        f.attrs["tetration_logical_bytes"] = LARGE_LOGICAL_BYTES
        dset = f.create_dataset(
            "data",
            shape=LARGE_SHAPE,
            dtype=np.float32,
            chunks=(CHUNK_ELEMS,),
        )
        for start, end in _progress(
            _iter_slabs(total, CHUNK_ELEMS),
            total=n_slabs,
            desc=path.name,
            quiet=quiet,
            unit="slab",
        ):
            dset[start:end] = rng.random(end - start, dtype=np.float32)

    if not quiet:
        print(f"  finished {path}")


def _write_nc_large(path: Path, *, quiet: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED_LARGE)
    total = LARGE_ELEM_COUNT
    n_slabs = _slab_count(total, CHUNK_ELEMS)
    giB = LARGE_LOGICAL_BYTES / (1024**3)

    if not quiet:
        print(f"NetCDF: {path} ({giB:.0f} GiB logical f32, {n_slabs} slabs)")

    with nc.Dataset(path, "w", format="NETCDF4") as ds:
        ds.setncattr("tetration_fixture", "large_20gb")
        ds.setncattr("tetration_logical_bytes", int(LARGE_LOGICAL_BYTES))
        ds.createDimension("i", total)
        var = ds.createVariable("data", "f4", ("i",), chunksizes=(CHUNK_ELEMS,))
        for start, end in _progress(
            _iter_slabs(total, CHUNK_ELEMS),
            total=n_slabs,
            desc=path.name,
            quiet=quiet,
            unit="slab",
        ):
            var[start:end] = rng.random(end - start, dtype=np.float32)

    if not quiet:
        print(f"  finished {path}")


def generate_small(*, quiet: bool) -> None:
    jobs: list[tuple[str, Callable[[Path, int], None], Path, int]] = []
    for ndim in (3, 4, 5):
        jobs.append(("h5", _write_h5_small, SMALL_H5 / f"tensor_{ndim}d.h5", ndim))
        jobs.append(("nc", _write_nc_small, SMALL_NC / f"tensor_{ndim}d.nc", ndim))

    if not quiet:
        dtypes = ", ".join(NUMERIC_DTYPES)
        print(
            f"Generating {len(jobs)} small fixtures under {ROOT / 'small'} "
            f"({dtypes} per file)"
        )

    for fmt, writer, path, ndim in _progress(
        jobs, total=len(jobs), desc="small", quiet=quiet
    ):
        if not quiet:
            tqdm.write(f"  {fmt} {ndim}d -> {path.relative_to(ROOT)}")
        writer(path, ndim)

    if not quiet:
        print(f"Done: {ROOT / 'small'}")


def generate_large(*, quiet: bool) -> None:
    if not quiet:
        print(f"Generating large fixtures under {ROOT / 'large'} (gitignored)")
    _write_h5_large(LARGE_H5 / "tensor_20gb.h5", quiet=quiet)
    _write_nc_large(LARGE_NC / "tensor_20gb.nc", quiet=quiet)
    if not quiet:
        print(f"Done: {ROOT / 'large'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "target",
        nargs="?",
        choices=("small", "large", "all"),
        default="small",
        help="small: 3d/4d/5d trackable files; large: 20 GiB each (untracked); all: both",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="no tqdm bars or status lines",
    )
    args = parser.parse_args(argv)
    if args.target in ("small", "all"):
        generate_small(quiet=args.quiet)
    if args.target in ("large", "all"):
        generate_large(quiet=args.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())

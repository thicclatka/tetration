"""Shared helpers for fixture generation."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TypeVar

import numpy as np
from tqdm import tqdm

from generate.constants import (
    CHUNK_ELEMS,
    NUMERIC_DTYPES,
    SEED_SMALL,
    SMALL_SHAPES,
    DtypeName,
)

T = TypeVar("T")


def slab_count(total: int, slab_elems: int = CHUNK_ELEMS) -> int:
    return (total + slab_elems - 1) // slab_elems


def iter_slabs(total: int, slab_elems: int = CHUNK_ELEMS) -> Iterator[tuple[int, int]]:
    for start in range(0, total, slab_elems):
        end = min(start + slab_elems, total)
        yield start, end


def progress(
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


def seed_for(ndim: int, dtype: DtypeName) -> int:
    return SEED_SMALL + ndim * 17 + NUMERIC_DTYPES.index(dtype) * 997


def numpy_dtype(dtype: DtypeName) -> np.dtype:
    return np.dtype({"f32": "f4", "f64": "f8", "i32": "i4", "i64": "i8"}[dtype])


def netcdf_dtype(dtype: DtypeName) -> str:
    return {"f32": "f4", "f64": "f8", "i32": "i4", "i64": "i8"}[dtype]


def chunk_shape_for_small(shape: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(min(32, dim) for dim in shape)


# Large stress zarr: raw little-endian f32 chunks (matches uncompressed h5/nc bench fixtures).
ZARR_RAW_ARRAY_KWARGS: dict[str, object] = {"compressors": None}


def small_array(ndim: int, dtype: DtypeName) -> np.ndarray:
    shape = SMALL_SHAPES[ndim]
    n = int(np.prod(shape))
    rng = np.random.default_rng(seed_for(ndim, dtype))
    np_dtype = numpy_dtype(dtype)

    if dtype in ("f32", "f64"):
        base = np.linspace(0.0, 1.0, num=n, dtype=np_dtype)
        noise = rng.standard_normal(shape, dtype=np_dtype) * np_dtype.type(0.01)
        return (base.reshape(shape) + noise).astype(np_dtype, copy=False)

    modulus = np.int32(1_000) if dtype == "i32" else np.int64(1_000_000)
    base = (np.arange(n, dtype=np_dtype) % modulus).reshape(shape)
    noise = rng.integers(-3, 4, size=shape, dtype=np_dtype)
    return base + noise

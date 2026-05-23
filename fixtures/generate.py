#!/usr/bin/env python3
"""Generate small (tracked) and large (gitignored) HDF5 / NetCDF / Zarr fixture tensors."""

from __future__ import annotations

import argparse
import shutil
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Literal, TypeVar

import h5py
import netCDF4 as nc
import numpy as np
import zarr
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
SMALL_H5 = ROOT / "small" / "h5"
SMALL_NC = ROOT / "small" / "netcdf"
SMALL_ZARR = ROOT / "small" / "zarr"
LARGE_H5 = ROOT / "large" / "h5"
LARGE_NC = ROOT / "large" / "netcdf"
LARGE_ZARR = ROOT / "large" / "zarr"
EXTRA_LARGE_H5 = ROOT / "extra_large" / "h5"
EXTRA_LARGE_NC = ROOT / "extra_large" / "netcdf"
EXTRA_LARGE_ZARR = ROOT / "extra_large" / "zarr"

SMALL_SHAPES: dict[int, tuple[int, ...]] = {
    3: (32, 32, 32),
    4: (16, 16, 16, 16),
    5: (8, 8, 8, 8, 8),
}

NUMERIC_DTYPES: tuple[str, ...] = ("f32", "f64", "i32", "i64")

DtypeName = Literal["f32", "f64", "i32", "i64"]

# Large stress suite: ~20 GiB logical f32 **total** split across HDF5, NetCDF, and Zarr.
LARGE_SUITE_LOGICAL_BYTES = 20 * 1024**3
LARGE_FORMATS = 3
LARGE_PER_FORMAT_BYTES = LARGE_SUITE_LOGICAL_BYTES // LARGE_FORMATS
FLOAT32_BYTES = 4
LARGE_PER_FORMAT_ELEMS = LARGE_PER_FORMAT_BYTES // FLOAT32_BYTES

# Single-file stress tensors (one format, full 20 GiB logical f32).
EXTRA_LARGE_LOGICAL_BYTES = 20 * 1024**3
EXTRA_LARGE_ELEM_COUNT = EXTRA_LARGE_LOGICAL_BYTES // FLOAT32_BYTES

# ~64 MiB of f32 per write slab (keeps peak RAM low while building multi-GiB files).
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


def _chunk_shape_for_small(shape: tuple[int, ...]) -> tuple[int, ...]:
    # Match convert defaults: modest tiles without tiny fragments.
    return tuple(min(32, dim) for dim in shape)


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


def _set_tetration_small_attrs(obj, *, fixture: str, ndim: int) -> None:
    obj.attrs["tetration_fixture"] = fixture
    obj.attrs["tetration_ndim"] = ndim
    obj.attrs["tetration_dtypes"] = ",".join(NUMERIC_DTYPES)


def _write_h5_small(path: Path, ndim: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        _set_tetration_small_attrs(f, fixture=f"small_{ndim}d", ndim=ndim)
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


def _write_h5_groups_small(path: Path, ndim: int) -> None:
    """Nested groups for Phase 5 richer HDF5 import tests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        _set_tetration_small_attrs(f, fixture=f"groups_{ndim}d", ndim=ndim)
        primary = f.create_group("primary")
        for dtype in NUMERIC_DTYPES:
            data = _small_array(ndim, dtype)
            dset = primary.create_dataset(
                dtype,
                data=data,
                compression="gzip",
                compression_opts=4,
            )
            dset.attrs["tetration_dtype"] = dtype
            dset.attrs["long_name"] = f"primary {dtype} field"
        aux = f.create_group("aux")
        aux.create_dataset(
            "scale",
            data=np.array([1.0, 1.5, 2.0], dtype=np.float32),
        )
        meta = f.create_group("meta")
        meta.attrs["version"] = 1
        meta.attrs["source"] = "tetration-fixtures"


def _write_nc_groups_small(path: Path, ndim: int) -> None:
    """NetCDF-4 groups mirroring the HDF5 groups layout."""
    path.parent.mkdir(parents=True, exist_ok=True)
    dim_names = tuple(f"d{i}" for i in range(ndim))
    shape = SMALL_SHAPES[ndim]
    with nc.Dataset(path, "w", format="NETCDF4") as ds:
        ds.setncattr("tetration_fixture", f"groups_{ndim}d")
        ds.setncattr("tetration_ndim", ndim)
        ds.setncattr("tetration_dtypes", ",".join(NUMERIC_DTYPES))
        for name, size in zip(dim_names, shape, strict=True):
            ds.createDimension(name, size)
        primary = ds.createGroup("primary")
        for dtype in NUMERIC_DTYPES:
            data = _small_array(ndim, dtype)
            var = primary.createVariable(
                dtype,
                _netcdf_dtype(dtype),
                dim_names,
                zlib=True,
                complevel=4,
            )
            var.setncattr("tetration_dtype", dtype)
            var.long_name = f"primary {dtype} field"
            var[:] = data
        aux = ds.createGroup("aux")
        scale = aux.createVariable("scale", "f4", ())
        scale[:] = np.float32(1.25)


def _write_h5_cf_small(path: Path) -> None:
    """CF-style attrs, coordinates, and packed storage (3-D only)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ndim = 3
    shape = SMALL_SHAPES[ndim]
    dim_names = ("time", "lat", "lon")
    with h5py.File(path, "w") as f:
        _set_tetration_small_attrs(f, fixture="cf_3d", ndim=ndim)
        coords = f.create_group("coordinates")
        coords.create_dataset("time", data=np.arange(shape[0], dtype=np.float64))
        coords.create_dataset("lat", data=np.linspace(-90.0, 90.0, shape[1], dtype=np.float32))
        coords.create_dataset("lon", data=np.linspace(-180.0, 180.0, shape[2], dtype=np.float32))
        coords["time"].attrs["units"] = "days since 2020-01-01"
        coords["lat"].attrs["units"] = "degrees_north"
        coords["lon"].attrs["units"] = "degrees_east"

        physical = _small_array(ndim, "f32")
        scale = np.float32(0.01)
        offset = np.float32(273.15)
        fill = np.float32(-9999.0)
        stored = np.where(
            physical > 0.9,
            fill,
            (physical - offset) / scale,
        ).astype(np.float32)

        sst = f.create_dataset(
            "temperature",
            data=stored,
            compression="gzip",
            compression_opts=4,
        )
        sst.attrs["scale_factor"] = float(scale)
        sst.attrs["add_offset"] = float(offset)
        sst.attrs["_FillValue"] = float(fill)
        sst.attrs["units"] = "K"
        sst.attrs["long_name"] = "sea surface temperature"
        sst.attrs["coordinates"] = "time lat lon"

        # Plain root-level dtypes remain for baseline convert until CF decode lands.
        for dtype in NUMERIC_DTYPES:
            data = _small_array(ndim, dtype)
            dset = f.create_dataset(dtype, data=data, compression="gzip", compression_opts=4)
            dset.attrs["tetration_dtype"] = dtype


def _write_nc_cf_small(path: Path) -> None:
    """CF conventions: coords, scale_factor, add_offset, _FillValue."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ndim = 3
    shape = SMALL_SHAPES[ndim]
    dim_names = ("time", "lat", "lon")
    with nc.Dataset(path, "w", format="NETCDF4") as ds:
        ds.setncattr("tetration_fixture", "cf_3d")
        ds.setncattr("tetration_ndim", ndim)
        ds.setncattr("tetration_dtypes", ",".join(NUMERIC_DTYPES))
        for name, size in zip(dim_names, shape, strict=True):
            ds.createDimension(name, size)

        time = ds.createVariable("time", "f8", ("time",))
        time.units = "days since 2020-01-01"
        time[:] = np.arange(shape[0], dtype=np.float64)

        lat = ds.createVariable("lat", "f4", ("lat",))
        lat.units = "degrees_north"
        lat.standard_name = "latitude"
        lat[:] = np.linspace(-90.0, 90.0, shape[1], dtype=np.float32)

        lon = ds.createVariable("lon", "f4", ("lon",))
        lon.units = "degrees_east"
        lon.standard_name = "longitude"
        lon[:] = np.linspace(-180.0, 180.0, shape[2], dtype=np.float32)

        physical = _small_array(ndim, "f32")
        scale = np.float32(0.01)
        offset = np.float32(273.15)
        fill = np.float32(-9999.0)
        stored = np.where(
            physical > 0.9,
            fill,
            (physical - offset) / scale,
        ).astype(np.float32)

        sst = ds.createVariable(
            "temperature",
            "f4",
            dim_names,
            fill_value=fill,
            zlib=True,
            complevel=4,
        )
        sst.setncattr("scale_factor", float(scale))
        sst.setncattr("add_offset", float(offset))
        sst.units = "K"
        sst.long_name = "sea surface temperature"
        sst[:] = stored

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


def _write_zarr_small(path: Path, ndim: int, *, grouped: bool = False) -> None:
    """Zarr v3 directory store; optional nested group layout."""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    shape = SMALL_SHAPES[ndim]
    chunks = _chunk_shape_for_small(shape)
    fixture = f"groups_{ndim}d" if grouped else f"small_{ndim}d"
    root = zarr.open_group(str(path), mode="w")
    root.attrs.update(
        {
            "tetration_fixture": fixture,
            "tetration_ndim": ndim,
            "tetration_dtypes": ",".join(NUMERIC_DTYPES),
        }
    )
    target = root.create_group("primary") if grouped else root
    for dtype in NUMERIC_DTYPES:
        data = _small_array(ndim, dtype)
        arr = target.create_array(
            dtype,
            shape=shape,
            chunks=chunks,
            dtype=_numpy_dtype(dtype),
        )
        arr[:] = data
        arr.attrs["tetration_dtype"] = dtype


def _write_h5_f32_slabs(
    path: Path,
    *,
    logical_bytes: int,
    fixture: str,
    suite_total_bytes: int | None,
    quiet: bool,
    seed: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    total = logical_bytes // FLOAT32_BYTES
    shape = (total,)
    n_slabs = _slab_count(total, CHUNK_ELEMS)
    gib = logical_bytes / (1024**3)

    if not quiet:
        print(f"HDF5: {path} ({gib:.2f} GiB logical f32, {n_slabs} slabs)")

    with h5py.File(path, "w") as f:
        f.attrs["tetration_fixture"] = fixture
        f.attrs["tetration_logical_bytes"] = int(logical_bytes)
        if suite_total_bytes is not None:
            f.attrs["tetration_suite_total_bytes"] = int(suite_total_bytes)
        dset = f.create_dataset(
            "data",
            shape=shape,
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


def _write_nc_f32_slabs(
    path: Path,
    *,
    logical_bytes: int,
    fixture: str,
    suite_total_bytes: int | None,
    quiet: bool,
    seed: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    total = logical_bytes // FLOAT32_BYTES
    n_slabs = _slab_count(total, CHUNK_ELEMS)
    gib = logical_bytes / (1024**3)

    if not quiet:
        print(f"NetCDF: {path} ({gib:.2f} GiB logical f32, {n_slabs} slabs)")

    with nc.Dataset(path, "w", format="NETCDF4") as ds:
        ds.setncattr("tetration_fixture", fixture)
        ds.setncattr("tetration_logical_bytes", int(logical_bytes))
        if suite_total_bytes is not None:
            ds.setncattr("tetration_suite_total_bytes", int(suite_total_bytes))
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


def _write_zarr_f32_slabs(
    path: Path,
    *,
    logical_bytes: int,
    fixture: str,
    suite_total_bytes: int | None,
    quiet: bool,
    seed: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        shutil.rmtree(path)
    rng = np.random.default_rng(seed)
    total = logical_bytes // FLOAT32_BYTES
    shape = (total,)
    n_slabs = _slab_count(total, CHUNK_ELEMS)
    gib = logical_bytes / (1024**3)

    if not quiet:
        print(f"Zarr: {path} ({gib:.2f} GiB logical f32, {n_slabs} slabs)")

    root = zarr.open_group(str(path), mode="w")
    attrs: dict[str, object] = {
        "tetration_fixture": fixture,
        "tetration_logical_bytes": int(logical_bytes),
    }
    if suite_total_bytes is not None:
        attrs["tetration_suite_total_bytes"] = int(suite_total_bytes)
    root.attrs.update(attrs)
    arr = root.create_array(
        "data",
        shape=shape,
        chunks=(CHUNK_ELEMS,),
        dtype="f4",
    )
    for start, end in _progress(
        _iter_slabs(total, CHUNK_ELEMS),
        total=n_slabs,
        desc=path.name,
        quiet=quiet,
        unit="slab",
    ):
        arr[start:end] = rng.random(end - start, dtype=np.float32)

    if not quiet:
        print(f"  finished {path}")


def _write_h5_large(path: Path, *, quiet: bool) -> None:
    _write_h5_f32_slabs(
        path,
        logical_bytes=LARGE_PER_FORMAT_BYTES,
        fixture="large_suite_h5",
        suite_total_bytes=LARGE_SUITE_LOGICAL_BYTES,
        quiet=quiet,
        seed=SEED_LARGE,
    )


def _write_nc_large(path: Path, *, quiet: bool) -> None:
    _write_nc_f32_slabs(
        path,
        logical_bytes=LARGE_PER_FORMAT_BYTES,
        fixture="large_suite_nc",
        suite_total_bytes=LARGE_SUITE_LOGICAL_BYTES,
        quiet=quiet,
        seed=SEED_LARGE + 1,
    )


def _write_zarr_large(path: Path, *, quiet: bool) -> None:
    _write_zarr_f32_slabs(
        path,
        logical_bytes=LARGE_PER_FORMAT_BYTES,
        fixture="large_suite_zarr",
        suite_total_bytes=LARGE_SUITE_LOGICAL_BYTES,
        quiet=quiet,
        seed=SEED_LARGE + 2,
    )


def generate_extra_large_h5(*, quiet: bool) -> None:
    if not quiet:
        gib = EXTRA_LARGE_LOGICAL_BYTES / (1024**3)
        print(f"Generating extra-large HDF5 under {EXTRA_LARGE_H5} ({gib:.0f} GiB)")
    _write_h5_f32_slabs(
        EXTRA_LARGE_H5 / "tensor_20gb.h5",
        logical_bytes=EXTRA_LARGE_LOGICAL_BYTES,
        fixture="extra_large_h5",
        suite_total_bytes=None,
        quiet=quiet,
        seed=SEED_LARGE + 10,
    )
    if not quiet:
        print(f"Done: {EXTRA_LARGE_H5}")


def generate_extra_large_netcdf(*, quiet: bool) -> None:
    if not quiet:
        gib = EXTRA_LARGE_LOGICAL_BYTES / (1024**3)
        print(f"Generating extra-large NetCDF under {EXTRA_LARGE_NC} ({gib:.0f} GiB)")
    _write_nc_f32_slabs(
        EXTRA_LARGE_NC / "tensor_20gb.nc",
        logical_bytes=EXTRA_LARGE_LOGICAL_BYTES,
        fixture="extra_large_nc",
        suite_total_bytes=None,
        quiet=quiet,
        seed=SEED_LARGE + 11,
    )
    if not quiet:
        print(f"Done: {EXTRA_LARGE_NC}")


def generate_extra_large_zarr(*, quiet: bool) -> None:
    if not quiet:
        gib = EXTRA_LARGE_LOGICAL_BYTES / (1024**3)
        print(f"Generating extra-large Zarr under {EXTRA_LARGE_ZARR} ({gib:.0f} GiB)")
    _write_zarr_f32_slabs(
        EXTRA_LARGE_ZARR / "tensor_20gb",
        logical_bytes=EXTRA_LARGE_LOGICAL_BYTES,
        fixture="extra_large_zarr",
        suite_total_bytes=None,
        quiet=quiet,
        seed=SEED_LARGE + 12,
    )
    if not quiet:
        print(f"Done: {EXTRA_LARGE_ZARR}")


def generate_small(*, quiet: bool) -> None:
    jobs: list[tuple[str, Callable[..., None], tuple, dict]] = []

    for ndim in (3, 4, 5):
        jobs.append(
            ("h5 baseline", _write_h5_small, (SMALL_H5 / f"tensor_{ndim}d.h5", ndim), {})
        )
        jobs.append(
            ("nc baseline", _write_nc_small, (SMALL_NC / f"tensor_{ndim}d.nc", ndim), {})
        )
        jobs.append(
            (
                "zarr baseline",
                _write_zarr_small,
                (SMALL_ZARR / f"tensor_{ndim}d", ndim),
                {"grouped": False},
            )
        )

    jobs.append(("h5 groups", _write_h5_groups_small, (SMALL_H5 / "groups_3d.h5", 3), {}))
    jobs.append(("nc groups", _write_nc_groups_small, (SMALL_NC / "groups_3d.nc", 3), {}))
    jobs.append(
        (
            "zarr groups",
            _write_zarr_small,
            (SMALL_ZARR / "groups_3d", 3),
            {"grouped": True},
        )
    )
    jobs.append(("h5 cf", _write_h5_cf_small, (SMALL_H5 / "cf_3d.h5",), {}))
    jobs.append(("nc cf", _write_nc_cf_small, (SMALL_NC / "cf_3d.nc",), {}))

    if not quiet:
        print(f"Generating {len(jobs)} small fixtures under {ROOT / 'small'}")

    for label, writer, args, kwargs in _progress(
        iter(jobs), total=len(jobs), desc="small", quiet=quiet
    ):
        if not quiet:
            rel = args[0].relative_to(ROOT)
            tqdm.write(f"  {label} -> {rel}")
        writer(*args, **kwargs)

    if not quiet:
        print(f"Done: {ROOT / 'small'}")


def generate_large(*, quiet: bool) -> None:
    suite_gib = LARGE_SUITE_LOGICAL_BYTES / (1024**3)
    per_gib = LARGE_PER_FORMAT_BYTES / (1024**3)
    if not quiet:
        print(
            f"Generating large suite under {ROOT / 'large'} "
            f"({suite_gib:.0f} GiB total ≈ {per_gib:.2f} GiB per format)"
        )
    _write_h5_large(LARGE_H5 / "tensor_large.h5", quiet=quiet)
    _write_nc_large(LARGE_NC / "tensor_large.nc", quiet=quiet)
    _write_zarr_large(LARGE_ZARR / "tensor_large", quiet=quiet)
    if not quiet:
        print(f"Done: {ROOT / 'large'}")


def generate_large_h5(*, quiet: bool) -> None:
    if not quiet:
        gib = LARGE_PER_FORMAT_BYTES / (1024**3)
        print(f"Generating large HDF5 under {LARGE_H5} ({gib:.2f} GiB logical f32)")
    _write_h5_large(LARGE_H5 / "tensor_large.h5", quiet=quiet)
    if not quiet:
        print(f"Done: {LARGE_H5}")


def generate_large_netcdf(*, quiet: bool) -> None:
    if not quiet:
        gib = LARGE_PER_FORMAT_BYTES / (1024**3)
        print(f"Generating large NetCDF under {LARGE_NC} ({gib:.2f} GiB logical f32)")
    _write_nc_large(LARGE_NC / "tensor_large.nc", quiet=quiet)
    if not quiet:
        print(f"Done: {LARGE_NC}")


def generate_large_zarr(*, quiet: bool) -> None:
    if not quiet:
        gib = LARGE_PER_FORMAT_BYTES / (1024**3)
        print(f"Generating large Zarr under {LARGE_ZARR} ({gib:.2f} GiB logical f32)")
    _write_zarr_large(LARGE_ZARR / "tensor_large", quiet=quiet)
    if not quiet:
        print(f"Done: {LARGE_ZARR}")


ExtraLargeTarget = Literal["extra-large-h5", "extra-large-netcdf", "extra-large-zarr"]
LargeSingleTarget = Literal["large-h5", "large-netcdf", "large-zarr"]

EXTRA_LARGE_TARGETS: dict[ExtraLargeTarget, Callable[..., None]] = {
    "extra-large-h5": generate_extra_large_h5,
    "extra-large-netcdf": generate_extra_large_netcdf,
    "extra-large-zarr": generate_extra_large_zarr,
}

LARGE_SINGLE_TARGETS: dict[LargeSingleTarget, Callable[..., None]] = {
    "large-h5": generate_large_h5,
    "large-netcdf": generate_large_netcdf,
    "large-zarr": generate_large_zarr,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "target",
        nargs="?",
        choices=(
            "small",
            "large",
            "large-h5",
            "large-netcdf",
            "large-zarr",
            "all",
            "extra-large-h5",
            "extra-large-netcdf",
            "extra-large-zarr",
        ),
        default="small",
        help=(
            "small: tracked baseline + groups/cf/zarr; "
            "large: ~20 GiB total across h5/nc/zarr (untracked); "
            "large-*: one ~6.67 GiB file for that format (untracked); "
            "extra-large-*: one 20 GiB file for that format (untracked); "
            "all: small + large suite only"
        ),
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
    if args.target in LARGE_SINGLE_TARGETS:
        LARGE_SINGLE_TARGETS[args.target](quiet=args.quiet)
    if args.target in EXTRA_LARGE_TARGETS:
        EXTRA_LARGE_TARGETS[args.target](quiet=args.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())

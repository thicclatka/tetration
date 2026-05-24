"""Tracked small HDF5 / NetCDF / Zarr fixtures."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

import h5py
import netCDF4 as nc
import numpy as np
import zarr
from tqdm import tqdm

from generate.constants import (
    FIXTURES_ROOT,
    NUMERIC_DTYPES,
    SMALL_H5,
    SMALL_NC,
    SMALL_SHAPES,
    SMALL_ZARR,
    DtypeName,
)
from generate.util import (
    chunk_shape_for_small,
    netcdf_dtype,
    numpy_dtype,
    progress,
    small_array,
)


def _set_tetration_small_attrs(obj, *, fixture: str, ndim: int) -> None:
    obj.attrs["tetration_fixture"] = fixture
    obj.attrs["tetration_ndim"] = ndim
    obj.attrs["tetration_dtypes"] = ",".join(NUMERIC_DTYPES)


def write_h5_small(path: Path, ndim: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        _set_tetration_small_attrs(f, fixture=f"small_{ndim}d", ndim=ndim)
        for dtype in NUMERIC_DTYPES:
            data = small_array(ndim, dtype)
            dset = f.create_dataset(
                dtype,
                data=data,
                compression="gzip",
                compression_opts=4,
            )
            dset.attrs["tetration_dtype"] = dtype


def write_nc_small(path: Path, ndim: int) -> None:
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
            data = small_array(ndim, dtype)
            var = ds.createVariable(
                dtype,
                netcdf_dtype(dtype),
                dim_names,
                zlib=True,
                complevel=4,
            )
            var.setncattr("tetration_dtype", dtype)
            var[:] = data


def write_h5_groups_small(path: Path, ndim: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        _set_tetration_small_attrs(f, fixture=f"groups_{ndim}d", ndim=ndim)
        primary = f.create_group("primary")
        for dtype in NUMERIC_DTYPES:
            data = small_array(ndim, dtype)
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


def write_nc_groups_small(path: Path, ndim: int) -> None:
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
            data = small_array(ndim, dtype)
            var = primary.createVariable(
                dtype,
                netcdf_dtype(dtype),
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


def write_h5_cf_small(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ndim = 3
    shape = SMALL_SHAPES[ndim]
    with h5py.File(path, "w") as f:
        _set_tetration_small_attrs(f, fixture="cf_3d", ndim=ndim)
        coords = f.create_group("coordinates")
        coords.create_dataset("time", data=np.arange(shape[0], dtype=np.float64))
        coords.create_dataset("lat", data=np.linspace(-90.0, 90.0, shape[1], dtype=np.float32))
        coords.create_dataset("lon", data=np.linspace(-180.0, 180.0, shape[2], dtype=np.float32))
        coords["time"].attrs["units"] = "days since 2020-01-01"
        coords["lat"].attrs["units"] = "degrees_north"
        coords["lon"].attrs["units"] = "degrees_east"

        physical = small_array(ndim, "f32")
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

        for dtype in NUMERIC_DTYPES:
            data = small_array(ndim, dtype)
            dset = f.create_dataset(dtype, data=data, compression="gzip", compression_opts=4)
            dset.attrs["tetration_dtype"] = dtype


def write_nc_cf_small(path: Path) -> None:
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

        physical = small_array(ndim, "f32")
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
            data = small_array(ndim, dtype)
            var = ds.createVariable(
                dtype,
                netcdf_dtype(dtype),
                dim_names,
                zlib=True,
                complevel=4,
            )
            var.setncattr("tetration_dtype", dtype)
            var[:] = data


def write_zarr_small(path: Path, ndim: int, *, grouped: bool = False) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    shape = SMALL_SHAPES[ndim]
    chunks = chunk_shape_for_small(shape)
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
        data = small_array(ndim, dtype)
        arr = target.create_array(
            dtype,
            shape=shape,
            chunks=chunks,
            dtype=numpy_dtype(dtype),
        )
        arr[:] = data
        arr.attrs["tetration_dtype"] = dtype


def generate_small(*, quiet: bool) -> None:
    jobs: list[tuple[str, Callable[..., None], tuple, dict]] = []

    for ndim in (3, 4, 5):
        jobs.append(
            ("h5 baseline", write_h5_small, (SMALL_H5 / f"tensor_{ndim}d.h5", ndim), {})
        )
        jobs.append(
            ("nc baseline", write_nc_small, (SMALL_NC / f"tensor_{ndim}d.nc", ndim), {})
        )
        jobs.append(
            (
                "zarr baseline",
                write_zarr_small,
                (SMALL_ZARR / f"tensor_{ndim}d", ndim),
                {"grouped": False},
            )
        )

    jobs.append(("h5 groups", write_h5_groups_small, (SMALL_H5 / "groups_3d.h5", 3), {}))
    jobs.append(("nc groups", write_nc_groups_small, (SMALL_NC / "groups_3d.nc", 3), {}))
    jobs.append(
        (
            "zarr groups",
            write_zarr_small,
            (SMALL_ZARR / "groups_3d", 3),
            {"grouped": True},
        )
    )
    jobs.append(("h5 cf", write_h5_cf_small, (SMALL_H5 / "cf_3d.h5",), {}))
    jobs.append(("nc cf", write_nc_cf_small, (SMALL_NC / "cf_3d.nc",), {}))

    if not quiet:
        print(f"Generating {len(jobs)} small fixtures under {FIXTURES_ROOT / 'small'}")

    for label, writer, args, kwargs in progress(
        iter(jobs), total=len(jobs), desc="small", quiet=quiet
    ):
        if not quiet:
            rel = args[0].relative_to(FIXTURES_ROOT)
            tqdm.write(f"  {label} -> {rel}")
        writer(*args, **kwargs)

    if not quiet:
        print(f"Done: {FIXTURES_ROOT / 'small'}")

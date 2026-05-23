"""Generate target dispatch."""

from __future__ import annotations

from collections.abc import Callable

from generate.constants import ExtraLargeTarget, GenerateTarget, LargeSingleTarget
from generate.large import (
    generate_extra_large_h5,
    generate_extra_large_netcdf,
    generate_extra_large_zarr,
    generate_large,
    generate_large_h5,
    generate_large_netcdf,
    generate_large_zarr,
)
from generate.small import generate_small

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

GENERATE_TARGETS: tuple[GenerateTarget, ...] = (
    "small",
    "large",
    "large-h5",
    "large-netcdf",
    "large-zarr",
    "all",
    "extra-large-h5",
    "extra-large-netcdf",
    "extra-large-zarr",
)


def run_target(target: GenerateTarget, *, quiet: bool) -> None:
    if target in ("small", "all"):
        generate_small(quiet=quiet)
    if target in ("large", "all"):
        generate_large(quiet=quiet)
    if target in LARGE_SINGLE_TARGETS:
        LARGE_SINGLE_TARGETS[target](quiet=quiet)  # type: ignore[index]
    if target in EXTRA_LARGE_TARGETS:
        EXTRA_LARGE_TARGETS[target](quiet=quiet)  # type: ignore[index]

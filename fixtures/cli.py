#!/usr/bin/env python3
"""Unified CLI for fixture generation and large-format benchmarks."""

from __future__ import annotations

import argparse
import os
import sys

from benchmark.constants import DEFAULT_FORMATS, FormatName
from benchmark.dispatch import run_benchmark
from generate.dispatch import GENERATE_TARGETS, run_target


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="tetration fixture generation and large-format benchmarks",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser(
        "generate",
        help="generate HDF5 / NetCDF / Zarr fixtures",
        description="Generate small (tracked) and large (gitignored) fixture tensors.",
    )
    gen.add_argument(
        "target",
        nargs="?",
        choices=GENERATE_TARGETS,
        default="small",
        help=(
            "small: tracked baseline + groups/cf/zarr; "
            "large: ~20 GiB total across h5/nc/zarr (untracked); "
            "large-*: one ~6.67 GiB file for that format (untracked); "
            "extra-large-*: one 20 GiB file for that format (untracked); "
            "all: small + large suite only"
        ),
    )
    gen.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="no tqdm bars or status lines",
    )

    bench = sub.add_parser(
        "bench",
        help="benchmark native source ops vs `.tet` query",
        description=(
            "For each tier/format: generate once, convert once, then time tier-A/B "
            "scalar ops on source (when comparable) and `.tet`."
        ),
    )
    bench.add_argument(
        "formats",
        nargs="*",
        choices=DEFAULT_FORMATS,
        metavar="format",
        help="formats to bench (default: h5 netcdf zarr)",
    )
    bench.add_argument(
        "--ops",
        nargs="*",
        metavar="OP",
        help="comma-separated ops (default: mean,sum,min,max,count,std,var)",
    )
    bench.add_argument(
        "--skip-ops",
        action="store_true",
        help="time convert only (skip source and .tet ops)",
    )
    bench.add_argument(
        "--skip-mean",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    bench.add_argument(
        "--skip-query",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    bench.add_argument(
        "--jobs",
        type=int,
        default=int(os.environ.get("BENCH_JOBS", "0")),
        help="tet convert --jobs (0 = auto: host parallelism, max 64)",
    )
    bench.add_argument(
        "--run-id",
        metavar="ID",
        help="archive under bench_results/runs/ID/ (default: <git>_<UTC timestamp>)",
    )
    bench.add_argument(
        "--no-clobber",
        action="store_true",
        help="fail if --run-id directory already exists",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "generate":
        run_target(args.target, quiet=args.quiet)
        return 0
    if args.command == "bench":
        skip_ops = args.skip_ops or args.skip_mean or args.skip_query
        formats: list[FormatName] = list(args.formats or DEFAULT_FORMATS)
        return run_benchmark(
            formats=formats,
            ops=args.ops,
            skip_ops=skip_ops,
            jobs=args.jobs,
            run_id=args.run_id,
            no_clobber=args.no_clobber,
        )
    return 1


def main_generate(argv: list[str] | None = None) -> int:
    """Backward-compatible entry for `generate-fixtures`."""
    return main(["generate", *(argv if argv is not None else sys.argv[1:])])


def main_bench(argv: list[str] | None = None) -> int:
    """Backward-compatible entry for `bench-large`."""
    return main(["bench", *(argv if argv is not None else sys.argv[1:])])


if __name__ == "__main__":
    sys.exit(main())

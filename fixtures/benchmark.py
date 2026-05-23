#!/usr/bin/env python3
"""Large / extra_large benchmarks: native mean vs `.tet` mean (+ convert).

Compares full-tensor **mean** three ways on the same synthetic `data` slab:

1. **Source** — chunked read from HDF5 / NetCDF / Zarr (no full-array load).
2. **`.tet` query** — `tet query --execute` streaming fold (parallel when many chunks).
3. **Convert** — one-time import source → `.tet` (not part of the mean race).

Runs one format and tier at a time; deletes source right after convert and wipes
each format directory before moving to the next format.
Results: `fixtures/bench_results/latest.md` (gitignored).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

from generate import (
    CHUNK_ELEMS,
    EXTRA_LARGE_LOGICAL_BYTES,
    EXTRA_LARGE_TARGETS,
    LARGE_PER_FORMAT_BYTES,
    LARGE_SINGLE_TARGETS,
)

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent
RESULTS_DIR = FIXTURES / "bench_results"
RESULTS_FILE = RESULTS_DIR / "latest.md"

QUERY_JSON = json.dumps(
    {
        "dataset": "data",
        "layout_version": 1,
        "operation": {"mean": {"axes": []}},
    }
)

FormatName = str  # h5 | netcdf | zarr
TierName = str  # large | extra

MEAN_TOLERANCE = 1e-4
STEPS_PER_CASE = 6


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[bench {ts}] {msg}", flush=True)


def log_step(case: BenchCase, step: int, msg: str) -> None:
    log(f"{case.format}/{case.tier} step {step}/{STEPS_PER_CASE}: {msg}")


def format_elapsed(seconds: float) -> str:
    return f"{seconds:.3f}s"


def generate_for_bench(target: str, case: BenchCase) -> float:
    log_step(
        case,
        1,
        f"generating ~{case.logical_gib} GiB source ({target}) — tqdm below",
    )
    t0 = time.perf_counter()
    if target in LARGE_SINGLE_TARGETS:
        LARGE_SINGLE_TARGETS[target](quiet=False)  # type: ignore[arg-type]
    elif target in EXTRA_LARGE_TARGETS:
        EXTRA_LARGE_TARGETS[target](quiet=False)  # type: ignore[arg-type]
    else:
        raise ValueError(f"unknown generate target: {target}")
    elapsed = time.perf_counter() - t0
    log_step(case, 1, f"generate done in {format_elapsed(elapsed)} → {case.src}")
    return elapsed


@dataclass(frozen=True)
class BenchCase:
    format: FormatName
    tier: TierName
    src: Path
    tet_out: Path
    gen_target: str
    logical_bytes: int

    @property
    def logical_gib(self) -> str:
        return f"{self.logical_bytes / (1024**3):.2f}"


def cases_for(format: FormatName) -> tuple[BenchCase, BenchCase]:
    large = BenchCase(
        format=format,
        tier="large",
        src=FIXTURES / "large" / format / _large_basename(format),
        tet_out=FIXTURES / "large" / format / "tensor_large.tet",
        gen_target=f"large-{format}",
        logical_bytes=LARGE_PER_FORMAT_BYTES,
    )
    extra = BenchCase(
        format=format,
        tier="extra",
        src=FIXTURES / "extra_large" / format / _extra_basename(format),
        tet_out=FIXTURES / "extra_large" / format / "tensor_20gb.tet",
        gen_target=f"extra-large-{format}",
        logical_bytes=EXTRA_LARGE_LOGICAL_BYTES,
    )
    return large, extra


def _large_basename(format: FormatName) -> str:
    if format == "zarr":
        return "tensor_large"
    return f"tensor_large.{format if format != 'netcdf' else 'nc'}"


def _extra_basename(format: FormatName) -> str:
    if format == "zarr":
        return "tensor_20gb"
    return f"tensor_20gb.{format if format != 'netcdf' else 'nc'}"


def tet_bin() -> Path:
    path = Path(os.environ.get("TET_BIN", ROOT / "target/release/tet"))
    if not path.is_file() or not os.access(path, os.X_OK):
        print(f"missing release binary: {path} (run: cargo build --release)", file=sys.stderr)
        sys.exit(1)
    return path


def git_short_rev() -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def generate_quiet(target: str) -> None:
    """Legacy helper; benchmark uses [`generate_for_bench`] with progress bars."""
    if target in LARGE_SINGLE_TARGETS:
        LARGE_SINGLE_TARGETS[target](quiet=True)  # type: ignore[arg-type]
    elif target in EXTRA_LARGE_TARGETS:
        EXTRA_LARGE_TARGETS[target](quiet=True)  # type: ignore[arg-type]
    else:
        raise ValueError(f"unknown generate target: {target}")


def time_warm_mean_logged(label: str, compute: Callable[[], float]) -> tuple[float, float]:
    """First pass warms page cache; second pass is timed."""
    log(f"{label}: warm pass (untimed, fills page cache) …")
    compute()
    log(f"{label}: timed pass …")
    t0 = time.perf_counter()
    mean = compute()
    elapsed = time.perf_counter() - t0
    log(f"{label}: done in {format_elapsed(elapsed)}  mean={mean}")
    return elapsed, mean


def iter_slabs(length: int) -> list[tuple[int, int]]:
    return [
        (start, min(start + CHUNK_ELEMS, length))
        for start in range(0, length, CHUNK_ELEMS)
    ]


def mean_from_slabs(
    length: int,
    read_slab: Callable[[int, int], tuple[float, int]],
) -> float:
    total = 0.0
    count = 0
    for start, end in iter_slabs(length):
        slab_sum, slab_count = read_slab(start, end)
        total += slab_sum
        count += slab_count
    return total / count


def mean_source_h5(path: Path) -> tuple[float, float]:
    import h5py

    with h5py.File(path, "r") as f:
        d = f["data"]
        length = int(d.shape[0])

        def compute() -> float:
            def read_slab(start: int, end: int) -> tuple[float, int]:
                slab = d[start:end]
                return float(slab.sum()), int(slab.size)

            return mean_from_slabs(length, read_slab)

        return time_warm_mean_logged(
            f"source mean ({CHUNK_ELEMS // (1024 * 1024)} MiB slabs, h5py)",
            compute,
        )


def mean_source_netcdf(path: Path) -> tuple[float, float]:
    import netCDF4 as nc

    with nc.Dataset(path, "r") as ds:
        var = ds.variables["data"]
        length = int(var.shape[0])

        def compute() -> float:
            def read_slab(start: int, end: int) -> tuple[float, int]:
                slab = var[start:end]
                return float(slab.sum()), int(slab.size)

            return mean_from_slabs(length, read_slab)

        return time_warm_mean_logged(
            f"source mean ({CHUNK_ELEMS // (1024 * 1024)} MiB slabs, netCDF4)",
            compute,
        )


def mean_source_zarr(path: Path) -> tuple[float, float]:
    import zarr

    root = zarr.open_group(str(path), mode="r")
    arr = root["data"]
    length = int(arr.shape[0])

    def compute() -> float:
        def read_slab(start: int, end: int) -> tuple[float, int]:
            slab = arr[start:end]
            return float(slab.sum()), int(end - start)

        return mean_from_slabs(length, read_slab)

    return time_warm_mean_logged(
        f"source mean ({CHUNK_ELEMS // (1024 * 1024)} MiB slabs, zarr)",
        compute,
    )


def mean_source(case: BenchCase) -> tuple[float, float]:
    match case.format:
        case "h5":
            return mean_source_h5(case.src)
        case "netcdf":
            return mean_source_netcdf(case.src)
        case "zarr":
            return mean_source_zarr(case.src)
        case _:
            raise ValueError(f"unknown format: {case.format}")


def run_convert(tet: Path, src: Path, out: Path, jobs: int) -> float:
    env = {**os.environ, "TET_NO_QUERY_HISTORY": "1"}
    t0 = time.perf_counter()
    subprocess.run(
        [str(tet), "convert", str(src), str(out), "--jobs", str(jobs)],
        check=True,
        env=env,
    )
    return time.perf_counter() - t0


def run_tet_mean(tet: Path, tet_path: Path, case: BenchCase) -> tuple[float, float | None, int | None]:
    """Warm-cache second run; returns (seconds, mean, chunk_count)."""
    env = {**os.environ, "TET_NO_QUERY_HISTORY": "1"}
    cmd = [str(tet), "query", "--tet", str(tet_path), "--execute", "--preview-f32", "0"]

    def once() -> dict:
        proc = subprocess.run(
            cmd,
            input=QUERY_JSON,
            text=True,
            capture_output=True,
            env=env,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr or proc.stdout or "tet query failed")
        return json.loads(proc.stdout)

    log(
        f".tet mean: warm pass (untimed) — `tet query` streaming fold over ~{case.logical_gib} GiB …"
    )
    try:
        once()
    except RuntimeError:
        pass

    log(".tet mean: timed pass …")
    t0 = time.perf_counter()
    payload = once()
    elapsed = time.perf_counter() - t0

    execution = payload.get("execution") or {}
    read_plan = payload.get("read_plan") or {}
    mean = execution.get("operation_mean")
    chunks = read_plan.get("chunk_count")
    strategy = execution.get("memory_strategy")
    log(
        f".tet mean: done in {format_elapsed(elapsed)}  mean={mean}  "
        f"chunks={chunks}  strategy={strategy}"
    )
    return elapsed, mean, chunks


def delete_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def delete_path_logged(path: Path, label: str) -> None:
    if not path.exists():
        log(f"{label}: nothing to delete ({path})")
        return
    log(f"{label}: deleting {path} …")
    t0 = time.perf_counter()
    delete_path(path)
    log(f"{label}: deleted in {format_elapsed(time.perf_counter() - t0)}")


def cleanup_format_tree(format: FormatName) -> None:
    """Remove any leftover artifacts for one format before the next format run."""
    for tier_root in (FIXTURES / "large", FIXTURES / "extra_large"):
        fmt_dir = tier_root / format
        if fmt_dir.exists():
            shutil.rmtree(fmt_dir)
        try:
            tier_root.rmdir()
        except OSError:
            pass


@dataclass
class BenchRow:
    case: BenchCase
    source_mean_s: float | None
    source_mean: float | None
    convert_s: float
    tet_mean_s: float | None
    tet_mean: float | None
    chunks: int | None


def bench_case(
    tet: Path,
    case: BenchCase,
    *,
    jobs: int,
    skip_mean: bool,
) -> BenchRow:
    log(f"=== {case.format} / {case.tier} (~{case.logical_gib} GiB logical f32) ===")
    generate_for_bench(case.gen_target, case)

    source_mean_s: float | None = None
    source_mean: float | None = None
    tet_mean_s: float | None = None
    tet_mean: float | None = None
    chunks: int | None = None

    if not skip_mean:
        log_step(case, 2, "source mean (Python chunked read, warm 2nd pass)")
        source_mean_s, source_mean = mean_source(case)
    else:
        log_step(case, 2, "skipped (--skip-mean)")

    log_step(
        case,
        3,
        f"convert {case.src.name} → {case.tet_out.name} (tet convert --jobs {jobs})",
    )
    convert_s = run_convert(tet, case.src, case.tet_out, jobs)
    log_step(case, 3, f"convert done in {format_elapsed(convert_s)}")

    delete_path_logged(case.src, f"{case.format}/{case.tier} step 4/6")

    if not skip_mean:
        log_step(
            case,
            5,
            ".tet mean (tet query --execute, warm 2nd pass) — this can take tens of seconds",
        )
        tet_mean_s, tet_mean, chunks = run_tet_mean(tet, case.tet_out, case)
        if source_mean is not None and tet_mean is not None:
            delta = abs(source_mean - tet_mean)
            if delta > MEAN_TOLERANCE:
                log(
                    f"warning: mean mismatch source vs .tet (|Δ|={delta:.6g})",
                )
    else:
        log_step(case, 5, "skipped (--skip-mean)")

    delete_path_logged(case.tet_out, f"{case.format}/{case.tier} step 6/6")
    return BenchRow(
        case,
        source_mean_s,
        source_mean,
        convert_s,
        tet_mean_s,
        tet_mean,
        chunks,
    )


def write_report(
    *,
    tet: Path,
    jobs: int,
    skip_mean: bool,
    rows: list[BenchRow],
) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    slab_mib = CHUNK_ELEMS // (1024 * 1024)
    lines = [
        "# Large / extra_large benchmark",
        "",
        f"- **Date (UTC):** {datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        f"- **Host:** {socket.gethostname()}",
        f"- **Git:** {git_short_rev()}",
        f"- **tet:** {tet}",
        f"- **convert jobs:** {jobs}",
        f"- **Source mean:** Python chunked slabs ({slab_mib} MiB), warm 2nd pass, dataset `data`",
        "- **`.tet` mean:** `tet query --execute` streaming fold, warm 2nd pass",
        "- **Convert:** source → `.tet` (one-time; not comparable to mean wall time)",
        f"- **Means skipped:** {skip_mean}",
        "",
        "| format | tier | logical GiB | source mean (s) | .tet mean (s) | convert (s) | mean | chunks |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        sm = f"{row.source_mean_s:.3f}" if row.source_mean_s is not None else "-"
        tm = f"{row.tet_mean_s:.3f}" if row.tet_mean_s is not None else "-"
        m = f"{row.tet_mean:.6f}" if row.tet_mean is not None else "-"
        c = str(row.chunks) if row.chunks is not None else "-"
        lines.append(
            f"| {row.case.format} | {row.case.tier} | {row.case.logical_gib} "
            f"| {sm} | {tm} | {row.convert_s:.3f} | {m} | {c} |"
        )
    text = "\n".join(lines) + "\n"
    RESULTS_FILE.write_text(text, encoding="utf-8")
    print(f"\nWrote {RESULTS_FILE}")
    print(text)


DEFAULT_FORMATS = ("h5", "netcdf", "zarr")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "formats",
        nargs="*",
        choices=DEFAULT_FORMATS,
        metavar="format",
        help="formats to bench (default: h5 netcdf zarr)",
    )
    parser.add_argument(
        "--skip-mean",
        action="store_true",
        help="time convert only (skip source and .tet mean)",
    )
    parser.add_argument(
        "--skip-query",
        action="store_true",
        help=argparse.SUPPRESS,  # alias for --skip-mean
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=int(os.environ.get("BENCH_JOBS", "0")),
        help="tet convert --jobs (default: 0 = auto)",
    )
    args = parser.parse_args(argv)
    skip_mean = args.skip_mean or args.skip_query
    formats = args.formats or list(DEFAULT_FORMATS)

    tet = tet_bin()
    log(f"benchmark start — formats={formats}  tet={tet}  jobs={args.jobs}  skip_mean={skip_mean}")
    rows: list[BenchRow] = []
    for fmt in formats:
        log(f"######## format: {fmt} ########")
        for case in cases_for(fmt):
            rows.append(bench_case(tet, case, jobs=args.jobs, skip_mean=skip_mean))
        log(f"cleaning format tree: large/{fmt}, extra_large/{fmt}")
        cleanup_format_tree(fmt)
        log(f"format {fmt} complete")

    write_report(tet=tet, jobs=args.jobs, skip_mean=skip_mean, rows=rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())

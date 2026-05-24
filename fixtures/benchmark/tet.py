"""`.tet` convert and query helpers."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from benchmark.case import BenchCase
from benchmark.constants import OpName, REPO_ROOT, TET_EXECUTION_KEY
from benchmark.log import format_elapsed, log


def tet_bin() -> Path:
    path = Path(os.environ.get("TET_BIN", REPO_ROOT / "target/release/tet"))
    if not path.is_file() or not os.access(path, os.X_OK):
        print(f"missing release binary: {path} (run: cargo build --release)", file=sys.stderr)
        sys.exit(1)
    return path


def git_short_rev() -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def query_json(op: OpName) -> str:
    return json.dumps(
        {
            "dataset": "data",
            "layout_version": 1,
            op: [],
        }
    )


def run_tet_op(
    tet: Path,
    tet_path: Path,
    case: BenchCase,
    op: OpName,
) -> tuple[float, float | None, int | None]:
    env = {**os.environ, "TET_NO_QUERY_HISTORY": "1"}
    cmd = [str(tet), "query", "--tet", str(tet_path), "--execute", "--preview-f32", "0"]
    body = query_json(op)

    def once() -> dict:
        proc = subprocess.run(
            cmd,
            input=body,
            text=True,
            capture_output=True,
            env=env,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr or proc.stdout or "tet query failed")
        return json.loads(proc.stdout)

    log(f".tet {op}: warm pass over ~{case.logical_gib} GiB …")
    try:
        once()
    except RuntimeError:
        pass

    log(f".tet {op}: timed pass …")
    t0 = time.perf_counter()
    payload = once()
    elapsed = time.perf_counter() - t0

    execution = payload.get("execution") or {}
    read_plan = payload.get("read_plan") or {}
    key = TET_EXECUTION_KEY[op]
    raw = execution.get(key)
    value = float(raw) if raw is not None else None
    chunks = read_plan.get("chunk_count")
    log(
        f".tet {op}: done in {format_elapsed(elapsed)}  value={value}  "
        f"strategy={execution.get('memory_strategy')}"
    )
    return elapsed, value, chunks


def run_convert(tet: Path, src: Path, out: Path, jobs: int) -> float:
    env = {**os.environ, "TET_NO_QUERY_HISTORY": "1"}
    t0 = time.perf_counter()
    subprocess.run(
        [str(tet), "convert", str(src), str(out), "--jobs", str(jobs)],
        check=True,
        env=env,
    )
    return time.perf_counter() - t0

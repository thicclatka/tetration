"""Benchmark entry and format loop."""

from __future__ import annotations

import os

from benchmark.case import cases_for
from benchmark.constants import DEFAULT_FORMATS, FormatName
from benchmark.log import log
from benchmark.report import parse_ops, write_report
from benchmark.runner import bench_case, cleanup_format_tree
from benchmark.tet import tet_bin


def run_benchmark(
    *,
    formats: list[FormatName] | None = None,
    ops: list[str] | None = None,
    skip_ops: bool = False,
    jobs: int | None = None,
    run_id: str | None = None,
    no_clobber: bool = False,
) -> int:
    fmt_list: list[FormatName] = list(formats or DEFAULT_FORMATS)
    op_list = parse_ops(ops)
    job_count = jobs if jobs is not None else int(os.environ.get("BENCH_JOBS", "0"))

    tet = tet_bin()
    log(f"start formats={fmt_list} ops={op_list} tet={tet} jobs={job_count} skip_ops={skip_ops}")
    rows = []
    for fmt in fmt_list:
        log(f"######## {fmt} ########")
        for case in cases_for(fmt):
            rows.append(bench_case(tet, case, jobs=job_count, ops=op_list, skip_ops=skip_ops))
        cleanup_format_tree(fmt)

    write_report(
        jobs=job_count,
        ops=op_list,
        skip_ops=skip_ops,
        rows=rows,
        run_id=run_id,
        no_clobber=no_clobber,
    )
    return 0

"""Markdown report rendering."""

from __future__ import annotations

import socket
from datetime import UTC, datetime
from pathlib import Path

from benchmark.constants import (
    DEFAULT_OPS,
    OpName,
    RESULTS_DIR,
    RESULTS_FILE,
)
from benchmark.runner import BenchRow
from benchmark.tet import git_short_rev
from generate.constants import CHUNK_ELEMS


def _cell(seconds: float | None) -> str:
    return f"{seconds:.3f}" if seconds is not None else "-"


def _val(value: float | None) -> str:
    if value is None:
        return "-"
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.6g}"


def format_md_table(
    headers: list[str],
    rows: list[list[str]],
    *,
    aligns: list[str] | None = None,
) -> list[str]:
    if not headers:
        return []
    aligns = aligns or ["left"] * len(headers)
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def pad(cell: str, width: int, align: str) -> str:
        if align == "right":
            return cell.rjust(width)
        if align == "center":
            return cell.center(width)
        return cell.ljust(width)

    def row_line(cells: list[str]) -> str:
        parts = [pad(c, widths[i], aligns[i]) for i, c in enumerate(cells)]
        return "| " + " | ".join(parts) + " |"

    sep_cells: list[str] = []
    for i, align in enumerate(aligns):
        w = widths[i]
        if align == "right":
            sep_cells.append("-" * max(3, w) + ":")
        elif align == "center":
            sep_cells.append(":" + "-" * max(1, w - 2) + ":")
        else:
            sep_cells.append(":" + "-" * max(2, w - 1))

    return [
        row_line(headers),
        "| " + " | ".join(sep_cells) + " |",
        *(row_line(r) for r in rows),
    ]


def write_report(
    *,
    tet: Path,
    jobs: int,
    ops: tuple[OpName, ...],
    skip_ops: bool,
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
        f"- **Ops:** {', '.join(ops)} (tier-A/B streaming; population var/std ddof=0)",
        f"- **Source:** Python {slab_mib} MiB slabs, warm 2nd pass (skipped when not comparable)",
        "- **`.tet`:** `tet query --execute` streaming fold, warm 2nd pass",
        f"- **Ops skipped:** {skip_ops}",
        "",
        "## Notes",
        "",
        "- **zarr** source times use Python directory-store reads; compare `.tet` vs h5/netcdf, not zarr source as a ceiling.",
        "- Rows with `n/a` source skipped ops that do not line up for that format.",
        "- `⚠` = source vs `.tet` value mismatch beyond tolerance.",
        "",
        "## convert",
        "",
    ]

    convert_rows: list[list[str]] = []
    for row in rows:
        chunks = str(row.chunks) if row.chunks is not None else "-"
        convert_rows.append(
            [
                row.case.format,
                row.case.tier,
                row.case.logical_gib,
                f"{row.convert_s:.3f}",
                chunks,
            ]
        )
    lines.extend(
        format_md_table(
            ["format", "tier", "GiB", "convert (s)", "chunks"],
            convert_rows,
            aligns=["left", "left", "right", "right", "right"],
        )
    )
    lines.append("")

    op_headers = ["format", "tier", "GiB", "source (s)", ".tet (s)", "source", ".tet", "note"]
    op_aligns = ["left", "left", "right", "right", "right", "right", "right", "left"]

    for op in ops:
        lines.extend([f"## {op}", ""])
        op_rows: list[list[str]] = []
        for row in rows:
            r = row.ops.get(op)
            if r is None:
                continue
            note = r.note or ""
            if r.skip_source and r.source_s is None:
                src_s = "n/a"
                src_v = "n/a"
            else:
                src_s = _cell(r.source_s)
                src_v = _val(r.source_value)
            flag = " ⚠" if r.mismatch else ""
            tet_v = _val(r.tet_value) + flag
            op_rows.append(
                [
                    row.case.format,
                    row.case.tier,
                    row.case.logical_gib,
                    src_s,
                    _cell(r.tet_s),
                    src_v,
                    tet_v,
                    note,
                ]
            )
        lines.extend(format_md_table(op_headers, op_rows, aligns=op_aligns))
        lines.append("")

    text = "\n".join(lines) + "\n"
    RESULTS_FILE.write_text(text, encoding="utf-8")
    print(f"\nWrote {RESULTS_FILE}")
    print(text)


def parse_ops(raw: list[str] | None) -> tuple[OpName, ...]:
    if not raw:
        return DEFAULT_OPS
    out: list[OpName] = []
    for item in raw:
        for part in item.split(","):
            part = part.strip()
            if part in DEFAULT_OPS:
                out.append(part)  # type: ignore[arg-type]
            else:
                raise ValueError(f"unknown op {part!r}; choose from {DEFAULT_OPS}")
    return tuple(out)

"""Best-effort host hardware probe for benchmark reports."""

from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SystemInfo:
    cpu: str
    logical_cpus: int
    physical_cpus: int | None
    ram_total_gib: float | None
    ram_available_gib: float | None
    query_workers: str
    convert_jobs: str
    gpu: str | None
    gpu_vram_gib: float | None


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True)
        return out.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _sysctl(name: str) -> str | None:
    if sys.platform != "darwin":
        return None
    return _run(["sysctl", "-n", name])


def _gib_from_bytes(n: int) -> float:
    return n / (1024**3)


def _cpu_model() -> str:
    if sys.platform == "darwin":
        brand = _sysctl("machdep.cpu.brand_string")
        if brand:
            return brand
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine() or "unknown"


def _cpu_counts() -> tuple[int, int | None]:
    logical = os.cpu_count() or 1
    physical: int | None = None
    if sys.platform == "darwin":
        raw = _sysctl("hw.physicalcpu")
        if raw and raw.isdigit():
            physical = int(raw)
    return logical, physical


def _ram_darwin() -> tuple[float | None, float | None]:
    total_raw = _sysctl("hw.memsize")
    if not total_raw or not total_raw.isdigit():
        return None, None
    total = _gib_from_bytes(int(total_raw))
    page_size_raw = _sysctl("hw.pagesize")
    page_size = int(page_size_raw) if page_size_raw and page_size_raw.isdigit() else 4096
    vm = _run(["vm_stat"])
    if not vm:
        return total, None
    free_pages = 0
    inactive_pages = 0
    for line in vm.splitlines():
        m = re.match(r"^Pages free:\s+(\d+)\.", line)
        if m:
            free_pages = int(m.group(1))
        m = re.match(r"^Pages inactive:\s+(\d+)\.", line)
        if m:
            inactive_pages = int(m.group(1))
    avail = _gib_from_bytes((free_pages + inactive_pages) * page_size)
    return total, avail


def _ram_linux() -> tuple[float | None, float | None]:
    try:
        mem: dict[str, int] = {}
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, rest = line.split(":", 1)
            parts = rest.strip().split()
            if parts and parts[0].isdigit():
                mem[key] = int(parts[0]) * 1024
        total = _gib_from_bytes(mem["MemTotal"]) if "MemTotal" in mem else None
        avail = _gib_from_bytes(mem["MemAvailable"]) if "MemAvailable" in mem else None
        return total, avail
    except OSError:
        return None, None


def _ram() -> tuple[float | None, float | None]:
    if sys.platform == "darwin":
        return _ram_darwin()
    if sys.platform.startswith("linux"):
        return _ram_linux()
    return None, None


def _effective_worker_count(requested: int | None = None) -> int:
    if requested is not None and requested > 0:
        return min(max(requested, 1), 64)
    env = os.environ.get("RAYON_NUM_THREADS")
    if env and env.isdigit():
        return min(max(int(env), 1), 64)
    return min(max(os.cpu_count() or 1, 1), 64)


def _format_workers(_label: str, requested: int | None) -> str:
    effective = _effective_worker_count(requested)
    if requested is not None and requested > 0:
        return str(effective)
    if os.environ.get("RAYON_NUM_THREADS"):
        return f"auto ({effective}, RAYON_NUM_THREADS={os.environ['RAYON_NUM_THREADS']})"
    return f"auto ({effective})"


def _gpu_nvidia() -> tuple[str | None, float | None]:
    out = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    if not out:
        return None, None
    line = out.splitlines()[0]
    name, _, mem = line.partition(",")
    name = name.strip()
    mem = mem.strip()
    if not name:
        return None, None
    vram: float | None = None
    if mem:
        try:
            vram = float(mem) / 1024.0  # MiB → GiB
        except ValueError:
            vram = None
    return name, vram


def _gpu_apple() -> tuple[str | None, float | None]:
    chip = _sysctl("machdep.gpu.model") or _sysctl("machdep.cpu.brand_string")
    if chip and "Apple" in chip:
        return chip, None
    return None, None


def _gpu() -> tuple[str | None, float | None]:
    name, vram = _gpu_nvidia()
    if name:
        return name, vram
    if sys.platform == "darwin":
        return _gpu_apple()
    return None, None


def _fmt_gib(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f} GiB"


def probe_system(*, convert_jobs_requested: int) -> SystemInfo:
    logical, physical = _cpu_counts()
    ram_total, ram_avail = _ram()
    gpu, vram = _gpu()
    return SystemInfo(
        cpu=_cpu_model(),
        logical_cpus=logical,
        physical_cpus=physical,
        ram_total_gib=ram_total,
        ram_available_gib=ram_avail,
        query_workers=_format_workers("query", None),
        convert_jobs=_format_workers("convert", convert_jobs_requested),
        gpu=gpu,
        gpu_vram_gib=vram,
    )


def report_metadata_lines(*, convert_jobs_requested: int) -> list[str]:
    """Markdown bullet lines for the report header (no hostname or tet path)."""
    info = probe_system(convert_jobs_requested=convert_jobs_requested)
    cpu_line = info.cpu
    if info.physical_cpus is not None:
        cpu_line += f" ({info.physical_cpus}P / {info.logical_cpus}L)"
    else:
        cpu_line += f" ({info.logical_cpus} logical)"

    lines = [
        f"- **CPU:** {cpu_line}",
        f"- **RAM:** {_fmt_gib(info.ram_total_gib)} total, {_fmt_gib(info.ram_available_gib)} available",
        f"- **Query workers:** {info.query_workers}",
        f"- **Convert workers:** {info.convert_jobs}",
    ]
    if info.gpu:
        if info.gpu == info.cpu or info.gpu in info.cpu:
            lines.append("- **GPU:** integrated (unified memory)")
        else:
            vram = _fmt_gib(info.gpu_vram_gib) if info.gpu_vram_gib is not None else "n/a"
            lines.append(f"- **GPU:** {info.gpu} ({vram} VRAM)")
    else:
        lines.append("- **GPU:** n/a")
    return lines

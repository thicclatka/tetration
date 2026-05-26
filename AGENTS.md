# AGENTS.md — status and ops for contributors / automation

Compact handoff for humans and agents: what exists today, how to verify it, and what is still stubbed. **Roadmap checklist:** [`GETTING_STARTED.md`](GETTING_STARTED.md). **Query engine detail:** [`docs/query_engine.md`](docs/query_engine.md).

## Current status (May 2026)

| Area                                                                        | Status                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Phases 0–3 (spec, writer/reader, chunk addressing, zstd + index robustness) | **Done**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| Phase 4 (query execute)                                                     | **Done** — plan, mmap materialize, parallel multi-chunk decode, **adaptive streaming fold** (in-core parallel chunk fold; out-of-core **linear scan** over contiguous raw spans when logical size exceeds ~85% available RAM), memory-aware routing, mmap spill, streaming scalar + partial-axis folds, **`f32`** / **`f64`** / **`i32`** / **`i64`** execution, tier-C **`median`** / **`quantile`** / **`histogram`** (scalar + partial axes); SIMD bulk **`f32`** sum/sumsq + min/max in [`variance_simd.rs`](src/query/fold/variance_simd.rs) |
| Phase 5 (convert)                                                           | **Done** — HDF5 / NetCDF / Zarr v3 → `.tet` (extension or sniff), groups, CF decode, parallel import; raw or zstd Zarr chunks                                                                                                                                                                                                                                                                                                                                                                                                                     |
| Phase 6 (CLI & query UX)                                                    | **Done** — focused query stdout, `tet qhist` list/run/filter (`hist` alias), flat query JSON, `tet info` table/filters/`--json`; spawn-`tet` smoke in `src/tests/cli_info.rs`; **next:** optional TOML front-end                                                                                                                                                                                                                                                                                                                                  |
| Phase 7 (metadata & history)                                                | **Done** — footer `metadata` JSON (+ **`metadata_ref`** spill when inline JSON > 64 KiB); structured **`history`** objects (legacy triples still read); **`TetWriterSession`**, **`TetFile`**, **`execute_query_*`**, convert import                                                                                                                                                                                                                                                                                                              |
| Phase 8 (dtypes & file health)                                              | **Next** — `tet verify` / library health checks; additional wire dtypes (`u8`, …)                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| Phase 9 (query ops & interchange)                                           | **Later** — histogram edges, covariance/correlation, named axes, coord selection, export                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| Phase 10 (GPU)                                                              | **Later** — optional device materialize after CPU decode; VRAM guardrails                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| Phase 11 (bindings)                                                         | **Not started** — separate Python repo (PyPI rename), pins crates.io `tetration`; C ABI when needed                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| JSON security                                                               | **Done (v1)** — `QueryLimits`, `deny_unknown_fields`, caps in `document.rs`; proptest in `src/tests/query.rs`                                                                                                                                                                                                                                                                                                                                                                                                                                     |

**Branch:** `main` ([PR #1](https://github.com/thicclatka/tetration/pull/1) layout/query v1; [PR #2](https://github.com/thicclatka/tetration/pull/2) integer dtypes; [PR #7](https://github.com/thicclatka/tetration/pull/7) adaptive out-of-core linear scan + SIMD bulk folds).

## Project shape

- **Crate:** `tetration` (library) + binary **`tet`** (`default-run = "tet"` in `Cargo.toml`).
- **Rust:** `edition = "2024"`, `rust-version = "1.95"`. Toolchain pin: `.mise.toml` sets `rust = "1.95"`.

## Implemented (layout v1)

- **`utils`:** Crate-private helpers. **`utils::wire`** — LE primitives, `align8`, byte-span checks. **`utils::le_pod`** — macro-generated **`f32_le`** / **`f64_le`** / **`i32_le`** / **`i64_le`** typed LE reads and byte counts. **`utils::dtype`** — `ElementDtype` from wire tags. **`utils::host_memory`** — best-effort host RAM probe (Linux `MemAvailable`, macOS free+inactive pages).
- **`layout`:** 32-byte `TETR` superblock v1, mmap open (`mmap_file_read`), superblock parse, empty v1 file creation (`create_empty_v1_file`). See `docs/layout_v1.md`.
- **`catalog`:** Dataset directory, chunk index header/entries (including optional **execution settings** in TIDX bytes 16–31), validation (`validate_chunk_payloads`), `read_tet_summary_v1`. Writers in **`catalog/write.rs`**. Chunk geometry in **`catalog/tile.rs`** (`pub`): `chunk_coords_intersecting_global_box`, `chunk_coords_intersecting_strided`. **`ChunkPayloadCodecV1::encode_tile_payload` / `decode_tile_payload`** (raw + zstd).
- **Writers:**
  - `write_one_chunk_raw_file` — single chunk, raw **`f32`** / **`f64`** / **`i32`** / **`i64`**, `codec = 0`.
  - `write_raw_array_file` / `RawArrayWrite` — multi-chunk **`f32`** / **`f64`** / **`i32`** / **`i64`** grid; per-chunk **`chunk_codec`**; optional **`file_execution`** → TIDX header settings.
- **`query`:** Flat JSON wire in **`document_wire.rs`**; limits + **`validate_query`** in **`document.rs`** (`QueryLimits`). Wire types in **`query/types/`**. Submodules under **`src/query/`**:

  | Submodule          | Role                                                                                                                                                                                      |
  | ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
  | **`plan/`**        | `selection`, `read_plan` — global box + chunk read plan                                                                                                                                   |
  | **`decode/`**      | `chunk_decode`, `indexing` — mmap decode and row-major indexing                                                                                                                           |
  | **`materialize/`** | f32/f64/i32/i64 materialize, `parallel`, `int`, `stats` (tier-C)                                                                                                                          |
  | **`fold/`**        | `fold_policy` (I/O regime), `linear_scan` (out-of-core byte-stream fold), `reduction`, `variance_simd`, `parallel_fold`, `shared` (`FoldPlanOutcome`), `partial_fold`, `partial_geometry` |
  | **`dispatch.rs`**  | Dtype routing for materialize, spill, scalar/partial fold                                                                                                                                 |
  | **`engine/`**      | `run`, `operations`, `budget`, `spill_policy` — entrypoints + execution preview                                                                                                           |
  | **`cli/`**         | `history` (qhist formatters), `info`, `output/` — query history JSONL, `tet info` formatters, query stdout                                                                                |

  **Execution:** `planned_chunk_mmap_slices` (raw codec **0** only). Prefer new mmap logic under **`decode/`** or **`materialize/`**; orchestration under **`engine/`** + **`dispatch.rs`**. Materialize: sequential + parallel, `_into` caller buffers; capped preview allocates `min(cap, logical)` only. **`build_execution_preview`** picks **`MemoryStrategy`**: `streaming_fold` (tier-A/B ops), `in_memory_materialize` / `temp_spill_materialize` (tier-C ops: **`median`**, **`quantile`**, **`histogram`**), `mmap_spill` (export spill + single-pass preview), `capped_in_memory` (preview-only). Temp spills live under platform cache allowlist roots and are deleted when execution finishes. Budget from host RAM (default **25%**), query `execution.*`, or per-file TIDX settings. **Streaming fold I/O:** [`FoldIoPolicy`](src/query/fold/fold_policy.rs) — **in-core** → parallel chunk fold (Rayon when `chunk_count > 1`); **out-of-core** (logical bytes > ~85% available RAM) + full dense raw scan + contiguous payloads → **linear scan** (64 MiB windows, sequential `read` when CLI **`-t`** path is set, else mmap span). Query hint **`execution.fold_parallel: false`** forces offset-ordered sequential chunk visits (not linear scan). Execution stats: **`io_regime`**, **`fold_parallel`**, **`fold_linear_scan`**. **Operations (all supported dtypes):** tier-A/B streaming ops (integer values promoted to `f64` for aggregates) plus tier-C **`median`**, **`quantile`**, **`histogram`**. Population **`var` / `std`**, `ddof = 0` — tier-A/B **`var` / `std`** merge decoded slabs via single-pass SIMD sum+sumsq ([`variance_simd.rs`](src/query/fold/variance_simd.rs) on **`f32`**) into parallel Welford state; **`min` / `max`** use SIMD slab min/max on **`f32`**. Preview: CLI `--preview` / `--preview-f32 N` (default **64** for `full`/`json`, **0** for `stats`/`quiet` when omitted); fills dtype-matched preview arrays; **`0`** with `operation` skips preview bytes.

## CLI (`tet`)

| Command                                               | Behavior                                                                                                                                                     |
| ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `tet info <path.tet>`                                 | Default: dataset table; `--json` full summary; `--grep` / `--dataset`; `--layout` / `--chunks` / `--history` / `--all`; `-q` one line.                       |
| `tet query [QUERY]`                                   | `QUERY`: path to `.json`, inline JSON, `-`, or omit → stdin. Parse/validate; without **`-t`** → plan-only echo.                                              |
| `tet query … -t path.tet`                             | Catalog + `ReadPlan` (`dataset_*_bytes`, `file_execution` when matched).                                                                                     |
| `tet query … -t path.tet -x`                          | **`build_execution_preview`**: mmap tiles (raw **0** / zstd **1**), previews, **`operation_*`**; appends to platform query history on success (best-effort). |
| `tet query … --format full\|json\|stats\|plan\|quiet` | **`format_query_response`**; **`plan`** = read_plan summary only; catalog miss → stderr hint. **`-q`** = **`quiet`**.                                        |
| `tet query … --preview N`                             | Preview cap when **`-x`** (alias **`--preview-f32`**); **`0`** + **`operation`** or **`spill`** skips preview arrays, still aggregates/exports.              |
| `tet query … --spill-allow DIR`                       | Extra spill roots (repeatable; needs **`-x`** + **`-t`**). Default roots include **`.tet` parent tree** + platform cache dirs.                               |
| `tet qhist` / `qhist list` (`hist` alias)             | Compact table (`-n`, `--all`, **`--json`**); filters **`--dataset`**, **`--tet`**, **`--mode x\|p`**, **`--grep`**; indices **1** = newest in filtered view. |
| `tet qhist run N`                                     | Re-run saved row (**1** = newest in filtered list); **`-t`** / **`-x`** / **`--plan`** override; stdout from current **`--format`** / **`-q`**.              |
| `tet qhist --clear`                                   | Remove `query_history.jsonl`.                                                                                                                                |
| `tet convert <input> <output.tet> [--jobs N]`         | HDF5 / NetCDF / Zarr v3 directory → `.tet` (extension or sniff; **`--jobs 0`** = auto).                                                                      |

**Daily driver:** `tet query q.json -t data.tet -x -q` → one-line aggregates; `tet query … -x --format stats` → slim JSON without `read_plan.chunks[]` or preview arrays.

Run: `cargo run -- …` or `cargo build --release` then `./target/release/tet …`.

## Features (`Cargo.toml`)

- **Default:** `tetration-netcdf` and `tetration-hdf5`. CI runs `cargo test --all-features`.
- **docs.rs:** `no-default-features = true` to skip native NetCDF on docs build.
- **Dev:** `proptest` for catalog index bounds tests in `src/tests/catalog.rs`.

Local minimal build: `cargo build --no-default-features`.

## Verification (ops)

From repo root (matches **`.github/workflows/ci.yml`** unless noted):

```bash
cargo fmt -- --check
cargo clippy -- -D warnings
cargo test
```

Stricter local: `cargo clippy -- -D warnings -W clippy::pedantic`; `cargo test --all-features` for NetCDF-linked paths.

**Windows / NetCDF:** see **`.github/scripts/windows-openssl.ps1`** and **`.github/scripts/configure-netcdf-env.sh`**; CI uses conda-forge NetCDF on Windows.

**Integration tests** live in **`src/tests/`** (compiled with `cargo test --lib`):

| Module / file                   | Covers                                                                                                                      |
| ------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `src/tests/catalog.rs`          | Catalog roundtrip (f32/f64), index robustness, chunk tile geometry, proptest `validate_chunk_payloads`, `le_pod` / `f32_le` |
| `src/tests/query.rs`            | Query JSON, mmap materialize, f64 + tier-C ops, memory budget, spill allowlist + temp spill cleanup                         |
| `src/tests/fold.rs`             | `FoldIoPolicy`, contiguous raw span detection, I/O regime routing                                                           |
| `src/tests/variance_simd.rs`    | SIMD vs scalar `f32` sum/sumsq and min/max                                                                                  |
| `src/tests/reduction.rs`        | Bulk variance accumulators vs elementwise Welford                                                                           |
| `src/tests/utils.rs`            | Host memory probe (`available_memory_bytes`)                                                                                |
| `src/tests/layout_roundtrip.rs` | Superblock / empty file                                                                                                     |
| `src/tests/convert.rs`          | HDF5 / NetCDF / Zarr import (`tensor_*`, `groups_3d`, `cf_3d`), parallel jobs, format sniff                                 |
| `src/tests/cli_output.rs`       | `format_query_response` for `full` / `json` / `stats` / `quiet`                                                             |
| `src/tests/cli_info.rs`         | `tet info` table, `--json`, filters, spawn `tet` smoke (debug/release/`cargo run` fallback)                                 |
| `src/tests/cli_history.rs`      | Platform query history JSONL append/list/clear                                                                              |
| `src/tests/fixture.rs`          | Shared temp `.tet` builders, `index_patch` helpers (wire offsets from `ChunkIndexEntryV1::WIRE_*`)                          |
| `src/tests/session.rs`          | `TetWriterSession` commit + history, `TetFile` + `execute_query_document`                                                   |

Examples: `cargo test --lib`, `cargo test --lib tests::catalog::index_bounds_proptest`, `cargo test --lib tests::fold`.

## Public API surface (high level)

Public API lives under **`tetration::catalog`**, **`tetration::convert`**, **`tetration::layout`**, and **`tetration::query`**. Common embedder imports are in **`tetration::prelude`** (`QueryDocument`, `parse_query_json`, `validate_query`, `plan_query_with_tet_mmap_ex`, `mmap_file_read`, `MAGIC`). Query helpers (`format_query_response`, `materialize_read_plan_*`, `ExecutionBudget`, …) re-export from **`query/mod.rs`**.

## Not implemented / intentional gaps

- **Query:** one dataset per query document; **decimal axis indices** only (`operation.axes`). Planned: **dimension names** (Phase 9) and **coordinate labels** (Phase 7 storage, Phase 9 slice/filter) — see [`docs/query_engine.md`](docs/query_engine.md#dimension-names-vs-coordinate-labels-planned).
- **Memory:** no cgroup-aware RAM probe.
- **Codecs:** only raw (**0**) and zstd (**1**); `write_one_chunk_raw_file` remains raw-only.
- **`tet convert`:** richer HDF5/NetCDF metadata (attrs, non-CF edge cases); no other Zarr codecs beyond raw bytes and zstd.
- **CLI / query UX (Phase 6+):** optional TOML/line-oriented query front-end, optional query `preview` stdout table; **`qhist run --plan`** on saved op rows (needs plan-only path without op gate).
- **Dtypes & verify (Phase 8):** only `f32`/`f64`/`i32`/`i64` today; no `tet verify` — see [`GETTING_STARTED.md` — Phase 8](GETTING_STARTED.md#phase-8--dtypes--file-health-next).
- **Bindings (Phase 11):** separate Python repo, PyPI rename, pins published `tetration`; convert via `h5py` / `netCDF4` / `zarr` extras — later.
- **Metadata (Phase 7):** `THST` footer `metadata` (+ spill via `metadata_ref`); structured `history`; **`tet convert`** + **`TetWriterSession`**; **`query::execute_query_*`**.

When behavior changes, keep **README.md**, **`GETTING_STARTED.md`**, **`docs/layout_v1.md`**, **`docs/query_engine.md`**, and this file aligned. New cross-cutting helpers → **`src/utils/`**; query mmap logic → **`src/query/`** (`decode/`, `materialize/`, … — not a flat `engine/` tree).

# AGENT.md — status and ops for contributors / automation

Compact handoff for humans and agents: what exists today, how to verify it, and what is still stubbed. **Roadmap checklist:** [`GETTING_STARTED.md`](GETTING_STARTED.md). **Query engine detail:** [`docs/query_engine.md`](docs/query_engine.md).

## Current status (May 2026)

| Area                                                                        | Status                                                                                                                                                                                                                                                                           |
| --------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Phases 0–3 (spec, writer/reader, chunk addressing, zstd + index robustness) | **Done**                                                                                                                                                                                                                                                                         |
| Phase 4 (query execute)                                                     | **Done** — plan, mmap materialize, parallel multi-chunk decode, memory-aware routing, mmap spill, streaming scalar + partial-axis folds, **`f32`** / **`f64`** / **`i32`** / **`i64`** execution, tier-C **`median`** / **`quantile`** / **`histogram`** (scalar + partial axes) |
| Phase 5 (convert / bindings)                                                | **Not started** (CLI stubs only)                                                                                                                                                                                                                                                 |
| JSON security                                                               | **Done (v1)** — `QueryLimits`, `deny_unknown_fields`, caps in `document.rs`; proptest in `tests/query.rs`                                                                                                                                                                        |

**Branch:** `main` ([PR #1](https://github.com/thicclatka/tetration/pull/1) layout/query v1; [PR #2](https://github.com/thicclatka/tetration/pull/2) integer dtypes).

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
- **`query`:** JSON parse/validate in **`document.rs`** (`QueryLimits`). Wire types in **`query/types/`**. Submodules under **`src/query/`**:

  | Submodule          | Role                                                                            |
  | ------------------ | ------------------------------------------------------------------------------- |
  | **`plan/`**        | `selection`, `read_plan` — global box + chunk read plan                         |
  | **`decode/`**      | `chunk_decode`, `indexing` — mmap decode and row-major indexing                 |
  | **`materialize/`** | f32/f64/i32/i64 materialize, `parallel`, `int`, `stats` (tier-C)                |
  | **`fold/`**        | `reduction`, `shared` (`FoldPlanOutcome`), `partial_fold`, `partial_geometry`   |
  | **`dispatch.rs`**  | Dtype routing for materialize, spill, scalar/partial fold                       |
  | **`engine/`**      | `run`, `operations`, `budget`, `spill_policy` — entrypoints + execution preview |

  **Execution:** `planned_chunk_mmap_slices` (raw codec **0** only). Prefer new mmap logic under **`decode/`** or **`materialize/`**; orchestration under **`engine/`** + **`dispatch.rs`**. Materialize: sequential + parallel, `_into` caller buffers; capped preview allocates `min(cap, logical)` only. **`build_execution_preview`** picks **`MemoryStrategy`**: `streaming_fold` (tier-A/B ops), `in_memory_materialize` / `temp_spill_materialize` (tier-C ops: **`median`**, **`quantile`**, **`histogram`**), `mmap_spill` (export spill + single-pass preview), `capped_in_memory` (preview-only). Temp spills live under platform cache allowlist roots and are deleted when execution finishes. Budget from host RAM (default **25%**), query `execution.*`, or per-file TIDX settings. **Operations (all supported dtypes):** tier-A/B streaming ops (integer values promoted to `f64` for aggregates) plus tier-C **`median`**, **`quantile`**, **`histogram`**. Population **`var` / `std`**, `ddof = 0`. Preview: `--preview-f32 N` (default **64**; fills `f32_preview`, `f64_preview`, `i32_preview`, or `i64_preview` by dtype); **`0`** with `operation` skips preview arrays.

## CLI (`tet`)

| Command                                     | Behavior                                                                                                                                                                                                       |
| ------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tet info <path.tet>`                       | Mmap, v1 superblock + catalog summary → pretty JSON.                                                                                                                                                           |
| `tet query [-f path \| stdin]`              | Parse/validate query JSON; plan-only echo without `--tet`.                                                                                                                                                     |
| `tet query … --tet path.tet`                | Catalog + `ReadPlan` in response (`dataset_f32_bytes` / `dataset_f64_bytes`, `file_execution` when matched).                                                                                                   |
| `tet query … --tet path.tet --execute`      | Plan + **`build_execution_preview`**: mmap tiles (raw **0** / zstd **1**), capped **`f32_preview`** / **`f64_preview`**, optional **`operation_*`** / **`operation_reduced_*`**, budget fields in `execution`. |
| `tet convert h5 …` / `tet convert netcdf …` | Placeholder errors (not implemented).                                                                                                                                                                          |

Run: `cargo run -- …` or `cargo build --release` then `./target/release/tet …`.

## Features (`Cargo.toml`)

- **Default:** `tetration-netcdf` (optional `netcdf` for future importers; writers/tests do not require it).
- **docs.rs:** `no-default-features = true` to skip native NetCDF on docs build.
- **Dev:** `proptest` for catalog index bounds tests in `tests/catalog.rs`.

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

**Integration tests (consolidated):**

| File                        | Covers                                                                                                                      |
| --------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `tests/catalog.rs`          | Catalog roundtrip (f32/f64), index robustness, chunk tile geometry, proptest `validate_chunk_payloads`, `le_pod` / `f32_le` |
| `tests/query.rs`            | Query JSON, mmap materialize, f64 + tier-C ops, memory budget, spill allowlist + temp spill cleanup                         |
| `tests/utils.rs`            | Host memory probe (`available_memory_bytes`)                                                                                |
| `tests/layout_roundtrip.rs` | Superblock / empty file                                                                                                     |
| `tests/fixture.rs`          | Shared temp `.tet` builders, `index_patch` helpers (wire offsets from `ChunkIndexEntryV1::WIRE_*`)                          |

Examples: `cargo test --test catalog`, `cargo test --test query`, `cargo test catalog::index_bounds_proptest`.

## Public API surface (high level)

Re-exported from `src/lib.rs`: layout (`MAGIC`, mmap, `create_empty_v1_file`, …), catalog (writers, codecs, `FileExecutionSettingsV1`, index types, `read_tet_summary_v1`, `validate_chunk_payloads`, chunk coord helpers, `f32_tensor_bytes_from_shape`, `f64_tensor_bytes_from_shape`, `i32_tensor_bytes_from_shape`, `i64_tensor_bytes_from_shape`, `DATASET_DTYPE_TAG_V1`), query (`QueryDocument`, `QueryResponse`, `ReadPlan`, `QueryExecutionPreview`, `ExecutionBudget`, `QueryLimits`, `SpillPathAllowlist`, parse/validate/plan, f32/f64/i32/i64 materialize + spill, `plan_query_with_tet_mmap_ex`, `planned_chunk_mmap_slices`).

## Not implemented / intentional gaps

- **Query:** named dimension labels for `operation.axes` (decimal indices only).
- **Memory:** no cgroup-aware RAM probe.
- **Codecs:** only raw (**0**) and zstd (**1**); `write_one_chunk_raw_file` remains raw-only.
- **`tet convert`:** stubs only; HDF5 / NetCDF → `.tet` not wired.
- **Bindings:** C ABI / Python per README — later.

When behavior changes, keep **README.md**, **`GETTING_STARTED.md`**, **`docs/layout_v1.md`**, **`docs/query_engine.md`**, and this file aligned. New cross-cutting helpers → **`src/utils/`**; query mmap logic → **`src/query/`** (`decode/`, `materialize/`, … — not a flat `engine/` tree).

# AGENT.md — status and ops for contributors / automation

Compact handoff for humans and agents: what exists today, how to verify it, and what is still stubbed. **Roadmap checklist:** [`GETTING_STARTED.md`](GETTING_STARTED.md). **Query engine detail:** [`docs/query_engine.md`](docs/query_engine.md).

## Current status (May 2026)

| Area                                                                        | Status                                                                                                                                                               |
| --------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Phases 0–3 (spec, writer/reader, chunk addressing, zstd + index robustness) | **Done**                                                                                                                                                             |
| Phase 4 (query execute)                                                     | **Core done** — plan, mmap materialize, parallel multi-chunk decode, `sum`/`mean`/`min`/`max`/`count` (scalar + partial axes), scalar fold without full tensor alloc |
| Phase 4 remainder                                                           | Spill/streaming, more ops (`var`, `product`, …), richer `QueryResponse`, non-`f32` dtypes                                                                            |
| Phase 5 (convert / bindings)                                                | **Not started** (CLI stubs only)                                                                                                                                     |
| JSON security                                                               | **Documented** in `docs/query_engine.md`; limits / `deny_unknown_fields` / fuzzing **not enforced** yet                                                              |

**Branch:** `dev` → `main` ([PR #1](https://github.com/thicclatka/tetration/pull/1)).

## Project shape

- **Crate:** `tetration` (library) + binary **`tet`** (`default-run = "tet"` in `Cargo.toml`).
- **Rust:** `edition = "2024"`, `rust-version = "1.95"`. Toolchain pin: `.mise.toml` sets `rust = "1.95"`.

## Implemented (layout v1)

- **`utils`:** Crate-private `src/utils/` (not re-exported from crate root except `#[doc(hidden)]` `f32_le` helpers). **`utils::wire`** — LE `u32`/`u64` cursors, alignment — for **`layout`** / **`catalog`**. **`utils::f32_le`** — `read_f32_le_at`, `try_cast_f32_le`, `f32_count` (aligned fast path + unaligned-safe reads in materialize).
- **`layout`:** 32-byte `TETR` superblock v1, mmap open (`mmap_file_read`), superblock parse, empty v1 file creation (`create_empty_v1_file`). See `docs/layout_v1.md`.
- **`catalog`:** Dataset directory, chunk index header/entries, validation (`validate_chunk_payloads` — also public for integration tests), `read_tet_summary_v1`. Chunk coords: `chunk_coords_intersecting_global_box`, `chunk_coords_intersecting_strided` (`catalog/tile.rs`).
- **Writers:**
  - `write_one_chunk_raw_file` — single chunk, raw `f32`, `codec = 0`.
  - `write_raw_array_file` / `RawArrayWrite` — multi-chunk `f32` grid; per-chunk **`chunk_codec`**: **`CHUNK_PAYLOAD_CODEC_V1.raw`** (0) or **`.zstd`** (1).
- **`tile` (in `catalog`):** Chunk grid, linear chunk index, local coords, `extract_f32_tile_row_major`; strided / box intersection for planning.
- **`query`:** JSON `QueryDocument` parse/validate (`document.rs`; module doc points at JSON security section). **Engine** (`src/query/engine/`):

  | Module           | Role                                                                          |
  | ---------------- | ----------------------------------------------------------------------------- |
  | `run.rs`         | `plan_query_empty`, `plan_query_with_tet_mmap` → `QueryResponse`              |
  | `selection.rs`   | JSON selection → global box + step                                            |
  | `read_plan.rs`   | `ReadPlan` (chunk I/O rows + logical geometry)                                |
  | `indexing.rs`    | Row-major index ↔ coords                                                      |
  | `materialize.rs` | Raw/zstd decode, logical row-major scatter; `fold_read_plan_scalar_operation` |
  | `parallel.rs`    | Rayon `materialize_read_plan_f32_le_parallel` / `_into_parallel`              |
  | `reduction.rs`   | `ReductionKind`, `ScalarAccum`, preview field mapping                         |
  | `operations.rs`  | `build_execution_preview`, `apply_operation`, `reduce_along_axes`             |

  **Execution:** `planned_chunk_mmap_slices` (raw codec **0** only). Materialize: sequential + parallel, `_into` caller buffers. **`build_execution_preview`** uses **parallel** decode when `read_plan.chunks.len() > 1` (including CLI `--execute`). **Operations (f32 only):** `sum`, `mean`, `min`, `max`, `count` — scalar (`axes: []`) via scalar fold (no full logical `Vec`); partial axes (`axes: ["0",…]`) via full materialize + `reduce_along_axes`. Preview: `--preview-f32 N` (default **64**); **`0`** with `operation` skips preview floats.

## CLI (`tet`)

| Command                                     | Behavior                                                                                                                                                                                       |
| ------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tet info <path.tet>`                       | Mmap, v1 superblock + catalog summary → pretty JSON.                                                                                                                                           |
| `tet query [-f path \| stdin]`              | Parse/validate query JSON; plan-only echo without `--tet`.                                                                                                                                     |
| `tet query … --tet path.tet`                | Catalog + `ReadPlan` in response.                                                                                                                                                              |
| `tet query … --tet path.tet --execute`      | Plan + **`build_execution_preview`**: mmap tiles (raw **0** / zstd **1**), capped `f32_preview`, optional **`operation_*`** / **`operation_reduced_*`**. Multi-chunk plans decode in parallel. |
| `tet convert h5 …` / `tet convert netcdf …` | Placeholder errors (not implemented).                                                                                                                                                          |

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

| File                        | Covers                                                                                                                  |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `tests/catalog.rs`          | Catalog roundtrip, index robustness (truncated spans, codec/length lies), proptest `validate_chunk_payloads`, `f32_le`  |
| `tests/query.rs`            | Query JSON parse/validate/plan, mmap materialize (sequential vs parallel parity, zstd), strided chunk-touch, operations |
| `tests/layout_roundtrip.rs` | Superblock / empty file                                                                                                 |
| `tests/fixture.rs`          | Shared temp `.tet` builders, `index_patch` helpers                                                                      |

Examples: `cargo test --test catalog`, `cargo test --test query`, `cargo test catalog::index_bounds_proptest`.

## Public API surface (high level)

Re-exported from `src/lib.rs`: layout (`MAGIC`, mmap, `create_empty_v1_file`, …), catalog (writers, codecs, index types, `read_tet_summary_v1`, `validate_chunk_payloads`, chunk coord helpers), query (`QueryDocument`, `QueryResponse`, `ReadPlan`, `QueryExecutionPreview`, parse/validate/plan, materialize + parallel twins, `planned_chunk_mmap_slices`).

## Not implemented / intentional gaps

- **Query:** dtypes other than **`f32`**; named dimension labels for `operation.axes` (decimal indices only); tier-1+ ops from [operations roadmap](docs/query_engine.md#operations-roadmap-planned) (`var`, `product`, …); disk spill / partial-axis streaming for huge selections.
- **JSON hardening:** size/depth caps, `deny_unknown_fields`, fuzzing — see [JSON security](docs/query_engine.md#json-security-input-and-output).
- **Codecs:** only raw (**0**) and zstd (**1**); `write_one_chunk_raw_file` remains raw-only.
- **`tet convert`:** stubs only; HDF5 / NetCDF → `.tet` not wired.
- **Bindings:** C ABI / Python per README — later.

When behavior changes, keep **README.md**, **`GETTING_STARTED.md`**, **`docs/layout_v1.md`**, **`docs/query_engine.md`**, and this file aligned. New cross-cutting helpers → **`src/utils/`**; query mmap logic → **`src/query/engine/`** before growing `document.rs` / `types.rs`.

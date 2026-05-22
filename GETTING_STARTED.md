# Getting started — Tetration

Use this as a working checklist. The repo today has a **v1 `.tet` layout** (superblock + dataset directory + chunk index + payloads), **catalog mmap I/O**, a **JSON query** control plane with **read planning** and **optional capped `f32` execution preview** (`tet query --tet … --execute`), and **optional NetCDF** behind the default feature flag.

## Environment

- [x] Install Rust **1.95+** (see `rust-version` in `Cargo.toml`; `.mise.toml` pins **1.95**).
- [x] Clone the repo and run `cargo test` to confirm the baseline passes.
- [x] Skim `README.md` for non-goals (no full SQL-on-files day one, etc.) so scope stays focused.

## Phase 0 — Spec before bytes

- [x] Write a short **layout v1** note (`docs/layout_v1.md`): magic bytes, endianness, alignment, chunk index wire.
- [x] **Superblock + bootstrap:** fixed **32-byte** `TETR` block; `chunk_index_offset` / `chunk_index_length`; empty-file rules.
- [x] **Dataset record:** `name`, `dtype`, `shape`, `chunk_shape` (v1 reference writers; see spec tables).
- [x] **Chunk index entry:** grid coords → `payload_offset`, `raw_byte_len`, `stored_byte_len`, `codec`, reserved.
- [x] **Concurrency** (informative): documented in `docs/layout_v1.md` + README (exclusive create, no v1 locking spec).

## Phase 1 — Minimal writer / reader (no compression required)

- [x] **`layout` + `catalog`** (+ shared **`src/utils/wire.rs`** via **`crate::utils::wire`**): binary structs for superblock + index (hand-rolled LE; **rkyv** is a dependency for later metadata, not required for v1 catalog hot path). **`src/utils/`** is the home for crate-private helpers—keep **chunk/dataset/query** logic in `catalog` / `query`.
- [x] **`create` path:** `create_empty_v1_file`, `write_one_chunk_raw_file`, `write_raw_array_file` / `RawArrayWrite` (multi-chunk raw `f32`).
- [x] **`open` + mmap** (`memmap2`): `mmap_file_read`, `read_superblock_v1`, `read_tet_summary_v1`.
- [x] **`tet info`** and library APIs dump catalog / superblock JSON.

## Phase 2 — Chunk addressing

- [x] **Logical slice → chunk coordinates:** `chunk_coords_intersecting_global_box`, `chunk_coords_intersecting_strided` (see `catalog/tile.rs`).
- [x] **Rayon** over independent chunk reads in execution: **`materialize_read_plan_f32_le_parallel`** / **`_into_parallel`**; **`build_execution_preview`** uses parallel decode when the read plan has more than one chunk (`tet query --execute`).
- [x] **`plan_query_with_tet_mmap`:** produces **`ReadPlan`** (payload offsets, `stored_byte_len`, `raw_byte_len`, `codec` per touched chunk).

## Phase 3 — Compression and robustness (complete)

- [x] **Per-chunk zstd** (`codec = 1`): `RawArrayWrite::chunk_codec` vs **`CHUNK_PAYLOAD_CODEC_V1`** (`raw` / `zstd`); index stores `raw_byte_len` vs `stored_byte_len`; query materialization decompresses for `f32` preview.
- [x] Fuzz or property-test **index bounds** vs file length: `tests/catalog.rs` (property tests + hand-patched robustness cases).
- [x] **`bytemuck`** for `f32` payloads: `src/utils/f32_le.rs` (`read_f32_le_at` / `try_cast_f32_le` when 4-byte aligned); materialize uses unaligned-safe reads; covered in `tests/catalog.rs`.

## Phase 4 — Query execution

- [x] **Mmap + plan + read:** `plan_query_with_tet_mmap`, `materialize_read_plan_f32_le` / **`materialize_read_plan_f32_le_into`**, parallel twins **`materialize_read_plan_f32_le_parallel`** / **`_into_parallel`**, CLI **`--execute`** / **`--preview-f32`** (raw and zstd-backed `f32` chunks; **`--preview-f32 0`** with **`operation`** skips preview bytes). Decoded layout is **logical row-major** over the strided selection. **`operation`:** `sum`, `mean`, `min`, `max`, `count`, `var`, `std`, `product` with **`axes: []`** (scalar) or **`axes: ["0",…]`** (partial reductions → **`operation_reduced_*`**; population **`var` / `std`**, `ddof = 0`).
- [x] **Scalar reductions** (`sum`, `mean`, `min`, `max`, `count`, `var`, `std`, `product` with `axes: []`) without full logical tensor allocation (`reduction.rs` + `fold_read_plan_scalar_operation` in `materialize.rs`, orchestrated by `build_execution_preview`).
- [ ] **Full materialization** ergonomics (disk spill, partial-axis streaming) for very large selections; richer **`Operation`** kinds (see [operations roadmap](docs/query_engine.md#operations-roadmap-planned)).
- [ ] Return richer **`QueryResponse`** / **`execution`** fields as operations grow (e.g. named-axis reductions, non-`f32` dtypes).

## Phase 5 — Interop and bindings (later)

- [ ] **`tet convert h5`**: depend on HDF5 stack, chunked read → `.tet` writer (feature-gated if heavy).
- [ ] **`tet convert netcdf`**: same pattern with `netcdf` / `netcdf-sys` (optional dep already present).
- [ ] **C ABI** (`cdylib`) + **Python** (PyO3/maturin) per README—after layout + query JSON churn slows.

## Ongoing hygiene

- [x] Integration tests: temp `.tet`, mmap, catalog (`tests/catalog.rs`), query (`tests/query.rs`), layout (`tests/layout_roundtrip.rs`); shared fixtures in `tests/fixture.rs`.
- [ ] Keep **README**, **`docs/layout_v1.md`**, **`docs/query_engine.md`**, and this file aligned when `layout_version`, codecs, or query JSON change. Prefer **`src/utils/`** for small shared non-domain code (see `utils/mod.rs`).
- [x] JSON hardening: `MAX_QUERY_JSON_BYTES`, `MAX_QUERY_JSON_DEPTH`, `deny_unknown_fields`, dataset/selection/axis caps, proptest in `tests/query.rs` ([query engine — JSON security](docs/query_engine.md#json-security-input-and-output)).
- [ ] When the format stabilizes: publish **docs.rs** examples that match on-disk guarantees.

---

**Suggested next PR-sized slices (pick one):**

1. **Operations:** tier-1 ops in [query engine roadmap](docs/query_engine.md#operations-roadmap-planned) (`product`, …); or execution depth (spill / partial-axis streaming).
2. **Robustness:** targeted tests for bad/truncated zstd payloads and index/file length mismatch (truncated zstd frame covered in `tests/catalog.rs`).
3. **Interop:** stub a real `tet convert netcdf` behind `--features tetration-netcdf` reading a tiny variable.

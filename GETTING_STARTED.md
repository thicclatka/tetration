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
- [x] **Rayon** over independent chunk reads in execution: **`materialize_read_plan_f32_le_parallel`** / **`_into_parallel`** in `query/engine/parallel.rs` (library API; CLI preview path still sequential).
- [x] **`plan_query_with_tet_mmap`:** produces **`ReadPlan`** (payload offsets, `stored_byte_len`, `raw_byte_len`, `codec` per touched chunk).

## Phase 3 — Compression and robustness

- [x] **Per-chunk zstd** (`codec = 1`): `RawArrayWrite::chunk_codec` vs **`CHUNK_PAYLOAD_CODEC_V1`** (`raw` / `zstd`); index stores `raw_byte_len` vs `stored_byte_len`; query materialization decompresses for `f32` preview.
- [ ] Fuzz or property-test **index bounds** vs file length; expand negative tests for truncation/corruption. (`tests/catalog_robustness.rs` covers truncation, corrupt zstd, `raw_byte_len` vs decoded size, and a short mmap slice vs `ReadPlan`.)
- [ ] Optional: **`bytemuck`** views only where alignment + dtype rules are guaranteed.

## Phase 4 — Query execution

- [x] **Mmap + plan + read:** `plan_query_with_tet_mmap`, `materialize_read_plan_f32_le` / **`materialize_read_plan_f32_le_into`**, parallel twins **`materialize_read_plan_f32_le_parallel`** / **`_into_parallel`**, CLI **`--execute`** / **`--preview-f32`** (raw and zstd-backed `f32` chunks; **`--preview-f32 0`** with **`operation`** skips preview bytes). Decoded layout is **logical row-major** over the strided selection. **`operation`:** `sum` / `mean` with **`axes: []`** (scalar) or **`axes: ["0",…]`** (decimal dimension indices; partial reductions → **`operation_reduced_*`**).
- [ ] **Full materialization** ergonomics (streaming / disk spill) for very large selections; richer **`Operation`** kinds; wire parallel materialize into **`--execute`** when worthwhile.
- [ ] Return richer **`QueryResponse`** / **`execution`** fields as operations grow (e.g. named-axis reductions, non-`f32` dtypes).

## Phase 5 — Interop and bindings (later)

- [ ] **`tet convert h5`**: depend on HDF5 stack, chunked read → `.tet` writer (feature-gated if heavy).
- [ ] **`tet convert netcdf`**: same pattern with `netcdf` / `netcdf-sys` (optional dep already present).
- [ ] **C ABI** (`cdylib`) + **Python** (PyO3/maturin) per README—after layout + query JSON churn slows.

## Ongoing hygiene

- [x] Integration tests: temp `.tet`, mmap, catalog roundtrip, query planning + `f32` materialization (see `tests/`).
- [ ] Keep **README**, **`docs/layout_v1.md`**, **`docs/query_engine.md`**, **`AGENT.md`**, and this file aligned when `layout_version`, codecs, or query JSON change. Prefer **`src/utils/`** for small shared non-domain code (see `utils/mod.rs`).
- [ ] When the format stabilizes: publish **docs.rs** examples that match on-disk guarantees.

---

**Suggested next PR-sized slices (pick one):**

1. **Execution depth:** wire parallel materialize into **`--execute`**; streaming or spill-to-disk for huge logical tensors; **`Operation`** beyond sum/mean or beyond decimal axis indices.
2. **Robustness:** targeted tests for bad/truncated zstd payloads and index/file length mismatch.
3. **Interop:** stub a real `tet convert netcdf` behind `--features tetration-netcdf` reading a tiny variable.

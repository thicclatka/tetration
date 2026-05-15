# Getting started — Tetration

Use this as a working checklist. Today the repo has a **JSON query control plane** (`tetration::query`, `tet query`) and dependencies chosen for the **mmap + chunked** story, but **no `.tet` on-disk layout or I/O** yet.

## Environment

- [ ] Install Rust **1.95+** (see `rust-version` in `Cargo.toml`).
- [ ] Clone the repo and run `cargo test` to confirm the baseline passes.
- [ ] Skim `README.md` for non-goals (no full SQL-on-files day one, etc.) so scope stays focused.

## Phase 0 — Spec before bytes

- [ ] Write a short **layout v1** note (magic bytes, endianness, `layout_version` field alignment with query JSON).
- [ ] Decide **fixed header** vs **bootstrap block** (how big is the header, where does the chunk index start).
- [ ] Define **dataset record**: `dataset_id`, `name`, `shape`, `dtype`, **chunk grid** (shape or explicit per-axis chunk sizes).
- [ ] Define **chunk index entry**: `(dataset_id, i0, …, i_{n-1})` → `(file_offset, compressed_len, raw_len, codec)`.
- [ ] Document **concurrency**: who may write which regions (exclusive create, append-only tail, per-chunk locks, etc.).

## Phase 1 — Minimal writer / reader (no compression)

- [ ] Add a `format` (or `layout`) module with **serialized structs** for header + index (serde for interchange; decide where **rkyv** lands first—header only vs also index).
- [ ] Implement **`create` path**: write magic, header, empty or single-dataset stub, **uncompressed** chunk payloads laid out contiguously or via index.
- [ ] Implement **`open` + mmap** (`memmap2`): validate magic/version, parse header + index into memory.
- [ ] Add **`tet info <file.tet>`** (or library API only at first) to dump dataset names, shapes, chunking—proves the path works.

## Phase 2 — Chunk addressing

- [ ] Implement **logical slice → chunk coordinate set** (hyperslab / stepping); unit tests for edge cases (partial chunks, boundaries).
- [ ] Use **Rayon** where chunk reads are independent; keep APIs explicit about parallelism.
- [ ] Wire **`plan_query`** (or a sibling) to produce a **physical read plan**: list of byte ranges / chunks (even if execution still returns JSON for a while).

## Phase 3 — Compression and robustness

- [ ] **Per-chunk zstd** (already a dependency): store compressed vs raw sizes in the index; decode into caller buffers or scratch.
- [ ] Fuzz or property-test **index bounds** vs file length; clear errors on truncation/corruption.
- [ ] Optional: **`bytemuck`** views only where alignment + dtype rules are guaranteed.

## Phase 4 — Query execution

- [ ] Connect **`tet query`** / `plan_query` execution to a real `.tet`: mmap, map selection to chunks, read/decompress, apply **`Operation`** where defined (start with `Sum` / `Mean` on small in-memory slices).
- [ ] Return **`QueryResponse`** fields that reflect real work (`status`, errors, maybe byte ranges touched)—evolve schema carefully.

## Phase 5 — Interop and bindings (later)

- [ ] **`tet convert h5`**: depend on `hdf5` (or similar), chunked read from HDF5 → `.tet` writer (feature-gated if heavy).
- [ ] **`tet convert netcdf`**: same pattern with `netcdf` / `netcdf-sys`.
- [ ] **C ABI** (`cdylib`) + **Python** (PyO3/maturin) per README—after layout is stable enough that churn won’t break FFI.

## Ongoing hygiene

- [ ] Keep **README** and **layout note** in sync when `layout_version` or query JSON fields change.
- [ ] Add integration tests that build a temp `.tet`, mmap it, read a hyperslab, compare to expected bytes.
- [ ] When the format stabilizes: publish **docs.rs** examples that match on-disk guarantees.

---

**Suggested first concrete PR-sized slice:** Phase 0 (one markdown or `docs/layout_v1.md`) + Phase 1 writer that emits a **valid empty or tiny** `.tet` and a reader that mmap-parses it—still no compression until that path is boring and tested.

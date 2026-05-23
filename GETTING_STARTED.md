# Getting started — Tetration

Use this as a working checklist. The repo today has a **v1 `.tet` layout** (superblock + dataset directory + chunk index + payloads), **catalog mmap I/O**, a **JSON query** control plane with **read planning** and **execution** (`tet query --tet … --execute`), and **`tet convert`** from **HDF5 / NetCDF** (extension sniff, streaming + parallel chunk import).

**Fixtures:** tracked import tensors and generators live in [`fixtures/README.md`](fixtures/README.md) (Phase 5 convert tests + local 20 GiB stress).

## Environment

- [x] Install Rust **1.95+** (see `rust-version` in `Cargo.toml`; `.mise.toml` pins **1.95**).
- [x] Clone the repo and run `cargo test` to confirm the baseline passes.
- [x] Skim `README.md` for non-goals (no full SQL-on-files day one, etc.) so scope stays focused.

## Phase 0 — Spec before bytes

**Goal:** lock v1 wire layout and concurrency expectations before writers/readers diverge.

- [x] Write a short **layout v1** note (`docs/layout_v1.md`): magic bytes, endianness, alignment, chunk index wire.
- [x] **Superblock + bootstrap:** fixed **32-byte** `TETR` block; `chunk_index_offset` / `chunk_index_length`; empty-file rules.
- [x] **Dataset record:** `name`, `dtype`, `shape`, `chunk_shape` (v1 reference writers; see spec tables).
- [x] **Chunk index entry:** grid coords → `payload_offset`, `raw_byte_len`, `stored_byte_len`, `codec`, reserved.
- [x] **Concurrency** (informative): documented in `docs/layout_v1.md` + README (exclusive create, no v1 locking spec).

**Verify:** `tests/layout_roundtrip.rs`; `tet info` on empty or single-chunk files.

## Phase 1 — Minimal writer / reader (no compression required)

**Goal:** create, mmap-open, and introspect `.tet` without codecs.

- [x] **`layout` + `catalog`** (+ shared **`src/utils/wire.rs`** via **`crate::utils::wire`**): binary structs for superblock + index (hand-rolled LE; **rkyv** is a dependency for later metadata, not required for v1 catalog hot path). **`src/utils/`** is the home for crate-private helpers—keep **chunk/dataset/query** logic in `catalog` / `query`.
- [x] **`create` path:** `create_empty_v1_file`, `write_one_chunk_raw_file`, `write_raw_array_file` / `RawArrayWrite` (multi-chunk raw **`f32`** / **`f64`** / **`i32`** / **`i64`**; optional **`file_execution`** → TIDX header).
- [x] **`open` + mmap** (`memmap2`): `mmap_file_read`, `read_superblock_v1`, `read_tet_summary_v1`.
- [x] **`tet info`** and library APIs dump catalog / superblock JSON.

**Verify:** `tests/catalog.rs`, `tests/fixture.rs` temp builders; `cargo run -- info …`.

## Phase 2 — Chunk addressing

**Goal:** map logical hyperslabs to chunk coordinates and produce a **`ReadPlan`**.

- [x] **Logical slice → chunk coordinates:** `chunk_coords_intersecting_global_box`, `chunk_coords_intersecting_strided` (see `catalog/tile.rs`).
- [x] **Rayon** over independent chunk reads in execution: parallel materialize paths; **`build_execution_preview`** uses parallel decode when the read plan has more than one chunk and materialization is required (`tet query --execute`).
- [x] **`plan_query_with_tet_mmap`:** produces **`ReadPlan`** (payload offsets, `stored_byte_len`, `raw_byte_len`, `codec` per touched chunk).

**Verify:** `tests/query.rs` plan-only responses; strided / multi-chunk selections.

## Phase 3 — Compression and robustness (complete)

**Goal:** per-chunk zstd, safe index parsing, typed LE payload reads.

- [x] **Per-chunk zstd** (`codec = 1`): `RawArrayWrite::chunk_codec` vs **`CHUNK_PAYLOAD_CODEC_V1`** (`raw` / `zstd`); index stores `raw_byte_len` vs `stored_byte_len`; query materialization decompresses all supported dtypes.
- [x] Fuzz or property-test **index bounds** vs file length: `tests/catalog.rs` (property tests + hand-patched robustness cases).
- [x] **`bytemuck`** for **`f32`** / **`f64`** / **`i32`** / **`i64`** payloads: `src/utils/le_pod.rs`; materialize uses unaligned-safe reads; covered in `tests/catalog.rs`.

**Verify:** `cargo test --test catalog`; zstd roundtrip in catalog + query tests.

## Phase 4 — Query execution

**Goal:** JSON **`operation`** over mmap’d chunks with memory-aware routing (stream, cap, spill, temp materialize).

- [x] **Mmap + plan + read:** `plan_query_with_tet_mmap`, materialize **`f32` / `f64` / `i32` / `i64`** (sequential + parallel + `_into`); CLI **`--execute`** / **`--preview-f32`** (raw and zstd chunks; **`--preview-f32 0`** with **`operation`** skips preview bytes). Decoded layout is **logical row-major** over the strided selection.
- [x] **`operation`:** `sum`, `mean`, `min`, `max`, `count`, `var`, `std`, `product`, `norm_l1`, `norm_l2`, `all_finite`, `any_nan` with **`axes: []`** (scalar) or **`axes: ["0",…]`** (partial reductions → **`operation_reduced_*`**; population **`var` / `std`**, `ddof = 0`).
- [x] **Streaming reductions** — scalar and partial-axis folds without full logical tensor allocation; **`memory_strategy: streaming_fold`**.
- [x] **Memory budget** — `ExecutionBudget::resolve` (query `execution.*` → TIDX header → default **25%** host RAM); per-file settings via **`RawArrayWrite::file_execution`**.
- [x] **Mmap spill** — `output.preferred.spill_array { handle }` → dtype-native spill paths (`memory_strategy: mmap_spill`).
- [x] **Capped preview** without full logical-buffer allocation when `max_elements < logical`.
- [x] **Spill path allowlist** — `SpillPathAllowlist` + `plan_query_with_tet_mmap_ex`; CLI `--spill-allow DIR`.
- [x] **Tier-2 index ops** — `arg_min` / `arg_max` (scalar + partial axes).
- [x] **Tier-C stats** — scalar + partial **`median`**, **`quantile`**, **`histogram`** (equal-width bins per reduced cell); in-RAM or temp spill + cleanup.

**Verify:** `tests/query.rs`, `docs/query_engine.md`; programmatic `.tet` from `tests/fixture.rs` (no import fixtures required).

## Phase 5 — Interop (convert)

**Goal:** import chunked numeric arrays from common scientific containers into `.tet` (reuse streaming writer + parallel tile fill). **Fixtures:** [`fixtures/README.md`](fixtures/README.md).

### Baseline (done)

- [x] **`tet convert <input> <output.tet> [--jobs N]`** — format from extension / file signature (HDF5, NetCDF); history footer (`convert` / `h5` | `nc`).
- [x] **HDF5** (`tetration-hdf5`): root-level **`f32` / `f64` / `i32` / `i64`** datasets; chunked hyperslab read → `.tet` (feature-gated).
- [x] **NetCDF** (`tetration-netcdf`): same dtypes; **`get_raw_values_into`** tile path; feature-gated.
- [x] **Streaming write** — one chunk in RAM at a time (≈ **`jobs` × tile** under parallel import); sequential payload append when layout allows.
- [x] **Fixtures + tests** — `fixtures/small/` byte roundtrips (`tests/convert.rs`); `fixtures/large/` 20 GiB stress (gitignored, `mise run fixtures:large`).

### Phase 5 focus (next)

- [ ] **Richer HDF5** — groups / nested paths (`/group/var` → catalog names); reuse source chunking; clearer errors for unsupported dtypes.
- [ ] **Richer NetCDF** — groups; **CF** numeric conventions (`scale_factor`, `add_offset`, `_FillValue`) at import; coordinate variables as separate datasets where useful.
- [ ] **Zarr → `.tet`** — v2/v3 directory (or zip) store; map Zarr chunks to `.tet` tiles; add `fixtures/small/zarr/` when landed.

### Could add later (not Phase 5 yet)

Other dense-grid formats may follow the same pipeline if there is demand — e.g. **`.npy` / `.npz`**, **COG/GeoTIFF**, **GRIB2**, **NIfTI**. **CSV / Parquet** are poor fits (mixed or columnar types vs one dense dtype). Pick per domain after HDF5/NetCDF depth + Zarr.

## Phase 6 — Bindings

**Goal:** ship a **separate Python repo** (renamed on PyPI) that pins a published **`tetration`** crates.io version; Rust stays the format engine, Python owns ergonomics and ecosystem import.

### Python package (separate repo)

- [ ] **PyPI package** (PyO3 / maturin) — `tetration = "x.y.z"` from crates.io (`default-features = false` for lean wheels); NumPy buffer views where dtypes align.
- [ ] **Read / query** — open `.tet`, catalog summary, validate + plan + execute query JSON (parity with key `tet query --execute` paths).
- [ ] **Write path** — stable Rust writer API for tile/chunk append; Python fills buffers from NumPy.
- [ ] **Convert via Python stack** — optional extras (`h5py`, `netCDF4`, `xarray`, `zarr`, …) read foreign formats → numpy tiles → Rust writer; not the Rust `tetration-hdf5` / `tetration-netcdf` link chain.
- [ ] **Tests** — shared or submodule `fixtures/small/`; byte roundtrips + query golden cases against pinned crate releases.

### C ABI (`cdylib`) — when needed

- [ ] **Stable C headers** — narrow API: open, close, last error, list datasets, run query JSON, optional convert entrypoint.
- [ ] **Consumers** — Julia / R / Go / etc. via their FFI.

### Already available (no binding required)

- [x] **Documented layout** — [`docs/layout_v1.md`](docs/layout_v1.md) for standalone readers.
- [x] **JSON + CLI** — `tet query`, `tet info`, `tet convert`; shell out or HTTP-post query documents from any runtime.
- [x] **Rust convert** — `tet convert` for fast CLI import (parallel HDF5/NetCDF); Python convert is a separate, ecosystem-native path.

## Phase 7 — Metadata & history

**Goal:** rich, bounded **file- and dataset-level metadata** plus **write-time lineage** in the `.tet` footer — without slowing mmap hot paths. **Query replay history** lives in the CLI cache only ([`tet history`](#cli-query-history)), not in the file. See README “Recording lineage” and [`docs/layout_v1.md`](docs/layout_v1.md) history footer.

### Baseline (done)

- [x] **Optional history footer** — `THST` tail, JSON `{"history":[[op, source, unix_secs],…]}`, superblock **`flags` bit 1**; payload bounds exclude footer (`catalog/history.rs`).
- [x] **Convert provenance** — `append_convert_history` on `tet convert` (`convert` / `h5` | `nc` / timestamp); **not** used for read/query events.
- [x] **`tet info` / summary** — `read_tet_summary_v1` surfaces parsed `history` alongside superblock + catalog.
- [x] **CLI query history** — `tet history` lists last **10** queries from platform cache (`query_history.jsonl`); `TET_NO_QUERY_HISTORY=1` disables; `TET_QUERY_HISTORY_FILE` overrides path.

### Phase 7 focus (next)

- [ ] **File header metadata** — structured file-level blob (tool + library versions, creation time, optional git commit / hostname); spec in `layout_v1.md`, surfaced in `tet info`.
- [ ] **Dataset attributes** — per-dataset key/value metadata (units, `long_name`, CF-style attrs, arbitrary JSON-safe strings); read in catalog summary; writers set on create/convert.
- [ ] **Richer history events** — versioned event schema beyond `(op, source, ts)`: transforms, parent dataset refs, parameters, operator identity; forward-compatible unknown-field skip.
- [ ] **Session / writer API** — accumulate events in memory during a write session; flush to footer (or metadata chunk) on `commit` / `close` (Rust + Python bindings).
- [ ] **Size policy** — caps on header/history size; spill overflow to **metadata chunks** when the inline footer would grow too large.
- [ ] **Import preservation** — carry selected HDF5/NetCDF/Zarr attrs into dataset metadata on convert (Rust CLI and Python paths).

## Phase 8 — Beyond core (later)

Not scheduled; capture direction without blocking Phases 5–7.

### Query ops (stats lane only)

Extend tier A–C **`operation`** when the result is still a **reduction or QC stat** on a selection:

- [ ] **Histogram** — caller-supplied `min` / `max` bin edges (already on slice list).
- [ ] **Covariance / correlation** along an axis (tier C; materialize or multi-pass).
- [ ] **Named axis labels** — resolve `"time"` → index via Phase 7 dataset metadata.

### Out of scope for JSON `operation` (use bindings after materialize/spill)

- **Spectral / ML transforms** — FFT, CWT, convolution, `matmul`, `einsum`, training/inference → NumPy / SciPy / PyTorch / JAX on spilled or NumPy slabs (Phase 6 Python).
- **Optional client cache** — memoize `(catalog hash, query hash) → plan or result` in CLI session or Python; never append query logs to `.tet`.

### Other

- [ ] **Export** — `.tet` → Zarr directory or other interchange (inverse of Phase 5 import).
- [ ] **GPU-friendly materialize** — optional device copy after CPU decode (binding concern; format stays mmap-first).
- [ ] **Layout / codec evolution** — v2 only when v1 guarantees are insufficient (new dtypes, filters, dedicated metadata regions).

## CLI query history

Recent **`tet query`** documents are stored under the platform cache (`…/tetration/query_history.jsonl`), **not** in the `.tet` file:

```bash
tet query -f q.json --tet data.tet --execute   # appends on success (best-effort)
tet history                                     # last 10 (JSON)
tet history --clear                             # remove file
TET_NO_QUERY_HISTORY=1 tet query …              # disable recording
```

## Ongoing hygiene

- [x] Integration tests: temp `.tet`, mmap, catalog (`tests/catalog.rs`), query (`tests/query.rs`), convert (`tests/convert.rs`), layout (`tests/layout_roundtrip.rs`); shared builders in `tests/fixture.rs`.
- [ ] Keep **README**, **`docs/layout_v1.md`**, **`docs/query_engine.md`**, **`fixtures/README.md`**, and this file aligned when layout, codecs, convert, or query JSON change. Prefer **`src/utils/`** for small shared non-domain code (see `utils/mod.rs`).
- [x] JSON hardening: [`QueryLimits::DEFAULT`](../src/query/document.rs) (`max_json_bytes`, `max_json_depth`, dataset/axis caps), `deny_unknown_fields`, proptest in `tests/query.rs` ([query engine — JSON security](docs/query_engine.md#json-security-input-and-output)).
- [ ] When the format stabilizes: publish **docs.rs** examples that match on-disk guarantees.

---

**Suggested next PR-sized slices (pick one):**

1. ~~**Dtypes:** integer tags (`i32` / `i64`) on disk and in materialize.~~ **Done** — wire tags `3`/`4`, writers, query preview/spill/ops.
2. ~~**Convert baseline:** HDF5 + NetCDF → `.tet` with streaming + parallel import.~~ **Done** — `tet convert`, `tests/convert.rs`, [`fixtures/`](fixtures/README.md).
3. **Richer NetCDF/HDF5:** groups + CF decode at import (+ fixture variants in `generate.py`).
4. **Zarr import:** `fixtures/small/zarr/` + directory → `.tet` roundtrip.
5. **Metadata scaffold:** file header blob + one dataset attribute roundtrip in catalog / `tet info`.
6. **History events v2:** structured transform event + session flush API.
7. **Python repo scaffold:** separate repo, maturin, pinned `tetration`, `open` / `info` / one query execute smoke test.
8. **Histogram:** caller-supplied `min` / `max` for bin edges.

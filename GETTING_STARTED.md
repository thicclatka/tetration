# Getting started — Tetration

Use this as a working checklist. The repo today has a **v1 `.tet` layout** (superblock + dataset directory + chunk index + payloads), **catalog mmap I/O**, a **JSON query** control plane with **read planning** and **execution** (`tet query … -t … -x`), and **`tet convert`** from **HDF5 / NetCDF / Zarr v3** (extension or directory sniff, streaming + parallel chunk import).

**Fixtures:** tracked import tensors and generators live in [`fixtures/README.md`](fixtures/README.md) (Phase 5 convert tests + local 20 GiB stress). Tracked **`.tet`** smoke files for verify/repair/query: [`fixtures/small/tet/README.md`](fixtures/small/tet/README.md) (`mise run fixtures:small-tet` to regenerate).

## Environment

- [x] Install Rust **1.95+** (see `rust-version` in `Cargo.toml`; `.mise.toml` pins **1.95**).
- [x] Clone the repo and run `cargo test --lib` to confirm the baseline passes.
- [x] Skim `README.md` for non-goals (no full SQL-on-files day one, etc.) so scope stays focused.

## Phase 0 — Spec before bytes

**Goal:** lock v1 wire layout and concurrency expectations before writers/readers diverge.

- [x] Write a short **layout v1** note (`docs/layout_v1.md`): magic bytes, endianness, alignment, chunk index wire.
- [x] **Superblock + bootstrap:** fixed **32-byte** `TETR` block; `chunk_index_offset` / `chunk_index_length`; empty-file rules.
- [x] **Dataset record:** `name`, `dtype`, `shape`, `chunk_shape` (v1 reference writers; see spec tables).
- [x] **Chunk index entry:** grid coords → `payload_offset`, `raw_byte_len`, `stored_byte_len`, `codec`, reserved.
- [x] **Concurrency** (informative): documented in `docs/layout_v1.md` + README (exclusive create, no v1 locking spec).

**Verify:** `src/tests/layout_roundtrip.rs`; `tet info` on empty or single-chunk files.

## Phase 1 — Minimal writer / reader (no compression required)

**Goal:** create, mmap-open, and introspect `.tet` without codecs.

- [x] **`layout` + `catalog`** (+ shared **`src/utils/wire.rs`** via **`crate::utils::wire`**): binary structs for superblock + index (hand-rolled LE; **rkyv** is a dependency for later metadata, not required for v1 catalog hot path). **`src/utils/`** is the home for crate-private helpers—keep **chunk/dataset/query** logic in `catalog` / `query`.
- [x] **`create` path:** `create_empty_v1_file`, `write_one_chunk_raw_file`, `write_raw_array_file` / `RawArrayWrite` (multi-chunk raw payloads for all wire dtypes **`f32`–`u64`**; optional **`file_execution`** → TIDX header).
- [x] **`open` + mmap** (`memmap2`): `mmap_file_read`, `read_superblock_v1`, `read_tet_summary_v1`.
- [x] **`tet info`** and library APIs dump catalog / superblock JSON.

**Verify:** `src/tests/catalog.rs`, `src/tests/fixture.rs` temp builders; `cargo run -- info …`.

## Phase 2 — Chunk addressing

**Goal:** map logical hyperslabs to chunk coordinates and produce a **`ReadPlan`**.

- [x] **Logical slice → chunk coordinates:** `chunk_coords_intersecting_global_box`, `chunk_coords_intersecting_strided` (see `catalog/tile.rs`).
- [x] **Rayon** over independent chunk reads in execution: parallel materialize paths; **`build_execution_preview`** uses parallel decode when the read plan has more than one chunk and materialization is required (`tet query --execute`). Tier-A/B streaming fold uses parallel chunks when **in-core**; **out-of-core** full dense scans use sequential linear scan ([`fold_policy.rs`](src/query/fold/fold_policy.rs)).
- [x] **`plan_query_with_tet_mmap`:** produces **`ReadPlan`** (payload offsets, `stored_byte_len`, `raw_byte_len`, `codec` per touched chunk).

**Verify:** `src/tests/query.rs` plan-only responses; strided / multi-chunk selections.

## Phase 3 — Compression and robustness (complete)

**Goal:** per-chunk zstd, safe index parsing, typed LE payload reads.

- [x] **Per-chunk zstd** (`codec = 1`): `RawArrayWrite::chunk_codec` vs **`CHUNK_PAYLOAD_CODEC_V1`** (`raw` / `zstd`); index stores `raw_byte_len` vs `stored_byte_len`; query materialization decompresses all supported dtypes.
- [x] Fuzz or property-test **index bounds** vs file length: `src/tests/catalog.rs` (property tests + hand-patched robustness cases).
- [x] **`bytemuck`** for **`f32`** / **`f64`** / **`i32`** / **`i64`** payloads: `src/utils/le_pod.rs`; materialize uses unaligned-safe reads; covered in `src/tests/catalog.rs`.

**Verify:** `cargo test --lib`; zstd roundtrip in catalog + query tests.

## Phase 4 — Query execution

**Goal:** JSON **`operation`** over mmap’d chunks with memory-aware routing (stream, cap, spill, temp materialize).

- [x] **Mmap + plan + read:** `plan_query_with_tet_mmap`, materialize **`f32` / `f64` / `i32` / `i64`** (sequential + parallel + `_into`); CLI **`--execute`** / **`--preview-f32`** (raw and zstd chunks; **`--preview-f32 0`** with **`operation`** skips preview bytes). Decoded layout is **logical row-major** over the strided selection.
- [x] **Reductions (flat JSON):** top-level keys `sum`, `mean`, … — scalar **`"mean": []`**, partial **`"mean": 0`** or **`"sum": [0,1]`** → **`operation_*`** / **`operation_reduced_*`**; population **`var` / `std`**, `ddof = 0`.
- [x] **Streaming reductions** — scalar and partial-axis folds without full logical tensor allocation; **`memory_strategy: streaming_fold`**.
- [x] **Adaptive fold I/O** — [`FoldIoPolicy`](src/query/fold/fold_policy.rs): **in-core** parallel chunk fold (Rayon); **out-of-core** sequential **linear scan** over contiguous raw payloads ([`linear_scan.rs`](src/query/fold/linear_scan.rs), 64 MiB windows, file `read` when **`-t`** is set). Query **`execution.fold_parallel`** hint; stats **`io_regime`**, **`fold_linear_scan`**, **`fold_parallel`**.
- [x] **SIMD bulk folds** — [`variance_simd.rs`](src/query/fold/variance_simd.rs): tier-A/B slab paths for all supported float/integer wire tags (`f32`/`f16`, `i32`, `u8`/`u16`, `u32`/`i64`/`u64` on SSE2; NEON for `f32`/`i32` on aarch64).
- [x] **Memory budget** — `ExecutionBudget::resolve` (query `execution.*` → TIDX header → default **25%** host RAM); per-file settings via **`RawArrayWrite::file_execution`**.
- [x] **Mmap spill** — top-level `"spill": "path"` → dtype-native spill paths (`memory_strategy: mmap_spill`); preview cap **`0`** (default for **`stats`/`quiet`**) still exports when **`spill`** is set.
- [x] **Capped preview** without full logical-buffer allocation when `max_elements < logical`.
- [x] **Spill path allowlist** — `SpillPathAllowlist` + `plan_query_with_tet_mmap_ex`; CLI `--spill-allow DIR`.
- [x] **Tier-2 index ops** — `arg_min` / `arg_max` (scalar + partial axes).
- [x] **Tier-C stats** — scalar + partial **`median`**, **`quantile`**, **`histogram`** (equal-width bins per reduced cell); in-RAM or temp spill + cleanup.

**Verify:** `src/tests/query.rs`, `src/tests/fold.rs`, `docs/query_engine.md`; programmatic `.tet` from `src/tests/fixture.rs` (no import fixtures required).

## Phase 5 — Interop (convert)

**Goal:** import chunked numeric arrays from common scientific containers into `.tet` (reuse streaming writer + parallel tile fill). **Fixtures:** [`fixtures/README.md`](fixtures/README.md).

- [x] **`tet convert <input> <output.tet> [--jobs N]`** — HDF5 / NetCDF from extension or file signature; **Zarr v3** from directory store (root `zarr.json`); history footer (`convert` / `h5` | `nc` | `zarr`).
- [x] **HDF5** (`tetration-hdf5`): **`f32` / `f64` / `i32` / `i64`**; nested groups → slash catalog names (`primary/f32`); **CF** decode (`scale_factor`, `add_offset`, `_FillValue`) at import; chunked hyperslab read → `.tet`.
- [x] **NetCDF** (`tetration-netcdf`): same dtypes + groups + CF; **`get_raw_values_into`** tile path.
- [x] **Zarr v3 directory store** — regular chunk grid, chunk codecs **bytes** (raw) or **zstd**; nested groups; map Zarr chunks → `.tet` tiles. Fixture zarr uses uncompressed chunks for fair bench vs `.tet`.
- [x] **Streaming write** — one chunk in RAM at a time (≈ **`jobs` × tile** under parallel import); sequential payload append when layout allows.
- [x] **Fixtures + tests** — `fixtures/small/` (`tensor_*`, `groups_3d`, `cf_3d`, zarr) in `src/tests/convert.rs`; `fixtures/large/` / `fixtures/extra_large/` for local stress (gitignored, `mise run fixtures:large` / `fixtures:extra-large-*`).

**Local bench (extra_large f32 slab, `--jobs 0`, 320 × 64 MiB chunks, warm 2nd pass):** see [`fixtures/bench_results/latest.md`](fixtures/bench_results/latest.md). Regenerate with `mise run bench:h5` (or `bench:netcdf` / `bench:zarr`).

| Regime                         | Machine                         | 20 GiB `.tet` mean (approx.)            |
| ------------------------------ | ------------------------------- | --------------------------------------- |
| In-core (data fits page cache) | Mac Studio M4 Max, ~25 GiB free | **~0.5–0.6 s** (parallel chunk fold)    |
| Out-of-core                    | MacBook M5, ~6 GiB free         | **~4.0 s** (linear scan; ~2× HDF5 warm) |

### Could add later (not Phase 5)

Other dense-grid formats may follow the same pipeline if there is demand — e.g. **`.npy` / `.npz`**, **COG/GeoTIFF**, **GRIB2**, **NIfTI**. **CSV / Parquet** are poor fits (mixed or columnar types vs one dense dtype). Pick per domain after HDF5/NetCDF depth + Zarr.

## Phase 6 — CLI & query UX

**Goal:** make **`tet`** the polished daily driver — readable output, dependable history, and a query document format that is easier to author than raw JSON. The library keeps accepting JSON today; CLI improvements can add alternate front-end formats without breaking embedders.

### Baseline (done)

- [x] **`tet query`** — validate, plan, optional **`-x` / `--execute`**; default stdout is pretty full **`QueryResponse`** (`--format full`).
- [x] **Focused query output** — **`--format full|json|stats|quiet`** (or **`-q`** for quiet); library **`format_query_response`** in **`src/query/cli/output/`**; **`src/tests/cli_output.rs`**.
- [x] **`tet qhist`** — platform query cache (`query_history.jsonl`); **`hist`** alias; **`--clear`**, **`TET_NO_QUERY_HISTORY`**, **`TET_QUERY_HISTORY_FILE`** (see [CLI query history](#cli-query-history-tet-qhist)).
- [x] **`tet info` / `tet convert`** — default dataset table; `--json` full dump; `--grep` / `--dataset`; sections (`--layout`, `--chunks`, `--history`, `--all`); `-q` one-liner.

### Phase 6 focus (next)

- [x] **Query history list / run** — `tet qhist list` (default), `--all` for all retained rows; `tet qhist run N` (1 = newest); `TET_QUERY_HISTORY_MAX` caps rotation on append.
- [x] **History extras** — `list` filters (`--dataset`, `--tet`, `--mode`, `--grep`); indices match filtered view for `run N` (not in `.tet`).
- [x] **Flat query JSON (v1 wire)** — top-level op keys (`mean: 0`, `quantile: { q, axis }`); nested `"operation"` rejected; see [`docs/query_engine.md`](docs/query_engine.md#query-document-json).
- [x] **TOML query profile** — `.toml` path or non-JSON stdin/inline → same `QueryDocument` as JSON ([`parse_query_toml`](src/query/document_toml.rs)); line-oriented profile still later.
- [x] **CLI polish** — `error:` prefix on failures; catalog-miss **hint** on stderr with dataset list (`tet info` tip).
- [x] **`--format plan`** — slim JSON: catalog + read_plan summary (no `chunks[]`, no `execution` block).
- [x] **`tet info` UX** — table + filters; **`--history`** = on-disk footer (not `qhist`).
- [x] **`--format table`** — ASCII tables for query summary, read plan, aggregates, preview sample ([`table.rs`](src/query/cli/output/table.rs)).

**Verify:** spawn-`tet` smoke in `src/tests/cli_info.rs`; golden query docs in repo; `tet query -x -q` on large multi-chunk **`operation_*`** responses.

**Examples:**

```bash
tet query q.json -t data.tet              # plan (full JSON)
tet query q.json -t data.tet -x -q        # execute, one-line stdout
tet query q.json -t data.tet --format plan
tet query q.json -t data.tet -x --format stats
tet query q.json -t data.tet -x --format json | jq .
tet info data.tet
tet info data.tet --grep temp --chunks -n 8
tet info data.tet --json | jq '.summary.datasets'
```

## Phase 7 — Metadata & history

**Goal:** rich, bounded **file- and dataset-level metadata** plus **write-time lineage** in the `.tet` footer — without slowing mmap hot paths. **Query replay history** is a Phase 6 CLI concern ([`tet qhist`](#cli-query-history-tet-qhist)), not on-disk format. See README “Recording lineage” and [`docs/layout_v1.md`](docs/layout_v1.md) (footer events; `tet info --history`).

### Baseline (done)

- [x] **Optional history footer** — `THST` tail, JSON `{"history":[[op, source, unix_secs],…]}`, superblock **`flags` bit 1**; payload bounds exclude footer (`catalog/history.rs`).
- [x] **Convert provenance** — `append_convert_history` on `tet convert` (`convert` / `h5` | `nc` | `zarr` / timestamp); **not** used for read/query events.
- [x] **`tet info` / summary** — `read_tet_summary_v1` surfaces parsed `history` alongside superblock + catalog.

### Phase 7 remainder (done)

- [x] **Structured history events** — footer `history` as JSON objects (`op`, `source`, `at`, optional `parents` / `params`); legacy `[op, source, at]` triples still read.
- [x] **Footer size policy** — inline JSON capped at **64 KiB**; oversized `metadata` spills to a raw blob before `THST` (`metadata_ref` in footer JSON).

### Phase 7 baseline (done)

- [x] **Rust embedder examples** — `create_and_query`, `inspect_catalog`, `session_write`; run with `cargo run --example …`.
- [x] **Rust embedder session API (baseline)** — [`TetWriterSession`](src/catalog/session.rs) / [`TetFile`](src/catalog/session.rs), [`execute_query_document`](src/query/execute.rs) / [`execute_query_json`](src/query/execute.rs), [`FileMetadataDraft`](src/catalog/session.rs) (in-memory until wire spec); [`prelude`](src/lib.rs) re-exports; tests in `src/tests/session.rs`.
- [x] **Rust embedder create + use (wire)** — [`TetWriterSession::open_append`](src/catalog/session.rs); streaming via [`commit_with_fill`](src/catalog/session.rs); catalog [`append_multi_raw_array_file`](src/catalog/append.rs).
- [x] **File header metadata (footer JSON)** — `metadata.file` in `THST` footer (`tool`, `library_version`, `created_at`); [`TetWriterSession::metadata`](src/catalog/session.rs); `tet info` text + `--json`.
- [x] **Dataset attributes (footer JSON)** — `metadata.datasets[name].attrs` + optional `dim_names`; session flush; [`read_tet_summary_v1`](src/catalog/mod.rs) / `tet info` roundtrip.
- [x] **Axis metadata (baseline)** — `dim_names` + inline **`coords`** (≤64 labels/axis) in footer metadata on **`tet convert`** (HDF5 CF `coordinates`, 1D coord arrays) and [`TetWriterSession`](src/catalog/session.rs) commit; see [`docs/layout_v1.md`](docs/layout_v1.md#axis-metadata-phase-7-baseline).
- [x] **Session / writer API** — [`TetWriterSession`](src/catalog/session.rs) queues attrs / `dim_names` / `coords`, optional [`push_history_event`](src/catalog/session.rs) (default `write` + path on commit when empty); footer flush on `commit` / `commit_with_fill`.
- [x] **Import preservation (baseline)** — HDF5/NetCDF/Zarr v3 scalar attrs → footer `metadata.datasets` on `tet convert`; NetCDF `dim_names` from dimension names.

## Phase 8 — Dtypes & file health (done)

**Goal:** **`tet verify`** / **`tet repair`** and additional **wire dtypes** (`u8`, `u16`, …) end-to-end (writers, convert, query). Distinct from Phase 7 metadata; Phase 9 (named axes, coord selection, interchange) is **done**.

**Baseline (May 2026):** file health + wire tags `1`–`10` (`f32`–`u64`, including **`f16`** tag `9`); booleans import as **`u8`**. `tet verify` is a **quick scan** (first 128 chunk decode-checks on large files); use **`tet verify --deep`** for a full payload decode pass.

### File health / verification

- [x] **`tet verify <path.tet>`** — findings + recommendations; `--json` / `-q` / `--repair`; [`verify_tet_file`](src/verify/mod.rs).
- [x] **`tet repair <path.tet>`** — plan by default; `--apply <code>` (`footer_invalid` today); [`tetration::repair`](src/repair/mod.rs).
- [x] **Library API** — [`tetration::verify`](src/verify/mod.rs): layout parse, chunk index/payload checks, decode integrity (≤128 chunks deep), footer + [`MetadataLimitsV1`](src/catalog/metadata.rs) on resolved metadata (incl. spill).
- [x] **Index vs payloads** — in-bounds payloads (parse + decode walk), duplicate offsets, optional contiguous-order warning.
- [x] **CI / tests** — [`src/tests/verify_fixtures.rs`](src/tests/verify_fixtures.rs); `assert_tet_verify_ok` after convert helpers in [`src/tests/convert.rs`](src/tests/convert.rs) (`cargo test --all-features`); committed smoke in [`src/tests/small_tet_fixtures.rs`](src/tests/small_tet_fixtures.rs) + [`fixtures/small/tet/`](fixtures/small/tet/).
- [x] **Deep decode** — `tet verify --deep` / [`VerifyOptions::deep_decode`](src/verify/options.rs) decodes every chunk; default samples first [`DEEP_DECODE_MAX_CHUNKS`](src/verify/chunks.rs) (128) on larger files.
- [x] **Dataset tensor bytes** — per-dataset chunk grid count + per-tile `raw_byte_len` + sum vs logical tensor size ([`check_dataset_tensor_bytes`](src/verify/datasets.rs)).

### Element dtypes (wire + execution)

Today: **`f32`**, **`f64`**, **`i32`**, **`i64`**, **`u8`** (`5`), **`u16`** (`6`), **`i16`** (`7`), **`u32`** (`8`), **`f16`** (`9`), **`u64`** (`10`) ([`ElementDtype`](src/utils/dtype.rs), [`DATASET_DTYPE_TAG_V1`](src/catalog/mod.rs)).

- [x] **`u8` / `u16` / `i16` wire tags** — catalog tags `5`–`7`; documented in [`docs/layout_v1.md`](docs/layout_v1.md).
- [x] **Writers** — same byte-span API for all integer tags via `write_raw_array_file` / session paths.
- [x] **Convert** — HDF5 signed/unsigned 8/16-bit + `Boolean`→`u8`; NetCDF `byte`/`short`/`ushort`; Zarr `int8`/`uint8`/`bool`→`u8`, `int16`/`uint16`.
- [x] **Query** — materialize, streaming fold, tier-A/B/C, spill, dtype-matched previews (`u8_preview`, `u16_preview`, `i16_preview`).
- [x] **Tests** — catalog roundtrip, query sum/preview, verify fixture gate per tag.
- [x] **More dtypes (`u32`, `f16`, `u64`)** — wire tags `8`/`9`/`10`; query materialize/fold, convert (Zarr `float16`/`uint32`/`uint64`, HDF5 unsigned `U4`/`U8`), verify fixtures.
- [x] **Integer SIMD (bulk sum/var/min-max)** — [`variance_simd.rs`](src/query/fold/variance_simd.rs): `f32`/`f16` (via `f32` chunks), `i32` (SSE2/NEON), `u8`/`u16` (SSE2 unpack), `u32`/`i64`/`u64` (SSE2 pairs); slab [`push_*_le_bytes`](src/query/fold/reduction.rs) + [`linear_scan.rs`](src/query/fold/linear_scan.rs) for all wire integer/float tags on tier-A/B ops.

## Phase 9 — Query ops & interchange

**Goal:** extend tier A–C **`operation`** and export paths when the result is still a **reduction, QC stat, or interchange artifact** — builds on Phase 7 metadata and Phase 8 dtype/verify baseline.

### Stats lane

- [x] **Histogram** — caller-supplied `min` / `max` bin edges (already on slice list).
- [x] **Covariance / correlation** — rank-2 selection; `covariance` / `correlation` with one observation `axis`.
- [x] **Dimension names in query** — resolve `"mean": "time"` → axis index via Phase 7 metadata (decimal indices remain the internal wire).
- [x] **Coordinate-aware selection** — `selection[].start_label` / `stop_label` resolve via footer `coords` (+ `dim_names` axis key) at plan time.

### QC / missing-data counts (Phase 9)

v1 already ships boolean **`any_nan`** and **`all_finite`** (scalar + partial axes). Phase 9 adds **count** reductions for data-quality workflows (tier A/B streaming fold where possible):

- [x] **`nan_count`** — count of **NaN** elements (`nan_count`); integers contribute 0.
- [x] **`null_count`** — count of elements equal to **fill** (query `fill` or attrs `_FillValue` / `missing_value` / `fill_value`).
- [x] **`inf_count`** — count of ±infinity elements (`inf_count`; integers contribute 0).
- [ ] **Related counts (as needed)** — e.g. `finite_count` or combined non-finite tallies; deferred.

See [`docs/query_engine.md` — Phase 9 ops](docs/query_engine.md#phase-9-ops-shipped).

See [dimension names vs coordinate labels](docs/query_engine.md#dimension-names-vs-coordinate-labels).

### Interchange & format

- [x] **Export** — `.tet` → Zarr v3 directory (`tet export in.tet out.zarr/`; inverse of Phase 5 import).
- [ ] **Layout / codec evolution (beyond Phase 8)** — v2 only when v1 cannot be extended (filters, dedicated metadata regions, breaking layout changes).

### Out of scope for JSON `operation`

- **Spectral / ML transforms** — FFT, CWT, convolution, `matmul`, `einsum`, training/inference → NumPy / SciPy / PyTorch / JAX on spilled slabs (Phase 11 Python).
- **Optional client cache** — memoize `(catalog hash, query hash) → plan or result` in CLI session or bindings; never append query logs to `.tet`.

### Already shipped (Phase 4)

- [x] **Parallel streaming fold** — Rayon over chunks for tier-A/B scalar + partial-axis ops when in-core and `chunk_count > 1` ([`parallel/`](src/query/fold/parallel/mod.rs); see [`docs/query_engine.md`](docs/query_engine.md#streaming-fold-performance)).
- [x] **Out-of-core linear scan** — sequential byte-stream fold when logical size exceeds available RAM headroom and payloads are contiguous raw ([`linear_scan.rs`](src/query/fold/linear_scan.rs), [PR #7](https://github.com/thicclatka/tetration/pull/7)).

## Phase 10 — GPU (later)

**Goal:** optional **device materialization** after CPU decode — format stays mmap-first; GPU is a binding/runtime choice, not a different wire layout.

- [ ] **Explicit device routing** — CLI flag or API knob (`cpu` / `cuda:0` / auto with fallback); document when transfer overhead dominates.
- [ ] **Batched host→device copy** — overlap decode/decompress on CPU with async copies where frameworks allow.
- [ ] **VRAM guardrails** — cap in-flight bytes, check free device memory, fall back to CPU on OOM.
- [ ] **Alignment / dtype notes** — document row-major chunk payloads and `f32` / `f16` expectations for fast paths (see README “GPUs and tensors”).

## Phase 11 — Bindings (Python & C ABI)

**Goal:** ship **language bindings** after the CLI and on-disk story are stable — separate Python repo (PyPI rename) pinning published **`tetration`** on crates.io; optional **`cdylib`** for other FFIs.

### Python package (separate repo)

- [ ] **PyPI package** (PyO3 / maturin) — `tetration = "x.y.z"` from crates.io (`default-features = false` for lean wheels); NumPy buffer views where dtypes align.
- [ ] **Read / query** — open `.tet`, catalog summary, validate + plan + execute query documents (parity with key `tet query --execute` paths).
- [ ] **Write path** — stable Rust writer API for tile/chunk append; Python fills buffers from NumPy.
- [ ] **Convert via Python stack** — optional extras (`h5py`, `netCDF4`, `xarray`, `zarr`, …) read foreign formats → numpy tiles → Rust writer; not the Rust `tetration-hdf5` / `tetration-netcdf` link chain.
- [ ] **Tests** — shared or submodule `fixtures/small/`; byte roundtrips + query golden cases against pinned crate releases.

### C ABI (`cdylib`) — when needed

- [ ] **Stable C headers** — narrow API: open, close, last error, list datasets, run query JSON, optional convert entrypoint.
- [ ] **Consumers** — Julia / R / Go / etc. via their FFI.

### Already available (no binding required)

- [x] **Documented layout** — [`docs/layout_v1.md`](docs/layout_v1.md) for standalone readers.
- [x] **JSON + CLI** — `tet query`, `tet info`, `tet convert`; shell out or HTTP-post query documents from any runtime.
- [x] **Rust convert** — `tet convert` for fast CLI import (parallel HDF5 / NetCDF / Zarr); Python convert is a separate, ecosystem-native path.

## CLI query history (`tet qhist`)

Recent **`tet query`** documents are stored under the platform cache (`…/tetration/query_history.jsonl`), **not** in the `.tet` file. Use **`tet info --history`** for convert/provenance events in the file footer.

```bash
tet query q.json -t data.tet -x                # appends on success (best-effort)
tet qhist                                       # same as `qhist list`
tet qhist list                                  # compact table (default)
tet qhist list --all                            # every retained row in table
tet qhist list --dataset temperature --mode x
tet qhist list --grep tensor_3d                 # dataset / tet path / op label
tet qhist list --json                           # full JSON (scripts)
tet qhist run 1 -q                              # re-run newest; today's stdout flags
tet qhist run 1 --plan -q                       # plan-only replay (no -x); fails if row has an op — use query without op or `run` without --plan
tet qhist list --clear                          # remove history file
# `tet hist` is an alias for `qhist`
TET_QUERY_HISTORY_MAX=50 tet query …            # keep up to 50 rows on disk
TET_NO_QUERY_HISTORY=1 tet query …              # disable recording
TET_QUERY_HISTORY_FILE=/tmp/tet_history.jsonl tet query …
```

## Ongoing hygiene

- [x] Integration tests: temp `.tet`, mmap, catalog (`src/tests/catalog.rs`), query (`src/tests/query.rs`), fold policy (`src/tests/fold.rs`), SIMD/reduction refs (`src/tests/variance_simd.rs`, `src/tests/reduction.rs`), convert (`src/tests/convert.rs`), layout (`src/tests/layout_roundtrip.rs`), CLI output/history/info (`src/tests/cli_output.rs`, `src/tests/cli_history.rs`, `src/tests/cli_info.rs`); shared builders in `src/tests/fixture.rs`. Run with **`cargo test --lib`**.
- [ ] Keep **README**, **`docs/layout_v1.md`**, **`docs/query_engine.md`**, **`fixtures/README.md`**, and this file aligned when layout, codecs, convert, or query JSON change. Prefer **`src/utils/`** for small shared non-domain code (see `utils/mod.rs`).
- [x] JSON hardening: [`QueryLimits::DEFAULT`](../src/query/document.rs) (`max_json_bytes`, `max_json_depth`, dataset/axis caps), `deny_unknown_fields`, proptest in `src/tests/query.rs` ([query engine — JSON security](docs/query_engine.md#json-security-input-and-output)).
- [ ] When the format stabilizes: publish **docs.rs** examples that match on-disk guarantees.

## Rust API by phase

Quick map for embedders and contributors. **Phases 0–9** are **done** unless marked _later_. User-facing summary: [README — Library use](README.md#library-use) (roadmap table + JSON/TOML query notes).

### Roadmap at a glance

| Area            | Status                                                            |
| --------------- | ----------------------------------------------------------------- |
| Phases **0–3**  | **Done** — layout, writers, `ReadPlan`, zstd                      |
| Phase **4**     | **Done** — query execute                                          |
| Phase **5**     | **Done** — `tet convert`                                          |
| Phase **6**     | **Done** — CLI UX; JSON + TOML query profiles → `QueryDocument` |
| Phase **7**     | **Done** — `TetWriterSession` / `TetFile`, footer metadata        |
| Phase **8**     | **Done** — verify/repair, dtypes                                  |
| Phase **9**     | **Done** — named axes, coord labels, export                       |
| Phase **10–11** | _Later_ — GPU, Python bindings                                    |

### Per-phase Rust / CLI

| Phase  | Status  | You get                                             | Primary Rust / CLI                                                                                                                                                                                 |
| ------ | ------- | --------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **0**  | Done    | Layout v1 wire spec                                 | [`docs/layout_v1.md`](docs/layout_v1.md)                                                                                                                                                           |
| **1**  | Done    | **Write `.tet` bytes** (no session type)            | [`create_empty_v1_file`](src/layout.rs), [`write_raw_array_file`](src/catalog/write.rs), [`write_one_chunk_raw_file`](src/catalog/write.rs); tests: [`src/tests/fixture.rs`](src/tests/fixture.rs) |
| **2**  | Done    | Slice → chunk coords → `ReadPlan`                   | [`plan_query_with_tet_mmap`](src/query/engine/run.rs), [`catalog/tile.rs`](src/catalog/tile.rs)                                                                                                    |
| **3**  | Done    | Per-chunk zstd + index robustness                   | [`ChunkPayloadCodecV1`](src/catalog/mod.rs), [`src/tests/catalog.rs`](src/tests/catalog.rs)                                                                                                        |
| **4**  | Done    | Query plan + execute (fold, spill, tier-C)          | [`build_execution_preview`](src/query/engine/run.rs), [`src/query/`](src/query/); CLI `tet query … -x`                                                                                             |
| **5**  | Done    | HDF5 / NetCDF / Zarr v3 import                      | [`src/convert/`](src/convert/), CLI `tet convert`                                                                                                                                                  |
| **6**  | Done    | CLI stdout modes + query history; JSON/TOML queries | [`format_query_response`](src/query/cli/output/mod.rs), `tet qhist`, [`parse_query_toml`](src/query/document_toml.rs)                                                                             |
| **7**  | Done    | **`TetWriterSession` / `TetFile`** embedder API     | [`src/catalog/session.rs`](src/catalog/session.rs), [`execute_query_*`](src/query/execute.rs), [`prelude`](src/lib.rs); examples + [`src/tests/session.rs`](src/tests/session.rs)                  |
| **8**  | Done    | `tet verify` / `tet repair`, dtypes **`f32`–`u64`** | [`src/verify/`](src/verify/), [`src/repair/`](src/repair/)                                                                                                                                         |
| **9**  | Done    | Named axes, coord labels, QC counts, `tet export`   | [`resolve_axes.rs`](src/query/resolve_axes.rs), [`resolve_selection.rs`](src/query/resolve_selection.rs), [`src/export/`](src/export/)                                                             |
| **10** | _Later_ | GPU materialize hooks                               | —                                                                                                                                                                                                  |
| **11** | _Later_ | Python bindings repo                                | —                                                                                                                                                                                                  |

**Typical embedder path:** Phase **7** on top of **1** (bytes) and **4** (query engine):

1. `TetWriterSession::create` → `push_dataset` → `commit()` (or `commit_with_fill` for streaming).
2. `TetFile::open` → `execute_query_json` → `QueryResponse`.

```bash
cargo run --example create_and_query
cargo run --example session_write
cargo test --lib tests::session
```

**Query input:** JSON or TOML (`tet query q.toml -t data.tet -x`):

```toml
dataset = "temperature"
mean = []
```

---

**Suggested next PR-sized slices (pick one):**

1. ~~**Dtypes:** integer tags (`i32` / `i64`) on disk and in materialize.~~ **Done** — wire tags `3`/`4`, writers, query preview/spill/ops.
2. ~~**Convert (Phase 5):** HDF5 + NetCDF + Zarr → `.tet` with streaming + parallel import; groups, CF decode, `src/tests/convert.rs`, [`fixtures/`](fixtures/README.md).~~ **Done**
3. ~~**Parallel streaming fold:** Rayon over chunks for tier-A/B ops.~~ **Done** — see [`parallel/`](src/query/fold/parallel/mod.rs).
4. ~~**Adaptive out-of-core fold:** linear scan + SIMD bulk tier-A/B when data oversubscribes RAM.~~ **Done** — [PR #7](https://github.com/thicclatka/tetration/pull/7); see [`fold_policy.rs`](src/query/fold/fold_policy.rs), [`linear_scan.rs`](src/query/fold/linear_scan.rs).
5. ~~**CLI focused output (Phase 6):** `--format` / `-q`, `format_query_response`.~~ **Done** — see Phase 6 baseline above.
6. ~~**Query TOML front-end:** `.toml` → `QueryDocument`.~~ **Done** — optional line-oriented profile + golden files still open.
7. ~~**Rust embedder workflows (Phase 7):** examples + session API.~~ **Done**
8. ~~**Metadata + history (Phase 7):** footer JSON, structured history, spill.~~ **Done**
9. ~~**History (Phase 7):** structured events + metadata spill.~~ **Done**
10. ~~**File health (Phase 8):** `tet verify` / `tet repair` + verify fixtures.~~ **Done** — includes `--deep` decode and dataset tensor byte cross-check.
11. ~~**Dtypes (Phase 8):** additional wire tags through write → query smoke.~~ **Done** — tags `5`–`10` (`u8`–`u64`, incl. `f16`); booleans import as `u8`; SIMD slab paths for tier-A/B ops.
12. ~~**Named axes (Phase 9):** `"mean": "time"` via footer `dim_names`.~~ **Done**
13. ~~**Histogram (Phase 9):** caller-supplied `min` / `max` for bin edges.~~ **Done**
14. ~~**QC counts + export (Phase 9):** `nan_count`, `null_count`, `inf_count`; `tet export` → Zarr v3; covariance/correlation.~~ **Done** — deferred: `finite_count` / combined non-finite tallies.
15. **Python repo scaffold (Phase 11):** separate repo, maturin, pinned `tetration`, `open` / `info` / one query execute smoke test.

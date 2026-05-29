# Tetration

[![Crates.io](https://img.shields.io/crates/v/tetration.svg)](https://crates.io/crates/tetration)
[![docs.rs](https://img.shields.io/docsrs/tetration)](https://docs.rs/tetration)
![Build](https://github.com/Latka-Industries/tetration/workflows/Build/badge.svg)
![Rust](https://img.shields.io/badge/rust-1.95-orange.svg)

[_For those who are more cur..._](https://bookshop.org/p/books/book-of-numbers-a-novel-joshua-cohen/af5aa739b0fac506?ean=9780812986655&next=t)

**_STILL IN DEVELOPMENT — layout v1 and query JSON/TOML may change before 1.0._**

**HDF5-shaped** persistence (many large arrays in one durable file), **Zarr-shaped** chunking (regular grid, per-chunk compression, parallel I/O)—in a **single mmap-friendly `.tet` file`**, not a directory of shard blobs.

## What it does today (v1)

- **On-disk layout** — superblock, dataset directory, chunk index, raw or zstd payloads ([`docs/layout_v1.md`](docs/layout_v1.md)).
- **Mmap + read planning** — logical slices → chunk coordinates → [`ReadPlan`](https://docs.rs/tetration/latest/tetration/query/struct.ReadPlan.html).
- **JSON / TOML query + execute** — flat query documents (paired examples in [`fixtures/queries/`](fixtures/queries/)), streaming reductions, tier-C stats, spill export, **`transform`** (zscore, minmax, l1/l2, center, scale, log1p, sqrt, softmax) and **`nan_mean`** / **`nan_std`**; **named axes**, **coord label** selection, QC counts (`nan_count`, `null_count`, `inf_count`, `any_inf`), **covariance** / **correlation** ([`docs/query_engine.md`](docs/query_engine.md)).
- **Import / export** — `tet convert` from HDF5, NetCDF, Zarr v3; **`tet export`** back to Zarr v3 (stored chunk bytes, nested groups).
- **File health** — `tet verify` (quick scan; **`--deep`** decodes every chunk), `tet repair` (plan / `--apply` safe fixes).
- **CLI** — `tet info`, `tet verify`, `tet repair`, `tet query`, `tet qhist`, `tet convert`, `tet export`.
- **Optional GPU (Phase 10, experimental)** — `execution.device` / `tet query --device` for tier-A/B **`f32`** (and **`f16`** on device); Metal (`tetration-metal`, macOS), CUDA (`tetration-gpu`), streaming fold + multi-GPU when host RAM does not fit a dense buffer. CPU streaming fold remains the default for large selections.
- **C ABI** — feature **`tetration-ffi`**, [`include/tetration.h`](include/tetration.h), [`docs/ffi.md`](docs/ffi.md).

**Wire dtypes** (tags `1`–`10`, row-major chunks): **`f32`**, **`f64`**, **`i32`**, **`i64`**, **`u8`**, **`u16`**, **`i16`**, **`u32`**, **`f16`**, **`u64`**. Booleans import as **`u8`**. See [`docs/layout_v1.md`](docs/layout_v1.md#dataset-record-concatenated-variable-length-per-record).

## Quick start

### macOS — Homebrew (recommended)

One-time tap (this repo ships `Formula/tetration.rb`; pulls in **HDF5** and **NetCDF** for `tet convert`):

```bash
brew tap Latka-Industries/tetration https://github.com/Latka-Industries/tetration
brew install tetration
tet --help
```

Upgrade later: `brew upgrade tetration`.

From a local clone (no tap): `brew install --build-from-source Formula/tetration.rb`

### `cargo install`

**Default features** need **system HDF5 and NetCDF** dev libraries (`.h5` / `.nc` convert; Zarr v3 is Rust + bundled **zstd**):

| Platform             | Typical packages                                                                                                                |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| **Debian / Ubuntu**  | `libhdf5-dev`, `libnetcdf-dev`, `pkg-config`, `build-essential`                                                                 |
| **macOS (Homebrew)** | `brew install hdf5 netcdf pkg-config`                                                                                           |
| **Windows**          | OpenSSL + NetCDF/HDF5 (e.g. [vcpkg](https://vcpkg.io/) or conda-forge); see [`.github/scripts/`](.github/scripts/) for CI hints |

```bash
cargo install tetration
```

Without HDF5/NetCDF libs: **`cargo install tetration --no-default-features`** — `tet info` / `tet query` on `.tet` files and **Zarr** import still work.

### Build from source

```bash
git clone https://github.com/Latka-Industries/tetration.git
cd tetration
cargo build --release
export PATH="$PWD/target/release:$PATH"   # or: alias tet="$PWD/target/release/tet"
```

### First commands

```bash
tet convert volume.h5 volume.tet          # HDF5 / NetCDF / Zarr v3 → .tet
tet export volume.tet volume.zarr/      # .tet → Zarr v3 directory (empty or new dir)

tet info volume.tet
tet verify volume.tet
tet verify --deep volume.tet -q    # full chunk decode (large files sample 128 by default)
tet query fixtures/queries/mean_temperature.toml -t volume.tet -x -q   # after convert; <name> from info
tet query '{"dataset":"<name>","inf_count":[]}' -t volume.tet -x -q
```

**Daily driver:** plan + execute with readable stdout:

```bash
tet query fixtures/queries/mean_temperature.toml -t data.tet -x -q
tet query q.json -t data.tet -x --format stats              # slim JSON (no chunk list)
tet query q.toml -t data.tet -x --format table --preview 6  # ASCII tables + slice grid
tet query q.json -t data.tet --format plan                  # catalog + read_plan only
```

Query documents are **flat** JSON or TOML (e.g. `"mean": []` / `mean = []`, `"spill": "slice.bin"`); nested `"operation"` objects are rejected. Details: [query document](docs/query_engine.md#query-document-json-and-toml).

## `tet` commands

Full flag lists: **`tet -h`** and **`tet <command> -h`** (always match the installed binary).

| Command                                            | Alias  | Role                                                        |
| -------------------------------------------------- | ------ | ----------------------------------------------------------- |
| [`tet info`](#tet-info) `<path.tet>`               | —      | Summarize a file (default: dataset table)                   |
| [`tet verify`](#tet-verify) `<path.tet>`           | —      | Layout health check (exit 1 on failure); `--json` / `-q`    |
| [`tet repair`](#tet-repair) `<path.tet>`           | —      | Plan or apply safe in-place fixes (e.g. bad footer)         |
| [`tet query`](#tet-query) `[QUERY]`                | `q`    | Validate JSON/TOML; optional catalog + execute against `-t` |
| [`tet qhist`](#tet-qhist) `[list\|run]`            | `hist` | Recent queries (platform cache; **not** the `.tet` footer)  |
| [`tet convert`](#tet-convert) `<in> <out.tet>`     | —      | HDF5 / NetCDF / Zarr v3 → `.tet`                            |
| [`tet export`](#tet-export) `<in.tet> <out.zarr/>` | —      | `.tet` → Zarr v3 directory store                            |

### `tet info`

| Flag                                                                 | Effect                                                            |
| -------------------------------------------------------------------- | ----------------------------------------------------------------- |
| _(default)_                                                          | Dataset catalog table                                             |
| `--json`                                                             | Full pretty JSON (superblock, catalog, chunks, history)           |
| `-q`, `--quiet`                                                      | One-line summary                                                  |
| `--all`                                                              | All text sections                                                 |
| `--layout` / `--execution` / `--datasets` / `--chunks` / `--history` | One section each (`--history` = convert footer; not `qhist`)      |
| `-n`, `--limit N`                                                    | Max chunk rows with `--chunks` or `--all` (default 32; `0` = all) |
| `--dataset`, `--grep`                                                | Case-insensitive filters on dataset name (and dtype for `--grep`) |

### `tet verify`

| Flag        | Effect                                                                               |
| ----------- | ------------------------------------------------------------------------------------ |
| _(default)_ | Human-readable check list + summary (decodes up to **128** chunks on large files)    |
| `--deep`    | Decode **every** chunk payload (not just the quick sample)                           |
| `--repair`  | After verify, apply safe in-place repairs for repairable findings (see `tet repair`) |
| `--json`    | Pretty JSON [`TetVerifyReport`](src/verify/report.rs)                                |
| `-q`        | One line (`status=ok` / `failed`)                                                    |

Exit code **1** when verification fails (CI-friendly). Manual smoke fixtures: [`fixtures/small/tet/README.md`](fixtures/small/tet/README.md).

### `tet repair`

| Flag           | Effect                                                                   |
| -------------- | ------------------------------------------------------------------------ |
| _(default)_    | Plan from verify recommendations (no writes)                             |
| `--apply CODE` | Apply fix (repeatable); today: `footer_invalid` strips a bad `THST` tail |
| `--dry-run`    | With `--apply`, show changes without writing                             |
| `--json`       | Pretty JSON plan or repair report                                        |

### `tet query`

`QUERY`: path to `.json` / `.toml`, inline JSON/TOML, `-` for stdin, or omit to read stdin. Leading `{` → JSON; `.toml` extension → TOML.

| Flag                | Effect                                                                                                              |
| ------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `-t`, `--tet PATH`  | Attach catalog / read plan (required for `-x`)                                                                      |
| `-x`, `--execute`   | Decode tiles, run `operation`, attach `execution`                                                                   |
| `--format`          | `full` (default), `json`, `stats`, `plan`, `quiet`, `table`                                                         |
| `-q`, `--quiet`     | Shorthand for `--format quiet` (one-line stdout)                                                                    |
| `--preview N`       | Cap preview sample values when executing (`--preview-f32` alias; default 64 for full/json, 0 for quiet/stats/table) |
| `--spill-allow DIR` | Extra spill roots (repeatable; needs `-x` and `-t`)                                                                 |

### `tet qhist`

Stored under the platform cache (`query_history.jsonl`), not in the `.tet` file. Env: `TET_NO_QUERY_HISTORY`, `TET_QUERY_HISTORY_FILE`, `TET_QUERY_HISTORY_MAX`. Details: [`docs/query_engine.md` — end-to-end flow](docs/query_engine.md#end-to-end-flow) (`tet qhist`); roadmap row under [operations](docs/query_engine.md#operations-roadmap-planned).

| Subcommand / flag                                                | Effect                                                                                                              |
| ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `list` _(default)_                                               | Compact table of recent queries                                                                                     |
| `run N`                                                          | Re-run saved row (`1` = newest in filtered view); honors today's `--format` / `-q`; `-t` / `-x` / `--plan` override |
| `--clear`                                                        | Remove the history file                                                                                             |
| `list --all`, `--dataset`, `--tet`, `--mode`, `--grep`, `--json` | Filters / full JSON export on `list`                                                                                |

### `tet convert`

| Input   | Sniff / extensions                                        |
| ------- | --------------------------------------------------------- |
| HDF5    | `.h5`, `.hdf5`, `.hdf`, `.he2`, `.he5`, or file signature |
| NetCDF  | `.nc`, `.netcdf`, `.nc4`, `.nc3`, `.cdf`, or signature    |
| Zarr v3 | Directory with root `zarr.json`                           |

| Flag       | Effect                                                                         |
| ---------- | ------------------------------------------------------------------------------ |
| `--jobs N` | Parallel chunk read workers (`0` = host `available_parallelism`, capped at 64) |

### `tet export`

| Flag / arg | Effect                                                                                 |
| ---------- | -------------------------------------------------------------------------------------- |
| `<in.tet>` | Source file (mmap read + catalog summary)                                              |
| `<out>`    | Zarr v3 **directory**; must be missing or **empty** (creates `zarr.json` + chunk tree) |
| _(stderr)_ | Progress line: dataset count, chunks written, elapsed seconds                          |

Preserves per-dataset **raw** or **zstd** chunk bytes; slash-separated dataset names become nested groups (`primary/f32`). Library: [`export_tet_to_zarr`](https://docs.rs/tetration/latest/tetration/export/fn.export_tet_to_zarr.html).

More examples: [`fixtures/queries/`](fixtures/queries/), [`docs/query_engine.md`](docs/query_engine.md).

## Documentation map

| Doc                                                | Contents                                                                                                                                             |
| -------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`docs/layout_v1.md`](docs/layout_v1.md)           | On-disk **layout v1**: superblock, dataset catalog, chunk index, codecs, footer metadata/history, concurrency                                        |
| [`docs/query_engine.md`](docs/query_engine.md)     | **Query** JSON/TOML wire, planning, fold/spill execution, optional GPU, `tet` CLI mapping, JSON security                                             |
| [`docs/ffi.md`](docs/ffi.md)                       | **C ABI** (`tetration-ffi`): [`include/tetration.h`](include/tetration.h), linking, [`examples/ffi_query.c`](examples/ffi_query.c), release archives |
| [docs.rs / `tetration`](https://docs.rs/tetration) | Rust crate API (`prelude`, `TetFile`, `execute_query_json`, convert, verify, …)                                                                      |

## Design stance (short)

**Partial I/O is the default case** — mmap payload regions, touch only chunks that intersect the selection, parallel decode across disjoint tiles. Full-array loads into RAM are not required for planning or tier-A/B aggregates.

**JSON/TOML is the control plane**, not the storage encoding: hosts validate input, cap size, and enforce spill path policy ([security notes](docs/query_engine.md#json-security-input-and-output)).

## Concurrency and scale

**Read-many / write-once** is the supported scale model for v1:

| Role          | Contract                                                                                                                                                                                                                                                                            |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Writer**    | One process (or coordinated [`TetWriterSession`](https://docs.rs/tetration/latest/tetration/catalog/struct.TetWriterSession.html) / `tet convert`) finishes the file before readers rely on it. v1 defines **no** file locking or live append protocol.                             |
| **Reader**    | Any number of processes or hosts may **mmap read-only** the same sealed `.tet` and run independent queries. The OS shares cold pages via the page cache; each query touches only chunks in its [`ReadPlan`](https://docs.rs/tetration/latest/tetration/query/struct.ReadPlan.html). |
| **Per query** | Tier-A/B folds merge **chunk-local** partials (parallel Rayon when in-core; linear scan when out-of-core). Temp spills use unique paths (`pid` + timestamp); export **`spill`** paths must differ per worker.                                                                       |

**Not supported without extra coordination:** multiple writers on one file, read-while-write, or two workers writing the same export spill path.

**CPU workers:** scale out with **N processes × independent queries** (or datasets), not by sharding one query inside the engine today. **Phase 10 GPU** uses the same chunk-parallel shape: dense materialize when RAM allows, else **streaming device fold** ([`gpu/streaming_fold.rs`](src/query/gpu/streaming_fold.rs)); **`cuda:multi` / `rocm:multi`** shard chunks across devices — see [query engine — scalability](docs/query_engine.md#scalability-read-many-and-phase-10).

Wire details: [`docs/layout_v1.md` — Concurrency](docs/layout_v1.md#concurrency-informative).

**Non-goals (v1):** SQL-on-files, arbitrary codec plugins, GPU codecs in the file format. Optional GPU: [`docs/query_engine.md`](docs/query_engine.md#phase-10--optional-gpu-experimental).

## Library use

```toml
[dependencies]
tetration = "0.1.6"
```

```rust
use tetration::prelude::*;
// TetWriterSession, TetFile, parse_query_json, parse_query_toml, execute_query_json, verify_tet_file, …
```

### Embedder flow

1. **Write** — `TetWriterSession::create` → `push_dataset` → `commit()` (or `commit_with_fill` for streaming).
2. **Read / aggregate** — `TetFile::open` → `execute_query_json` → [`QueryResponse`](https://docs.rs/tetration/latest/tetration/query/struct.QueryResponse.html).

```bash
cargo run --example create_and_query
cargo run --example session_write
```

### Query input: JSON or TOML

Flat JSON and TOML profiles compile to the same [`QueryDocument`](https://docs.rs/tetration/latest/tetration/query/struct.QueryDocument.html). `tet query` accepts `.json` / `.toml` paths, inline text, or stdin; leading `{` selects JSON, otherwise TOML (extension overrides).

```json
{ "dataset": "temperature", "mean": [] }
```

```toml
dataset = "temperature"
mean = [] # scalar reduction
```

Library: [`parse_query_json`](https://docs.rs/tetration/latest/tetration/query/fn.parse_query_json.html), [`parse_query_toml`](https://docs.rs/tetration/latest/tetration/query/fn.parse_query_toml.html), [`parse_query_text`](https://docs.rs/tetration/latest/tetration/query/fn.parse_query_text.html) (auto-detect).

See **Documentation map** above for layout, query engine, FFI, fixtures, and [docs.rs](https://docs.rs/tetration).

---

## To do

- [ ] **Python wrapper** — separate repository (TBA); will pin crates.io `tetration`
- [ ] **docs.rs examples** — match on-disk guarantees when the format stabilizes

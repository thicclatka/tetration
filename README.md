# Tetration

[![Crates.io](https://img.shields.io/crates/v/tetration.svg)](https://crates.io/crates/tetration)
[![docs.rs](https://img.shields.io/docsrs/tetration)](https://docs.rs/tetration)
![Build](https://github.com/thicclatka/tetration/workflows/Build/badge.svg)
![Rust](https://img.shields.io/badge/rust-1.95-orange.svg)

[_For those who are more cur..._](https://bookshop.org/p/books/book-of-numbers-a-novel-joshua-cohen/af5aa739b0fac506?ean=9780812986655&next=t)

**_STILL IN DEVELOPMENT — layout v1 and query JSON may change before 1.0._**

**HDF5-shaped** persistence (many large arrays in one durable file), **Zarr-shaped** chunking (regular grid, per-chunk compression, parallel I/O)—in a **single mmap-friendly `.tet` file`**, not a directory of shard blobs.

## What it does today (v1)

- **On-disk layout** — superblock, dataset directory, chunk index, raw or zstd payloads ([`docs/layout_v1.md`](docs/layout_v1.md)).
- **Mmap + read planning** — logical slices → chunk coordinates → [`ReadPlan`](https://docs.rs/tetration/latest/tetration/query/struct.ReadPlan.html).
- **JSON query + execute** — flat query documents, streaming reductions, tier-C stats, spill export; **named axes**, **coord label** selection, QC counts (`nan_count`, `null_count`, `inf_count`), **covariance** / **correlation** ([`docs/query_engine.md`](docs/query_engine.md)).
- **Import / export** — `tet convert` from HDF5, NetCDF, Zarr v3; **`tet export`** back to Zarr v3 (stored chunk bytes, nested groups).
- **File health** — `tet verify` (quick scan; **`--deep`** decodes every chunk), `tet repair` (plan / `--apply` safe fixes).
- **CLI** — `tet info`, `tet verify`, `tet repair`, `tet query`, `tet qhist`, `tet convert`, `tet export`.

**Wire dtypes** (tags `1`–`10`, row-major chunks): **`f32`**, **`f64`**, **`i32`**, **`i64`**, **`u8`**, **`u16`**, **`i16`**, **`u32`**, **`f16`**, **`u64`**. Booleans import as **`u8`**. See [`docs/layout_v1.md`](docs/layout_v1.md#dataset-record-concatenated-variable-length-per-record).

## Quick start

### macOS — Homebrew (recommended)

One-time tap (this repo ships `Formula/tetration.rb`; pulls in **HDF5** and **NetCDF** for `tet convert`):

```bash
brew tap thicclatka/tetration https://github.com/thicclatka/tetration
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
git clone https://github.com/thicclatka/tetration.git
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
tet query '{"dataset":"<name>","mean":[]}' -t volume.tet -x -q   # <name> from info output
tet query '{"dataset":"<name>","inf_count":[]}' -t volume.tet -x -q
```

**Daily driver:** plan + execute with readable stdout:

```bash
tet query q.json -t data.tet -x -q              # one-line aggregate
tet query q.json -t data.tet -x --format stats  # slim JSON (no chunk list)
tet query q.json -t data.tet --format plan      # catalog + read_plan only
```

Query JSON is **flat** (e.g. `"mean": []`, `"spill": "slice.bin"`); nested `"operation"` objects are rejected. Details: [query document](docs/query_engine.md#query-document-json).

## `tet` commands

Full flag lists: **`tet -h`** and **`tet <command> -h`** (always match the installed binary).

| Command                                            | Alias  | Role                                                       |
| -------------------------------------------------- | ------ | ---------------------------------------------------------- |
| [`tet info`](#tet-info) `<path.tet>`               | —      | Summarize a file (default: dataset table)                  |
| [`tet verify`](#tet-verify) `<path.tet>`           | —      | Layout health check (exit 1 on failure); `--json` / `-q`   |
| [`tet repair`](#tet-repair) `<path.tet>`           | —      | Plan or apply safe in-place fixes (e.g. bad footer)        |
| [`tet query`](#tet-query) `[QUERY]`                | `q`    | Validate JSON; optional catalog + execute against `-t`     |
| [`tet qhist`](#tet-qhist) `[list\|run]`            | `hist` | Recent queries (platform cache; **not** the `.tet` footer) |
| [`tet convert`](#tet-convert) `<in> <out.tet>`     | —      | HDF5 / NetCDF / Zarr v3 → `.tet`                           |
| [`tet export`](#tet-export) `<in.tet> <out.zarr/>` | —      | `.tet` → Zarr v3 directory store                           |

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

`QUERY`: path to `.json`, inline JSON, `-` for stdin, or omit to read stdin.

| Flag                | Effect                                                                                                        |
| ------------------- | ------------------------------------------------------------------------------------------------------------- |
| `-t`, `--tet PATH`  | Attach catalog / read plan (required for `-x`)                                                                |
| `-x`, `--execute`   | Decode tiles, run `operation`, attach `execution`                                                             |
| `--format`          | `full` (default), `json`, `stats`, `plan`, `quiet`                                                            |
| `-q`, `--quiet`     | Shorthand for `--format quiet` (one-line stdout)                                                              |
| `--preview N`       | Cap preview sample values when executing (`--preview-f32` alias; default 64 for full/json, 0 for quiet/stats) |
| `--spill-allow DIR` | Extra spill roots (repeatable; needs `-x` and `-t`)                                                           |

### `tet qhist`

Stored under the platform cache (`query_history.jsonl`), not in the `.tet` file. Env: `TET_NO_QUERY_HISTORY`, `TET_QUERY_HISTORY_FILE`, `TET_QUERY_HISTORY_MAX`. Details: [`GETTING_STARTED.md` — qhist](GETTING_STARTED.md#cli-query-history-tet-qhist).

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

More examples and roadmap: [`GETTING_STARTED.md`](GETTING_STARTED.md).

## Documentation map

| Doc                                            | Contents                                                                               |
| ---------------------------------------------- | -------------------------------------------------------------------------------------- |
| [`GETTING_STARTED.md`](GETTING_STARTED.md)     | Phased checklist, verification, CLI history, what's next                               |
| [`docs/layout_v1.md`](docs/layout_v1.md)       | Wire layout, superblock, chunk index, footer history                                   |
| [`docs/query_engine.md`](docs/query_engine.md) | Planning, execution strategies, spill allowlist, JSON security                         |
| [`fixtures/README.md`](fixtures/README.md)     | Test tensors, convert fixtures, [`small/tet/`](fixtures/small/tet/) verify/query smoke |

## Design stance (short)

**Partial I/O is the default case** — mmap payload regions, touch only chunks that intersect the selection, parallel decode across disjoint tiles. Full-array loads into RAM are not required for planning or tier-A/B aggregates.

**JSON is the control plane**, not the storage encoding: hosts validate input, cap size, and enforce spill path policy ([security notes](docs/query_engine.md#json-security-input-and-output)).

**Non-goals (v1):** SQL-on-files, arbitrary codec plugins, GPU codecs in the file format. GPU use is “materialize on CPU (or spill), then copy to device” in bindings—see Phase 10 in [`GETTING_STARTED.md`](GETTING_STARTED.md). **Phases 8–9** are **done** (verify/repair, wire dtypes through **`u64`/`f16`**, named axes, coord labels, histogram edges, QC counts, covariance/correlation, **`tet export`**). **Next:** Phase 10 GPU hooks, Phase 11 Python wheels + narrow C ABI; the layout spec is the portable floor.

## Library use

```toml
[dependencies]
tetration = "0.1"
```

```rust
use tetration::prelude::*;
// TetWriterSession, TetFile, parse_query_json, execute_query_json, verify_tet_file, …
```

### Rust API by phase

Phases **0–9** below are **done** in this repo unless marked _later_. Full checklist: [`GETTING_STARTED.md`](GETTING_STARTED.md).

| Phase  | Status  | You get                                                                                              | Rust / CLI entry points                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| ------ | ------- | ---------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **1**  | Done    | **Create `.tet` on disk** (superblock, catalog, chunk payloads)                                      | [`create_empty_v1_file`](https://docs.rs/tetration/latest/tetration/layout/fn.create_empty_v1_file.html), [`write_raw_array_file`](https://docs.rs/tetration/latest/tetration/catalog/fn.write_raw_array_file.html), [`write_one_chunk_raw_file`](https://docs.rs/tetration/latest/tetration/catalog/fn.write_one_chunk_raw_file.html) — low-level; tests/fixtures use these directly                                                                                                            |
| **4**  | Done    | **Query engine** (plan, mmap decode, streaming fold, spill, tier-C stats)                            | [`plan_query_with_tet_mmap_ex`](https://docs.rs/tetration/latest/tetration/query/fn.plan_query_with_tet_mmap_ex.html), [`build_execution_preview`](https://docs.rs/tetration/latest/tetration/query/fn.build_execution_preview.html); CLI `tet query … -x`                                                                                                                                                                                                                                       |
| **5**  | Done    | **Import** HDF5 / NetCDF / Zarr v3 → `.tet`                                                          | [`tetration::convert`](https://docs.rs/tetration/latest/tetration/convert/index.html); CLI `tet convert`                                                                                                                                                                                                                                                                                                                                                                                         |
| **7**  | Done    | **Embedder session types** — queue datasets + footer metadata/history, commit, mmap-open for queries | [`TetWriterSession`](https://docs.rs/tetration/latest/tetration/catalog/struct.TetWriterSession.html), [`TetDatasetWrite`](https://docs.rs/tetration/latest/tetration/catalog/struct.TetDatasetWrite.html), [`TetFile`](https://docs.rs/tetration/latest/tetration/catalog/struct.TetFile.html), [`execute_query_json`](https://docs.rs/tetration/latest/tetration/query/fn.execute_query_json.html) — re-exported in [`prelude`](https://docs.rs/tetration/latest/tetration/prelude/index.html) |
| **8**  | Done    | **File health** + wire dtypes through **`u64` / `f16`**                                              | [`verify_tet_file`](https://docs.rs/tetration/latest/tetration/verify/fn.verify_tet_file.html), [`repair_tet_file`](https://docs.rs/tetration/latest/tetration/repair/fn.repair_tet_file.html); CLI `tet verify` / `tet repair` (`VerifyOptions::deep_decode` = `tet verify --deep`)                                                                                                                                                                                                             |
| **9**  | Done    | Named axes, coord **label** selection, QC counts, covariance/correlation, **Zarr export**            | Resolved at plan time inside `execute_query_*`; CLI `tet export`; details in [`docs/query_engine.md`](docs/query_engine.md)                                                                                                                                                                                                                                                                                                                                                                      |
| **10** | _Later_ | Optional GPU materialize after CPU decode                                                            | —                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| **11** | _Later_ | Python wheels (+ narrow C ABI when needed)                                                           | Separate repo; pins crates.io `tetration`                                                                                                                                                                                                                                                                                                                                                                                                                                                        |

**Typical embedder flow (Phase 7 on top of Phase 1 + 4):**

1. **Write** — `TetWriterSession::create` → `push_dataset` → `commit()` (or `commit_with_fill` for streaming tiles).
2. **Read / aggregate** — `TetFile::open` → `execute_query_json` → [`QueryResponse`](https://docs.rs/tetration/latest/tetration/query/struct.QueryResponse.html) (CLI: [`format_query_response`](https://docs.rs/tetration/latest/tetration/query/fn.format_query_response.html) for stdout).

**Examples:** `cargo run --example create_and_query`, `session_write`, `inspect_catalog` (see [`src/catalog/session.rs`](src/catalog/session.rs)). **Smoke fixtures:** `cargo run --example gen_small_tet_fixtures` → [`fixtures/small/tet/`](fixtures/small/tet/).

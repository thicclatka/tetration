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
- **JSON query + execute** — flat query documents, streaming reductions, tier-C stats, spill export ([`docs/query_engine.md`](docs/query_engine.md)).
- **Import** — `tet convert` from HDF5, NetCDF, Zarr v3 directory stores.
- **CLI** — `tet info`, `tet verify`, `tet query`, `tet qhist`, `tet convert`.

Dtypes on disk and in query execution: **`f32`**, **`f64`**, **`i32`**, **`i64`**.

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

tet info volume.tet
tet verify volume.tet
tet query '{"dataset":"<name>","mean":[]}' -t volume.tet -x -q   # <name> from info output
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

| Command                                        | Alias  | Role                                                       |
| ---------------------------------------------- | ------ | ---------------------------------------------------------- |
| [`tet info`](#tet-info) `<path.tet>`           | —      | Summarize a file (default: dataset table)                  |
| [`tet verify`](#tet-verify) `<path.tet>`       | —      | Layout health check (exit 1 on failure); `--json` / `-q`   |
| [`tet query`](#tet-query) `[QUERY]`            | `q`    | Validate JSON; optional catalog + execute against `-t`     |
| [`tet qhist`](#tet-qhist) `[list\|run]`        | `hist` | Recent queries (platform cache; **not** the `.tet` footer) |
| [`tet convert`](#tet-convert) `<in> <out.tet>` | —      | HDF5 / NetCDF / Zarr v3 → `.tet`                           |

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

| Flag        | Effect                                                |
| ----------- | ----------------------------------------------------- |
| _(default)_ | Human-readable check list + summary                   |
| `--json`    | Pretty JSON [`TetVerifyReport`](src/verify/report.rs) |
| `-q`        | One line (`status=ok` / `failed`)                     |

Exit code **1** when verification fails (CI-friendly).

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

More examples and roadmap: [`GETTING_STARTED.md`](GETTING_STARTED.md).

## Documentation map

| Doc                                            | Contents                                                       |
| ---------------------------------------------- | -------------------------------------------------------------- |
| [`GETTING_STARTED.md`](GETTING_STARTED.md)     | Phased checklist, verification, CLI history, what's next       |
| [`docs/layout_v1.md`](docs/layout_v1.md)       | Wire layout, superblock, chunk index, footer history           |
| [`docs/query_engine.md`](docs/query_engine.md) | Planning, execution strategies, spill allowlist, JSON security |
| [`fixtures/README.md`](fixtures/README.md)     | Test tensors, convert fixtures, local bench sizes              |

## Design stance (short)

**Partial I/O is the default case** — mmap payload regions, touch only chunks that intersect the selection, parallel decode across disjoint tiles. Full-array loads into RAM are not required for planning or tier-A/B aggregates.

**JSON is the control plane**, not the storage encoding: hosts validate input, cap size, and enforce spill path policy ([security notes](docs/query_engine.md#json-security-input-and-output)).

**Non-goals (v1):** SQL-on-files, arbitrary codec plugins, GPU codecs in the file format. GPU use is “materialize on CPU (or spill), then copy to device” in bindings—see Phase 10 in [`GETTING_STARTED.md`](GETTING_STARTED.md). **Phase 8** (file health + `f32`–`i16` wire dtypes) is done; **Phase 9** is named axes and richer query ops; Python wheels and a narrow C ABI are Phase 11; the layout spec is the portable floor.

## Library use

```toml
[dependencies]
tetration = "0.1"
```

```rust
use tetration::prelude::*;
// or: tetration::layout::mmap_file_read, tetration::query::{parse_query_json, …}
```

Embedders get the full [`QueryResponse`](https://docs.rs/tetration/latest/tetration/query/struct.QueryResponse.html); the CLI uses [`format_query_response`](https://docs.rs/tetration/latest/tetration/query/fn.format_query_response.html) for stdout modes. **Session API:** [`TetWriterSession`](https://docs.rs/tetration/latest/tetration/catalog/struct.TetWriterSession.html), [`TetFile`](https://docs.rs/tetration/latest/tetration/catalog/struct.TetFile.html), [`execute_query_json`](https://docs.rs/tetration/latest/tetration/query/fn.execute_query_json.html) (or [`prelude`](https://docs.rs/tetration/latest/tetration/prelude/index.html)). **Examples:** `cargo run --example create_and_query`, `inspect_catalog`, `session_write`. **File health:** `tet verify` / `tet repair` (`--deep` for full chunk decode); see [GETTING_STARTED.md — Phase 9](GETTING_STARTED.md#phase-9--query-ops--interchange-later) for query-semantics next steps.

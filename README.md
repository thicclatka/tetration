# Tetration

[![Crates.io](https://img.shields.io/crates/v/tetration.svg)](https://crates.io/crates/tetration)
[![docs.rs](https://img.shields.io/docsrs/tetration)](https://docs.rs/tetration)
![Build](https://github.com/thicclatka/tetration/workflows/Build/badge.svg)
![Rust](https://img.shields.io/badge/rust-1.95-orange.svg)

[_For those who are more cur..._](https://bookshop.org/p/books/book-of-numbers-a-novel-joshua-cohen/af5aa739b0fac506?ean=9780812986655&next=t)

**_STILL IN DEVELOPMENT, PRE 0.1.0 STATE_**

**HDF5-shaped** persistence (many large arrays in one durable file), **Zarr-shaped** chunking (regular grid, per-chunk compression, parallel I/O)—in a **single mmap-friendly `.tet` file`**, not a directory of shard blobs.

## What it does today (v1)

- **On-disk layout** — superblock, dataset directory, chunk index, raw or zstd payloads ([`docs/layout_v1.md`](docs/layout_v1.md)).
- **Mmap + read planning** — logical slices → chunk coordinates → [`ReadPlan`](https://docs.rs/tetration/latest/tetration/struct.ReadPlan.html).
- **JSON query + execute** — flat query documents, streaming reductions, tier-C stats, spill export ([`docs/query_engine.md`](docs/query_engine.md)).
- **Import** — `tet convert` from HDF5, NetCDF, Zarr v3 directory stores.
- **CLI** — `tet info`, `tet query`, `tet qhist`, `tet convert`.

Dtypes on disk and in query execution: **`f32`**, **`f64`**, **`i32`**, **`i64`**.

## Quick start

```bash
git clone git@github.com:thicclatka/tetration.git
cd path/to/tetration
cargo build

tet info data.tet
tet query '{"dataset":"f32","mean":[]}' -t data.tet -x -q
tet convert volume.h5 volume.tet
```

**Daily driver:** plan + execute with readable stdout:

```bash
tet query q.json -t data.tet -x -q              # one-line aggregate
tet query q.json -t data.tet -x --format stats  # slim JSON (no chunk list)
tet query q.json -t data.tet --format plan      # catalog + read_plan only
```

Query JSON is **flat** (e.g. `"mean": []`, `"spill": "slice.bin"`); nested `"operation"` objects are rejected. Details: [query document](docs/query_engine.md#query-document-json).

## `tet` commands (summary)

| Command                      | Role                                                                              |
| ---------------------------- | --------------------------------------------------------------------------------- |
| `tet info <file.tet>`        | Dataset table (default); `--json`, `--grep`, section flags                        |
| `tet query [QUERY]`          | Validate JSON; `-t` file; `-x` execute; `--format full\|json\|stats\|plan\|quiet` |
| `tet qhist`                  | Platform query history (`list`, `run N`, filters); `hist` alias                   |
| `tet convert <in> <out.tet>` | HDF5 / NetCDF / Zarr v3 → `.tet` (`--jobs 0` = auto)                              |

More flags, env vars, and examples: [`GETTING_STARTED.md`](GETTING_STARTED.md) (phased roadmap + [`tet qhist`](GETTING_STARTED.md#cli-query-history-tet-qhist)). Contributor ops: [`AGENTS.md`](AGENTS.md).

## Documentation map

| Doc                                            | Contents                                                       |
| ---------------------------------------------- | -------------------------------------------------------------- |
| [`GETTING_STARTED.md`](GETTING_STARTED.md)     | Phased checklist, verification, CLI history, what's next       |
| [`docs/layout_v1.md`](docs/layout_v1.md)       | Wire layout, superblock, chunk index, footer history           |
| [`docs/query_engine.md`](docs/query_engine.md) | Planning, execution strategies, spill allowlist, JSON security |
| [`fixtures/README.md`](fixtures/README.md)     | Test tensors, convert fixtures, local bench sizes              |
| [docs.rs](https://docs.rs/tetration)           | Rust API reference                                             |

## Design stance (short)

**Partial I/O is the default case** — mmap payload regions, touch only chunks that intersect the selection, parallel decode across disjoint tiles. Full-array loads into RAM are not required for planning or tier-A/B aggregates.

**JSON is the control plane**, not the storage encoding: hosts validate input, cap size, and enforce spill path policy ([security notes](docs/query_engine.md#json-security-input-and-output)).

**Non-goals (v1):** SQL-on-files, arbitrary codec plugins, GPU codecs in the file format. GPU use is “materialize on CPU (or spill), then copy to device” in bindings—see Phase 9 in [`GETTING_STARTED.md`](GETTING_STARTED.md). Python wheels and a narrow C ABI are planned (Phase 10); the layout spec is the portable floor.

## Library use

```toml
[dependencies]
tetration = "0.1"
```

```rust
use tetration::{mmap_file_read, parse_query_json, plan_query_with_tet_mmap_ex, validate_query};
```

Embedders get the full [`QueryResponse`](https://docs.rs/tetration/latest/tetration/struct.QueryResponse.html); the CLI uses [`format_query_response`](https://docs.rs/tetration/latest/tetration/fn.format_query_response.html) for stdout modes.

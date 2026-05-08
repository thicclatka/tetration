# Tetration

[![Crates.io](https://img.shields.io/crates/v/zahirscan.svg)](https://crates.io/crates/zahirscan)
[![docs.rs](https://img.shields.io/docsrs/zahirscan)](https://docs.rs/zahirscan)
![Build](https://github.com/thicclatka/zahirscan/workflows/Build/badge.svg)
![Rust](https://img.shields.io/badge/rust-1.95-orange.svg)

[_For those who are more cur_](https://bookshop.org/p/books/book-of-numbers-a-novel-joshua-cohen/af5aa739b0fac506?ean=9780812986655&next=t)

New file format for tensors.

**Naming:** the Rust **library** on crates.io is **`tetration`** (`use tetration::…` in code, `tetration` in `Cargo.toml` dependencies). The **command-line tool** that ships in the same package is **`tet`**—short, typed often, and distinct from the library name so docs and issues can say “fix in tetration” vs “run `tet query`” without ambiguity.

## Vision: a modern, mmap-first n-dimensional store

The goal is not to clone HDF5’s API or feature matrix, but to occupy a similar _role_ in the stack: a **durable container for large, structured numeric data** (tensors, grids, tables-as-arrays) that you can treat as part of the address space instead of always streaming through a library. Where classic HDF5 leans on rich metadata trees, plugins, and a long compatibility tail, this project aims at **predictable on-disk layout, cheap random access, and parallelism-friendly chunking** from day one.

### What “like HDF5” means here

- **n-dimensional datasets** — logical arrays with shape, dtype, and optional attributes; multiple named arrays in one file or family of files.
- **Rich metadata headers (process and lineage)** — file- and dataset-level metadata are first-class and can be **verbose about how data came to be**: run ids, tool and library versions, parameters, timestamps, operator graphs, and **relations between datasets**. For example, when one dataset is a **morph** (resample, warp, reprojection, broadcasted view materialized to storage, or any other defined transform) of another, the header can record **parent references**, transform kind, and enough context to reconstruct **history** or audit trails without relying on a separate notebook or ad hoc filenames alone.
- **Chunked storage** — data are not one giant contiguous slab you must read end-to-end; they are split into **fixed, predetermined chunks** (regular grid in index space) so offsets and sizes are computable without scanning.
- **mmap-friendly** — a reader can **memory-map** payload (and key index regions) and dereference slices that correspond to chunks or sub-chunks, relying on the OS page cache instead of bespoke buffering for many access patterns.
- **Multi-threaded by construction** — because chunk boundaries are known and stable, **different threads (or tasks) can touch disjoint chunk ranges** without coordinating on a mutable global cursor; writers still need clear rules (e.g. exclusive file creation, append-only regions, or per-chunk serialization) so the format stays coherent.
- **JSON-described queries and basic operations** — callers can **submit JSON documents** that name a dataset, describe **slices / hyperslabs / stepping**, request **metadata**, or ask for a small, fixed set of **built-in operations** (aggregations, dtype views, bounds checks, etc.). That gives you a language-agnostic control plane—HTTP bodies, CLI stdin, worker queues—without requiring a custom binary protocol for everyday reads and light server-side work.

### JSON query plane (how it fits)

The on-disk layout stays binary and mmap-oriented; JSON sits **above** it as a **stable, versioned interchange** for “what to read or compute,” not as the storage encoding of tensor bytes. A query document might carry `layout_version`, `dataset`, `selection` (per-axis start/stop/step or chunk ranges), optional `operation` (e.g. `sum` / `mean` along named axes), and `output` hints (inline JSON stats vs spill to a new array handle). The implementation will validate requests, map them to chunk index sets, then execute I/O and ops in parallel where selections are disjoint. Over time the set of allowed JSON fields and operations can grow while older clients keep working against documented schema levels.

### What this will entail (technical direction)

1. **On-disk layout** — a small header plus **index tables** that map `(dataset_id, chunk_index_0, …, chunk_index_{n-1})` to byte ranges (offset, length), optionally compressed per chunk. Layout should be **64-bit aligned** where it matters for `bytemuck` / zero-copy views and for mmap page granularity.
2. **Deterministic chunking** — chunk shape (or chunk grid) is part of the dataset definition; clients can **precompute** which file bytes back a logical hyperslab. That enables **Rayon**-style parallel decode or memcpy over disjoint regions without a central “chunk allocator” at read time.
3. **Compression as a per-chunk decision** — e.g. **zstd** on chunk payloads keeps cold data small while preserving mmap + parallel decode for uncompressed or lightly compressed cases; the index must store compressed vs raw sizes so mmap views can target decompressed staging buffers when needed.
4. **Serialization / interchange** — **rkyv** and **serde** carry not only schema, shapes, dtypes, and user attributes but also **optional, structured provenance**: links between datasets (derivation, “morphed from”), ordered **history** or append-only **event lists**, and process blobs small enough to sit in header space or spill to dedicated metadata chunks. The numeric bulk stays **hot** and mmap-simple; metadata can be **cold** yet still rich enough for reproducibility and for UIs or services that explain _why_ two arrays belong together.
5. **Ergonomics vs HDF5** — explicit non-goals early on (arbitrary plugin ecosystem, full SQL-on-files, etc.) in favor of **Rust-first safety**, **reproducible byte layout**, and **documentation of concurrency semantics** (who may write which chunks, when).
6. **JSON query spec** — a **documented JSON schema** (or equivalent contract) for query and operation requests: parsing via **serde**, strict validation, clear error responses, and mapping from logical selections to physical chunk reads—again enabling **Rayon** where the plan says the work is embarrassingly parallel.

### Who this is for

Teams that want **HDF5-like persistence** (big arrays, partial I/O, shared analysis) but are willing to adopt a **smaller, opinionated format** optimized for **mmap + parallel chunk read/write** on local disks or object-store–backed block devices, with a clear path to binding in Rust (and later other languages via a documented layout spec).

### Multi-language embedding (calling into Tetration from Python and beyond)

The Rust crate **`tetration`** stays the **reference implementation**, but other languages are first-class targets. The plan is layered so bindings stay maintainable:

1. **Documented on-disk layout** — A versioned **file spec** (magic, endianness, header, chunk index, compression flags) lets any language implement a **standalone reader** if needed; it also locks semantics so FFI and Rust cannot drift silently.

2. **Small, stable C ABI** — Expose a **`cdylib`** with a narrow C API (e.g. open path, list or resolve datasets, read chunk bytes into caller-owned memory, query last error, close). C is the **portable FFI floor**: Python, Julia, Go, JVM, .NET, R, etc. can all call it via their usual native interop without depending on Rust ABI stability across toolchains.

3. **Python next** — A **`tetration`-py** style package (PyO3 / maturin) links the same core library and ships wheels that bundle the shared object. NumPy-friendly buffer protocols and thin wrappers follow from that stack.

4. **Control plane without in-process Rust** — The **JSON query** format and the **`tet`** CLI remain valid integration paths: other runtimes can shell out or HTTP-post JSON when a native binding is not worth shipping yet.

Together, **spec + C ABI + Python** cover “used as a library from other langs” without promising that every language gets a hand-written idiomatic wrapper on day one.

### CLI: `tet`

The **`tetration`** crate compiles both the library and a binary named **`tet`**. `cargo install tetration` (or `cargo build --release`) produces a `tet` executable; `default-run` is set to **`tet`**, so `cargo run -- …` runs the CLI without `--bin tet`. The CLI is the ergonomic front door for **JSON querying** and, later, **foreign-format conversion**; embedders link **`tetration`** as a normal dependency.

- **`tet query`** — Load a query document from **`--file` / `-f`** (path to JSON) or from **standard input** (omit `-f`, or pass `-`). The tool parses JSON, runs the same validation rules as the library, and prints a **pretty-printed JSON plan** (status, echoed `dataset`, selection axis count, optional `operation`, and a short message). Full execution against a mmap’d `.tet` file will attach here once the on-disk layout and chunk engine are implemented; until then the CLI is still useful for contract checks and scripting against the query schema.
- **`tet convert h5 <input.h5> <output.tet>`** and **`tet convert netcdf <input.nc> <output.tet>`** — Declared commands for **HDF5 → Tetration** and **NetCDF → Tetration** pipelines. They are **placeholders** today: they parse paths and exit with a clear “not implemented” explanation so CI and installers can ship one binary while importers are developed behind a stable `.tet` writer and optional native dependencies.

Examples:

```bash
cargo run -- query --file query.json
echo '{"dataset":"temperature","layout_version":1}' | cargo run -- query
cargo run -- convert h5 volumes.h5 volumes.tet
cargo run -- convert netcdf model.nc model.tet
```

Query JSON follows the shapes described above (`dataset`, optional `layout_version`, `selection` as per-axis `{ start, stop, step }`, optional `operation` such as `{ "mean": { "axes": ["time"] } }`, optional `output.preferred` for result delivery hints). Exact field evolution will stay aligned with the library types in `tetration::query`.

If that matches your use case, the crate is the place where those ideas become concrete types, layout version numbers, and eventually stability guarantees.

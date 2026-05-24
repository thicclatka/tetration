# Tetration

[![Crates.io](https://img.shields.io/crates/v/tetration.svg)](https://crates.io/crates/tetration)
[![docs.rs](https://img.shields.io/docsrs/tetration)](https://docs.rs/tetration)
![Build](https://github.com/thicclatka/tetration/workflows/Build/badge.svg)
![Rust](https://img.shields.io/badge/rust-1.95-orange.svg)

[_For those who are more cur..._](https://bookshop.org/p/books/book-of-numbers-a-novel-joshua-cohen/af5aa739b0fac506?ean=9780812986655&next=t)

New file format for tensors: **HDF5-shaped** in the stack (one durable home for many large arrays and metadata), **Zarr-shaped** in mechanics (regular chunk grid, per-chunk compression, parallel-friendly I/O)—but **one mmap-friendly file** instead of a directory store of chunk blobs.

**Naming:** the Rust **library** on crates.io is **`tetration`** (`use tetration::…` in code, `tetration` in `Cargo.toml` dependencies). The **command-line tool** that ships in the same package is **`tet`**—short, typed often, and distinct from the library name so docs and issues can say “fix in tetration” vs “run `tet query`” without ambiguity.

**Docs (v1 today):**

- On-disk layout — [`docs/layout_v1.md`](docs/layout_v1.md)
- JSON query + mmap engine — [`docs/query_engine.md`](docs/query_engine.md) (includes [operations roadmap](docs/query_engine.md#operations-roadmap-planned))

## Vision: a modern, mmap-first n-dimensional store

The goal is not to clone HDF5’s API or feature matrix, but to occupy a similar _role_ in the stack: a **durable container for large, structured numeric data** (tensors, grids, tables-as-arrays) that you can treat as part of the address space instead of always streaming through a library. Conceptually it is also close to **[Zarr](https://zarr.dev/)**: **chunk-addressed arrays** with **metadata-first** ergonomics—but **bundled into a single artifact** for mmap, versioning, and “email this file” workflows, rather than exposing chunk files across a filesystem or object-prefix layout. Where classic HDF5 leans on rich metadata trees, plugins, and a long compatibility tail, this project aims at **predictable on-disk layout, cheap random access, and parallelism-friendly chunking** from day one.

### What “too big to read” means here

Often a dataset is called “impossible to read” because something in the stack tries to **load the entire array into RAM** or **scan it end-to-end** when the work only needs a **hyperslab** or a **subset of chunks**. Tetration treats **partial access as the normal case**: **memory-map** large payload regions so the **OS page cache** supplies working sets, **map logical slices to chunk coordinates** so I/O tracks what you touched, and use **parallel reads** across disjoint chunks. That is the main performance story for **very large n-D numeric bulk**. Relational workloads (multiple tables, **join** semantics, SQL-style queries across datasets) stay **orthogonal**—they belong in callers or optional layers—not as distractions from **making huge files routinely usable**.

### What “like HDF5” means here

- **n-dimensional datasets** — logical arrays with shape, dtype, and optional attributes; multiple named arrays in one file or family of files.
- **Rich metadata headers (process and lineage)** — file- and dataset-level metadata are first-class and can be **verbose about how data came to be**: run ids, tool and library versions, parameters, timestamps, operator graphs, and **relations between datasets**. For example, when one dataset is a **morph** (resample, warp, reprojection, broadcasted view materialized to storage, or any other defined transform) of another, the header can record **parent references**, transform kind, and enough context to reconstruct **history** or audit trails without relying on a separate notebook or ad hoc filenames alone.
- **Chunked storage** — data are not one giant contiguous slab you must read end-to-end; they are split into **fixed, predetermined chunks** (regular grid in index space) so offsets and sizes are computable without scanning.
- **mmap-friendly** — a reader can **memory-map** payload (and key index regions) and dereference slices that correspond to chunks or sub-chunks, relying on the OS page cache instead of bespoke buffering for many access patterns.
- **Multi-threaded by construction** — because chunk boundaries are known and stable, **different threads (or tasks) can touch disjoint chunk ranges** without coordinating on a mutable global cursor; writers still need clear rules (e.g. exclusive file creation, append-only regions, or per-chunk serialization) so the format stays coherent.
- **JSON-described queries and basic operations** — callers can **submit JSON documents** that name a dataset, describe **slices / hyperslabs / stepping**, request **metadata**, or ask for a small, fixed set of **built-in operations** (aggregations, dtype views, bounds checks, etc.). That gives you a language-agnostic control plane—HTTP bodies, CLI stdin, worker queues—without requiring a custom binary protocol for everyday reads and light server-side work.

### Single-file Zarr (mental model)

[Zarr](https://zarr.dev/) is usually experienced as **chunked arrays plus JSON (or similar) metadata**, often spread across **many files or keys** in a store. Tetration keeps that **chunk grid and per-chunk compression story**—deterministic chunk indices, embarrassingly parallel reads, optional filters—but puts **payload, chunk index, and rich headers in one `.tet` file** you can mmap end-to-end. The JSON **query** layer is not a replacement for Zarr’s on-disk metadata tree; it is the **control plane** (“what to read or compute”) sitting above a **binary** layout, similar in spirit to how Zarr users think in JSON for structure while bytes live in separate chunk objects.

### JSON query plane (how it fits)

The on-disk layout stays binary and mmap-oriented; JSON sits **above** it as a **stable, versioned interchange** for “what to read or compute,” not as the storage encoding of tensor bytes. A query document carries `layout_version`, `dataset`, optional `selection` (per-axis start/stop/step), and at most one **top-level reduction key** (e.g. **`"mean": []`** for scalar, **`"mean": 0`** over axis 0, **`"quantile": { "q": 0.95, "axis": 0 }`** — streaming ops **`sum`**, **`mean`**, **`min`**, **`max`**, **`count`**, **`var`**, **`std`**, **`product`**, **`norm_l1`**, **`norm_l2`**, **`all_finite`**, **`any_nan`**, **`arg_min`**, **`arg_max`**; tier-C **`median`**, **`quantile`**, **`histogram`**; population **`var` / `std`**, `ddof = 0`), optional **`execution`** (**`memory_budget_bytes`**, **`memory_budget_percent`** where **100** = 100% of host RAM), and optional **`"spill": "path"`** for full logical decode to a caller file. Nested `"operation"` / `"output"` objects are not supported. The **`src/query/`** tree (`plan/`, `decode/`, `materialize/`, `fold/`, `dispatch.rs`, `engine/`) validates against a mmap’d catalog, builds a **`ReadPlan`**, resolves a memory budget (query JSON → per-file TIDX header → default **25%** of host RAM), and decodes **`f32`**, **`f64`**, **`i32`**, or **`i64`** tiles (raw or zstd) in **logical row-major** selection order; the **`tet`** CLI exposes this via **`--tet`** and **`--execute`**. Tier-A/B **`operation`** queries use **streaming fold**; tier-C ops materialize in RAM or temp spill; preview-only paths are **capped in memory** when the selection fits the budget; spill requests write dtype-native bytes via mmap. Multi-chunk materialize paths decode chunks in parallel (Rayon). Query JSON is **not executable**—hosts should still cap input size, validate spill paths, and treat responses as data when embedding in shells, HTML, or logs ([JSON security](docs/query_engine.md#json-security-input-and-output)).

### What this will entail (technical direction)

1. **On-disk layout** — a small header plus **index tables** that map `(dataset_id, chunk_index_0, …, chunk_index_{n-1})` to byte ranges (offset, length), optionally compressed per chunk. Layout should be **64-bit aligned** where it matters for `bytemuck` / zero-copy views and for mmap page granularity.
2. **Deterministic chunking** — chunk shape (or chunk grid) is part of the dataset definition; clients can **precompute** which file bytes back a logical hyperslab. That enables **Rayon**-style parallel decode or memcpy over disjoint regions without a central “chunk allocator” at read time.
3. **Compression as a per-chunk decision** — e.g. **zstd** on chunk payloads keeps cold data small while preserving mmap + parallel decode for uncompressed or lightly compressed cases; the index must store compressed vs raw sizes so mmap views can target decompressed staging buffers when needed.
4. **Serialization / interchange** — **rkyv** and **serde** carry not only schema, shapes, dtypes, and user attributes but also **optional, structured provenance**: links between datasets (derivation, “morphed from”), ordered **history** or append-only **event lists**, and process blobs small enough to sit in header space or spill to dedicated metadata chunks. The numeric bulk stays **hot** and mmap-simple; metadata can be **cold** yet still rich enough for reproducibility and for UIs or services that explain _why_ two arrays belong together.
5. **Ergonomics vs HDF5 / directory Zarr** — explicit non-goals early on (arbitrary plugin ecosystem, full SQL-on-files, etc.) in favor of **Rust-first safety**, **reproducible byte layout**, **one-file distribution**, and **documentation of concurrency semantics** (who may write which chunks, when).
6. **JSON query spec** — a **documented JSON schema** (or equivalent contract) for query and operation requests: parsing via **serde**, strict validation, clear error responses, and mapping from logical selections to physical chunk reads—again enabling **Rayon** where the plan says the work is embarrassingly parallel.

### Who this is for

Teams that hit **datasets larger than RAM** (or larger than patience for full-file reads) and want **HDF5-like persistence** (big arrays, partial I/O, shared analysis) or **Zarr-like chunking** without managing a **directory or key-prefix store** of shards—instead a **single file** optimized for **mmap + parallel chunk read/write** on local disks or object-store–backed block devices, with a clear path to binding in Rust (and later other languages via a documented layout spec).

### GPUs and tensors (same file, optional path)

N-dimensional tensors and GPUs often show up together. Tetration does **not** need a separate “GPU file format.” The **on-disk layout stays mmap-first** for partial I/O and the OS page cache: you map what you need, touch the pages you read, and avoid pretending the whole array must sit in RAM at once.

When it is **time to use** a selection, a reader **materializes** it: read or mmap the relevant bytes, **decompress per chunk if needed**, then either **keep the result on the CPU** or **copy it to a GPU** (host-to-device). That is a **binding / library choice**, not a different wire format—one `.tet`, one index; only the **destination** changes.

**Why that helps:** workloads where the **file is much larger than RAM** but each step only needs a **hyperslab or batch of chunks** get cheap I/O on the CPU side, then can land the working set on a GPU for training or inference without an extra hand-rolled pipeline for “file → tensor → CUDA.”

**What moving to a GPU actually optimizes:** compute can read **device memory** at the bandwidth your kernels need; you can **overlap** transfers with work (for example async host-to-device while the previous batch still runs); frameworks can run fused ops on device tensors without shuttling through host memory again. **What it does not fix by itself:** decompression is still usually **CPU work** unless you add optional GPU-oriented codecs later; **GPU memory is finite**, so you still stream batches rather than loading unlimited data into VRAM.

**Choosing CPU vs GPU:** the practical rule is **explicit beats magic**—for example a `device` argument or an environment variable so users say “CUDA device 0” or “CPU only.” An **automatic** mode can **probe** whether GPU infrastructure is available (initialize CUDA or ask PyTorch, and so on) and **fall back to CPU** if not. Optional heuristics can skip the GPU for **tiny** selections where transfer overhead dominates. None of that belongs in the byte-for-byte file header as hard requirements; it lives in **APIs and docs**.

**Guardrails:** sensible defaults can cap **how much** is in flight to the GPU at once, limit **concurrent** copies, check **free VRAM** before committing, and **fall back** on out-of-memory. Power users should still get **knobs** (limits, thresholds, opt-in-only GPU) so production jobs stay predictable.

**Format details that play nicely with GPUs** (over time, in the spec): predictable **row-major** contiguous chunk payloads, **dtype** support that matches ML stacks (for example `f32` / `f16`), and **alignment** rules that make copies and optional fast paths (such as vendor direct-storage setups) less painful. Those help **everyone** who wants zero-copy or memcpy-friendly bytes, not only GPU users.

### Recording lineage (sessions and wrappers)

The format already treats **rich headers** and **optional history** as first-class (see **Serialization / interchange** above). A practical way to populate them—**in Rust and in Python**—is a **first-party writer or session type**: it wraps the real open / read / write APIs, **appends structured provenance events** in memory (what ran, with what parameters, parent dataset ids, and so on), and on **`commit` / `close`** serializes that list into the **file or dataset metadata** (or into **dedicated metadata chunks** if the log outgrows a small header).

**What “automatic” can mean here:** reliably **automatic only for work that goes through that API**—for example “wrote these chunks,” “applied this transform,” “copied lineage from parent.” **Cheap extras** on write are still worth doing without tracing: **tool versions**, **timestamps**, **git commit**, **hostname**, narrow **environment** snapshots.

**What is not a goal:** silently recording **every** statement in an arbitrary user script across arbitrary libraries; that needs heavy, fragile instrumentation and still misses subprocesses and side effects.

**Product discipline:** cap or **spill** long event lists so metadata stays bounded; use a **versioned, forward-compatible** event shape so old readers skip unknown fields; treat the log as **best-effort audit and reproducibility**, not tamper-proof provenance unless you add signing as a separate concern. Optional **explicit** hooks (`log_step`, context managers) sit well next to session defaults so power users can name steps the library cannot infer.

### Multi-language embedding (calling into Tetration from Python and beyond)

The Rust crate **`tetration`** stays the **reference implementation**, but other languages are first-class targets. The plan is layered so bindings stay maintainable:

1. **Documented on-disk layout** — A versioned **file spec** ([`docs/layout_v1.md`](docs/layout_v1.md): magic, endianness, superblock, dataset directory, chunk index, per-chunk codecs) lets any language implement a **standalone reader** if needed; it also locks semantics so FFI and Rust cannot drift silently.

2. **Small, stable C ABI** — Expose a **`cdylib`** with a narrow C API (e.g. open path, list or resolve datasets, read chunk bytes into caller-owned memory, query last error, close). C is the **portable FFI floor**: Python, Julia, Go, JVM, .NET, R, etc. can all call it via their usual native interop without depending on Rust ABI stability across toolchains.

3. **Python next** — A **`tetration`-py** style package (PyO3 / maturin) links the same core library and ships wheels that bundle the shared object. NumPy-friendly buffer protocols and thin wrappers follow from that stack.

4. **Control plane without in-process Rust** — The **JSON query** format and the **`tet`** CLI remain valid integration paths: other runtimes can shell out or HTTP-post JSON when a native binding is not worth shipping yet.

Together, **spec + C ABI + Python** cover “used as a library from other langs” without promising that every language gets a hand-written idiomatic wrapper on day one.

### CLI: `tet`

The **`tetration`** crate compiles both the library and a binary named **`tet`**. `cargo install tetration` (or `cargo build --release`) produces a `tet` executable; `default-run` is set to **`tet`**, so `cargo run -- …` runs the CLI without `--bin tet`. The CLI is the ergonomic front door for **JSON querying** and **foreign-format conversion**; embedders link **`tetration`** as a normal dependency.

- **`tet info <path.tet>`** — Catalog summary: default **dataset table**; **`--json`** for the full superblock + catalog dump; **`--grep`** / **`--dataset`** filters; **`--layout`**, **`--chunks`**, **`--history`**, **`--all`** sections; **`-q`** one-line summary.
- **`tet query [QUERY]`** — `QUERY` is a path to `.json`, inline JSON, `-`, or omit for stdin. Parses JSON with the same validation as the library; stdout format is **`--format full|json|stats|quiet`** (default **`full`**; **`-q`** = quiet one-liner). **Without `-t`**, **`plan_query_empty`** echoes the validated plan only. **With `-t path.tet`**, **`plan_query_with_tet_mmap`** adds **`catalog`** and **`read_plan`**. **With `-t`** and **`-x` / `--execute`**, the engine attaches **`execution`** (previews, budget, **`operation_*`**). Preview cap: **`--preview N`** (alias **`--preview-f32`**; default **64** for `full`/`json`, **0** for `stats`/`quiet` when omitted). The library still returns a full [`QueryResponse`](https://docs.rs/tetration/latest/tetration/struct.QueryResponse.html); embedders can call **`format_query_response`** for the same stdout modes.
- **`tet convert <input> <output.tet> [--jobs N]`** — Import **HDF5**, **NetCDF**, or **Zarr v3** directory stores into `.tet`. Format is detected from extension or file signature (`--jobs 0` = host parallelism, capped at 64). Parallel chunked read → streaming tile write; append convert provenance to the history footer. See [`fixtures/README.md`](fixtures/README.md) for tracked roundtrip fixtures and local stress sizes.

Examples:

```bash
cargo run -- info data.tet
cargo run -- query query.json
echo '{"dataset":"temperature","layout_version":1}' | cargo run -- query
cargo run -- query query.json -t data.tet
cargo run -- query query.json -t data.tet -x --preview 128
# daily driver: aggregates without chunk dump or previews:
cargo run -- query query-with-mean.json -t data.tet -x -q
cargo run -- query query-with-mean.json -t data.tet -x --format stats
cargo run -- convert volumes.h5 volumes.tet
cargo run -- convert model.nc model.tet
cargo run -- convert tensor_large/ tensor_large.tet --jobs 4
```

Query JSON follows the shapes described above (`dataset`, optional `layout_version`, `selection` as per-axis `{ start, stop, step }`, flat reduction keys such as `"min": []`, `"quantile": { "q": 0.95 }`, optional `execution` memory hints, optional `"spill": "slice.bin"`). Library entrypoints: **`plan_query_empty`**, **`plan_query_with_tet_mmap`**, **`ExecutionBudget`**, f32/f64 **`materialize_read_plan_*_le`** (+ parallel / `_into` twins), **`spill_read_plan_f32_le`** (see [`docs/query_engine.md`](docs/query_engine.md)).

If that matches your use case, the crate is the place where those ideas become concrete types, layout version numbers, and eventually stability guarantees.

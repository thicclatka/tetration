# Tetration on-disk layout — version 1

This document describes the **v1** `.tet` container: a fixed **32-byte superblock** at offset 0, followed (when `dataset_count > 0`) by a **dataset directory blob**, an **8-byte-aligned** **chunk index** region, and **raw chunk payloads**.

## File map

Byte offsets increase left → right. All integers are **little-endian**.  
`align8(n) = (n + 7) & !7` (round up to an 8-byte boundary).

### Regions at a glance

| # | Region | Starts at | Size | When present |
| - | ------ | --------- | ---- | ------------ |
| 1 | **Superblock** | `0` | `32` B fixed | always |
| 2 | **Dataset directory** | `32` | `8 + dataset_blob_len` | `dataset_count > 0` |
| 3 | **Padding** | `40 + dataset_blob_len` | `0 … 7` B | only bytes needed so the next region is 8-byte aligned |
| 4 | **Chunk index** | `chunk_index_offset` (= `align8(40 + dataset_blob_len)`) | `chunk_index_length` (= `32 + entry_count × 104`) | `dataset_count > 0` |
| 5 | **Chunk payloads** | each `payload_offset` from index | each `stored_byte_len` | one span per index row; may be non-contiguous in general |

**Empty file** (`dataset_count = 0`): regions 2–5 are absent; `chunk_index_offset = 32`, `chunk_index_length = 0`; the file may end at byte 32.

**Populated file**: the reference writer packs payloads after the index, but v1 only requires `payload_offset + stored_byte_len ≤ file_len` for every row.

**How pieces connect:** the superblock’s `chunk_index_offset` / `chunk_index_length` bound region ④. Each index row’s `payload_offset` points at one payload span in region ⑤. Dataset metadata in region ② supplies `shape`, `chunk_shape`, and `dtype` for interpreting those bytes (see [query engine](query_engine.md) for read planning).

## Endianness and alignment

- All multi-byte integers are **little-endian** (`u32`, `u64`).
- The superblock is **32 bytes**. Chunk index entries and payload offsets assume **8-byte alignment** for index base offsets computed from the dataset directory.

## Magic and `layout_version`

- Bytes `0..4` must be ASCII **`TETR`**.
- Bytes `4..8` are **`layout_version`** (`u32` LE). Only **`1`** is defined today.

This matches the optional `layout_version` field in JSON query documents (`tetration::query::QueryDocument`).

## Superblock v1 (32 bytes)

| Offset | Size | Type     | Field                | Notes                                                                           |
| ------ | ---- | -------- | -------------------- | ------------------------------------------------------------------------------- |
| 0      | 4    | `[u8;4]` | `magic`              | Must be `TETR`.                                                                 |
| 4      | 4    | `u32`    | `layout_version`     | Must be `1`.                                                                    |
| 8      | 4    | `u32`    | `dataset_count`      | Number of dataset records in the directory blob; **0** means no directory blob. |
| 12     | 4    | `u32`    | `flags`              | Reserved; write **0**.                                                          |
| 16     | 8    | `u64`    | `chunk_index_offset` | Byte offset to the **chunk index** region (see below).                          |
| 24     | 8    | `u64`    | `chunk_index_length` | Length in bytes of the chunk index region.                                      |

### Empty file (`dataset_count = 0`)

- `chunk_index_offset = 32`
- `chunk_index_length = 0`
- The file may end at byte 32 (no trailing bytes required).

Readers must ensure `chunk_index_offset + chunk_index_length` fits in the file and does not overflow.

**Typical populated file** (contrast with empty above): superblock with `dataset_count = N`, dataset records at offset 40, chunk index at `align8(40 + dataset_blob_len)`, payloads referenced by index rows.

## Dataset directory (only when `dataset_count > 0`)

When `dataset_count > 0`, bytes starting at offset **32** are:

| Offset | Size | Field              | Notes                                                                                   |
| ------ | ---- | ------------------ | --------------------------------------------------------------------------------------- |
| 32     | 8    | `dataset_blob_len` | Length in bytes of the concatenated **dataset records** that immediately follow.        |
| 40     | \*   | `dataset_records`  | Total size = `dataset_blob_len`. Must contain exactly `dataset_count` records in order. |

### Dataset record (concatenated, variable length per record)

Each record is:

| Field         | Type            | Notes                                                                                                                                                                                |
| ------------- | --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `name_len`    | `u32` LE        | Byte length of UTF-8 `name` that follows.                                                                                                                                            |
| `dtype`       | `u32` LE        | `1` = IEEE754 **`f32`**, row-major within each chunk (more later).                                                                                                                   |
| `ndim`        | `u32` LE        | Rank in `1 … 8`.                                                                                                                                                                     |
| `reserved`    | `u32` LE        | Write **0**.                                                                                                                                                                         |
| `name`        | `[u8]`          | UTF-8 of length `name_len`.                                                                                                                                                          |
| _padding_     | `0…7`           | Zero bytes so the **total byte length from the start of this record** to the first byte of `shape[0]` is a multiple of **8**.                                                        |
| `shape`       | `ndim × u64` LE | Global array shape.                                                                                                                                                                  |
| `chunk_shape` | `ndim × u64` LE | Chunk size along each axis (must tile `shape`; `write_one_chunk_raw_file` requires a single tile `chunk_shape == shape`; `write_raw_array_file` supports regular multi-chunk grids). |

Records are concatenated in catalog order; `dataset_id` in the chunk index is the **0-based index** into this list (`0` = first record).

## Chunk index region

- Starts at `chunk_index_offset`, which **must** equal `align8(40 + dataset_blob_len)` when `dataset_count > 0`.
- `align8(n) = (n + 7) & !7`.

### Chunk index header (32 bytes)

| Offset | Size | Field           | Notes                         |
| ------ | ---- | --------------- | ----------------------------- |
| 0      | 4    | magic           | ASCII **`TIDX`**.             |
| 4      | 4    | `index_version` | `u32` LE; must be **1**.      |
| 8      | 8    | `entry_count`   | Number of fixed-size entries. |
| 16     | 16   | reserved        | Write **0**.                  |

The total chunk index byte length must be exactly:

`32 + entry_count * entry_wire_len`

### Chunk index entry (fixed `entry_wire_len` bytes, LE)

Layout (see `ChunkIndexEntryV1::WIRE_LEN` in `src/catalog/index.rs`):

- `dataset_id`: `u64`
- `chunk_index[8]`: eight `u64` values — chunk grid coordinates `i0 … i7`; unused slots are **0** (rank comes from the dataset record’s `ndim`).
- `payload_offset`: `u64` — file offset to **stored** bytes for this chunk.
- `raw_byte_len`: `u64` — logical uncompressed size of the chunk tensor in bytes.
- `stored_byte_len`: `u64` — bytes on disk at `payload_offset`.
- `codec`: `u32` — **`0`** = raw copy (`stored_byte_len` **must** equal `raw_byte_len`). **`1`** = **zstd**–compressed payload at `payload_offset`; `stored_byte_len` is the compressed size on disk and **`raw_byte_len`** is the uncompressed tensor size in bytes after decode.
- `reserved`: `u32` — **0**.

Each entry is **104 bytes** on the wire (`ChunkIndexEntryV1::WIRE_LEN`).

Payload bytes must lie fully inside the file: `payload_offset + stored_byte_len ≤ file_len`.

**Index on disk:** `TIDX` header (32 B) then `entry_count` fixed-size rows (104 B each). See field list above.

**Reader resolution** (how a query reaches bytes; details in [query engine](query_engine.md)):

```mermaid
flowchart LR
    Q["dataset name"] --> D["dataset_id"]
    D --> G["chunk coords"]
    G --> L["index row"]
    L --> R["mmap payload"]
```

Each index row’s `payload_offset` selects the byte span in region ⑤.

### Per-chunk payload codecs

| Codec | `stored_byte_len` vs `raw_byte_len` | On-disk bytes |
| ----- | ----------------------------------- | ------------- |
| **0** raw | must be equal | tensor payload (LE `f32` in v1 writers) |
| **1** zstd | `stored` = compressed, `raw` = decoded size | zstd frame; decode to `raw_byte_len` bytes |

See also [`query_engine.md`](query_engine.md) for how the query engine materializes planned chunks.

## Reference subset (current Rust writer)

The `write_one_chunk_raw_file` helper in `tetration::catalog` writes exactly **one** dataset and **one** chunk: `chunk_shape` must equal `shape` so the chunk grid has a single tile; payloads are always **raw** (`codec = 0`).

`write_raw_array_file` / `RawArrayWrite` accept per-chunk **`chunk_codec`**: compare to **`CHUNK_PAYLOAD_CODEC_V1.raw`** (**0**, raw tiles) or **`CHUNK_PAYLOAD_CODEC_V1.zstd`** (**1**, zstd-compressed frames; `stored_byte_len` may differ from `raw_byte_len`). The Rust API exposes this as the `ChunkPayloadCodecV1` struct plus the `CHUNK_PAYLOAD_CODEC_V1` constant in `tetration::catalog`.

## Concurrency (informative)

Writers should use **exclusive create** or clearly documented append rules before parallel writers touch payloads. v1 does not define locking; see the main **README** for the long-term concurrency story.

# Tetration on-disk layout — version 1

This document describes the **v1** `.tet` container: a fixed **32-byte superblock** at offset 0, followed (when `dataset_count > 0`) by a **dataset directory blob**, an **8-byte-aligned** **chunk index** region, and **raw chunk payloads**.

## Endianness and alignment

- All multi-byte integers are **little-endian** (`u32`, `u64`).
- The superblock is **32 bytes**. Chunk index entries and payload offsets assume **8-byte alignment** for index base offsets computed from the dataset directory.

## Magic and `layout_version`

- Bytes `0..4` must be ASCII **`TETR`**.
- Bytes `4..8` are **`layout_version`** (`u32` LE). Only **`1`** is defined today.

This matches the optional `layout_version` field in JSON query documents (`tetration::query::QueryDocument`).

## Superblock v1 (32 bytes)

| Offset | Size | Type     | Field                  | Notes                                                                               |
| ------ | ---- | -------- | ---------------------- | ----------------------------------------------------------------------------------- |
| 0      | 4    | `[u8;4]` | `magic`                | Must be `TETR`.                                                                     |
| 4      | 4    | `u32`    | `layout_version`       | Must be `1`.                                                                       |
| 8      | 4    | `u32`    | `dataset_count`        | Number of dataset records in the directory blob; **0** means no directory blob.   |
| 12     | 4    | `u32`    | `flags`                | Reserved; write **0**.                                                             |
| 16     | 8    | `u64`    | `chunk_index_offset`   | Byte offset to the **chunk index** region (see below).                             |
| 24     | 8    | `u64`    | `chunk_index_length`   | Length in bytes of the chunk index region.                                         |

### Empty file (`dataset_count = 0`)

- `chunk_index_offset = 32`
- `chunk_index_length = 0`
- The file may end at byte 32 (no trailing bytes required).

Readers must ensure `chunk_index_offset + chunk_index_length` fits in the file and does not overflow.

## Dataset directory (only when `dataset_count > 0`)

When `dataset_count > 0`, bytes starting at offset **32** are:

| Offset | Size | Field                 | Notes                                                                                   |
| ------ | ---- | --------------------- | --------------------------------------------------------------------------------------- |
| 32     | 8    | `dataset_blob_len`    | Length in bytes of the concatenated **dataset records** that immediately follow.       |
| 40     | *    | `dataset_records`     | Total size = `dataset_blob_len`. Must contain exactly `dataset_count` records in order. |

### Dataset record (concatenated, variable length per record)

Each record is:

| Field        | Type     | Notes                                                                 |
| ------------ | -------- | --------------------------------------------------------------------- |
| `name_len`   | `u32` LE | Byte length of UTF-8 `name` that follows.                             |
| `dtype`      | `u32` LE | `1` = IEEE754 **`f32`**, row-major within each chunk (more later).    |
| `ndim`       | `u32` LE | Rank in `1 … 8`.                                                      |
| `reserved`   | `u32` LE | Write **0**.                                                          |
| `name`       | `[u8]`   | UTF-8 of length `name_len`.                                           |
| *padding*    | `0…7`    | Zero bytes so the **total byte length from the start of this record** to the first byte of `shape[0]` is a multiple of **8**. |
| `shape`      | `ndim × u64` LE | Global array shape.                                           |
| `chunk_shape`| `ndim × u64` LE | Chunk size along each axis (v1 reference writer requires one chunk). |

## Chunk index region

- Starts at `chunk_index_offset`, which **must** equal `align8(40 + dataset_blob_len)` when `dataset_count > 0`.
- `align8(n) = (n + 7) & !7`.

### Chunk index header (32 bytes)

| Offset | Size | Field           | Notes                          |
| ------ | ---- | --------------- | ------------------------------ |
| 0      | 4    | magic           | ASCII **`TIDX`**.              |
| 4      | 4    | `index_version` | `u32` LE; must be **1**.       |
| 8      | 8    | `entry_count`   | Number of fixed-size entries. |
| 16     | 16   | reserved        | Write **0**.                   |

The total chunk index byte length must be exactly:

`32 + entry_count * entry_wire_len`

### Chunk index entry (fixed `entry_wire_len` bytes, LE)

Layout (see `ChunkIndexEntryV1::WIRE_LEN` in `src/catalog.rs`):

- `dataset_id`: `u64`
- `chunk_index[8]`: eight `u64` values — chunk grid coordinates `i0 … i7`; unused slots are **0** (rank comes from the dataset record’s `ndim`).
- `payload_offset`: `u64` — file offset to **stored** bytes for this chunk.
- `raw_byte_len`: `u64` — logical uncompressed size of the chunk tensor in bytes.
- `stored_byte_len`: `u64` — bytes on disk at `payload_offset`.
- `codec`: `u32` — **`0`** = raw copy (`stored_byte_len` **must** equal `raw_byte_len`).
- `reserved`: `u32` — **0**.

Payload bytes must lie fully inside the file: `payload_offset + stored_byte_len ≤ file_len`.

## Reference subset (current Rust writer)

The `write_one_chunk_raw_file` helper in `tetration::catalog` writes exactly **one** dataset and **one** chunk: `chunk_shape` must equal `shape` so the chunk grid has a single tile. Compression is not implemented (`codec = 0` only).

## Concurrency (informative)

Writers should use **exclusive create** or clearly documented append rules before parallel writers touch payloads. v1 does not define locking; see the main **README** for the long-term concurrency story.

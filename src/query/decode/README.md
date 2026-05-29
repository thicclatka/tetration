# `query/decode` — chunk payload decode

Low-level bridge from **mmap'd file bytes** to decoded tile payloads and scatter into logical layouts.

## Files

| File              | Role                                                                          |
| ----------------- | ----------------------------------------------------------------------------- |
| `chunk_decode.rs` | Decode one chunk (raw/zstd via `catalog` codecs); `planned_chunk_mmap_slices` |
| `indexing.rs`     | Map global logical indices ↔ chunk-local offsets                              |
| `dense_visit.rs`  | Visit elements in row-major logical order                                     |

## Role in the pipeline

```text
ReadPlan.planned_chunks
    → slice mmap [payload_offset, payload_offset + stored_len)
    → decode_tile_payload → row-major tile bytes
    → scatter into materialized buffer OR fold partial OR GPU upload
```

Does **not** interpret query operations — only bytes → logical tensor pieces.

## Related

- Codec rules: `catalog::ChunkPayloadCodecV1`
- Dense assembly: `materialize/`
- Streaming without full buffer: `fold/linear_scan`, `gpu/streaming_fold`

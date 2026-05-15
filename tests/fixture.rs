//! Shared `.tet` test fixtures (integration tests only).

use std::path::Path;

use tetration::{CHUNK_PAYLOAD_CODEC_V1, DTYPE_F32, RawArrayWrite, write_raw_array_file};

/// Shape `[2, 3]` used across multi-chunk examples.
pub const SHAPE_2X3: [u64; 2] = [2, 3];
/// Chunk shape `[2, 2]` (two tiles along the last axis).
pub const CHUNK_2X2: [u64; 2] = [2, 2];

/// Row-major `f32` tensor values 1..=6 as little-endian bytes (`shape` [2, 3]).
pub fn le_row_major_2x3_f32_one_to_six() -> Vec<u8> {
    let mut data = vec![0u8; 24];
    for (slot, n) in data.chunks_exact_mut(4).zip(1_u8..=6) {
        slot.copy_from_slice(&f32::from(n).to_le_bytes());
    }
    data
}

/// Write a single-dataset `[2,3]` / `[2,2]` multi-chunk raw `f32` file (values 1..6).
pub fn write_multichunk_2x3_tiles(path: &Path, dataset_name: &str) {
    let data = le_row_major_2x3_f32_one_to_six();
    write_raw_array_file(
        path,
        &RawArrayWrite {
            name: dataset_name,
            dtype: DTYPE_F32,
            shape: &SHAPE_2X3,
            chunk_shape: &CHUNK_2X2,
            chunk_codec: CHUNK_PAYLOAD_CODEC_V1.raw,
            data: &data,
        },
    )
    .unwrap();
}

/// Same geometry as [`write_multichunk_2x3_tiles`], but chunk payloads are **zstd**-compressed
/// (all-zero `f32` tensor so frames shrink on disk).
pub fn write_multichunk_2x3_zero_zstd(path: &Path, dataset_name: &str) {
    let data = vec![0u8; 24];
    write_raw_array_file(
        path,
        &RawArrayWrite {
            name: dataset_name,
            dtype: DTYPE_F32,
            shape: &SHAPE_2X3,
            chunk_shape: &CHUNK_2X2,
            chunk_codec: CHUNK_PAYLOAD_CODEC_V1.zstd,
            data: &data,
        },
    )
    .unwrap();
}

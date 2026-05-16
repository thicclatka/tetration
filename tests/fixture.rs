//! Shared `.tet` test fixtures (integration tests only).

use std::path::Path;

use tetration::{
    CHUNK_PAYLOAD_CODEC_V1, DTYPE_F32, MAX_NDIM, RawArrayWrite, mmap_file_read,
    read_tet_summary_v1, write_raw_array_file,
};

/// Patch helpers for corrupting chunk index entries in on-disk fixtures.
#[allow(dead_code)]
pub mod index_patch {
    use super::*;

    pub const CHUNK_INDEX_HDR_LEN: usize = 32;
    /// Byte offset of `raw_byte_len` within a chunk index entry.
    pub const ENTRY_RAW_BYTE_LEN_OFFSET: usize = 8 + MAX_NDIM * 8 + 8;
    pub const ENTRY_STORED_BYTE_LEN_OFFSET: usize = ENTRY_RAW_BYTE_LEN_OFFSET + 8;
    pub const ENTRY_PAYLOAD_OFFSET: usize = 8 + MAX_NDIM * 8;
    pub const ENTRY_CODEC_OFFSET: usize = ENTRY_STORED_BYTE_LEN_OFFSET + 8;

    pub fn first_index_field_offset(path: &Path, field_offset_in_entry: usize) -> usize {
        let mmap0 = mmap_file_read(path).unwrap();
        let s = read_tet_summary_v1(&mmap0).unwrap();
        let idx = usize::try_from(s.superblock.chunk_index_offset).unwrap();
        idx + CHUNK_INDEX_HDR_LEN + field_offset_in_entry
    }

    pub fn patch_first_index_entry_u64(path: &Path, field_offset_in_entry: usize, value: u64) {
        let patch_at = first_index_field_offset(path, field_offset_in_entry);
        let mut bytes = std::fs::read(path).unwrap();
        assert!(
            patch_at + 8 <= bytes.len(),
            "patch offset out of range for test fixture"
        );
        bytes[patch_at..patch_at + 8].copy_from_slice(&value.to_le_bytes());
        std::fs::write(path, &bytes).unwrap();
    }

    pub fn patch_first_index_entry_raw_and_stored(
        path: &Path,
        raw_byte_len: u64,
        stored_byte_len: u64,
    ) {
        let raw_at = first_index_field_offset(path, ENTRY_RAW_BYTE_LEN_OFFSET);
        let stored_at = first_index_field_offset(path, ENTRY_STORED_BYTE_LEN_OFFSET);
        let mut bytes = std::fs::read(path).unwrap();
        assert!(stored_at + 8 <= bytes.len(), "patch offset out of range");
        bytes[raw_at..raw_at + 8].copy_from_slice(&raw_byte_len.to_le_bytes());
        bytes[stored_at..stored_at + 8].copy_from_slice(&stored_byte_len.to_le_bytes());
        std::fs::write(path, &bytes).unwrap();
    }

    pub fn truncate_file(path: &Path, new_len: u64) {
        std::fs::OpenOptions::new()
            .write(true)
            .open(path)
            .unwrap()
            .set_len(new_len)
            .unwrap();
    }
}

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
#[allow(dead_code)]
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

//! Shared `.tet` test fixtures (integration tests only).

use std::path::Path;

use tetration::{
    CHUNK_INDEX_HEADER_V1, CHUNK_PAYLOAD_CODEC_V1, ChunkIndexEntryV1, DATASET_DTYPE_TAG_V1,
    FileExecutionSettingsV1, RawArrayWrite, mmap_file_read, read_tet_summary_v1,
    write_raw_array_file,
};

/// Patch helpers for corrupting chunk index entries in on-disk fixtures.
#[allow(dead_code)]
pub mod index_patch {
    use super::*;

    pub const CHUNK_INDEX_HDR_LEN: usize = CHUNK_INDEX_HEADER_V1.header_len;
    /// Byte offset of `raw_byte_len` within a chunk index entry.
    pub const ENTRY_RAW_BYTE_LEN_OFFSET: usize = ChunkIndexEntryV1::WIRE_RAW_BYTE_LEN_OFFSET;
    pub const ENTRY_STORED_BYTE_LEN_OFFSET: usize = ChunkIndexEntryV1::WIRE_STORED_BYTE_LEN_OFFSET;
    pub const ENTRY_PAYLOAD_OFFSET: usize = ChunkIndexEntryV1::WIRE_PAYLOAD_OFFSET;
    pub const ENTRY_CODEC_OFFSET: usize = ChunkIndexEntryV1::WIRE_CODEC_OFFSET;

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

/// Row-major `i32` tensor values 1..=6 as little-endian bytes (`shape` [2, 3]).
pub fn le_row_major_2x3_i32_one_to_six() -> Vec<u8> {
    let mut data = vec![0u8; 24];
    for (slot, n) in data.chunks_exact_mut(4).zip(1_i32..=6) {
        slot.copy_from_slice(&n.to_le_bytes());
    }
    data
}

/// Row-major `i64` tensor values 1..=6 as little-endian bytes (`shape` [2, 3]).
pub fn le_row_major_2x3_i64_one_to_six() -> Vec<u8> {
    let mut data = vec![0u8; 48];
    for (slot, n) in data.chunks_exact_mut(8).zip(1_i64..=6) {
        slot.copy_from_slice(&n.to_le_bytes());
    }
    data
}

/// Row-major `f64` tensor values 1..=6 as little-endian bytes (`shape` [2, 3]).
pub fn le_row_major_2x3_f64_one_to_six() -> Vec<u8> {
    let mut data = vec![0u8; 48];
    for (slot, n) in data.chunks_exact_mut(8).zip(1_u64..=6) {
        slot.copy_from_slice(&(n as f64).to_le_bytes());
    }
    data
}

fn write_multichunk_2x3(
    path: &Path,
    dataset_name: &str,
    chunk_codec: u32,
    dtype: u32,
    data: &[u8],
) {
    write_raw_array_file(
        path,
        &RawArrayWrite {
            name: dataset_name,
            dtype,
            shape: &SHAPE_2X3,
            chunk_shape: &CHUNK_2X2,
            chunk_codec,
            data,
            file_execution: None,
        },
    )
    .unwrap();
}

fn write_multichunk_2x3_f32(path: &Path, dataset_name: &str, chunk_codec: u32, data: &[u8]) {
    write_multichunk_2x3(
        path,
        dataset_name,
        chunk_codec,
        DATASET_DTYPE_TAG_V1.f32,
        data,
    );
}

/// Write a single-dataset `[2,3]` / `[2,2]` multi-chunk raw `f32` file (values 1..6).
pub fn write_multichunk_2x3_tiles(path: &Path, dataset_name: &str) {
    write_multichunk_2x3_f32(
        path,
        dataset_name,
        CHUNK_PAYLOAD_CODEC_V1.raw,
        &le_row_major_2x3_f32_one_to_six(),
    );
}

/// Write a single-dataset `[2,3]` / `[2,2]` multi-chunk raw `f64` file (values 1..6).
pub fn write_multichunk_2x3_f64_tiles(path: &Path, dataset_name: &str) {
    write_multichunk_2x3_f64(
        path,
        dataset_name,
        CHUNK_PAYLOAD_CODEC_V1.raw,
        &le_row_major_2x3_f64_one_to_six(),
    );
}

fn write_multichunk_2x3_f64(path: &Path, dataset_name: &str, chunk_codec: u32, data: &[u8]) {
    write_multichunk_2x3(
        path,
        dataset_name,
        chunk_codec,
        DATASET_DTYPE_TAG_V1.f64,
        data,
    );
}

/// Write a single-dataset `[2,3]` / `[2,2]` multi-chunk raw `i32` file (values 1..6).
#[allow(dead_code)]
pub fn write_multichunk_2x3_i32_tiles(path: &Path, dataset_name: &str) {
    write_multichunk_2x3(
        path,
        dataset_name,
        CHUNK_PAYLOAD_CODEC_V1.raw,
        DATASET_DTYPE_TAG_V1.i32,
        &le_row_major_2x3_i32_one_to_six(),
    );
}

/// Write a single-dataset `[2,3]` / `[2,2]` multi-chunk raw `i64` file (values 1..6).
#[allow(dead_code)]
pub fn write_multichunk_2x3_i64_tiles(path: &Path, dataset_name: &str) {
    write_multichunk_2x3(
        path,
        dataset_name,
        CHUNK_PAYLOAD_CODEC_V1.raw,
        DATASET_DTYPE_TAG_V1.i64,
        &le_row_major_2x3_i64_one_to_six(),
    );
}

/// Same geometry as [`write_multichunk_2x3_tiles`], but chunk payloads are **zstd**-compressed.
pub fn write_multichunk_2x3_zstd(path: &Path, dataset_name: &str, data: &[u8]) {
    write_multichunk_2x3_f32(path, dataset_name, CHUNK_PAYLOAD_CODEC_V1.zstd, data);
}

/// Same geometry as [`write_multichunk_2x3_tiles`], with per-file execution settings in the index header.
#[allow(dead_code)]
pub fn write_multichunk_2x3_with_execution(
    path: &Path,
    dataset_name: &str,
    file_execution: FileExecutionSettingsV1,
) {
    write_raw_array_file(
        path,
        &RawArrayWrite {
            name: dataset_name,
            dtype: DATASET_DTYPE_TAG_V1.f32,
            shape: &SHAPE_2X3,
            chunk_shape: &CHUNK_2X2,
            chunk_codec: CHUNK_PAYLOAD_CODEC_V1.raw,
            data: &le_row_major_2x3_f32_one_to_six(),
            file_execution: Some(file_execution),
        },
    )
    .unwrap();
}

/// Same geometry as [`write_multichunk_2x3_tiles`], but chunk payloads are **zstd**-compressed
/// (all-zero `f32` tensor so frames shrink on disk).
#[allow(dead_code)]
pub fn write_multichunk_2x3_zero_zstd(path: &Path, dataset_name: &str) {
    write_multichunk_2x3_zstd(path, dataset_name, &vec![0u8; 24]);
}

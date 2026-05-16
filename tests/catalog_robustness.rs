//! Negative tests: truncated files, corrupt zstd, index `raw_byte_len` lying about decoded size.

mod fixture;

use tetration::{
    CatalogError, MAX_NDIM, materialize_read_plan_f32_le, mmap_file_read, parse_query_json,
    plan_query_with_tet_mmap, read_tet_summary_v1, validate_query,
};

use fixture::{write_multichunk_2x3_tiles, write_multichunk_2x3_zero_zstd};

const CHUNK_INDEX_HDR_LEN: usize = 32;
/// Byte offset of `raw_byte_len` within a chunk index entry (after `dataset_id` + `chunk_index` + `payload_offset`).
const ENTRY_RAW_BYTE_LEN_OFFSET: usize = 8 + MAX_NDIM * 8 + 8;
const ENTRY_STORED_BYTE_LEN_OFFSET: usize = ENTRY_RAW_BYTE_LEN_OFFSET + 8;
const ENTRY_CODEC_OFFSET: usize = ENTRY_STORED_BYTE_LEN_OFFSET + 8;

fn first_index_field_offset(path: &std::path::Path, field_offset_in_entry: usize) -> usize {
    let mmap0 = mmap_file_read(path).unwrap();
    let s = read_tet_summary_v1(&mmap0).unwrap();
    let idx = usize::try_from(s.superblock.chunk_index_offset).unwrap();
    idx + CHUNK_INDEX_HDR_LEN + field_offset_in_entry
}

fn patch_first_index_entry_u64(path: &std::path::Path, field_offset_in_entry: usize, value: u64) {
    let patch_at = first_index_field_offset(path, field_offset_in_entry);
    let mut bytes = std::fs::read(path).unwrap();
    assert!(
        patch_at + 8 <= bytes.len(),
        "patch offset out of range for test fixture"
    );
    bytes[patch_at..patch_at + 8].copy_from_slice(&value.to_le_bytes());
    std::fs::write(path, &bytes).unwrap();
}

fn patch_first_index_entry_raw_and_stored(
    path: &std::path::Path,
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

#[test]
fn read_tet_summary_rejects_chunk_payload_span_past_eof() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("trunc.tet");
    write_multichunk_2x3_tiles(&path, "a");

    let len = std::fs::metadata(&path).unwrap().len();
    assert!(len > 2, "fixture file unexpectedly tiny");
    std::fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .unwrap()
        .set_len(len - 1)
        .unwrap();

    let mmap = mmap_file_read(&path).unwrap();
    let err = read_tet_summary_v1(&mmap).unwrap_err();
    assert!(
        matches!(err, CatalogError::PayloadOutOfBounds { .. }),
        "expected PayloadOutOfBounds, got {err:?}"
    );
}

#[test]
fn materialize_rejects_corrupt_zstd_payload() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("badzstd.tet");
    write_multichunk_2x3_zero_zstd(&path, "z");

    let mut bytes = std::fs::read(&path).unwrap();
    let p0 = {
        let mmap0 = mmap_file_read(&path).unwrap();
        let s = read_tet_summary_v1(&mmap0).unwrap();
        usize::try_from(s.chunks[0].payload_offset).unwrap()
    };
    bytes[p0] ^= 0xff;
    std::fs::write(&path, &bytes).unwrap();

    let mmap = mmap_file_read(&path).unwrap();
    assert!(
        read_tet_summary_v1(&mmap).is_ok(),
        "catalog parse should not eagerly validate zstd frames"
    );

    let doc = parse_query_json(r#"{"dataset":"z"}"#).unwrap();
    validate_query(&doc).unwrap();
    let plan = plan_query_with_tet_mmap(&doc, None, &mmap, None).unwrap();
    let rp = plan.read_plan.as_ref().unwrap();
    let err = materialize_read_plan_f32_le(&mmap, rp, None).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("zstd decode") && msg.contains("chunk_index"),
        "unexpected error (want zstd decode + chunk_index): {msg}"
    );
}

#[test]
fn materialize_rejects_zstd_when_raw_byte_len_mismatches_decompressed_size() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("liar_raw_len.tet");
    write_multichunk_2x3_zero_zstd(&path, "z");

    patch_first_index_entry_u64(&path, ENTRY_RAW_BYTE_LEN_OFFSET, 999);

    let mmap = mmap_file_read(&path).unwrap();
    read_tet_summary_v1(&mmap).expect("index still self-consistent on disk");

    let doc = parse_query_json(r#"{"dataset":"z"}"#).unwrap();
    validate_query(&doc).unwrap();
    let plan = plan_query_with_tet_mmap(&doc, None, &mmap, None).unwrap();
    let rp = plan.read_plan.as_ref().unwrap();
    let err = materialize_read_plan_f32_le(&mmap, rp, None).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("decoded length") && msg.contains("raw_byte_len"),
        "unexpected error: {msg}"
    );
}

#[test]
fn read_tet_summary_rejects_inflated_stored_byte_len_past_eof() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("big_stored.tet");
    write_multichunk_2x3_tiles(&path, "a");

    let huge = {
        let mmap0 = mmap_file_read(&path).unwrap();
        let s = read_tet_summary_v1(&mmap0).unwrap();
        let c0 = &s.chunks[0];
        c0.payload_offset
            .checked_add(c0.stored_byte_len)
            .expect("fixture span")
            + 1024
    };
    // Raw codec requires raw_byte_len == stored_byte_len before span checks run.
    patch_first_index_entry_raw_and_stored(&path, huge, huge);

    let mmap = mmap_file_read(&path).unwrap();
    let err = read_tet_summary_v1(&mmap).unwrap_err();
    assert!(
        matches!(err, CatalogError::PayloadOutOfBounds { .. }),
        "expected PayloadOutOfBounds, got {err:?}"
    );
}

#[test]
fn read_tet_summary_rejects_raw_stored_len_mismatch_for_raw_codec() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("raw_mismatch.tet");
    write_multichunk_2x3_tiles(&path, "a");

    let stored = {
        let mmap0 = mmap_file_read(&path).unwrap();
        read_tet_summary_v1(&mmap0).unwrap().chunks[0].stored_byte_len
    };
    patch_first_index_entry_u64(&path, ENTRY_STORED_BYTE_LEN_OFFSET, stored + 4);

    let mmap = mmap_file_read(&path).unwrap();
    let err = read_tet_summary_v1(&mmap).unwrap_err();
    assert!(
        matches!(err, CatalogError::RawStoredMismatch { .. }),
        "expected RawStoredMismatch, got {err:?}"
    );
}

#[test]
fn read_tet_summary_rejects_unsupported_codec_tag() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bad_codec.tet");
    write_multichunk_2x3_tiles(&path, "a");

    patch_first_index_entry_u64(&path, ENTRY_CODEC_OFFSET, 99);

    let mmap = mmap_file_read(&path).unwrap();
    let err = read_tet_summary_v1(&mmap).unwrap_err();
    assert!(
        matches!(err, CatalogError::UnsupportedCodec { codec: 99 }),
        "expected UnsupportedCodec, got {err:?}"
    );
}

#[test]
fn materialize_rejects_mmap_shorter_than_planned_stored_span() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("short_mmap.tet");
    write_multichunk_2x3_tiles(&path, "a");

    let mmap_full = mmap_file_read(&path).unwrap();
    let doc = parse_query_json(r#"{"dataset":"a"}"#).unwrap();
    validate_query(&doc).unwrap();
    let plan = plan_query_with_tet_mmap(&doc, None, &mmap_full, None).unwrap();
    let rp = plan.read_plan.as_ref().unwrap();

    let trim = mmap_full.len().saturating_sub(1);
    let short = &mmap_full[..trim];
    let err = materialize_read_plan_f32_le(short, rp, None).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("chunk_index") && msg.contains("extends past mmap"),
        "unexpected error: {msg}"
    );
}

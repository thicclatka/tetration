//! Negative tests: truncated files, corrupt zstd, index `raw_byte_len` lying about decoded size.

mod fixture;

use tetration::{
    CatalogError, materialize_read_plan_f32_le, mmap_file_read, parse_query_json,
    plan_query_with_tet_mmap, read_tet_summary_v1, validate_query,
};

use fixture::{write_multichunk_2x3_tiles, write_multichunk_2x3_zero_zstd};

/// Byte offset within the first chunk index entry for `raw_byte_len` (`u64` LE), after the 32-byte `TIDX` header.
const ENTRY0_RAW_LEN_OFFSET: usize = 32 + 8 + (8 * 8) + 8;

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

    let patch_at = {
        let mmap0 = mmap_file_read(&path).unwrap();
        let s = read_tet_summary_v1(&mmap0).unwrap();
        let idx = usize::try_from(s.superblock.chunk_index_offset).unwrap();
        idx + ENTRY0_RAW_LEN_OFFSET
    };

    let mut bytes = std::fs::read(&path).unwrap();
    assert!(
        patch_at + 8 <= bytes.len(),
        "patch offset out of range for test fixture"
    );
    bytes[patch_at..patch_at + 8].copy_from_slice(&999u64.to_le_bytes());
    std::fs::write(&path, &bytes).unwrap();

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

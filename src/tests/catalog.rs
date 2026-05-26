//! Catalog integration tests: roundtrip, robustness, index property tests, f32 helpers.

use super::fixture::{
    self, SHAPE_2X3,
    index_patch::{
        self, ENTRY_CODEC_OFFSET, ENTRY_PAYLOAD_OFFSET, ENTRY_RAW_BYTE_LEN_OFFSET,
        ENTRY_STORED_BYTE_LEN_OFFSET,
    },
    le_row_major_2x3_f32_one_to_six, write_multichunk_2x3_f64_tiles, write_multichunk_2x3_tiles,
    write_multichunk_2x3_zero_zstd,
};
use crate::catalog::{
    CHUNK_PAYLOAD_CODEC_V1, CatalogError, ChunkIndexEntryV1, DATASET_DTYPE_TAG_V1, MAX_NDIM,
    OneChunkRawWrite, chunk_coords_intersecting_global_box, chunk_coords_intersecting_strided,
    read_tet_summary_v1, validate_chunk_payloads, write_one_chunk_raw_file,
};
use crate::layout::{create_empty_v1_file, mmap_file_read};
use crate::query::{
    materialize_read_plan_f32_le, parse_query_json, plan_query_with_tet_mmap, validate_query,
};
use crate::utils::f32_le::{read_f32_le_at, try_cast_f32_le};
use proptest::prelude::*;

// --- roundtrip ---

fn assert_one_chunk_roundtrip(
    path: &std::path::Path,
    dtype: u32,
    name: &str,
    payload: &[u8],
    shape: &[u64],
) {
    let mmap = mmap_file_read(path).unwrap();
    let s = read_tet_summary_v1(&mmap).unwrap();
    assert_eq!(s.datasets.len(), 1);
    assert_eq!(s.datasets[0].name, name);
    assert_eq!(s.datasets[0].dtype, dtype);
    assert_eq!(s.datasets[0].shape, shape);
    assert_eq!(s.chunks.len(), 1);
    let off = usize::try_from(s.chunks[0].payload_offset).unwrap();
    let len = usize::try_from(s.chunks[0].stored_byte_len).unwrap();
    assert_eq!(&mmap[off..off + len], payload);
}

#[test]
fn empty_file_summary() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("empty.tet");
    create_empty_v1_file(&path).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let s = read_tet_summary_v1(&mmap).unwrap();
    assert_eq!(s.superblock.dataset_count, 0);
    assert!(s.datasets.is_empty());
    assert!(s.chunks.is_empty());
}

#[test]
fn one_chunk_f32_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("grid.tet");
    let shape = SHAPE_2X3;
    let chunk_shape = [2u64, 3];
    let payload = le_row_major_2x3_f32_one_to_six();
    write_one_chunk_raw_file(
        &path,
        &OneChunkRawWrite {
            name: "temperature",
            dtype: DATASET_DTYPE_TAG_V1.f32,
            shape: &shape,
            chunk_shape: &chunk_shape,
            payload: &payload,
        },
    )
    .unwrap();

    assert_one_chunk_roundtrip(
        &path,
        DATASET_DTYPE_TAG_V1.f32,
        "temperature",
        &payload,
        &shape,
    );
}

#[test]
fn one_chunk_f64_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("grid_f64.tet");
    let shape = SHAPE_2X3;
    let chunk_shape = [2u64, 3];
    let payload = fixture::le_row_major_2x3_f64_one_to_six();
    write_one_chunk_raw_file(
        &path,
        &OneChunkRawWrite {
            name: "pressure",
            dtype: DATASET_DTYPE_TAG_V1.f64,
            shape: &shape,
            chunk_shape: &chunk_shape,
            payload: &payload,
        },
    )
    .unwrap();

    assert_one_chunk_roundtrip(
        &path,
        DATASET_DTYPE_TAG_V1.f64,
        "pressure",
        &payload,
        &shape,
    );
}

#[test]
fn one_chunk_i32_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("grid_i32.tet");
    let shape = SHAPE_2X3;
    let chunk_shape = [2u64, 3];
    let payload = fixture::le_row_major_2x3_i32_one_to_six();
    write_one_chunk_raw_file(
        &path,
        &OneChunkRawWrite {
            name: "counts",
            dtype: DATASET_DTYPE_TAG_V1.i32,
            shape: &shape,
            chunk_shape: &chunk_shape,
            payload: &payload,
        },
    )
    .unwrap();
    assert_one_chunk_roundtrip(&path, DATASET_DTYPE_TAG_V1.i32, "counts", &payload, &shape);
}

#[test]
fn one_chunk_u8_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("grid_u8.tet");
    let shape = SHAPE_2X3;
    let chunk_shape = [2u64, 3];
    let payload = fixture::le_row_major_2x3_u8_one_to_six();
    write_one_chunk_raw_file(
        &path,
        &OneChunkRawWrite {
            name: "counts",
            dtype: DATASET_DTYPE_TAG_V1.u8,
            shape: &shape,
            chunk_shape: &chunk_shape,
            payload: &payload,
        },
    )
    .unwrap();
    assert_one_chunk_roundtrip(&path, DATASET_DTYPE_TAG_V1.u8, "counts", &payload, &shape);
}

#[test]
fn one_chunk_u16_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("grid_u16.tet");
    let shape = SHAPE_2X3;
    let chunk_shape = [2u64, 3];
    let payload = fixture::le_row_major_2x3_u16_one_to_six();
    write_one_chunk_raw_file(
        &path,
        &OneChunkRawWrite {
            name: "counts",
            dtype: DATASET_DTYPE_TAG_V1.u16,
            shape: &shape,
            chunk_shape: &chunk_shape,
            payload: &payload,
        },
    )
    .unwrap();
    assert_one_chunk_roundtrip(&path, DATASET_DTYPE_TAG_V1.u16, "counts", &payload, &shape);
}

#[test]
fn multi_chunk_u16_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiles_u16.tet");
    fixture::write_multichunk_2x3_u16_tiles(&path, "a");
    let mmap = mmap_file_read(&path).unwrap();
    let s = read_tet_summary_v1(&mmap).unwrap();
    assert_eq!(s.datasets[0].dtype, DATASET_DTYPE_TAG_V1.u16);
    assert_eq!(s.chunks.len(), 2);
}

#[test]
fn multi_chunk_i16_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiles_i16.tet");
    fixture::write_multichunk_2x3_i16_tiles(&path, "a");
    let mmap = mmap_file_read(&path).unwrap();
    let s = read_tet_summary_v1(&mmap).unwrap();
    assert_eq!(s.datasets[0].dtype, DATASET_DTYPE_TAG_V1.i16);
    assert_eq!(s.chunks.len(), 2);
}

#[test]
fn multi_chunk_u8_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiles_u8.tet");
    fixture::write_multichunk_2x3_u8_tiles(&path, "a");
    let mmap = mmap_file_read(&path).unwrap();
    let s = read_tet_summary_v1(&mmap).unwrap();
    assert_eq!(s.datasets[0].dtype, DATASET_DTYPE_TAG_V1.u8);
    assert_eq!(s.chunks.len(), 2);
}

#[test]
fn multi_chunk_i32_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiles_i32.tet");
    fixture::write_multichunk_2x3_i32_tiles(&path, "a");
    let mmap = mmap_file_read(&path).unwrap();
    let s = read_tet_summary_v1(&mmap).unwrap();
    assert_eq!(s.datasets[0].dtype, DATASET_DTYPE_TAG_V1.i32);
    assert_eq!(s.chunks.len(), 2);
}

#[test]
fn multi_chunk_f64_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiles_f64.tet");
    write_multichunk_2x3_f64_tiles(&path, "a");

    let mmap = mmap_file_read(&path).unwrap();
    let s = read_tet_summary_v1(&mmap).unwrap();
    assert_eq!(s.datasets[0].dtype, DATASET_DTYPE_TAG_V1.f64);
    assert_eq!(s.chunks.len(), 2);
}

#[test]
fn multi_chunk_f32_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiles.tet");
    write_multichunk_2x3_tiles(&path, "a");

    let mmap = mmap_file_read(&path).unwrap();
    let s = read_tet_summary_v1(&mmap).unwrap();
    assert_eq!(s.chunks.len(), 2);
    assert_eq!(s.chunks[0].stored_byte_len, 16);
    assert_eq!(s.chunks[1].stored_byte_len, 8);

    let o0 = usize::try_from(s.chunks[0].payload_offset).unwrap();
    let o1 = usize::try_from(s.chunks[1].payload_offset).unwrap();
    let mut expect0 = [0u8; 16];
    let mut expect1 = [0u8; 8];
    for (slot, n) in expect0.chunks_exact_mut(4).zip([1_u8, 2, 4, 5]) {
        slot.copy_from_slice(&f32::from(n).to_le_bytes());
    }
    for (slot, n) in expect1.chunks_exact_mut(4).zip([3_u8, 6]) {
        slot.copy_from_slice(&f32::from(n).to_le_bytes());
    }
    assert_eq!(&mmap[o0..o0 + 16], expect0.as_slice());
    assert_eq!(&mmap[o1..o1 + 8], expect1.as_slice());
}

#[test]
fn multi_chunk_zstd_zeros_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("zstd_zeros.tet");
    write_multichunk_2x3_zero_zstd(&path, "z");

    let mmap = mmap_file_read(&path).unwrap();
    let s = read_tet_summary_v1(&mmap).unwrap();
    assert_eq!(s.chunks.len(), 2);
    for ch in &s.chunks {
        assert_eq!(ch.codec, CHUNK_PAYLOAD_CODEC_V1.zstd);
    }

    let doc = parse_query_json(r#"{"dataset":"z","layout_version":1}"#).unwrap();
    validate_query(&doc).unwrap();
    let plan = plan_query_with_tet_mmap(&doc, None, &mmap, None).unwrap();
    let rp = plan.read_plan.as_ref().unwrap();
    let (vals, truncated, disk) = materialize_read_plan_f32_le(&mmap, rp, None).unwrap();
    assert!(!truncated);
    assert_eq!(vals.len(), 6);
    assert_eq!(
        disk,
        s.chunks[0].stored_byte_len + s.chunks[1].stored_byte_len
    );
    assert!(vals.iter().all(|&x| x == 0.0));
}

#[test]
fn multi_chunk_zstd_write_explicit_raw_array() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("zstd_explicit.tet");
    fixture::write_multichunk_2x3_zstd(&path, "t", &le_row_major_2x3_f32_one_to_six());
    let mmap = mmap_file_read(&path).unwrap();
    let doc = parse_query_json(r#"{"dataset":"t"}"#).unwrap();
    validate_query(&doc).unwrap();
    let plan = plan_query_with_tet_mmap(&doc, None, &mmap, None).unwrap();
    let rp = plan.read_plan.as_ref().unwrap();
    let (vals, _, _) = materialize_read_plan_f32_le(&mmap, rp, None).unwrap();
    let want = [1f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    for (a, b) in vals.iter().zip(want.iter()) {
        assert!((a - b).abs() < 1e-5);
    }
}

// --- robustness ---

fn allowed_catalog_error(err: &CatalogError) -> bool {
    matches!(
        err,
        CatalogError::TooShort { .. }
            | CatalogError::PayloadOutOfBounds { .. }
            | CatalogError::RawStoredMismatch { .. }
            | CatalogError::UnsupportedCodec { .. }
            | CatalogError::BadIndexLength { .. }
            | CatalogError::ChunkIndexOffsetMismatch { .. }
            | CatalogError::DatasetBlobMismatch { .. }
            | CatalogError::MissingDatasetDirectory { .. }
            | CatalogError::Layout(_)
    )
}

#[test]
fn read_tet_summary_rejects_chunk_payload_span_past_eof() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("trunc.tet");
    write_multichunk_2x3_tiles(&path, "a");

    let len = std::fs::metadata(&path).unwrap().len();
    assert!(len > 2, "fixture file unexpectedly tiny");
    index_patch::truncate_file(&path, len - 1);

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
fn materialize_rejects_truncated_zstd_payload() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("trunc_zstd.tet");
    write_multichunk_2x3_zero_zstd(&path, "z");

    let stored = {
        let mmap0 = mmap_file_read(&path).unwrap();
        let s = read_tet_summary_v1(&mmap0).unwrap();
        s.chunks[0].stored_byte_len
    };
    assert!(stored > 1, "fixture zstd payload unexpectedly tiny");
    index_patch::patch_first_index_entry_u64(&path, ENTRY_STORED_BYTE_LEN_OFFSET, stored - 1);

    let mmap = mmap_file_read(&path).unwrap();
    read_tet_summary_v1(&mmap).expect("catalog parse should not eagerly validate zstd frames");

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

    index_patch::patch_first_index_entry_u64(&path, ENTRY_RAW_BYTE_LEN_OFFSET, 999);

    let mmap = mmap_file_read(&path).unwrap();
    read_tet_summary_v1(&mmap).expect("index still self-consistent on disk");

    let doc = parse_query_json(r#"{"dataset":"z"}"#).unwrap();
    validate_query(&doc).unwrap();
    let plan = plan_query_with_tet_mmap(&doc, None, &mmap, None).unwrap();
    let rp = plan.read_plan.as_ref().unwrap();
    let err = materialize_read_plan_f32_le(&mmap, rp, None).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("decoded") && msg.contains("raw_byte_len"),
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
    index_patch::patch_first_index_entry_raw_and_stored(&path, huge, huge);

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
    index_patch::patch_first_index_entry_u64(&path, ENTRY_STORED_BYTE_LEN_OFFSET, stored + 4);

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

    index_patch::patch_first_index_entry_u64(&path, ENTRY_CODEC_OFFSET, 99);

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

// --- index property tests ---

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 64,
        ..ProptestConfig::default()
    })]

    #[test]
    fn validate_chunk_payloads_never_panics(
        offset in any::<u64>(),
        raw in any::<u64>(),
        stored in any::<u64>(),
        codec in any::<u32>(),
        file_len in 0u64..16_000_000,
    ) {
        let entry = ChunkIndexEntryV1 {
            dataset_id: 0,
            chunk_index: [0; MAX_NDIM],
            payload_offset: offset,
            raw_byte_len: raw,
            stored_byte_len: stored,
            codec,
        };
        let r = validate_chunk_payloads(&[entry], file_len);
        if let Err(e) = r {
            let ok = matches!(
                e,
                CatalogError::UnsupportedCodec { .. }
                    | CatalogError::RawStoredMismatch { .. }
                    | CatalogError::PayloadOutOfBounds { .. }
            );
            prop_assert!(ok, "unexpected catalog error: {e:?}");
        } else if CHUNK_PAYLOAD_CODEC_V1.is_supported(codec) {
            let end = offset.saturating_add(stored);
            prop_assert!(end <= file_len);
            if CHUNK_PAYLOAD_CODEC_V1.is_raw(codec) {
                prop_assert_eq!(raw, stored);
            }
        }
    }

    #[test]
    fn valid_fixture_always_parses(file_tag in 0u8..2) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ok.tet");
        write_multichunk_2x3_tiles(&path, if file_tag == 0 { "a" } else { "b" });
        let mmap = mmap_file_read(&path).unwrap();
        let s = read_tet_summary_v1(&mmap).expect("valid writer output");
        for c in &s.chunks {
            let end = c.payload_offset.saturating_add(c.stored_byte_len);
            prop_assert!(end <= mmap.len() as u64);
        }
    }

    #[test]
    fn truncation_never_panics(trim in 1u64..4096) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("trunc.tet");
        write_multichunk_2x3_tiles(&path, "a");
        let len = std::fs::metadata(&path).unwrap().len();
        prop_assume!(len > 33);
        let keep = len.saturating_sub(trim % len).max(32);
        index_patch::truncate_file(&path, keep);
        let mmap = mmap_file_read(&path).unwrap();
        let r = read_tet_summary_v1(&mmap);
        if r.is_err() {
            prop_assert!(allowed_catalog_error(&r.unwrap_err()));
        }
    }

    #[test]
    fn patched_stored_span_rejects_or_accepts(
        stored in any::<u64>(),
        raw in any::<u64>(),
    ) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("patch_stored.tet");
        write_multichunk_2x3_tiles(&path, "a");
        index_patch::patch_first_index_entry_raw_and_stored(&path, raw, stored);
        let mmap = mmap_file_read(&path).unwrap();
        match read_tet_summary_v1(&mmap) {
            Ok(s) => {
                for c in &s.chunks {
                    let end = c.payload_offset.saturating_add(c.stored_byte_len);
                    prop_assert!(end <= mmap.len() as u64);
                    if CHUNK_PAYLOAD_CODEC_V1.is_raw(c.codec) {
                        prop_assert_eq!(c.raw_byte_len, c.stored_byte_len);
                    }
                }
            }
            Err(e) => prop_assert!(allowed_catalog_error(&e)),
        }
    }

    #[test]
    fn patched_payload_offset_rejects_or_accepts(offset in any::<u64>()) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("patch_off.tet");
        write_multichunk_2x3_tiles(&path, "a");
        index_patch::patch_first_index_entry_u64(&path, ENTRY_PAYLOAD_OFFSET, offset);
        let mmap = mmap_file_read(&path).unwrap();
        match read_tet_summary_v1(&mmap) {
            Ok(s) => {
                for c in &s.chunks {
                    let end = c.payload_offset.saturating_add(c.stored_byte_len);
                    prop_assert!(end <= mmap.len() as u64);
                }
            }
            Err(e) => prop_assert!(allowed_catalog_error(&e)),
        }
    }

    #[test]
    fn patched_raw_len_rejects_or_accepts(raw in any::<u64>()) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("patch_raw.tet");
        write_multichunk_2x3_tiles(&path, "a");
        index_patch::patch_first_index_entry_u64(&path, ENTRY_RAW_BYTE_LEN_OFFSET, raw);
        let mmap = mmap_file_read(&path).unwrap();
        let r = read_tet_summary_v1(&mmap);
        if r.is_err() {
            prop_assert!(allowed_catalog_error(&r.unwrap_err()));
        }
    }

    #[test]
    fn unsupported_codec_tag_rejects(codec in 2u32..256u32) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("codec.tet");
        write_multichunk_2x3_tiles(&path, "a");
        index_patch::patch_first_index_entry_u64(&path, ENTRY_CODEC_OFFSET, u64::from(codec));
        let mmap = mmap_file_read(&path).unwrap();
        let err = read_tet_summary_v1(&mmap).unwrap_err();
        match err {
            CatalogError::UnsupportedCodec { codec: c } => prop_assert_eq!(c, codec),
            other => prop_assert!(false, "expected UnsupportedCodec, got {other:?}"),
        }
    }
}

// --- f32_le helpers ---

#[test]
fn read_unaligned_matches_from_le_bytes() {
    let mut bytes = [0u8; 8];
    bytes[1..5].copy_from_slice(&1.5f32.to_le_bytes());
    assert_eq!(read_f32_le_at(&bytes[1..], 0), 1.5);
    assert!(try_cast_f32_le(&bytes[1..]).is_none());
}

#[test]
fn cast_aligned_tile() {
    let bytes: [u8; 8] = [0u8; 8];
    let aligned = &bytes[..];
    assert_eq!(
        aligned.as_ptr().align_offset(std::mem::align_of::<f32>()),
        0
    );
    let vals = try_cast_f32_le(aligned).expect("aligned");
    assert_eq!(vals.len(), 2);
    assert_eq!(read_f32_le_at(aligned, 0), vals[0]);
}

// --- chunk tile geometry (from src/catalog/tile.rs) ---

#[test]
fn strided_axis_fewer_chunks_than_dense() {
    let shape = [4u64, 3];
    let cs = [2u64, 3];
    let g0 = [1, 0];
    let g1 = [3, 3];
    let steps_strided = [2u64, 1];
    let steps_dense = [1u64, 1];
    let a = chunk_coords_intersecting_strided(&shape, &cs, &g0, &g1, &steps_strided).unwrap();
    let b = chunk_coords_intersecting_strided(&shape, &cs, &g0, &g1, &steps_dense).unwrap();
    assert!(a.len() < b.len());
    assert_eq!(a.len(), 1);
    assert_eq!(b.len(), 2);
}

#[test]
fn global_box_matches_strided_unit_steps() {
    let shape = [2u64, 3];
    let cs = [2u64, 2];
    let g0 = [0, 0];
    let g1 = [2, 2];
    let steps = [1u64, 1];
    let a = chunk_coords_intersecting_global_box(&shape, &cs, &g0, &g1).unwrap();
    let b = chunk_coords_intersecting_strided(&shape, &cs, &g0, &g1, &steps).unwrap();
    assert_eq!(a, b);
}

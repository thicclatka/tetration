mod fixture;

use tetration::{
    CHUNK_PAYLOAD_CODEC_V1, DTYPE_F32, OneChunkRawWrite, RawArrayWrite, create_empty_v1_file,
    materialize_read_plan_f32_le, mmap_file_read, parse_query_json, plan_query_with_tet_mmap,
    read_tet_summary_v1, validate_query, write_one_chunk_raw_file, write_raw_array_file,
};

use fixture::{
    CHUNK_2X2, SHAPE_2X3, le_row_major_2x3_f32_one_to_six, write_multichunk_2x3_tiles,
    write_multichunk_2x3_zero_zstd,
};

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
            dtype: DTYPE_F32,
            shape: &shape,
            chunk_shape: &chunk_shape,
            payload: &payload,
        },
    )
    .unwrap();

    let mmap = mmap_file_read(&path).unwrap();
    let s = read_tet_summary_v1(&mmap).unwrap();
    assert_eq!(s.datasets.len(), 1);
    assert_eq!(s.datasets[0].name, "temperature");
    assert_eq!(s.datasets[0].dtype, DTYPE_F32);
    assert_eq!(s.datasets[0].shape, vec![2, 3]);
    assert_eq!(s.chunks.len(), 1);
    let off = usize::try_from(s.chunks[0].payload_offset).unwrap();
    let len = usize::try_from(s.chunks[0].stored_byte_len).unwrap();
    assert_eq!(&mmap[off..off + len], &payload[..]);
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
    let data = le_row_major_2x3_f32_one_to_six();
    write_raw_array_file(
        &path,
        &RawArrayWrite {
            name: "t",
            dtype: DTYPE_F32,
            shape: &SHAPE_2X3,
            chunk_shape: &CHUNK_2X2,
            chunk_codec: CHUNK_PAYLOAD_CODEC_V1.zstd,
            data: &data,
        },
    )
    .unwrap();
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

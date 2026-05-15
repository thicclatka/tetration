mod fixture;

use tetration::{
    DTYPE_F32, OneChunkRawWrite, create_empty_v1_file, mmap_file_read, read_tet_summary_v1,
    write_one_chunk_raw_file,
};

use fixture::{SHAPE_2X3, le_row_major_2x3_f32_one_to_six, write_multichunk_2x3_tiles};

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

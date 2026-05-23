use std::io::Write;

use tetration::{
    LayoutError, SUPERBLOCK_V1_LEN, SuperblockV1, create_empty_v1_file, open_superblock_v1,
    read_superblock_v1,
};

#[test]
fn empty_file_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("empty.tet");
    create_empty_v1_file(&path).unwrap();

    let (_mmap, sb) = open_superblock_v1(&path).unwrap();
    assert_eq!(sb, SuperblockV1::empty_file());
}

#[test]
fn parse_rejects_bad_magic() {
    let mut buf = SuperblockV1::empty_file().to_bytes();
    buf[0] = b'X';
    let err = read_superblock_v1(&buf).unwrap_err();
    assert!(matches!(err, LayoutError::BadMagic(_)));
}

#[test]
fn parse_rejects_index_past_eof() {
    let mut sb = SuperblockV1::empty_file();
    sb.chunk_index_offset = 0;
    sb.chunk_index_length = 100;
    let err = read_superblock_v1(&sb.to_bytes()).unwrap_err();
    assert!(matches!(err, LayoutError::IndexOutOfBounds { .. }));
}

#[test]
fn parse_rejects_too_short() {
    let err = read_superblock_v1(&[0u8; 8]).unwrap_err();
    assert!(matches!(err, LayoutError::TooShort { .. }));
}

#[test]
fn mmap_handles_append_garbage_after_superblock() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("padded.tet");
    create_empty_v1_file(&path).unwrap();
    let mut f = std::fs::OpenOptions::new()
        .append(true)
        .open(&path)
        .unwrap();
    f.write_all(b"future metadata").unwrap();

    let (mmap, sb) = open_superblock_v1(&path).unwrap();
    assert_eq!(sb, SuperblockV1::empty_file());
    assert_eq!(mmap.len(), SUPERBLOCK_V1_LEN + b"future metadata".len());
}

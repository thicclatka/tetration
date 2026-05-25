//! Phase 7 embedder session + execute helpers.

use crate::catalog::{TetDatasetWrite, TetFile, TetWriterSession, read_tet_summary_v1};
use crate::query::{
    ExecuteQueryOptions, QueryOutputFormat, execute_query_document, format_query_response,
    parse_query_json, validate_query,
};

const SHAPE: [u64; 2] = [2, 3];
const CHUNK_SHAPE: [u64; 2] = [2, 2];

fn f32_one_to_six() -> Vec<u8> {
    let mut data = vec![0u8; 24];
    for (slot, n) in data.chunks_exact_mut(4).zip(1_u8..=6) {
        slot.copy_from_slice(&f32::from(n).to_le_bytes());
    }
    data
}

#[test]
fn writer_session_commit_and_query() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("session.tet");

    let mut session = TetWriterSession::create(&path);
    session.metadata.tool = Some("tetration-test".to_owned());
    session.push_history_event("write", "rust");
    session
        .push_dataset(
            TetDatasetWrite::f32_row_major("temperature", &SHAPE, &CHUNK_SHAPE, f32_one_to_six())
                .unwrap(),
        )
        .unwrap();

    let out = session.commit().unwrap();
    assert_eq!(out, path);

    let summary = read_tet_summary_v1(&std::fs::read(&path).unwrap()).unwrap();
    assert_eq!(summary.datasets.len(), 1);
    assert_eq!(summary.history.len(), 1);
    assert_eq!(summary.history[0].0, "write");

    let file = TetFile::open(&path).unwrap();
    let doc = parse_query_json(r#"{"dataset":"temperature","mean":[]}"#).unwrap();
    validate_query(&doc).unwrap();
    let response = execute_query_document(
        &doc,
        file.path(),
        file.mmap(),
        ExecuteQueryOptions::execute_no_preview(),
        None,
    )
    .unwrap();
    let line = format_query_response(&response, QueryOutputFormat::Quiet).unwrap();
    assert!(line.contains("mean=3.5"), "{line}");
}

#[test]
fn writer_session_requires_dataset() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("empty_commit.tet");
    let session = TetWriterSession::create(path);
    assert!(session.commit().is_err());
}

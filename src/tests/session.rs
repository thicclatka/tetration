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
    let mut ds =
        TetDatasetWrite::f32_row_major("temperature", &SHAPE, &CHUNK_SHAPE, f32_one_to_six())
            .unwrap();
    ds.attrs.insert("units".to_owned(), "1".to_owned());
    ds.attrs
        .insert("long_name".to_owned(), "demo temperature".to_owned());
    ds.dim_names = Some(vec!["row".to_owned(), "col".to_owned()]);
    session.push_dataset(ds).unwrap();

    let out = session.commit().unwrap();
    assert_eq!(out, path);

    let summary = read_tet_summary_v1(&std::fs::read(&path).unwrap()).unwrap();
    assert_eq!(summary.datasets.len(), 1);
    assert_eq!(summary.history.len(), 1);
    assert_eq!(summary.history[0].0, "write");
    assert_eq!(
        summary
            .metadata
            .file
            .as_ref()
            .and_then(|f| f.tool.as_deref()),
        Some("tetration-test")
    );
    let ds_meta = summary.metadata.datasets.get("temperature").unwrap();
    assert_eq!(ds_meta.attrs.get("units").map(String::as_str), Some("1"));
    assert_eq!(
        ds_meta.dim_names,
        Some(vec!["row".to_owned(), "col".to_owned()])
    );

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

//! CLI query output formatting (`QueryOutputFormat`).

mod fixture;

use tetration::{
    QueryOutputFormat, format_query_response, mmap_file_read, parse_query_json, plan_query_empty,
    plan_query_with_tet_mmap, validate_query,
};

#[test]
fn format_full_and_json_include_operation_fields() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("fmt.tet");
    fixture::write_multichunk_2x3_tiles(&path, "a");
    let doc = parse_query_json(r#"{"dataset":"a","operation":{"mean":{"axes":[]}}}"#).unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let response = plan_query_with_tet_mmap(&doc, None, &mmap, Some(0)).unwrap();

    let full = format_query_response(&response, QueryOutputFormat::Full).unwrap();
    assert!(full.contains("\"operation_mean\""));

    let compact = format_query_response(&response, QueryOutputFormat::Json).unwrap();
    assert!(!compact.contains('\n'));
    assert!(compact.contains("\"operation_mean\""));
}

#[test]
fn format_stats_omits_chunk_rows_and_previews() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("stats.tet");
    fixture::write_multichunk_2x3_tiles(&path, "a");
    let doc = parse_query_json(r#"{"dataset":"a","operation":{"sum":{"axes":[]}}}"#).unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let response = plan_query_with_tet_mmap(&doc, None, &mmap, Some(8)).unwrap();

    let stats = format_query_response(&response, QueryOutputFormat::Stats).unwrap();
    assert!(stats.contains("\"operation_sum\""));
    assert!(stats.contains("\"chunk_count\""));
    assert!(!stats.contains("\"chunks\""));
    assert!(!stats.contains("f32_preview"));
}

#[test]
fn format_quiet_scalar_mean() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("quiet.tet");
    fixture::write_multichunk_2x3_tiles(&path, "a");
    let doc = parse_query_json(r#"{"dataset":"a","operation":{"mean":{"axes":[]}}}"#).unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let response = plan_query_with_tet_mmap(&doc, None, &mmap, Some(0)).unwrap();

    let line = format_query_response(&response, QueryOutputFormat::Quiet).unwrap();
    assert!(line.contains("dataset=a"));
    assert!(line.contains("status=ok"));
    assert!(line.contains("op=mean"));
    assert!(line.contains("mean=3.5"));
    assert!(line.contains("elements=6"));
    assert!(!line.contains('\n'));
}

#[test]
fn format_quiet_partial_sum_along_axis() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("quiet_partial.tet");
    fixture::write_multichunk_2x3_tiles(&path, "a");
    let doc = parse_query_json(r#"{"dataset":"a","operation":{"sum":{"axes":["0"]}}}"#).unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let response = plan_query_with_tet_mmap(&doc, None, &mmap, Some(0)).unwrap();

    let line = format_query_response(&response, QueryOutputFormat::Quiet).unwrap();
    assert!(line.contains("op=sum"));
    assert!(line.contains("axes=[0]"));
    assert!(line.contains("values=[5,7,9]"));
    assert!(line.contains("reduced_shape=3"));
}

#[test]
fn format_quiet_plan_only() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("plan.tet");
    fixture::write_multichunk_2x3_tiles(&path, "a");
    let doc = parse_query_json(r#"{"dataset":"a"}"#).unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let response = plan_query_with_tet_mmap(&doc, None, &mmap, None).unwrap();

    let line = format_query_response(&response, QueryOutputFormat::Quiet).unwrap();
    assert!(line.contains("status=planned"));
    assert!(line.contains("chunks=2"));
    assert!(line.contains("logical_shape=2×3"));
}

#[test]
fn format_quiet_unmatched_dataset() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("miss.tet");
    fixture::write_multichunk_2x3_tiles(&path, "a");
    let doc = parse_query_json(r#"{"dataset":"missing"}"#).unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let response = plan_query_with_tet_mmap(&doc, None, &mmap, None).unwrap();

    let line = format_query_response(&response, QueryOutputFormat::Quiet).unwrap();
    assert!(line.contains("status=not_found"));
    assert!(line.contains("available=[a]"));
}

#[test]
fn format_quiet_validated_without_tet() {
    let doc = parse_query_json(r#"{"dataset":"a"}"#).unwrap();
    validate_query(&doc).unwrap();
    let response = plan_query_empty(&doc);

    let line = format_query_response(&response, QueryOutputFormat::Quiet).unwrap();
    assert!(line.contains("validated"));
    assert!(line.contains("hint=pass --tet"));
}

#[test]
fn format_quiet_preview_execute() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("prev.tet");
    fixture::write_multichunk_2x3_tiles(&path, "a");
    let doc = parse_query_json(r#"{"dataset":"a"}"#).unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let response = plan_query_with_tet_mmap(&doc, None, &mmap, Some(4)).unwrap();

    let line = format_query_response(&response, QueryOutputFormat::Quiet).unwrap();
    assert!(line.contains("status=preview"));
    assert!(line.contains("preview=4 f32"));
    assert!(line.contains("sample=[1,2,3,4]"));
}

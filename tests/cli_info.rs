//! `tet info` formatting and filters.

use std::process::Command;

use tetration::{
    InfoListFilter, InfoViewSections, format_info_json, format_info_quiet, format_info_text,
    mmap_file_read, read_tet_summary_v1,
};

mod fixture;

use fixture::write_multichunk_2x3_tiles;

#[test]
fn info_default_table_lists_datasets() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("a.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let mmap = mmap_file_read(&path).unwrap();
    let summary = read_tet_summary_v1(&mmap).unwrap();
    let text = format_info_text(
        Some(&path),
        mmap.len() as u64,
        &summary,
        None,
        InfoViewSections::default_table(),
        32,
    );
    assert!(text.contains("datasets:"));
    assert!(text.contains("a"));
    assert!(text.contains("f32"));
    assert!(!text.contains("\"superblock\""));
}

#[test]
fn info_json_includes_full_summary() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("one.tet");
    write_multichunk_2x3_tiles(&path, "temp");
    let mmap = mmap_file_read(&path).unwrap();
    let summary = read_tet_summary_v1(&mmap).unwrap();
    let json = format_info_json(Some(&path), mmap.len() as u64, &summary, None).unwrap();
    assert!(json.contains("\"superblock\""));
    assert!(json.contains("\"datasets\""));
    assert!(json.contains("temp"));
}

#[test]
fn info_grep_filters_datasets() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("a.tet");
    write_multichunk_2x3_tiles(&path, "temperature");
    let mmap = mmap_file_read(&path).unwrap();
    let summary = read_tet_summary_v1(&mmap).unwrap();
    let filter = InfoListFilter {
        dataset: None,
        grep: Some("temp".to_owned()),
    };
    let text = format_info_text(
        Some(&path),
        mmap.len() as u64,
        &summary,
        Some(&filter),
        InfoViewSections::default_table(),
        32,
    );
    assert!(text.contains("temperature"));
    assert!(text.contains("filter: grep~temp"));
}

#[test]
fn tet_info_binary_runs_on_fixture() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("a.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let tet = env!("CARGO_BIN_EXE_tet");
    let out = Command::new(tet)
        .args(["info", path.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(out.status.success(), "stderr={}", String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("datasets:"));
    assert!(stdout.contains("a"));
}

#[test]
fn info_quiet_one_line() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("a.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let mmap = mmap_file_read(&path).unwrap();
    let summary = read_tet_summary_v1(&mmap).unwrap();
    let line = format_info_quiet(Some(&path), mmap.len() as u64, &summary, None);
    assert!(line.contains("path="));
    assert!(line.contains("datasets=1"));
}

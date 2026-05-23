//! CLI query history (platform cache JSONL, not stored in `.tet`).

use std::env;
use std::sync::{Mutex, MutexGuard};

use tetration::{
    CLI_QUERY_HISTORY_MAX, append_cli_query_history, clear_cli_query_history,
    list_cli_query_history, parse_query_json, validate_query,
};

static HISTORY_ENV_LOCK: Mutex<()> = Mutex::new(());

fn history_lock() -> MutexGuard<'static, ()> {
    HISTORY_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn with_history_file<F, T>(f: F) -> T
where
    F: FnOnce() -> T,
{
    let _guard = history_lock();
    let temp = tempfile::NamedTempFile::new().unwrap();
    let path = temp.path().to_path_buf();
    // Test-only: isolate history path per test (Rust 2024: env mutators are unsafe).
    unsafe {
        env::set_var("TET_QUERY_HISTORY_FILE", &path);
    }
    let out = f();
    unsafe {
        env::remove_var("TET_QUERY_HISTORY_FILE");
    }
    out
}

fn sample_query_json() -> &'static str {
    r#"{"dataset":"temperature","layout_version":1}"#
}

#[test]
fn cli_query_history_append_trim_and_list() {
    with_history_file(|| {
        let doc = parse_query_json(sample_query_json()).unwrap();
        validate_query(&doc).unwrap();
        for i in 0..12 {
            append_cli_query_history(&doc, Some(&format!("/data/{i}.tet")), i % 2 == 0).unwrap();
        }
        let listed = list_cli_query_history(CLI_QUERY_HISTORY_MAX).unwrap();
        assert_eq!(listed.len(), CLI_QUERY_HISTORY_MAX);
        assert_eq!(listed[0].tet.as_deref(), Some("/data/11.tet"));
        assert_eq!(listed[9].tet.as_deref(), Some("/data/2.tet"));
    });
}

#[test]
fn cli_query_history_clear() {
    with_history_file(|| {
        let doc = parse_query_json(sample_query_json()).unwrap();
        validate_query(&doc).unwrap();
        append_cli_query_history(&doc, None, false).unwrap();
        assert_eq!(list_cli_query_history(10).unwrap().len(), 1);
        clear_cli_query_history().unwrap();
        assert!(list_cli_query_history(10).unwrap().is_empty());
    });
}

#[test]
fn cli_query_history_disabled_by_env() {
    with_history_file(|| {
        unsafe {
            env::set_var("TET_NO_QUERY_HISTORY", "1");
        }
        let doc = parse_query_json(sample_query_json()).unwrap();
        append_cli_query_history(&doc, None, false).unwrap();
        assert!(list_cli_query_history(10).unwrap().is_empty());
        unsafe {
            env::remove_var("TET_NO_QUERY_HISTORY");
        }
    });
}

//! CLI query history (platform cache JSONL, not stored in `.tet`).

use std::env;
use std::sync::{Mutex, MutexGuard};

use tetration::{
    HistorySettings, append_cli_query_history, clear_cli_query_history, cli_query_history_max,
    get_cli_query_history_entry, list_cli_query_history, parse_query_json, validate_query,
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
        env::remove_var("TET_QUERY_HISTORY_MAX");
    }
    let out = f();
    unsafe {
        env::remove_var("TET_QUERY_HISTORY_FILE");
        env::remove_var("TET_QUERY_HISTORY_MAX");
    }
    out
}

fn sample_query_json() -> &'static str {
    r#"{"dataset":"temperature","layout_version":1}"#
}

#[test]
fn history_settings_default_and_env() {
    with_history_file(|| {
        let d = HistorySettings::default();
        assert_eq!(d.cli_query_max, 10);
        assert_eq!(d.history_max_cap, 10_000);
        assert_eq!(d.history_file_name, "query_history.jsonl");
        unsafe {
            env::set_var("TET_QUERY_HISTORY_MAX", "3");
        }
        let from_env = HistorySettings::from_env();
        assert_eq!(from_env.cli_query_max, 3);
        assert_eq!(cli_query_history_max(), 3);
    });
}

#[test]
fn cli_query_history_append_trim_and_list() {
    with_history_file(|| {
        let doc = parse_query_json(sample_query_json()).unwrap();
        validate_query(&doc).unwrap();
        for i in 0..12 {
            append_cli_query_history(&doc, Some(&format!("/data/{i}.tet")), i % 2 == 0).unwrap();
        }
        let max = HistorySettings::default().cli_query_max;
        let listed = list_cli_query_history(max, false).unwrap();
        assert_eq!(listed.len(), max);
        assert_eq!(listed[0].tet.as_deref(), Some("/data/11.tet"));
        assert_eq!(listed[9].tet.as_deref(), Some("/data/2.tet"));
    });
}

#[test]
fn cli_query_history_max_env_trims() {
    with_history_file(|| {
        unsafe {
            env::set_var("TET_QUERY_HISTORY_MAX", "3");
        }
        assert_eq!(cli_query_history_max(), 3);
        let doc = parse_query_json(sample_query_json()).unwrap();
        for i in 0..5 {
            append_cli_query_history(&doc, Some(&format!("/data/{i}.tet")), false).unwrap();
        }
        let all = list_cli_query_history(100, true).unwrap();
        assert_eq!(all.len(), 3);
        assert_eq!(all[0].tet.as_deref(), Some("/data/4.tet"));
        assert_eq!(all[2].tet.as_deref(), Some("/data/2.tet"));
    });
}

#[test]
fn cli_query_history_list_all_flag() {
    with_history_file(|| {
        let doc = parse_query_json(sample_query_json()).unwrap();
        for _ in 0..5 {
            append_cli_query_history(&doc, None, false).unwrap();
        }
        assert_eq!(list_cli_query_history(2, false).unwrap().len(), 2);
        assert_eq!(list_cli_query_history(2, true).unwrap().len(), 5);
    });
}

#[test]
fn cli_query_history_get_entry_by_index() {
    with_history_file(|| {
        let doc = parse_query_json(sample_query_json()).unwrap();
        append_cli_query_history(&doc, Some("/a.tet"), true).unwrap();
        append_cli_query_history(&doc, Some("/b.tet"), false).unwrap();
        let newest = get_cli_query_history_entry(1).unwrap();
        assert_eq!(newest.tet.as_deref(), Some("/b.tet"));
        assert!(!newest.execute);
        let older = get_cli_query_history_entry(2).unwrap();
        assert_eq!(older.tet.as_deref(), Some("/a.tet"));
        assert!(older.execute);
        assert!(get_cli_query_history_entry(0).is_err());
        assert!(get_cli_query_history_entry(3).is_err());
    });
}

#[test]
fn cli_query_history_clear() {
    with_history_file(|| {
        let doc = parse_query_json(sample_query_json()).unwrap();
        validate_query(&doc).unwrap();
        append_cli_query_history(&doc, None, false).unwrap();
        assert_eq!(list_cli_query_history(10, false).unwrap().len(), 1);
        clear_cli_query_history().unwrap();
        assert!(list_cli_query_history(10, false).unwrap().is_empty());
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
        assert!(list_cli_query_history(10, false).unwrap().is_empty());
        unsafe {
            env::remove_var("TET_NO_QUERY_HISTORY");
        }
    });
}

//! CLI-only recent query log (platform cache file, not stored in `.tet`).

use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::query::engine::spill_policy::platform_tetration_cache_dir;
use crate::query::types::QueryDocument;

/// Default cap on retained CLI query rows (`tet history`).
pub const CLI_QUERY_HISTORY_MAX: usize = 10;

const HISTORY_FILE_NAME: &str = "query_history.jsonl";

/// One row in the CLI query history file (newest first when listed).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CliQueryHistoryEntry {
    /// Unix seconds (UTC).
    pub at: u64,
    /// `.tet` path when `--tet` was passed.
    pub tet: Option<String>,
    /// Whether `--execute` was set.
    pub execute: bool,
    /// Validated query document.
    pub query: QueryDocument,
}

/// Whether CLI query history recording is enabled (unset or any value other than `1` / `true`).
#[must_use]
pub fn cli_query_history_enabled() -> bool {
    !matches!(
        std::env::var("TET_NO_QUERY_HISTORY").ok().as_deref(),
        Some("1" | "true" | "TRUE" | "yes" | "YES")
    )
}

/// Path to the JSONL history file (`TET_QUERY_HISTORY_FILE` overrides platform cache).
#[must_use]
pub fn cli_query_history_path() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("TET_QUERY_HISTORY_FILE") {
        return Some(PathBuf::from(path));
    }
    platform_tetration_cache_dir().map(|dir| dir.join(HISTORY_FILE_NAME))
}

/// Append a successful query; trims to [`CLI_QUERY_HISTORY_MAX`]. Best-effort (creates parent dirs).
///
/// # Errors
///
/// Returns I/O or JSON errors when the history file cannot be read or rewritten.
pub fn append_cli_query_history(
    query: &QueryDocument,
    tet: Option<&str>,
    execute: bool,
) -> io::Result<()> {
    if !cli_query_history_enabled() {
        return Ok(());
    }
    let path = cli_query_history_path().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            "no platform cache directory for query history",
        )
    })?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let entry = CliQueryHistoryEntry {
        at: unix_timestamp_secs(),
        tet: tet.map(str::to_owned),
        execute,
        query: query.clone(),
    };

    let mut entries = read_entries(&path)?;
    entries.push(entry);
    if entries.len() > CLI_QUERY_HISTORY_MAX {
        let drop = entries.len() - CLI_QUERY_HISTORY_MAX;
        entries.drain(0..drop);
    }
    write_entries(&path, &entries)
}

/// List recent queries (newest first). Missing file → empty vec.
///
/// # Errors
///
/// Returns I/O or JSON errors when the history file cannot be read.
pub fn list_cli_query_history(limit: usize) -> io::Result<Vec<CliQueryHistoryEntry>> {
    let Some(path) = cli_query_history_path() else {
        return Ok(Vec::new());
    };
    if !path.is_file() {
        return Ok(Vec::new());
    }
    let mut entries = read_entries(&path)?;
    entries.reverse();
    entries.truncate(limit);
    Ok(entries)
}

/// Remove the CLI query history file if present.
///
/// # Errors
///
/// Returns I/O errors from [`std::fs::remove_file`].
pub fn clear_cli_query_history() -> io::Result<()> {
    let Some(path) = cli_query_history_path() else {
        return Ok(());
    };
    match fs::remove_file(&path) {
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
        other => other,
    }
}

fn read_entries(path: &Path) -> io::Result<Vec<CliQueryHistoryEntry>> {
    if !path.is_file() {
        return Ok(Vec::new());
    }
    let raw = fs::read_to_string(path)?;
    let mut out = Vec::new();
    for line in raw.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let entry: CliQueryHistoryEntry = serde_json::from_str(line).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("query history parse: {e}"),
            )
        })?;
        out.push(entry);
    }
    Ok(out)
}

fn write_entries(path: &Path, entries: &[CliQueryHistoryEntry]) -> io::Result<()> {
    let mut f = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)?;
    for entry in entries {
        let line = serde_json::to_string(entry)
            .map_err(|e| io::Error::other(format!("query history encode: {e}")))?;
        f.write_all(line.as_bytes())?;
        f.write_all(b"\n")?;
    }
    f.sync_all()?;
    Ok(())
}

fn unix_timestamp_secs() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

//! CLI-only recent query log (platform cache file, not stored in `.tet`).

use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::query::engine::spill_policy::platform_tetration_cache_dir;
use crate::query::types::QueryDocument;

/// CLI query history limits and file naming.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HistorySettings {
    /// Max rows kept on disk (oldest dropped on append).
    pub cli_query_max: usize,
    /// Upper bound for `cli_query_max` when parsing `TET_QUERY_HISTORY_MAX`.
    pub history_max_cap: usize,
    /// File name joined to the platform cache dir (full path via [`HistorySettings::path`]).
    pub history_file_name: String,
}

impl Default for HistorySettings {
    fn default() -> Self {
        Self {
            cli_query_max: 10,
            history_max_cap: 10_000,
            history_file_name: "query_history.jsonl".to_owned(),
        }
    }
}

impl HistorySettings {
    /// Defaults merged with `TET_QUERY_HISTORY_MAX` when set and in range.
    #[must_use]
    pub fn from_env() -> Self {
        let mut settings = Self::default();
        if let Ok(raw) = std::env::var("TET_QUERY_HISTORY_MAX")
            && let Ok(n) = raw.trim().parse::<usize>()
            && (1..=settings.history_max_cap).contains(&n)
        {
            settings.cli_query_max = n;
        }
        settings
    }

    /// Path to the JSONL file (`TET_QUERY_HISTORY_FILE` overrides platform cache + [`Self::history_file_name`]).
    #[must_use]
    pub fn path(&self) -> Option<PathBuf> {
        if let Ok(path) = std::env::var("TET_QUERY_HISTORY_FILE") {
            return Some(PathBuf::from(path));
        }
        platform_tetration_cache_dir().map(|dir| dir.join(&self.history_file_name))
    }

    /// Append a successful query; trims oldest rows to [`Self::cli_query_max`]. Best-effort.
    ///
    /// # Errors
    ///
    /// Returns I/O or JSON errors when the history file cannot be read or rewritten.
    pub fn append(
        &self,
        query: &QueryDocument,
        tet: Option<&str>,
        execute: bool,
    ) -> io::Result<()> {
        if !cli_query_history_enabled() {
            return Ok(());
        }
        let path = self.path().ok_or_else(|| {
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
        if entries.len() > self.cli_query_max {
            let drop = entries.len() - self.cli_query_max;
            entries.drain(0..drop);
        }
        write_entries(&path, &entries)
    }

    /// List recent queries (newest first). Missing file → empty vec.
    ///
    /// When `all` is true, returns every retained row (up to [`Self::cli_query_max`] on disk).
    ///
    /// # Errors
    ///
    /// Returns I/O or JSON errors when the history file cannot be read.
    pub fn list(&self, limit: usize, all: bool) -> io::Result<Vec<CliQueryHistoryEntry>> {
        let mut entries = self.read_newest_first()?;
        if all {
            return Ok(entries);
        }
        entries.truncate(limit);
        Ok(entries)
    }

    /// Fetch one row by display index (**1** = newest, same order as [`Self::list`]).
    ///
    /// # Errors
    ///
    /// Returns I/O or JSON errors when the file cannot be read. Returns
    /// [`io::ErrorKind::NotFound`] when the index is out of range or history is empty.
    pub fn get(&self, index: usize) -> io::Result<CliQueryHistoryEntry> {
        if index == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "history index must be >= 1 (1 = newest)",
            ));
        }
        let entries = self.read_newest_first()?;
        let have = entries.len();
        let pos = index - 1;
        entries.into_iter().nth(pos).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("history entry {index} not found (have {have} entr(y/ies))"),
            )
        })
    }

    /// Remove the history file if present.
    ///
    /// # Errors
    ///
    /// Returns I/O errors from [`std::fs::remove_file`].
    pub fn clear(&self) -> io::Result<()> {
        let Some(path) = self.path() else {
            return Ok(());
        };
        match fs::remove_file(&path) {
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
            other => other,
        }
    }

    fn read_newest_first(&self) -> io::Result<Vec<CliQueryHistoryEntry>> {
        let Some(path) = self.path() else {
            return Ok(Vec::new());
        };
        if !path.is_file() {
            return Ok(Vec::new());
        }
        let mut entries = read_entries(&path)?;
        entries.reverse();
        Ok(entries)
    }
}

/// One row in the CLI query history file (newest first when listed).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CliQueryHistoryEntry {
    /// Unix seconds (UTC).
    pub at: u64,
    /// `.tet` path when `-t` was passed.
    pub tet: Option<String>,
    /// Whether `-x` / `--execute` was set.
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

/// Resolved retention cap ([`HistorySettings::from_env`].[`cli_query_max`](HistorySettings::cli_query_max)).
#[must_use]
pub fn cli_query_history_max() -> usize {
    HistorySettings::from_env().cli_query_max
}

/// Path to the JSONL history file ([`HistorySettings::from_env`].[`path`](HistorySettings::path)).
#[must_use]
pub fn cli_query_history_path() -> Option<PathBuf> {
    HistorySettings::from_env().path()
}

/// Append using [`HistorySettings::from_env`].
pub fn append_cli_query_history(
    query: &QueryDocument,
    tet: Option<&str>,
    execute: bool,
) -> io::Result<()> {
    HistorySettings::from_env().append(query, tet, execute)
}

/// List using [`HistorySettings::from_env`].
pub fn list_cli_query_history(limit: usize, all: bool) -> io::Result<Vec<CliQueryHistoryEntry>> {
    HistorySettings::from_env().list(limit, all)
}

/// Get entry using [`HistorySettings::from_env`].
pub fn get_cli_query_history_entry(index: usize) -> io::Result<CliQueryHistoryEntry> {
    HistorySettings::from_env().get(index)
}

/// Clear using [`HistorySettings::from_env`].
pub fn clear_cli_query_history() -> io::Result<()> {
    HistorySettings::from_env().clear()
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

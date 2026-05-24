//! CLI-only recent query log (platform cache file, not stored in `.tet`).

use std::fmt::Write as _;
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::query::engine::spill_policy::platform_tetration_cache_dir;
use crate::query::types::{AxisSlice, Operation, QueryDocument};

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
            cli_query_max: 50,
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
        if let Some(last) = entries.last_mut()
            && entries_equivalent(last, &entry)
        {
            last.at = entry.at;
            return write_entries(&path, &entries);
        }
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

/// Compact table for `tet history list` (human default).
#[must_use]
pub fn format_history_list_text(
    entries: &[CliQueryHistoryEntry],
    path: Option<&Path>,
    settings: &HistorySettings,
) -> String {
    let mut out = String::new();
    if let Some(path) = path {
        let _ = writeln!(out, "file: {}", path.display());
    }
    let _ = writeln!(
        out,
        "shown: {}  keep: {} on disk",
        entries.len(),
        settings.cli_query_max
    );
    if entries.is_empty() {
        out.push_str("(empty — run `tet query … -t file.tet -x` to record)\n");
        return out;
    }
    out.push('\n');
    let _ = writeln!(
        out,
        "{:>3}  {:^1}  {:<18}  {:<10}  {:<8}  tet",
        "#", "x", "dataset", "op", "select"
    );
    for (i, e) in entries.iter().enumerate() {
        let mode = if e.execute { "x" } else { "p" };
        let op = operation_label(e.query.operation.as_ref());
        let sel = selection_label(e.query.selection.as_ref());
        let tet = e
            .tet
            .as_deref()
            .map(|p| {
                Path::new(p)
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or(p)
            })
            .unwrap_or("-");
        let _ = writeln!(
            out,
            "{:>3}  {:^1}  {:<18}  {:<10}  {:<8}  {}",
            i + 1,
            mode,
            truncate_field(&e.query.dataset, 18),
            op,
            sel,
            tet
        );
    }
    out.push_str("\nreplay: tet history run <#>  (1 = newest)\n");
    out
}

/// Full JSON envelope for `tet history list --json`.
pub fn format_history_list_json(
    entries: &[CliQueryHistoryEntry],
    path: Option<&Path>,
    settings: &HistorySettings,
) -> Result<String, String> {
    let out = serde_json::json!({
        "path": path.map(|p| p.display().to_string()),
        "settings": settings,
        "shown": entries.len(),
        "entries": entries,
    });
    serde_json::to_string_pretty(&out).map_err(|e| e.to_string())
}

/// Same `.tet`, execute flag, and query document as an existing row (consecutive dedup).
fn entries_equivalent(a: &CliQueryHistoryEntry, b: &CliQueryHistoryEntry) -> bool {
    a.tet == b.tet && a.execute == b.execute && history_queries_equal(&a.query, &b.query)
}

fn history_queries_equal(a: &QueryDocument, b: &QueryDocument) -> bool {
    match (serde_json::to_string(a), serde_json::to_string(b)) {
        (Ok(left), Ok(right)) => left == right,
        _ => false,
    }
}

fn operation_label(op: Option<&Operation>) -> &'static str {
    match op {
        None => "-",
        Some(Operation::Sum { .. }) => "sum",
        Some(Operation::Mean { .. }) => "mean",
        Some(Operation::Min { .. }) => "min",
        Some(Operation::Max { .. }) => "max",
        Some(Operation::Count { .. }) => "count",
        Some(Operation::Var { .. }) => "var",
        Some(Operation::Std { .. }) => "std",
        Some(Operation::Product { .. }) => "product",
        Some(Operation::NormL1 { .. }) => "norm_l1",
        Some(Operation::NormL2 { .. }) => "norm_l2",
        Some(Operation::AllFinite { .. }) => "all_finite",
        Some(Operation::AnyNan { .. }) => "any_nan",
        Some(Operation::ArgMin { .. }) => "arg_min",
        Some(Operation::ArgMax { .. }) => "arg_max",
        Some(Operation::Median { .. }) => "median",
        Some(Operation::Quantile { .. }) => "quantile",
        Some(Operation::Histogram { .. }) => "histogram",
    }
}

fn selection_label(sel: Option<&Vec<AxisSlice>>) -> &'static str {
    match sel {
        None => "full",
        Some(v) if v.is_empty() => "full",
        Some(v) if v.iter().any(|s| s.step.is_some_and(|st| st > 1)) => "strided",
        Some(_) => "subset",
    }
}

fn truncate_field(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_owned();
    }
    let mut end = max.saturating_sub(1);
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &s[..end])
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

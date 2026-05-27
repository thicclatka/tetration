//! CLI-only recent query log (platform cache file, not stored in `.tet`).

use std::fmt::Write as _;
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::query::engine::spill_policy::platform_tetration_cache_dir;
use crate::query::types::{AxisSlice, Operation, QueryDocument};

use super::text::{contains_ascii_case_insensitive, truncate_field};

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
    /// When `all` is true, returns every retained row that matches `filter` (up to
    /// [`Self::cli_query_max`] on disk). Otherwise returns at most `limit` matching rows.
    ///
    /// # Errors
    ///
    /// Returns I/O or JSON errors when the history file cannot be read.
    pub fn list(
        &self,
        limit: usize,
        all: bool,
        filter: Option<&HistoryListFilter>,
    ) -> io::Result<Vec<CliQueryHistoryEntry>> {
        let mut entries = self.read_newest_first()?;
        if let Some(f) = filter {
            entries.retain(|e| f.matches(e));
        }
        if !all {
            entries.truncate(limit);
        }
        Ok(entries)
    }

    /// Fetch one row by display index (**1** = newest, same order as [`Self::list`] with the same `filter`).
    ///
    /// # Errors
    ///
    /// Returns I/O or JSON errors when the file cannot be read. Returns
    /// [`io::ErrorKind::NotFound`] when the index is out of range or history is empty.
    pub fn get(
        &self,
        index: usize,
        filter: Option<&HistoryListFilter>,
    ) -> io::Result<CliQueryHistoryEntry> {
        if index == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "history index must be >= 1 (1 = newest)",
            ));
        }
        let entries = self.list(usize::MAX, true, filter)?;
        let have = entries.len();
        let pos = index - 1;
        entries.into_iter().nth(pos).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("history entry {index} not found (have {have} matching entr(y/ies))"),
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

/// Optional filters for [`HistorySettings::list`] / [`HistorySettings::get`] (AND semantics).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HistoryListFilter {
    /// Case-insensitive substring on `query.dataset`.
    pub dataset: Option<String>,
    /// Case-insensitive substring on the saved `.tet` path.
    pub tet: Option<String>,
    /// Execute vs plan-only rows.
    pub mode: Option<HistoryExecuteFilter>,
    /// Case-insensitive substring across dataset, tet path, and operation label.
    pub grep: Option<String>,
}

impl HistoryListFilter {
    /// True when every set predicate matches `entry`.
    #[must_use]
    pub fn matches(&self, entry: &CliQueryHistoryEntry) -> bool {
        if let Some(needle) = self.dataset.as_deref()
            && !contains_ascii_case_insensitive(&entry.query.dataset, needle)
        {
            return false;
        }
        if let Some(needle) = self.tet.as_deref() {
            let hay = entry.tet.as_deref().unwrap_or("");
            if !contains_ascii_case_insensitive(hay, needle) {
                return false;
            }
        }
        if let Some(mode) = self.mode {
            let is_execute = entry.execute;
            match mode {
                HistoryExecuteFilter::Execute if !is_execute => return false,
                HistoryExecuteFilter::Plan if is_execute => return false,
                _ => {}
            }
        }
        if let Some(needle) = self.grep.as_deref() {
            let op = operation_label(entry.query.operation.as_ref());
            let hay = format!(
                "{} {} {}",
                entry.query.dataset,
                entry.tet.as_deref().unwrap_or(""),
                op
            );
            if !contains_ascii_case_insensitive(&hay, needle) {
                return false;
            }
        }
        true
    }

    /// Human summary for list headers (empty when no predicates).
    #[must_use]
    pub fn summary(&self) -> String {
        let mut parts = Vec::new();
        if let Some(d) = &self.dataset {
            parts.push(format!("dataset~{d}"));
        }
        if let Some(t) = &self.tet {
            parts.push(format!("tet~{t}"));
        }
        if let Some(m) = self.mode {
            let execute = matches!(m, HistoryExecuteFilter::Execute);
            parts.push(format!("mode={}", history_entry_mode(execute)));
        }
        if let Some(g) = &self.grep {
            parts.push(format!("grep~{g}"));
        }
        parts.join(" ")
    }

    /// True when no filter fields are set.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.dataset.is_none() && self.tet.is_none() && self.mode.is_none() && self.grep.is_none()
    }
}

/// `x` / execute vs `p` / plan-only filter for history list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HistoryExecuteFilter {
    Execute,
    Plan,
}

/// Parse `x`, `execute`, `p`, or `plan` for `--mode`.
///
/// # Errors
///
/// Returns a message when `s` is not a known mode token.
pub fn parse_history_execute_filter(s: &str) -> Result<HistoryExecuteFilter, String> {
    match s.trim().to_ascii_lowercase().as_str() {
        "x" | "execute" => Ok(HistoryExecuteFilter::Execute),
        "p" | "plan" => Ok(HistoryExecuteFilter::Plan),
        other => Err(format!(
            "unknown history mode {other:?}; expected x, execute, p, or plan"
        )),
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

/// `x` = execute (`-x`), `p` = plan-only (matches `tet qhist list` mode column).
#[must_use]
pub fn history_entry_mode(execute: bool) -> &'static str {
    if execute { "x" } else { "p" }
}

const HISTORY_MODE_LEGEND: &str = "mode: x = had -x (execute), p = plan only (no -x)";

/// Row shape for `tet qhist list --json` (adds human `mode` alongside stored `execute`).
#[derive(Serialize)]
struct HistoryListRow<'a> {
    at: u64,
    mode: &'static str,
    tet: Option<&'a str>,
    query: &'a QueryDocument,
}

impl<'a> From<&'a CliQueryHistoryEntry> for HistoryListRow<'a> {
    fn from(e: &'a CliQueryHistoryEntry) -> Self {
        Self {
            at: e.at,
            mode: history_entry_mode(e.execute),
            tet: e.tet.as_deref(),
            query: &e.query,
        }
    }
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
///
/// # Errors
///
/// Same as [`HistorySettings::append`].
pub fn append_cli_query_history(
    query: &QueryDocument,
    tet: Option<&str>,
    execute: bool,
) -> io::Result<()> {
    HistorySettings::from_env().append(query, tet, execute)
}

/// List using [`HistorySettings::from_env`].
///
/// # Errors
///
/// Same as [`HistorySettings::list`].
pub fn list_cli_query_history(
    limit: usize,
    all: bool,
    filter: Option<&HistoryListFilter>,
) -> io::Result<Vec<CliQueryHistoryEntry>> {
    HistorySettings::from_env().list(limit, all, filter)
}

/// Get entry using [`HistorySettings::from_env`].
///
/// # Errors
///
/// Same as [`HistorySettings::get`].
pub fn get_cli_query_history_entry(
    index: usize,
    filter: Option<&HistoryListFilter>,
) -> io::Result<CliQueryHistoryEntry> {
    HistorySettings::from_env().get(index, filter)
}

/// Clear using [`HistorySettings::from_env`].
///
/// # Errors
///
/// Same as [`HistorySettings::clear`].
pub fn clear_cli_query_history() -> io::Result<()> {
    HistorySettings::from_env().clear()
}

/// Compact table for `tet qhist list` (human default).
#[must_use]
pub fn format_history_list_text(
    entries: &[CliQueryHistoryEntry],
    path: Option<&Path>,
    settings: &HistorySettings,
    filter: Option<&HistoryListFilter>,
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
    if let Some(f) = filter.filter(|f| !f.is_empty()) {
        let _ = writeln!(out, "filter: {}", f.summary());
    }
    if entries.is_empty() {
        if filter.is_some_and(|f| !f.is_empty()) {
            out.push_str("(no rows match filter)\n");
        } else {
            out.push_str("(empty — run `tet query … -t file.tet -x` to record)\n");
        }
        return out;
    }
    out.push('\n');
    let _ = writeln!(
        out,
        "{:>3}  {:^4}  {:<18}  {:<10}  {:<8}  tet",
        "#", "mode", "dataset", "op", "select"
    );
    for (i, e) in entries.iter().enumerate() {
        let mode = history_entry_mode(e.execute);
        let op = operation_label(e.query.operation.as_ref());
        let sel = selection_label(e.query.selection.as_ref());
        let tet = e.tet.as_deref().map_or("-", |p| {
            Path::new(p)
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or(p)
        });
        let _ = writeln!(
            out,
            "{:>3}  {:^4}  {:<18}  {:<10}  {:<8}  {}",
            i + 1,
            mode,
            truncate_field(&e.query.dataset, 18),
            op,
            sel,
            tet
        );
    }
    out.push('\n');
    let _ = writeln!(out, "{HISTORY_MODE_LEGEND}");
    out.push_str("replay: tet qhist run <#>  (1 = newest)\n");
    out
}

/// Full JSON envelope for `tet qhist list --json`.
///
/// # Errors
///
/// Returns a serialization error string when JSON encoding fails.
pub fn format_history_list_json(
    entries: &[CliQueryHistoryEntry],
    path: Option<&Path>,
    settings: &HistorySettings,
    filter: Option<&HistoryListFilter>,
) -> Result<String, String> {
    let rows: Vec<HistoryListRow<'_>> = entries.iter().map(HistoryListRow::from).collect();
    let filter_summary = filter
        .filter(|f| !f.is_empty())
        .map(HistoryListFilter::summary);
    let out = serde_json::json!({
        "path": path.map(|p| p.display().to_string()),
        "settings": settings,
        "shown": entries.len(),
        "filter": filter_summary,
        "mode_key": {
            "x": "execute (-x was set)",
            "p": "plan only (no -x)",
        },
        "entries": rows,
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
        Some(op) => op.wire_key(),
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

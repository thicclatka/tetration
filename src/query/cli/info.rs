//! CLI formatting for `tet info` (`.tet` catalog summary).

use std::fmt::Write as _;
use std::path::Path;

use crate::catalog::{
    CHUNK_PAYLOAD_CODEC_V1, ChunkIndexEntryV1, DATASET_DTYPE_TAG_V1, DatasetRecordV1,
    FileExecutionSettingsV1, HistoryEventV1, TetFileSummaryV1,
};
use crate::layout::SuperblockV1;

use super::text::{contains_ascii_case_insensitive, truncate_field};

/// Default max chunk rows in the text table (`tet info --chunks`).
pub const DEFAULT_INFO_CHUNK_TABLE_LIMIT: usize = 32;

/// Which sections to print in text mode (`tet info` without `--json`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[allow(clippy::struct_excessive_bools)]
pub struct InfoViewSections {
    pub layout: bool,
    pub execution: bool,
    pub datasets: bool,
    pub chunks: bool,
    pub history: bool,
}

impl InfoViewSections {
    /// Default text view: dataset catalog table only (plus file header).
    #[must_use]
    pub const fn default_table() -> Self {
        Self {
            layout: false,
            execution: false,
            datasets: true,
            chunks: false,
            history: false,
        }
    }

    /// All text sections (chunk table still obeys `chunk_limit`).
    #[must_use]
    pub const fn all() -> Self {
        Self {
            layout: true,
            execution: true,
            datasets: true,
            chunks: true,
            history: true,
        }
    }
}

/// Filters for dataset / chunk listing (case-insensitive substrings).
#[derive(Debug, Clone, Default)]
pub struct InfoListFilter {
    /// Substring on dataset name.
    pub dataset: Option<String>,
    /// Substring on dataset name or dtype label.
    pub grep: Option<String>,
}

impl InfoListFilter {
    /// True when no predicates are set.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.dataset.is_none() && self.grep.is_none()
    }

    /// Human summary for headers (empty when no predicates).
    #[must_use]
    pub fn summary(&self) -> String {
        let mut parts = Vec::new();
        if let Some(d) = &self.dataset {
            parts.push(format!("dataset~{d}"));
        }
        if let Some(g) = &self.grep {
            parts.push(format!("grep~{g}"));
        }
        parts.join(" ")
    }

    /// Whether `rec` matches all set predicates.
    #[must_use]
    pub fn matches_dataset(&self, rec: &DatasetRecordV1) -> bool {
        if let Some(needle) = self.dataset.as_deref()
            && !contains_ascii_case_insensitive(&rec.name, needle)
        {
            return false;
        }
        if let Some(needle) = self.grep.as_deref() {
            let hay = format!("{} {}", rec.name, dtype_label(rec.dtype));
            if !contains_ascii_case_insensitive(&hay, needle) {
                return false;
            }
        }
        true
    }
}

/// Resolve text sections from CLI flags (`--all`, `--layout`, …).
#[must_use]
#[allow(clippy::fn_params_excessive_bools)]
pub fn info_view_sections_from_flags(
    all: bool,
    layout: bool,
    execution: bool,
    datasets: bool,
    chunks: bool,
    history: bool,
) -> InfoViewSections {
    if all {
        return InfoViewSections::all();
    }
    let any = layout || execution || datasets || chunks || history;
    if any {
        InfoViewSections {
            layout,
            execution,
            datasets,
            chunks,
            history,
        }
    } else {
        InfoViewSections::default_table()
    }
}

/// Pretty JSON envelope (`tet info --json`).
///
/// # Errors
///
/// JSON serialization error.
pub fn format_info_json(
    path: Option<&Path>,
    file_len: u64,
    summary: &TetFileSummaryV1,
    filter: Option<&InfoListFilter>,
) -> Result<String, String> {
    let filtered = filtered_summary(summary, filter);
    let out = serde_json::json!({
        "path": path.map(|p| p.display().to_string()),
        "file_len": file_len,
        "filter": filter.filter(|f| !f.is_empty()).map(InfoListFilter::summary),
        "summary": filtered,
    });
    serde_json::to_string_pretty(&out).map_err(|e| e.to_string())
}

/// One-line summary (`tet info -q`).
#[must_use]
pub fn format_info_quiet(
    path: Option<&Path>,
    file_len: u64,
    summary: &TetFileSummaryV1,
    filter: Option<&InfoListFilter>,
) -> String {
    let filtered = filtered_summary(summary, filter);
    let path_s = path.map_or("-".to_owned(), |p| p.display().to_string());
    format!(
        "path={path_s} file_len={file_len} layout={} datasets={} chunks={} history={}",
        summary.superblock.layout_version,
        filtered.datasets.len(),
        filtered.chunks.len(),
        filtered.history.len()
    )
}

/// Multi-section text report (default `tet info`).
#[must_use]
pub fn format_info_text(
    path: Option<&Path>,
    file_len: u64,
    summary: &TetFileSummaryV1,
    filter: Option<&InfoListFilter>,
    sections: InfoViewSections,
    chunk_limit: usize,
) -> String {
    let filtered = filtered_summary(summary, filter);
    let mut out = String::new();
    if let Some(p) = path {
        let _ = writeln!(out, "file: {}", p.display());
    }
    let _ = writeln!(
        out,
        "size: {} bytes  layout_version: {}  datasets: {}  chunks: {}",
        file_len,
        summary.superblock.layout_version,
        summary.datasets.len(),
        summary.chunks.len()
    );
    if let Some(f) = filter.filter(|f| !f.is_empty()) {
        let _ = writeln!(out, "filter: {}", f.summary());
    }
    if sections.layout {
        out.push('\n');
        out.push_str(&format_layout_section(&summary.superblock));
    }
    if sections.execution {
        out.push('\n');
        out.push_str(&format_execution_section(summary.file_execution));
    }
    if sections.datasets {
        out.push('\n');
        out.push_str(&format_datasets_table(
            &filtered.datasets,
            &filtered.chunks,
            filter,
        ));
    }
    if sections.chunks {
        out.push('\n');
        out.push_str(&format_chunks_table(
            &filtered.datasets,
            &filtered.chunks,
            chunk_limit,
        ));
    }
    if sections.history {
        out.push('\n');
        out.push_str(&format_history_section(&filtered.history));
    }
    if !out.ends_with('\n') {
        out.push('\n');
    }
    out
}

fn filtered_summary(
    summary: &TetFileSummaryV1,
    filter: Option<&InfoListFilter>,
) -> TetFileSummaryV1 {
    let Some(f) = filter.filter(|x| !x.is_empty()) else {
        return summary.clone();
    };
    let mut datasets = Vec::new();
    let mut id_map = std::collections::HashMap::new();
    for (old_id, rec) in summary.datasets.iter().enumerate() {
        if f.matches_dataset(rec) {
            id_map.insert(old_id as u64, datasets.len() as u64);
            datasets.push(rec.clone());
        }
    }
    let chunks = summary
        .chunks
        .iter()
        .filter_map(|c| {
            let new_id = *id_map.get(&c.dataset_id)?;
            let mut ch = c.clone();
            ch.dataset_id = new_id;
            Some(ch)
        })
        .collect();
    TetFileSummaryV1 {
        superblock: summary.superblock.clone(),
        datasets,
        chunks,
        file_execution: summary.file_execution,
        history: summary.history.clone(),
    }
}

fn format_layout_section(sb: &SuperblockV1) -> String {
    let mut out = String::new();
    out.push_str("layout:\n");
    let _ = writeln!(out, "  layout_version:    {}", sb.layout_version);
    let _ = writeln!(out, "  dataset_count:     {}", sb.dataset_count);
    let _ = writeln!(
        out,
        "  flags:             {} (history_footer={})",
        sb.flags,
        sb.flags & crate::layout::SUPERBLOCK_FLAG_HISTORY_FOOTER != 0
    );
    let _ = writeln!(out, "  chunk_index_offset: {}", sb.chunk_index_offset);
    let _ = writeln!(out, "  chunk_index_length: {}", sb.chunk_index_length);
    out
}

fn format_execution_section(ex: FileExecutionSettingsV1) -> String {
    let mut out = String::new();
    out.push_str("execution (file defaults):\n");
    if ex.memory_budget_bytes == 0 && ex.memory_budget_percent_bps == 0 {
        out.push_str("  (engine defaults — 25% host RAM when unset in query JSON)\n");
        return out;
    }
    if ex.memory_budget_percent_bps != 0 {
        let pct = f64::from(ex.memory_budget_percent_bps) / 100.0;
        let _ = writeln!(out, "  memory_budget_percent: {pct}");
    }
    if ex.memory_budget_bytes != 0 {
        let _ = writeln!(out, "  memory_budget_bytes: {}", ex.memory_budget_bytes);
    }
    out
}

fn format_datasets_table(
    datasets: &[DatasetRecordV1],
    chunks: &[ChunkIndexEntryV1],
    filter: Option<&InfoListFilter>,
) -> String {
    let mut out = String::new();
    if datasets.is_empty() {
        if filter.is_some_and(|f| !f.is_empty()) {
            out.push_str("datasets: (no rows match filter)\n");
        } else {
            out.push_str("datasets: (empty file)\n");
        }
        return out;
    }
    out.push_str("datasets:\n");
    let _ = writeln!(
        out,
        "  {:>3}  {:<20}  {:<5}  {:<16}  {:<12}  chunks",
        "id", "name", "dtype", "shape", "chunk_shape"
    );
    for (id, ds) in datasets.iter().enumerate() {
        let n_chunks = chunks.iter().filter(|c| c.dataset_id == id as u64).count();
        let _ = writeln!(
            out,
            "  {:>3}  {:<20}  {:<5}  {:<16}  {:<12}  {}",
            id,
            truncate_field(&ds.name, 20),
            dtype_label(ds.dtype),
            shape_label(&ds.shape),
            shape_label(&ds.chunk_shape),
            n_chunks
        );
    }
    out
}

fn format_chunks_table(
    datasets: &[DatasetRecordV1],
    chunks: &[ChunkIndexEntryV1],
    limit: usize,
) -> String {
    let mut out = String::new();
    if chunks.is_empty() {
        out.push_str("chunks: (none)\n");
        return out;
    }
    let show = if limit == 0 {
        chunks.len()
    } else {
        limit.min(chunks.len())
    };
    out.push_str("chunks:\n");
    if show < chunks.len() {
        let _ = writeln!(
            out,
            "  (showing {show} of {}; use -n 0 for all)",
            chunks.len()
        );
    }
    let _ = writeln!(
        out,
        "  {:>4}  {:<12}  {:<8}  {:>10}  {:>10}  {:>5}  offset",
        "#", "dataset", "coords", "raw", "stored", "codec"
    );
    for (i, ch) in chunks.iter().take(show).enumerate() {
        let ds_name = datasets
            .get(ch.dataset_id as usize)
            .map_or("?", |d| d.name.as_str());
        let _ = writeln!(
            out,
            "  {:>4}  {:<12}  {:<8}  {:>10}  {:>10}  {:>5}  {}",
            i,
            truncate_field(ds_name, 12),
            chunk_coords_label(ch, datasets.get(ch.dataset_id as usize)),
            ch.raw_byte_len,
            ch.stored_byte_len,
            codec_label(ch.codec),
            ch.payload_offset
        );
    }
    out
}

fn format_history_section(history: &[HistoryEventV1]) -> String {
    let mut out = String::new();
    out.push_str("history:\n");
    if history.is_empty() {
        out.push_str("  (none)\n");
        return out;
    }
    for (i, ev) in history.iter().enumerate() {
        let (kind, fmt, ts) = ev;
        let _ = writeln!(out, "  {:>3}  {kind}  {fmt}  at={ts}", i + 1);
    }
    out
}

fn dtype_label(tag: u32) -> &'static str {
    let t = DATASET_DTYPE_TAG_V1;
    if t.is_f32(tag) {
        "f32"
    } else if t.is_f64(tag) {
        "f64"
    } else if t.is_i32(tag) {
        "i32"
    } else if t.is_i64(tag) {
        "i64"
    } else {
        "?"
    }
}

fn codec_label(codec: u32) -> &'static str {
    let c = CHUNK_PAYLOAD_CODEC_V1;
    if c.is_raw(codec) {
        "raw"
    } else if c.is_zstd(codec) {
        "zstd"
    } else {
        "?"
    }
}

fn shape_label(shape: &[u64]) -> String {
    if shape.is_empty() {
        return "-".to_owned();
    }
    shape
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join("×")
}

fn chunk_coords_label(ch: &ChunkIndexEntryV1, ds: Option<&DatasetRecordV1>) -> String {
    let rank = ds.map_or(0, |d| d.shape.len());
    if rank == 0 {
        return "-".to_owned();
    }
    ch.chunk_index[..rank]
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

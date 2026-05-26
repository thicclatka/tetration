//! `tet verify` stdout formatters.

use std::fmt::Write;

use super::report::{TetVerifyReport, VerifySeverity};

/// Pretty JSON for `tet verify --json`.
///
/// # Errors
///
/// JSON serialization error.
pub fn format_verify_json(report: &TetVerifyReport) -> Result<String, String> {
    serde_json::to_string_pretty(report).map_err(|e| e.to_string())
}

/// Human-readable report for default `tet verify` stdout.
#[must_use]
pub fn format_verify_text(report: &TetVerifyReport) -> String {
    let mut out = String::new();
    if let Some(p) = &report.path {
        let _ = writeln!(out, "file: {p}");
    }
    let _ = writeln!(out, "size: {} bytes", report.file_len);
    let _ = write!(
        out,
        "{}",
        if report.ok {
            "status: ok\n"
        } else {
            "status: failed\n"
        }
    );

    if !report.findings.is_empty() {
        out.push_str("checks:\n");
        for f in &report.findings {
            let mark = if !f.ok {
                "FAIL"
            } else if f.severity == VerifySeverity::Warning {
                "warn"
            } else {
                "ok"
            };
            if let Some(d) = &f.detail {
                let _ = writeln!(out, "  {:<22} {mark} ({d})", f.check);
            } else {
                let _ = writeln!(out, "  {:<22} {mark}", f.check);
            }
        }
    }

    if !report.recommendations.is_empty() {
        out.push_str("recommendations:\n");
        for r in &report.recommendations {
            let _ = writeln!(out, "  [{}] {}", r.code, r.message);
            if let Some(fix) = &r.fix {
                let _ = writeln!(out, "       → {}", fix.summary);
                if let Some(cmd) = &fix.command {
                    let _ = writeln!(out, "       → {cmd}");
                }
            }
        }
    }

    if let Some(s) = &report.summary {
        let _ = writeln!(
            out,
            "summary: layout={} datasets={} chunks={} history={} metadata={} footer={} deep_decode={}",
            s.layout_version,
            s.dataset_count,
            s.chunk_count,
            s.history_events,
            s.has_metadata,
            s.history_footer,
            s.deep_chunk_decode
        );
    }
    out
}

/// One-line summary for `tet verify -q`.
#[must_use]
pub fn format_verify_quiet(report: &TetVerifyReport) -> String {
    let path = report.path.as_deref().unwrap_or("-");
    let status = if report.ok { "ok" } else { "failed" };
    let datasets = report
        .summary
        .as_ref()
        .map_or_else(|| "?".to_owned(), |s| s.dataset_count.to_string());
    let chunks = report
        .summary
        .as_ref()
        .map_or_else(|| "?".to_owned(), |s| s.chunk_count.to_string());
    let recs = report.recommendations.len();
    format!(
        "path={path} status={status} datasets={datasets} chunks={chunks} recommendations={recs}"
    )
}

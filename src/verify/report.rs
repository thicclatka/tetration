//! Verification report types (findings, recommendations, summary).

use serde::Serialize;

/// Severity of a verification finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum VerifySeverity {
    /// File is not safe to use; `TetVerifyReport::ok` is false.
    Error,
    /// Layout is usable but something is suboptimal or worth reviewing.
    Warning,
    /// Informational note (does not affect `ok`).
    Info,
}

/// One check result (pass or fail with detail).
#[derive(Debug, Clone, Serialize)]
pub struct VerifyFinding {
    pub check: String,
    pub severity: VerifySeverity,
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// Actionable guidance derived from findings; see [`crate::repair`] for in-place fixes.
#[derive(Debug, Clone, Serialize)]
pub struct VerifyRecommendation {
    /// Stable tag for scripts (`reconvert`, `truncate_file`, …).
    pub code: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fix: Option<VerifyFixHint>,
}

/// Human-readable repair hint; `command` is filled when `tet repair` supports the code.
#[derive(Debug, Clone, Serialize)]
pub struct VerifyFixHint {
    pub summary: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,
}

/// High-level counts when structural verification succeeds.
#[derive(Debug, Clone, Serialize)]
pub struct VerifySummary {
    pub layout_version: u32,
    pub dataset_count: usize,
    pub chunk_count: usize,
    pub history_events: usize,
    pub has_metadata: bool,
    pub history_footer: bool,
    pub deep_chunk_decode: bool,
}

/// Full result of [`super::verify_tet_bytes`] / [`super::verify_tet_file`].
#[derive(Debug, Clone, Serialize)]
pub struct TetVerifyReport {
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    pub file_len: u64,
    pub findings: Vec<VerifyFinding>,
    pub recommendations: Vec<VerifyRecommendation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<VerifySummary>,
}

impl TetVerifyReport {
    pub(crate) fn from_fatal(
        path: Option<String>,
        file_len: u64,
        check: &str,
        message: String,
    ) -> Self {
        let findings = vec![VerifyFinding {
            check: check.to_owned(),
            severity: VerifySeverity::Error,
            ok: false,
            detail: Some(message),
        }];
        Self {
            ok: false,
            path,
            file_len,
            findings,
            recommendations: Vec::new(),
            summary: None,
        }
        .finalize()
    }

    /// Structural failure before a full summary is available.
    pub(crate) fn incomplete(
        path: Option<String>,
        file_len: u64,
        findings: Vec<VerifyFinding>,
    ) -> Self {
        Self {
            ok: false,
            path,
            file_len,
            findings,
            recommendations: Vec::new(),
            summary: None,
        }
        .finalize()
    }

    pub(crate) fn finalize(mut self) -> Self {
        self.recommendations = super::recommend::recommendations_for_findings(&self.findings);
        if let Some(path_s) = self.path.clone() {
            crate::repair::enrich_verify_recommendations(std::path::Path::new(&path_s), &mut self);
        }
        self.ok = !self
            .findings
            .iter()
            .any(|f| !f.ok && f.severity == VerifySeverity::Error);
        self
    }
}

pub(crate) fn ok_finding(check: &str, detail: Option<String>) -> VerifyFinding {
    VerifyFinding {
        check: check.to_owned(),
        severity: VerifySeverity::Info,
        ok: true,
        detail,
    }
}

pub(crate) fn err_finding(check: &str, detail: String) -> VerifyFinding {
    VerifyFinding {
        check: check.to_owned(),
        severity: VerifySeverity::Error,
        ok: false,
        detail: Some(detail),
    }
}

pub(crate) fn warn_finding(check: &str, detail: String) -> VerifyFinding {
    VerifyFinding {
        check: check.to_owned(),
        severity: VerifySeverity::Warning,
        ok: true,
        detail: Some(detail),
    }
}

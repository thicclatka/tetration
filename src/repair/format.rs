//! `tet repair` stdout formatters.

use std::fmt::Write;

use super::{RepairPlan, TetRepairReport};

/// Pretty JSON for `tet repair --json`.
///
/// # Errors
///
/// JSON serialization error.
pub fn format_repair_json(report: &TetRepairReport) -> Result<String, String> {
    serde_json::to_string_pretty(report).map_err(|e| e.to_string())
}

/// Pretty JSON for a repair plan.
///
/// # Errors
///
/// JSON serialization error.
pub fn format_plan_json(plan: &RepairPlan) -> Result<String, String> {
    serde_json::to_string_pretty(plan).map_err(|e| e.to_string())
}

/// Human-readable repair report.
#[must_use]
pub fn format_repair_text(report: &TetRepairReport) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "file: {}", report.path.display());
    let _ = write!(
        out,
        "{}",
        if report.dry_run {
            "mode: dry-run\n"
        } else {
            "mode: apply\n"
        }
    );
    for a in &report.actions {
        let status = if a.applied {
            "applied"
        } else if a.dry_run {
            "planned"
        } else {
            "skipped"
        };
        let _ = writeln!(out, "  [{}] {status}: {}", a.code, a.message);
    }
    if let Some(ok) = report.verify_after_ok {
        let _ = writeln!(out, "verify_after: {}", if ok { "ok" } else { "failed" });
    }
    out
}

/// Human-readable repair plan (default `tet repair` with no `--apply`).
#[must_use]
pub fn format_plan_text(plan: &RepairPlan) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "file: {}", plan.path);
    out.push_str("repair plan:\n");
    if plan.actions.is_empty() {
        out.push_str("  (no recommendations — run tet verify first or file is ok)\n");
        return out;
    }
    for a in &plan.actions {
        let tag = if a.repairable { "repairable" } else { "manual" };
        let _ = writeln!(out, "  [{tag}] {} — {}", a.code, a.summary);
    }
    out.push_str("\nUse: tet repair <path> --apply <code>\n");
    out
}

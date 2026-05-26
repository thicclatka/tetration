//! Repair plans derived from verify reports.

use std::path::Path;

use serde::Serialize;

use crate::verify::TetVerifyReport;

use super::{is_repairable_code, repair_command_line};

/// One planned repair action.
#[derive(Debug, Clone, Serialize)]
pub struct RepairAction {
    pub code: String,
    pub repairable: bool,
    pub summary: String,
}

/// Planned repairs for a file (from the latest verify report).
#[derive(Debug, Clone, Serialize)]
pub struct RepairPlan {
    pub path: String,
    pub actions: Vec<RepairAction>,
}

/// Build a repair plan from a verify report.
#[must_use]
pub fn repair_plan_from_verify(path: &Path, verify: &TetVerifyReport) -> RepairPlan {
    let path_s = path.display().to_string();
    let mut actions = Vec::new();
    for rec in &verify.recommendations {
        let repairable = is_repairable_code(&rec.code);
        let summary = rec
            .fix
            .as_ref()
            .map_or_else(|| rec.message.clone(), |f| f.summary.clone());
        let mut action = RepairAction {
            code: rec.code.clone(),
            repairable,
            summary,
        };
        if repairable {
            action.summary = format!(
                "{} — try: {}",
                action.summary,
                repair_command_line(path, &rec.code, true)
            );
        }
        actions.push(action);
    }
    RepairPlan {
        path: path_s,
        actions,
    }
}

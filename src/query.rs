//! JSON query documents: validated plans for reads and basic operations.

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum TetError {
    #[error("invalid query JSON: {0}")]
    InvalidJson(#[from] serde_json::Error),
    #[error("validation: {0}")]
    Validation(String),
}

/// Per-axis slice: `start` inclusive, `stop` exclusive, `step` ≥ 1 when present.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AxisSlice {
    pub start: Option<u64>,
    pub stop: Option<u64>,
    pub step: Option<u64>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Operation {
    Sum { axes: Vec<String> },
    Mean { axes: Vec<String> },
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputHint {
    InlineJson,
    SpillArray { handle: String },
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct OutputHints {
    #[serde(default)]
    pub preferred: Option<OutputHint>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct QueryDocument {
    #[serde(default)]
    pub layout_version: Option<u32>,
    pub dataset: String,
    #[serde(default)]
    pub selection: Option<Vec<AxisSlice>>,
    #[serde(default)]
    pub operation: Option<Operation>,
    #[serde(default)]
    pub output: Option<OutputHints>,
}

#[derive(Debug, Clone, Serialize)]
pub struct QueryResponse {
    pub status: &'static str,
    pub accepted: bool,
    pub layout_version: Option<u32>,
    pub dataset: String,
    pub selection_axes: Option<usize>,
    pub operation: Option<Operation>,
    pub message: String,
}

pub fn parse_query_json(text: &str) -> Result<QueryDocument, TetError> {
    Ok(serde_json::from_str(text)?)
}

pub fn validate_query(doc: &QueryDocument) -> Result<(), TetError> {
    if doc.dataset.trim().is_empty() {
        return Err(TetError::Validation(
            "`dataset` must be a non-empty string".into(),
        ));
    }
    if let Some(axes) = &doc.selection {
        for (i, sl) in axes.iter().enumerate() {
            if let Some(step) = sl.step
                && step == 0
            {
                return Err(TetError::Validation(format!(
                    "selection[{i}].step must be >= 1, got 0"
                )));
            }
            match (sl.start, sl.stop) {
                (Some(a), Some(b)) if a >= b => {
                    return Err(TetError::Validation(format!(
                        "selection[{i}]: start must be < stop when both set (got {a} >= {b})"
                    )));
                }
                _ => {}
            }
        }
    }
    Ok(())
}

/// Build a response echoing the plan. Execution against a real file is not wired yet.
pub fn plan_query(doc: &QueryDocument) -> QueryResponse {
    let axes = doc.selection.as_ref().map(Vec::len);
    QueryResponse {
        status: "planned",
        accepted: true,
        layout_version: doc.layout_version,
        dataset: doc.dataset.clone(),
        selection_axes: axes,
        operation: doc.operation.clone(),
        message: "query accepted and validated; mmap chunk engine and on-disk `.tet` I/O are not connected in this build".into(),
    }
}

//! JSON query document wire types (`QueryDocument`, selection slices, operations).
//!
//! Parsed by [`crate::query::parse_query_json`] and validated by [`crate::query::validate_query`].
//! See `docs/query_engine.md` for the full schema.

use serde::{Deserialize, Serialize};

// `QueryDocument` JSON wire format: see `document_wire.rs` (flat op keys, `mean: 0`, …).

/// Per-axis slice: `start` inclusive, `stop` exclusive, `step` ≥ 1 when present.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AxisSlice {
    pub start: Option<u64>,
    pub stop: Option<u64>,
    pub step: Option<u64>,
}

/// Reduction or aggregate over the logical selection.
///
/// Each variant carries `axes`: decimal dimension indices (`"0"`, `"1"`, …) to reduce along.
/// An empty `axes` list reduces all elements to a scalar. Tier-A/B ops stream over chunks;
/// tier-C ops (`median`, `quantile`, `histogram`) may materialize the full logical tensor.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Operation {
    Sum { axes: Vec<String> },
    Mean { axes: Vec<String> },
    Min { axes: Vec<String> },
    Max { axes: Vec<String> },
    Count { axes: Vec<String> },
    Var { axes: Vec<String> },
    Std { axes: Vec<String> },
    Product { axes: Vec<String> },
    NormL1 { axes: Vec<String> },
    NormL2 { axes: Vec<String> },
    AllFinite { axes: Vec<String> },
    AnyNan { axes: Vec<String> },
    ArgMin { axes: Vec<String> },
    ArgMax { axes: Vec<String> },
    Median { axes: Vec<String> },
    Quantile { axes: Vec<String>, q: f64 },
    Histogram { axes: Vec<String>, bins: u32 },
}

impl Operation {
    /// Per-axis decimal dimension indices for this operation.
    #[must_use]
    pub fn axes(&self) -> &[String] {
        match self {
            Self::Sum { axes }
            | Self::Mean { axes }
            | Self::Min { axes }
            | Self::Max { axes }
            | Self::Count { axes }
            | Self::Var { axes }
            | Self::Std { axes }
            | Self::Product { axes }
            | Self::NormL1 { axes }
            | Self::NormL2 { axes }
            | Self::AllFinite { axes }
            | Self::AnyNan { axes }
            | Self::ArgMin { axes }
            | Self::ArgMax { axes }
            | Self::Median { axes }
            | Self::Quantile { axes, .. }
            | Self::Histogram { axes, .. } => axes,
        }
    }
}

/// Caller preference for where large query results should land.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum OutputHint {
    /// Keep results inline in the JSON response.
    InlineJson,
    /// Spill a dense array to a caller-managed file identified by `handle`.
    SpillArray { handle: String },
}

/// Optional output routing hints on a query document.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(deny_unknown_fields)]
pub struct OutputHints {
    #[serde(default)]
    pub preferred: Option<OutputHint>,
}

/// Top-level JSON query document accepted by the engine and `tet query`.
#[derive(Debug, Clone)]
pub struct QueryDocument {
    /// Optional layout version hint (currently informational).
    pub layout_version: Option<u32>,
    /// Dataset name as stored in the `.tet` catalog.
    pub dataset: String,
    /// Per-axis half-open slices; omitted means the full dataset extent on each axis.
    pub selection: Option<Vec<AxisSlice>>,
    /// Optional reduction over the logical selection.
    pub operation: Option<Operation>,
    /// Optional output routing hints.
    pub output: Option<OutputHints>,
    /// Host-side memory budget overrides for execution.
    pub execution: Option<ExecutionHints>,
}

/// Host-side execution limits (JSON query document).
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(deny_unknown_fields)]
pub struct ExecutionHints {
    /// Max anonymous RAM for a dense in-memory logical `f32` buffer; overrides `.tet` header and percent defaults.
    #[serde(default)]
    pub memory_budget_bytes: Option<u64>,
    /// Share of host available RAM in basis points (10000 = 100%); overrides `.tet` header percent when no fixed bytes.
    #[serde(default)]
    pub memory_budget_percent_bps: Option<u16>,
}

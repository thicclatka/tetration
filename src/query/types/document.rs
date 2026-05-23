use serde::{Deserialize, Serialize};

/// Per-axis slice: `start` inclusive, `stop` exclusive, `step` ≥ 1 when present.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AxisSlice {
    pub start: Option<u64>,
    pub stop: Option<u64>,
    pub step: Option<u64>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
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
            | Self::Median { axes } => axes,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum OutputHint {
    InlineJson,
    SpillArray { handle: String },
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(deny_unknown_fields)]
pub struct OutputHints {
    #[serde(default)]
    pub preferred: Option<OutputHint>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
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
    #[serde(default)]
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

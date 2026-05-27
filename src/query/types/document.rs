//! JSON query document wire types (`QueryDocument`, selection slices, operations).
//!
//! Parsed by [`crate::query::parse_query_json`] and validated by [`crate::query::validate_query`].
//! See `docs/query_engine.md` for the full schema.

use serde::{Deserialize, Serialize};

use super::error::TetError;

// `QueryDocument` JSON wire format: see `document_wire.rs` (flat op keys, `mean: 0`, `spill: "…"`, …).

/// Per-axis slice: `start` inclusive, `stop` exclusive, `step` ≥ 1 when present.
///
/// Coordinate labels (`start_label` / `stop_label`) resolve to indices at plan time when footer
/// `coords` exist (axis key = `dim_names[d]` or decimal `"d"`).
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AxisSlice {
    pub start: Option<u64>,
    pub stop: Option<u64>,
    pub step: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start_label: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_label: Option<String>,
}

/// Reduction or aggregate over the logical selection.
///
/// Each variant carries `axes`: decimal dimension indices (`"0"`, `"1"`, …) to reduce along.
/// An empty `axes` list reduces all elements to a scalar. Tier-A/B ops stream over chunks;
/// tier-C ops (`median`, `quantile`, `histogram`) may materialize the full logical tensor.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Operation {
    Sum {
        axes: Vec<String>,
    },
    Mean {
        axes: Vec<String>,
    },
    Min {
        axes: Vec<String>,
    },
    Max {
        axes: Vec<String>,
    },
    Count {
        axes: Vec<String>,
    },
    Var {
        axes: Vec<String>,
    },
    Std {
        axes: Vec<String>,
    },
    Product {
        axes: Vec<String>,
    },
    NormL1 {
        axes: Vec<String>,
    },
    NormL2 {
        axes: Vec<String>,
    },
    AllFinite {
        axes: Vec<String>,
    },
    AnyNan {
        axes: Vec<String>,
    },
    ArgMin {
        axes: Vec<String>,
    },
    ArgMax {
        axes: Vec<String>,
    },
    Median {
        axes: Vec<String>,
    },
    Quantile {
        axes: Vec<String>,
        q: f64,
    },
    Histogram {
        axes: Vec<String>,
        bins: u32,
        /// When both set, bin edges span `[min, max]`; otherwise data min/max.
        min: Option<f64>,
        max: Option<f64>,
    },
    /// Count of NaN elements (float/`f16` wire tags; integers contribute 0).
    NanCount {
        axes: Vec<String>,
    },
    /// Count of ±infinity elements (float/`f16`; integers contribute 0).
    InfCount {
        axes: Vec<String>,
    },
    /// Count of elements equal to `fill` (resolved from query or dataset attrs at plan time).
    NullCount {
        axes: Vec<String>,
        fill: Option<f64>,
    },
    /// Population covariance matrix (variables × variables) with samples along `axes[0]`.
    Covariance {
        axes: Vec<String>,
    },
    /// Pearson correlation matrix with samples along `axes[0]`.
    Correlation {
        axes: Vec<String>,
    },
}

macro_rules! operation_axes_match {
    ($op:expr) => {
        match $op {
            Operation::Sum { axes }
            | Operation::Mean { axes }
            | Operation::Min { axes }
            | Operation::Max { axes }
            | Operation::Count { axes }
            | Operation::Var { axes }
            | Operation::Std { axes }
            | Operation::Product { axes }
            | Operation::NormL1 { axes }
            | Operation::NormL2 { axes }
            | Operation::AllFinite { axes }
            | Operation::AnyNan { axes }
            | Operation::ArgMin { axes }
            | Operation::ArgMax { axes }
            | Operation::Median { axes }
            | Operation::Quantile { axes, .. }
            | Operation::Histogram { axes, .. }
            | Operation::NanCount { axes }
            | Operation::InfCount { axes }
            | Operation::NullCount { axes, .. }
            | Operation::Covariance { axes }
            | Operation::Correlation { axes } => axes,
        }
    };
}

impl Operation {
    /// Per-axis decimal dimension indices for this operation.
    #[must_use]
    pub fn axes(&self) -> &[String] {
        operation_axes_match!(self)
    }

    /// Mutable axis list for plan-time name → index resolution.
    pub(crate) fn axes_mut(&mut self) -> &mut Vec<String> {
        operation_axes_match!(self)
    }

    /// Top-level JSON wire key (`"mean"`, `"arg_min"`, …).
    #[must_use]
    pub fn wire_key(&self) -> &'static str {
        match self {
            Self::Sum { .. } => "sum",
            Self::Mean { .. } => "mean",
            Self::Min { .. } => "min",
            Self::Max { .. } => "max",
            Self::Count { .. } => "count",
            Self::Var { .. } => "var",
            Self::Std { .. } => "std",
            Self::Product { .. } => "product",
            Self::NormL1 { .. } => "norm_l1",
            Self::NormL2 { .. } => "norm_l2",
            Self::AllFinite { .. } => "all_finite",
            Self::AnyNan { .. } => "any_nan",
            Self::NanCount { .. } => "nan_count",
            Self::InfCount { .. } => "inf_count",
            Self::NullCount { .. } => "null_count",
            Self::ArgMin { .. } => "arg_min",
            Self::ArgMax { .. } => "arg_max",
            Self::Median { .. } => "median",
            Self::Quantile { .. } => "quantile",
            Self::Histogram { .. } => "histogram",
            Self::Covariance { .. } => "covariance",
            Self::Correlation { .. } => "correlation",
        }
    }

    /// Tier-C ops that require a full logical materialize (not streaming fold).
    #[must_use]
    pub fn requires_materialize(&self) -> bool {
        matches!(
            self,
            Self::Median { .. }
                | Self::Quantile { .. }
                | Self::Histogram { .. }
                | Self::Covariance { .. }
                | Self::Correlation { .. }
        )
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
    /// When `Some(true)`, force parallel streaming fold; `Some(false)` force sequential; `None` = auto from RAM vs selection size.
    #[serde(default)]
    pub fold_parallel: Option<bool>,
    /// Device routing for tier-A/B reductions: `cpu`, `auto`, `cuda`, or `cuda:N` (Phase 10; wire via [`crate::query::document_wire`]).
    #[serde(default, skip)]
    pub device: Option<ExecutionDeviceHint>,
}

/// Parsed `execution.device` or CLI `--device` token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionDeviceHint {
    Cpu,
    Auto,
    Cuda(usize),
}

impl ExecutionDeviceHint {
    /// Parse `cpu`, `auto`, `cuda`, or `cuda:N`.
    ///
    /// # Errors
    ///
    /// Returns [`TetError::Validation`] when the token is unknown or malformed.
    pub fn parse(token: &str) -> Result<Self, TetError> {
        let t = token.trim();
        if t.is_empty() {
            return Err(TetError::Validation(
                "device token must not be empty".into(),
            ));
        }
        if t.eq_ignore_ascii_case("cpu") {
            return Ok(Self::Cpu);
        }
        if t.eq_ignore_ascii_case("auto") {
            return Ok(Self::Auto);
        }
        if t.eq_ignore_ascii_case("cuda") {
            return Ok(Self::Cuda(0));
        }
        if let Some(rest) = t.strip_prefix("cuda:") {
            let idx = rest.parse::<usize>().map_err(|_| {
                TetError::Validation(format!(
                    "invalid device `{token}` (expected cuda:N with non-negative N)"
                ))
            })?;
            return Ok(Self::Cuda(idx));
        }
        Err(TetError::Validation(format!(
            "unknown device `{token}` (expected cpu, auto, cuda, or cuda:N)"
        )))
    }

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Auto => "auto",
            Self::Cuda(0) => "cuda:0",
            Self::Cuda(n) => {
                // Stable tokens for JSON; cuda:1+ formatted at serialize time in wire layer.
                let _ = n;
                "cuda"
            }
        }
    }

    /// Wire/JSON token (includes index for `cuda:N`).
    #[must_use]
    pub fn to_token(self) -> String {
        match self {
            Self::Cpu => "cpu".to_string(),
            Self::Auto => "auto".to_string(),
            Self::Cuda(0) => "cuda".to_string(),
            Self::Cuda(n) => format!("cuda:{n}"),
        }
    }
}

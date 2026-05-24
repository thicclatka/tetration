//! Flat JSON query document wire format (v1).
//!
//! Top-level reduction keys (`mean`, `sum`, …) replace nested `operation`. Axis specs accept
//! `0`, `[0, 1]`, or `[]` (scalar). Parametric ops use `{ "q": …, "axis": … }`.

use serde::de::{self, Deserialize, Deserializer};
use serde::ser::{Serialize, SerializeMap, Serializer};
use serde_json::{Map, Value};

use super::document::QueryLimits;
use super::types::{AxisSlice, ExecutionHints, Operation, OutputHints, QueryDocument, TetError};

const OP_KEYS: &[&str] = &[
    "sum",
    "mean",
    "min",
    "max",
    "count",
    "var",
    "std",
    "product",
    "norm_l1",
    "norm_l2",
    "all_finite",
    "any_nan",
    "arg_min",
    "arg_max",
    "median",
    "quantile",
    "histogram",
];

const RESERVED_KEYS: &[&str] = &[
    "layout_version",
    "dataset",
    "selection",
    "output",
    "execution",
];

/// Parse a JSON value into a [`QueryDocument`].
///
/// # Errors
///
/// [`TetError::Validation`] or [`TetError::InvalidJson`].
pub fn parse_query_value(value_ref: &Value) -> Result<QueryDocument, TetError> {
    let obj = value_ref
        .as_object()
        .ok_or_else(|| TetError::Validation("query document must be a JSON object".into()))?;

    if obj.contains_key("operation") {
        return Err(TetError::Validation(
            "nested `operation` is not supported; use a top-level op key (e.g. `\"mean\": 0`)"
                .into(),
        ));
    }

    for key in obj.keys() {
        if !is_allowed_key(key) {
            return Err(TetError::Validation(format!("unknown field `{key}`")));
        }
    }

    let dataset = obj
        .get("dataset")
        .and_then(Value::as_str)
        .ok_or_else(|| TetError::Validation("`dataset` must be a string".into()))?
        .to_owned();

    let layout_version = match obj.get("layout_version") {
        None => None,
        Some(v) => Some(parse_u32_field("layout_version", v)?),
    };

    let selection = match obj.get("selection") {
        None => None,
        Some(v) => Some(parse_selection(v)?),
    };

    let output = match obj.get("output") {
        None => None,
        Some(v) => {
            let hints: OutputHints = serde_json::from_value(v.clone())?;
            Some(hints)
        }
    };

    let execution = match obj.get("execution") {
        None => None,
        Some(v) => Some(parse_execution(v)?),
    };

    let mut operation = None;
    for key in OP_KEYS {
        if let Some(v) = obj.get(*key) {
            if operation.is_some() {
                return Err(TetError::Validation(
                    "query document must include at most one reduction key".into(),
                ));
            }
            operation = Some(parse_operation(key, v)?);
        }
    }

    Ok(QueryDocument {
        layout_version,
        dataset,
        selection,
        operation,
        output,
        execution,
    })
}

fn is_allowed_key(key: &str) -> bool {
    RESERVED_KEYS.contains(&key) || OP_KEYS.contains(&key)
}

fn parse_selection(v: &Value) -> Result<Vec<AxisSlice>, TetError> {
    let slices: Vec<AxisSlice> = serde_json::from_value(v.clone())?;
    Ok(slices)
}

fn parse_u32_field(name: &str, v: &Value) -> Result<u32, TetError> {
    match v {
        Value::Number(n) => n
            .as_u64()
            .and_then(|u| u32::try_from(u).ok())
            .ok_or_else(|| {
                TetError::Validation(format!("`{name}` must be a non-negative integer"))
            }),
        _ => Err(TetError::Validation(format!("`{name}` must be a number"))),
    }
}

fn parse_execution(v: &Value) -> Result<ExecutionHints, TetError> {
    let obj = v
        .as_object()
        .ok_or_else(|| TetError::Validation("`execution` must be a JSON object".into()))?;

    for key in obj.keys() {
        if !matches!(
            key.as_str(),
            "memory_budget_bytes" | "memory_budget_percent" | "memory_budget_percent_bps"
        ) {
            return Err(TetError::Validation(format!(
                "unknown field `execution.{key}`"
            )));
        }
    }

    let memory_budget_bytes = match obj.get("memory_budget_bytes") {
        None => None,
        Some(v) => Some(parse_u64_field("execution.memory_budget_bytes", v)?),
    };

    let mut memory_budget_percent_bps = None;
    if let Some(v) = obj.get("memory_budget_percent_bps") {
        memory_budget_percent_bps = Some(parse_percent_bps_field(
            "execution.memory_budget_percent_bps",
            v,
        )?);
    }
    if let Some(v) = obj.get("memory_budget_percent") {
        if memory_budget_percent_bps.is_some() {
            return Err(TetError::Validation(
                "use either `execution.memory_budget_percent` or `execution.memory_budget_percent_bps`, not both".into(),
            ));
        }
        memory_budget_percent_bps = Some(percent_to_bps(parse_percent_field(
            "execution.memory_budget_percent",
            v,
        )?)?);
    }

    Ok(ExecutionHints {
        memory_budget_bytes,
        memory_budget_percent_bps,
    })
}

fn parse_u64_field(name: &str, v: &Value) -> Result<u64, TetError> {
    match v {
        Value::Number(n) => n.as_u64().ok_or_else(|| {
            TetError::Validation(format!("`{name}` must be a non-negative integer"))
        }),
        _ => Err(TetError::Validation(format!("`{name}` must be a number"))),
    }
}

fn parse_percent_field(name: &str, v: &Value) -> Result<f64, TetError> {
    let pct = match v {
        Value::Number(n) => n.as_f64(),
        _ => None,
    }
    .ok_or_else(|| TetError::Validation(format!("`{name}` must be a number")))?;
    if !(0.0..=100.0).contains(&pct) {
        return Err(TetError::Validation(format!(
            "`{name}` must be in [0, 100], got {pct}"
        )));
    }
    Ok(pct)
}

fn parse_percent_bps_field(name: &str, v: &Value) -> Result<u16, TetError> {
    let bps = parse_u64_field(name, v)?;
    let bps =
        u16::try_from(bps).map_err(|_| TetError::Validation(format!("`{name}` exceeds u16")))?;
    if bps > 10_000 {
        return Err(TetError::Validation(format!(
            "`{name}` must be <= 10000 (100%)"
        )));
    }
    Ok(bps)
}

fn percent_to_bps(percent: f64) -> Result<u16, TetError> {
    let bps = (percent * 100.0).round();
    let bps = u16::try_from(bps as u64)
        .map_err(|_| TetError::Validation("memory_budget_percent out of range".into()))?;
    if bps > 10_000 {
        return Err(TetError::Validation(
            "memory_budget_percent must be <= 100".into(),
        ));
    }
    Ok(bps)
}

fn parse_operation(name: &str, v: &Value) -> Result<Operation, TetError> {
    match name {
        "quantile" => {
            let (axes, obj) = parse_parametric_op_object("quantile", v)?;
            let q = obj
                .get("q")
                .and_then(Value::as_f64)
                .ok_or_else(|| TetError::Validation("`quantile.q` is required".into()))?;
            Ok(Operation::Quantile { axes, q })
        }
        "histogram" => {
            let (axes, obj) = parse_parametric_op_object("histogram", v)?;
            let bins = obj
                .get("bins")
                .and_then(Value::as_u64)
                .and_then(|u| u32::try_from(u).ok())
                .ok_or_else(|| TetError::Validation("`histogram.bins` is required".into()))?;
            Ok(Operation::Histogram { axes, bins })
        }
        other => {
            let axes = parse_axis_spec(v)?;
            Ok(operation_from_axes(other, axes))
        }
    }
}

fn parse_parametric_op_object(
    op: &str,
    v: &Value,
) -> Result<(Vec<String>, Map<String, Value>), TetError> {
    let obj = v.as_object().ok_or_else(|| {
        TetError::Validation(format!(
            "`{op}` must be a JSON object (e.g. {{ \"q\": 0.95, \"axis\": 0 }})"
        ))
    })?;
    let axes = if let Some(axis) = obj.get("axis") {
        parse_axis_spec(axis)?
    } else if let Some(axes) = obj.get("axes") {
        parse_axis_spec(axes)?
    } else {
        Vec::new()
    };
    Ok((axes, obj.clone()))
}

fn operation_from_axes(name: &str, axes: Vec<String>) -> Operation {
    match name {
        "sum" => Operation::Sum { axes },
        "mean" => Operation::Mean { axes },
        "min" => Operation::Min { axes },
        "max" => Operation::Max { axes },
        "count" => Operation::Count { axes },
        "var" => Operation::Var { axes },
        "std" => Operation::Std { axes },
        "product" => Operation::Product { axes },
        "norm_l1" => Operation::NormL1 { axes },
        "norm_l2" => Operation::NormL2 { axes },
        "all_finite" => Operation::AllFinite { axes },
        "any_nan" => Operation::AnyNan { axes },
        "arg_min" => Operation::ArgMin { axes },
        "arg_max" => Operation::ArgMax { axes },
        "median" => Operation::Median { axes },
        _ => unreachable!("operation_from_axes: {name}"),
    }
}

fn parse_axis_spec(v: &Value) -> Result<Vec<String>, TetError> {
    match v {
        Value::Null => Err(TetError::Validation("axis spec must not be null".into())),
        Value::Number(n) => {
            let idx = parse_axis_index_number(n)?;
            Ok(vec![idx])
        }
        Value::String(s) => Ok(vec![validate_axis_token(s)?]),
        Value::Array(items) => {
            let mut out = Vec::with_capacity(items.len());
            for item in items {
                out.push(parse_axis_spec_item(item)?);
            }
            Ok(out)
        }
        Value::Object(map) => {
            if let Some(axis) = map.get("axis") {
                return parse_axis_spec(axis);
            }
            if let Some(axes) = map.get("axes") {
                return parse_axis_spec(axes);
            }
            Err(TetError::Validation(
                "axis object must include `axis` or `axes`".into(),
            ))
        }
        Value::Bool(_) => Err(TetError::Validation(
            "axis spec must be a number, string, or array".into(),
        )),
    }
}

fn parse_axis_spec_item(v: &Value) -> Result<String, TetError> {
    match v {
        Value::Number(n) => parse_axis_index_number(n),
        Value::String(s) => validate_axis_token(s),
        _ => Err(TetError::Validation(
            "axis list entries must be numbers or strings".into(),
        )),
    }
}

fn parse_axis_index_number(n: &serde_json::Number) -> Result<String, TetError> {
    let u = n
        .as_u64()
        .ok_or_else(|| TetError::Validation("axis index must be a non-negative integer".into()))?;
    Ok(u.to_string())
}

fn validate_axis_token(s: &str) -> Result<String, TetError> {
    if s.is_empty() {
        return Err(TetError::Validation("axis index must not be empty".into()));
    }
    if s.len() > QueryLimits::DEFAULT.max_operation_axis_label_len {
        return Err(TetError::Validation(format!(
            "axis index exceeds maximum length ({})",
            QueryLimits::DEFAULT.max_operation_axis_label_len
        )));
    }
    if !s.chars().all(|c| c.is_ascii_digit()) {
        return Err(TetError::Validation(format!(
            "invalid axis index {s:?} (decimal dimension indices only)"
        )));
    }
    Ok(s.to_owned())
}

// --- Serialize (flat wire) ---

struct QueryDocumentWire<'a> {
    doc: &'a QueryDocument,
}

impl<'a> From<&'a QueryDocument> for QueryDocumentWire<'a> {
    fn from(doc: &'a QueryDocument) -> Self {
        Self { doc }
    }
}

impl Serialize for QueryDocumentWire<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let doc = self.doc;
        let mut map = serializer.serialize_map(None)?;
        if let Some(v) = doc.layout_version {
            map.serialize_entry("layout_version", &v)?;
        }
        map.serialize_entry("dataset", &doc.dataset)?;
        if let Some(sel) = &doc.selection {
            map.serialize_entry("selection", sel)?;
        }
        if let Some(op) = &doc.operation {
            serialize_operation(&mut map, op)?;
        }
        if let Some(out) = &doc.output {
            map.serialize_entry("output", out)?;
        }
        if let Some(ex) = &doc.execution {
            serialize_execution(&mut map, ex)?;
        }
        map.end()
    }
}

fn serialize_operation<S>(map: &mut S, op: &Operation) -> Result<(), S::Error>
where
    S: SerializeMap,
{
    match op {
        Operation::Sum { axes } => map.serialize_entry("sum", &AxisSpecWire::from_axes(axes))?,
        Operation::Mean { axes } => map.serialize_entry("mean", &AxisSpecWire::from_axes(axes))?,
        Operation::Min { axes } => map.serialize_entry("min", &AxisSpecWire::from_axes(axes))?,
        Operation::Max { axes } => map.serialize_entry("max", &AxisSpecWire::from_axes(axes))?,
        Operation::Count { axes } => {
            map.serialize_entry("count", &AxisSpecWire::from_axes(axes))?;
        }
        Operation::Var { axes } => map.serialize_entry("var", &AxisSpecWire::from_axes(axes))?,
        Operation::Std { axes } => map.serialize_entry("std", &AxisSpecWire::from_axes(axes))?,
        Operation::Product { axes } => {
            map.serialize_entry("product", &AxisSpecWire::from_axes(axes))?;
        }
        Operation::NormL1 { axes } => {
            map.serialize_entry("norm_l1", &AxisSpecWire::from_axes(axes))?;
        }
        Operation::NormL2 { axes } => {
            map.serialize_entry("norm_l2", &AxisSpecWire::from_axes(axes))?;
        }
        Operation::AllFinite { axes } => {
            map.serialize_entry("all_finite", &AxisSpecWire::from_axes(axes))?;
        }
        Operation::AnyNan { axes } => {
            map.serialize_entry("any_nan", &AxisSpecWire::from_axes(axes))?;
        }
        Operation::ArgMin { axes } => {
            map.serialize_entry("arg_min", &AxisSpecWire::from_axes(axes))?;
        }
        Operation::ArgMax { axes } => {
            map.serialize_entry("arg_max", &AxisSpecWire::from_axes(axes))?;
        }
        Operation::Median { axes } => {
            map.serialize_entry("median", &AxisSpecWire::from_axes(axes))?;
        }
        Operation::Quantile { axes, q } => {
            let wire = ParametricOpWire {
                axes: AxisSpecWire::from_axes(axes),
                extra: vec![("q", serde_json::json!(q))],
            };
            map.serialize_entry("quantile", &wire)?;
        }
        Operation::Histogram { axes, bins } => {
            let wire = ParametricOpWire {
                axes: AxisSpecWire::from_axes(axes),
                extra: vec![("bins", serde_json::json!(bins))],
            };
            map.serialize_entry("histogram", &wire)?;
        }
    }
    Ok(())
}

fn serialize_execution<S>(map: &mut S, ex: &ExecutionHints) -> Result<(), S::Error>
where
    S: SerializeMap,
{
    let mut obj = serde_json::Map::new();
    if let Some(bytes) = ex.memory_budget_bytes {
        obj.insert("memory_budget_bytes".to_owned(), serde_json::json!(bytes));
    }
    if let Some(bps) = ex.memory_budget_percent_bps {
        let percent = f64::from(bps) / 100.0;
        obj.insert(
            "memory_budget_percent".to_owned(),
            serde_json::json!(percent),
        );
    }
    if !obj.is_empty() {
        map.serialize_entry("execution", &obj)?;
    }
    Ok(())
}

struct AxisSpecWire {
    axes: Vec<String>,
}

impl AxisSpecWire {
    fn from_axes(axes: &[String]) -> Self {
        Self {
            axes: axes.to_vec(),
        }
    }
}

impl Serialize for AxisSpecWire {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        if self.axes.is_empty() {
            return serializer.collect_seq(std::iter::empty::<u8>());
        }
        if self.axes.len() == 1
            && let Ok(idx) = self.axes[0].parse::<u64>()
        {
            return serializer.serialize_u64(idx);
        }
        let indices: Vec<u64> = self
            .axes
            .iter()
            .map(|s| s.parse::<u64>())
            .collect::<Result<Vec<_>, _>>()
            .map_err(serde::ser::Error::custom)?;
        serializer.collect_seq(indices.iter())
    }
}

struct ParametricOpWire {
    axes: AxisSpecWire,
    extra: Vec<(&'static str, Value)>,
}

impl Serialize for QueryDocument {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        QueryDocumentWire::from(self).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for QueryDocument {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        parse_query_value(&value).map_err(de::Error::custom)
    }
}

impl Serialize for ParametricOpWire {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = serializer.serialize_map(None)?;
        for (k, v) in &self.extra {
            map.serialize_entry(*k, v)?;
        }
        if !self.axes.axes.is_empty() {
            if self.axes.axes.len() == 1 {
                if let Ok(idx) = self.axes.axes[0].parse::<u64>() {
                    map.serialize_entry("axis", &idx)?;
                } else {
                    map.serialize_entry("axis", &self.axes.axes[0])?;
                }
            } else {
                map.serialize_entry("axes", &self.axes)?;
            }
        }
        map.end()
    }
}

//! TOML query front-end: parse to `toml::Value`, convert to JSON, reuse [`super::document_wire::parse_query_value`].

use serde_json::{Number, Value};

use super::document_wire::parse_query_value;
use super::types::{QueryDocument, TetError};

/// Map a parsed TOML value into `serde_json::Value` for the flat JSON wire parser.
fn toml_value_to_json(v: toml::Value) -> Value {
    match v {
        toml::Value::String(s) => Value::String(s),
        toml::Value::Integer(i) => {
            if let Some(n) = Number::from_i128(i128::from(i)) {
                Value::Number(n)
            } else {
                Value::String(i.to_string())
            }
        }
        toml::Value::Float(f) => Number::from_f64(f)
            .map(Value::Number)
            .unwrap_or(Value::String(f.to_string())),
        toml::Value::Boolean(b) => Value::Bool(b),
        toml::Value::Datetime(dt) => Value::String(dt.to_string()),
        toml::Value::Array(items) => {
            Value::Array(items.into_iter().map(toml_value_to_json).collect())
        }
        toml::Value::Table(table) => {
            let mut map = serde_json::Map::new();
            for (k, v) in table {
                map.insert(k, toml_value_to_json(v));
            }
            Value::Object(map)
        }
    }
}

/// Parse a TOML query document (same semantics as flat JSON after conversion).
///
/// # Errors
///
/// Same as [`super::document::parse_query_json`].
pub fn parse_query_toml(text: &str) -> Result<QueryDocument, TetError> {
    super::document::check_query_payload_size(text)?;
    let root: toml::Value = toml::from_str(text)?;
    let value = toml_value_to_json(root);
    super::document::check_query_value_depth(&value)?;
    parse_query_value(&value)
}

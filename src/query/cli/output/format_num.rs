//! Shared numeric / list formatting for CLI quiet output.

pub(super) const QUIET_VEC_INLINE_MAX: usize = 12;

pub(super) fn missing_field(field: &str) -> String {
    format!("table/quiet output: missing execution.{field} after --execute")
}

pub(super) fn fmt_f64(v: f64) -> String {
    if !v.is_finite() {
        return v.to_string();
    }
    let s = format!("{v:.6}");
    trim_float_string(s)
}

fn trim_float_string(s: String) -> String {
    if !s.contains('.') {
        return s;
    }
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}

pub(super) fn fmt_f64_list(values: &[f64], inline_max: usize) -> String {
    fmt_list(values, inline_max, fmt_f64)
}

pub(super) fn fmt_i64_list(values: &[i64], inline_max: usize) -> String {
    fmt_list(values, inline_max, |v| v.to_string())
}

pub(super) fn fmt_u64_list(values: &[u64], inline_max: usize) -> String {
    fmt_list(values, inline_max, |v| v.to_string())
}

pub(super) fn fmt_bool_list(values: &[bool], inline_max: usize) -> String {
    fmt_list(values, inline_max, |v| v.to_string())
}

fn fmt_list<T: Copy>(values: &[T], inline_max: usize, fmt_one: impl Fn(T) -> String) -> String {
    if values.is_empty() {
        return "[]".to_string();
    }
    if values.len() <= inline_max {
        let body = values
            .iter()
            .map(|v| fmt_one(*v))
            .collect::<Vec<_>>()
            .join(",");
        return format!("[{body}]");
    }
    let body = values[..inline_max]
        .iter()
        .map(|v| fmt_one(*v))
        .collect::<Vec<_>>()
        .join(",");
    format!("[{body},…+{}]", values.len() - inline_max)
}

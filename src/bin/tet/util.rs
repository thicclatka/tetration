//! Shared CLI helpers (`error:` prefix, query JSON input).

use std::fs;
use std::io::{self, Read};
use std::path::Path;

use crate::args::QueryStdoutFormat;
use tetration::QueryOutputFormat;

pub(crate) fn cli_error(message: impl std::fmt::Display) -> String {
    format!("error: {message}")
}

pub(crate) fn resolve_stdout(quiet: bool, format: QueryStdoutFormat) -> QueryOutputFormat {
    if quiet {
        QueryOutputFormat::Quiet
    } else {
        format.into()
    }
}

pub(crate) fn read_stdin_string() -> io::Result<String> {
    let mut buf = String::new();
    io::stdin().read_to_string(&mut buf)?;
    Ok(buf)
}

/// Read query JSON from a positional arg (file path or inline JSON) or stdin.
pub(crate) fn read_query_payload(query: Option<&str>) -> io::Result<String> {
    let Some(arg) = query else {
        return read_stdin_string();
    };
    if arg == "-" {
        return read_stdin_string();
    }
    let path = Path::new(arg);
    if path.is_file() {
        fs::read_to_string(path)
    } else {
        Ok(arg.to_owned())
    }
}

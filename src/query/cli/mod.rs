//! CLI-only helpers (`tet` history, query response formatting).

pub mod history;
pub mod output;

pub use history::{
    CLI_QUERY_HISTORY_MAX, CliQueryHistoryEntry, append_cli_query_history, clear_cli_query_history,
    cli_query_history_enabled, cli_query_history_path, list_cli_query_history,
};
pub use output::{QueryOutputFormat, format_query_response};

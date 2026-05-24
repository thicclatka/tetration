//! CLI-only helpers (`tet` history, query response formatting).

pub mod history;
pub mod output;

pub use history::{
    CliQueryHistoryEntry, HistorySettings, append_cli_query_history, clear_cli_query_history,
    cli_query_history_enabled, cli_query_history_max, cli_query_history_path,
    get_cli_query_history_entry, list_cli_query_history,
};
pub use output::{QueryOutputFormat, format_query_response};

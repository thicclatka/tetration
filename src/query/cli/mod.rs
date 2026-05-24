//! CLI-only helpers (`tet` history, query response formatting).

pub mod history;
pub mod output;

pub use history::{
    CliQueryHistoryEntry, HistoryExecuteFilter, HistoryListFilter, HistorySettings,
    append_cli_query_history, clear_cli_query_history, cli_query_history_enabled,
    cli_query_history_max, cli_query_history_path, format_history_list_json,
    format_history_list_text, get_cli_query_history_entry, history_entry_mode,
    list_cli_query_history, parse_history_execute_filter,
};
pub use output::{QueryOutputFormat, format_query_response, format_query_stderr_hints};

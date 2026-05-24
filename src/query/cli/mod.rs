//! CLI-only helpers (`tet` history, query response formatting).

pub mod history;
pub mod info;
pub mod output;
mod text;

pub use history::{
    CliQueryHistoryEntry, HistoryExecuteFilter, HistoryListFilter, HistorySettings,
    append_cli_query_history, clear_cli_query_history, cli_query_history_enabled,
    cli_query_history_max, cli_query_history_path, format_history_list_json,
    format_history_list_text, get_cli_query_history_entry, history_entry_mode,
    list_cli_query_history, parse_history_execute_filter,
};
pub use info::{
    DEFAULT_INFO_CHUNK_TABLE_LIMIT, InfoListFilter, InfoViewSections,
    format_info_json, format_info_quiet, format_info_text, info_view_sections_from_flags,
};
pub use output::{QueryOutputFormat, format_query_response, format_query_stderr_hints};

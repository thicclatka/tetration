//! **tetration** — Rust library for the Tetration mmap-oriented chunked tensor format.
//!
//! Public API: [`catalog`], [`convert`], [`layout`], [`query`], and [`prelude`] for common
//! embedder imports. The companion CLI is **`tet`** (`src/bin/tet.rs`, `src/bin/tet/`).
//!
//! **Embedder walkthrough:** `cargo run --example session_write` (or `create_and_query`,
//! `inspect_catalog`; Phase 7 — see `GETTING_STARTED.md`).

pub(crate) mod utils;

pub mod catalog;
pub mod convert;
pub mod layout;
pub mod query;

/// Common embedder imports: query document types, parse/validate/plan, mmap open.
pub mod prelude {
    pub use crate::catalog::{
        FileMetadataDraft, TetDatasetWrite, TetFile, TetWriterSession, read_tet_summary_v1,
    };
    pub use crate::layout::{MAGIC, mmap_file_read};
    pub use crate::query::{
        ExecuteQueryOptions, QueryDocument, QueryOutputFormat, QueryResponse, ReadPlan,
        execute_query_document, execute_query_json, format_query_response, parse_query_json,
        plan_query_with_tet_mmap_ex, validate_query,
    };
}

#[cfg(test)]
mod tests;

//! **tetration** — Rust library for the Tetration mmap-oriented chunked tensor format.
//!
//! Public API: [`catalog`], [`convert`], [`layout`], [`query`], [`verify`], [`repair`], and [`prelude`]
//! for common embedder imports. The companion CLI is **`tet`** (`src/bin/tet.rs`, `src/bin/tet/`).
//!
//! **Embedder walkthrough:** `cargo run --example session_write` (or `create_and_query`,
//! `inspect_catalog`; file health: `tet verify` / [`verify`](verify)).

pub(crate) mod utils;

pub mod catalog;
pub mod convert;
pub mod layout;
pub mod query;
pub mod repair;
pub mod verify;

/// Common embedder imports: query document types, parse/validate/plan, mmap open.
pub mod prelude {
    pub use crate::catalog::{
        FileMetadataDraft, StreamTileJob, TetDatasetStreamSpec, TetDatasetWrite, TetFile,
        TetWriterSession, read_tet_summary_v1,
    };
    pub use crate::layout::{MAGIC, mmap_file_read};
    pub use crate::query::{
        ExecuteQueryOptions, QueryDocument, QueryOutputFormat, QueryResponse, ReadPlan,
        execute_query_document, execute_query_json, format_query_response, parse_query_json,
        plan_query_with_tet_mmap_ex, validate_query,
    };
    pub use crate::verify::{TetVerifyReport, verify_tet_file};
}

#[cfg(test)]
mod tests;

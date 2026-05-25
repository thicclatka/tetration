//! **tetration** — Rust library for the Tetration mmap-oriented chunked tensor format.
//! The companion CLI binary is **`tet`** (see `src/bin/tet.rs` and `src/bin/tet/`).

pub(crate) mod utils;

pub mod catalog;
pub mod convert;
pub mod layout;
pub mod query;

/// Common embedder imports (`QueryDocument`, planning, mmap open).
pub mod prelude {
    pub use crate::layout::{MAGIC, mmap_file_read};
    pub use crate::query::{
        QueryDocument, QueryResponse, ReadPlan, parse_query_json, plan_query_with_tet_mmap_ex,
        validate_query,
    };
}

#[cfg(test)]
mod tests;

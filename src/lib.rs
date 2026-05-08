//! **tetration** — Rust library for the Tetration mmap-oriented chunked tensor format.
//! The companion CLI binary is **`tet`** (see `src/bin/tet.rs`).

pub mod query;

pub use query::{
    QueryDocument, QueryResponse, parse_query_json, plan_query, validate_query,
};

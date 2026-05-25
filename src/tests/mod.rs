//! In-crate integration tests (`cargo test`); uses `crate::` so `pub(crate)` helpers stay private.

#![allow(dead_code)]

mod catalog;
mod cli_history;
mod cli_info;
mod cli_output;
mod convert;
mod fixture;
mod fold;
mod layout_roundtrip;
mod query;
mod utils;

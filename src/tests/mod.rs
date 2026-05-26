//! In-crate integration tests (`cargo test --lib`); uses submodule paths and `pub(crate)` helpers.

#![allow(dead_code)]

mod catalog;
mod cli_history;
mod cli_info;
mod cli_output;
mod convert;
mod fixture;
mod fold;
mod layout_roundtrip;
mod metadata;
mod query;
mod reduction;
mod repair;
mod session;
mod small_tet_fixtures;
mod utils;
mod variance_simd;
mod verify;
mod verify_fixtures;

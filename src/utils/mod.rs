//! Shared low-level helpers used across layout, catalog, and related code.

pub mod dtype;
mod le_pod;
pub use le_pod::f32_le;
pub(crate) use le_pod::{f64_le, i32_le, i64_le};
pub mod host_memory;
pub(crate) mod wire;

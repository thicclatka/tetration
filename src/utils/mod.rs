//! Shared low-level helpers used across layout, catalog, and related code.

pub mod dtype;
pub(crate) mod f32_le;
pub(crate) mod f64_le;
pub mod host_memory;
pub(crate) mod wire;

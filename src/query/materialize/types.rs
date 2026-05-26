//! Logical backing stores and tier-C materialization enums.

use crate::query::engine::spill_policy::TempSpillFile;

pub(crate) enum LogicalF32Backing {
    InMemory(Vec<f32>),
    TempSpill(TempSpillFile),
}

pub(crate) enum LogicalF64Backing {
    InMemory(Vec<f64>),
    TempSpill(TempSpillFile),
}

/// Capped decode previews for all supported element types.
#[derive(Debug, Clone, Default)]
#[allow(clippy::struct_excessive_bools)]
pub(crate) struct DecodePreviewBundle {
    pub f32: Vec<f32>,
    pub f64: Vec<f64>,
    pub i32: Vec<i32>,
    pub i64: Vec<i64>,
    pub u8: Vec<u8>,
    pub u16: Vec<u16>,
    pub i16: Vec<i16>,
    pub f32_truncated: bool,
    pub f64_truncated: bool,
    pub i32_truncated: bool,
    pub i64_truncated: bool,
    pub u8_truncated: bool,
    pub u16_truncated: bool,
    pub i16_truncated: bool,
}

impl DecodePreviewBundle {
    #[must_use]
    pub fn empty() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn all_truncated(truncated: bool) -> Self {
        Self {
            f32_truncated: truncated,
            f64_truncated: truncated,
            i32_truncated: truncated,
            i64_truncated: truncated,
            u8_truncated: truncated,
            u16_truncated: truncated,
            i16_truncated: truncated,
            ..Self::default()
        }
    }
}

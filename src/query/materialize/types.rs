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

pub(crate) enum LogicalF16Backing {
    InMemory(Vec<half::f16>),
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
    pub u32: Vec<u32>,
    pub u64: Vec<u64>,
    pub f16: Vec<half::f16>,
    pub f32_truncated: bool,
    pub f64_truncated: bool,
    pub i32_truncated: bool,
    pub i64_truncated: bool,
    pub u8_truncated: bool,
    pub u16_truncated: bool,
    pub i16_truncated: bool,
    pub u32_truncated: bool,
    pub u64_truncated: bool,
    pub f16_truncated: bool,
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
            u32_truncated: truncated,
            u64_truncated: truncated,
            f16_truncated: truncated,
            ..Self::default()
        }
    }

    #[must_use]
    pub(crate) fn f32_preview(values: Vec<f32>, truncated: bool) -> Self {
        Self {
            f32: values,
            f32_truncated: truncated,
            ..Self::empty()
        }
    }

    #[must_use]
    pub(crate) fn f64_preview(values: Vec<f64>, truncated: bool) -> Self {
        Self {
            f64: values,
            f64_truncated: truncated,
            ..Self::empty()
        }
    }

    #[must_use]
    pub(crate) fn i32_preview(values: Vec<i32>, truncated: bool) -> Self {
        Self {
            i32: values,
            i32_truncated: truncated,
            ..Self::empty()
        }
    }

    #[must_use]
    pub(crate) fn i64_preview(values: Vec<i64>, truncated: bool) -> Self {
        Self {
            i64: values,
            i64_truncated: truncated,
            ..Self::empty()
        }
    }

    #[must_use]
    pub(crate) fn u8_preview(values: Vec<u8>, truncated: bool) -> Self {
        Self {
            u8: values,
            u8_truncated: truncated,
            ..Self::empty()
        }
    }

    #[must_use]
    pub(crate) fn u16_preview(values: Vec<u16>, truncated: bool) -> Self {
        Self {
            u16: values,
            u16_truncated: truncated,
            ..Self::empty()
        }
    }

    #[must_use]
    pub(crate) fn i16_preview(values: Vec<i16>, truncated: bool) -> Self {
        Self {
            i16: values,
            i16_truncated: truncated,
            ..Self::empty()
        }
    }

    #[must_use]
    pub(crate) fn u32_preview(values: Vec<u32>, truncated: bool) -> Self {
        Self {
            u32: values,
            u32_truncated: truncated,
            ..Self::empty()
        }
    }

    #[must_use]
    pub(crate) fn u64_preview(values: Vec<u64>, truncated: bool) -> Self {
        Self {
            u64: values,
            u64_truncated: truncated,
            ..Self::empty()
        }
    }

    #[must_use]
    pub(crate) fn f16_preview(values: Vec<half::f16>, truncated: bool) -> Self {
        Self {
            f16: values,
            f16_truncated: truncated,
            ..Self::empty()
        }
    }
}

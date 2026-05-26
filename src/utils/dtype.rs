//! On-disk element dtype tags for query execution.

use crate::catalog::DATASET_DTYPE_TAG_V1;
use crate::query::TetError;

/// Supported element types in the v1 query engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElementDtype {
    F32,
    F64,
    I32,
    I64,
    U8,
    U16,
    I16,
}

impl ElementDtype {
    /// Parse a catalog wire `dtype` tag.
    ///
    /// # Errors
    ///
    /// Returns [`TetError::Validation`] for unsupported tags.
    pub fn from_wire(dtype: u32) -> Result<Self, TetError> {
        Self::try_from_wire_tag(dtype).ok_or_else(|| {
            let tags = DATASET_DTYPE_TAG_V1;
            TetError::Validation(format!(
                "unsupported dataset dtype {dtype} (supported: f32={}, f64={}, i32={}, i64={}, u8={}, u16={}, i16={})",
                tags.f32,
                tags.f64,
                tags.i32,
                tags.i64,
                tags.u8,
                tags.u16,
                tags.i16
            ))
        })
    }

    /// Parse a wire tag, returning `None` for unsupported values.
    #[must_use]
    pub fn try_from_wire_tag(dtype: u32) -> Option<Self> {
        let tags = DATASET_DTYPE_TAG_V1;
        if tags.is_f32(dtype) {
            Some(Self::F32)
        } else if tags.is_f64(dtype) {
            Some(Self::F64)
        } else if tags.is_i32(dtype) {
            Some(Self::I32)
        } else if tags.is_i64(dtype) {
            Some(Self::I64)
        } else if tags.is_u8(dtype) {
            Some(Self::U8)
        } else if tags.is_u16(dtype) {
            Some(Self::U16)
        } else if tags.is_i16(dtype) {
            Some(Self::I16)
        } else {
            None
        }
    }

    /// `element_size * product(shape)` for this dtype, or `None` on overflow / bad shape.
    #[must_use]
    pub fn tensor_bytes_for_shape(self, shape: &[u64]) -> Option<u64> {
        let elems = shape.iter().try_fold(1u64, |a, &b| a.checked_mul(b))?;
        self.bytes_from_elem_count(elems)
    }

    #[must_use]
    pub const fn wire_tag(self) -> u32 {
        let tags = DATASET_DTYPE_TAG_V1;
        match self {
            Self::F32 => tags.f32,
            Self::F64 => tags.f64,
            Self::I32 => tags.i32,
            Self::I64 => tags.i64,
            Self::U8 => tags.u8,
            Self::U16 => tags.u16,
            Self::I16 => tags.i16,
        }
    }

    #[must_use]
    pub const fn elem_size(self) -> usize {
        match self {
            Self::U8 => 1,
            Self::U16 | Self::I16 => 2,
            Self::F32 | Self::I32 => 4,
            Self::F64 | Self::I64 => 8,
        }
    }

    /// Byte length for `count` logical elements, or `None` on overflow.
    #[must_use]
    pub fn bytes_from_elem_count(self, count: u64) -> Option<u64> {
        match self {
            Self::F32 => crate::utils::f32_le::bytes_from_elem_count(count),
            Self::F64 => crate::utils::f64_le::bytes_from_elem_count(count),
            Self::I32 => crate::utils::i32_le::bytes_from_elem_count(count),
            Self::I64 => crate::utils::i64_le::bytes_from_elem_count(count),
            Self::U8 => crate::utils::u8_le::bytes_from_elem_count(count),
            Self::U16 => crate::utils::u16_le::bytes_from_elem_count(count),
            Self::I16 => crate::utils::i16_le::bytes_from_elem_count(count),
        }
    }

    /// Whether streaming fold / tier-C stats use the native floating preview path.
    #[must_use]
    pub const fn uses_f32_preview(self) -> bool {
        matches!(self, Self::F32)
    }

    #[must_use]
    pub const fn uses_f64_preview(self) -> bool {
        matches!(self, Self::F64)
    }

    #[must_use]
    pub const fn uses_i32_preview(self) -> bool {
        matches!(self, Self::I32)
    }

    #[must_use]
    pub const fn uses_i64_preview(self) -> bool {
        matches!(self, Self::I64)
    }

    #[must_use]
    pub const fn uses_u8_preview(self) -> bool {
        matches!(self, Self::U8)
    }

    #[must_use]
    pub const fn uses_u16_preview(self) -> bool {
        matches!(self, Self::U16)
    }

    #[must_use]
    pub const fn uses_i16_preview(self) -> bool {
        matches!(self, Self::I16)
    }

    /// Integer dtypes promoted to `f64` for tier-A/B aggregates.
    #[must_use]
    pub const fn is_integer(self) -> bool {
        matches!(
            self,
            Self::I32 | Self::I64 | Self::U8 | Self::U16 | Self::I16
        )
    }
}

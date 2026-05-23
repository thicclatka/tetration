//! On-disk element dtype tags for query execution.

use crate::catalog::{DTYPE_F32, DTYPE_F64};
use crate::query::TetError;

/// Supported element types in the v1 query engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElementDtype {
    F32,
    F64,
}

impl ElementDtype {
    /// Parse a catalog wire `dtype` tag.
    ///
    /// # Errors
    ///
    /// Returns [`TetError::Validation`] for unsupported tags.
    pub fn from_wire(dtype: u32) -> Result<Self, TetError> {
        match dtype {
            DTYPE_F32 => Ok(Self::F32),
            DTYPE_F64 => Ok(Self::F64),
            _ => Err(TetError::Validation(format!(
                "unsupported dataset dtype {dtype} (supported: DTYPE_F32=1, DTYPE_F64=2)"
            ))),
        }
    }

    #[must_use]
    pub const fn wire_tag(self) -> u32 {
        match self {
            Self::F32 => DTYPE_F32,
            Self::F64 => DTYPE_F64,
        }
    }

    #[must_use]
    pub const fn elem_size(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F64 => 8,
        }
    }

    /// Byte length for `count` logical elements, or `None` on overflow.
    #[must_use]
    pub fn bytes_from_elem_count(self, count: u64) -> Option<u64> {
        match self {
            Self::F32 => crate::utils::f32_le::bytes_from_elem_count(count),
            Self::F64 => crate::utils::f64_le::bytes_from_elem_count(count),
        }
    }
}

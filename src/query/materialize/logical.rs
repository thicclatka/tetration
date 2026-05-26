//! Full logical selection backing (RAM or temp spill) for tier-C ops.

use crate::query::engine::budget::MemoryStrategy;

use super::int;
use super::types::{LogicalF32Backing, LogicalF64Backing};

pub(crate) enum MaterializedLogical {
    F32 {
        backing: LogicalF32Backing,
        total_bytes_read_from_disk: u64,
        strategy: MemoryStrategy,
    },
    F64 {
        backing: LogicalF64Backing,
        total_bytes_read_from_disk: u64,
        strategy: MemoryStrategy,
    },
    I32 {
        backing: int::LogicalI32Backing,
        total_bytes_read_from_disk: u64,
        strategy: MemoryStrategy,
    },
    I64 {
        backing: int::LogicalI64Backing,
        total_bytes_read_from_disk: u64,
        strategy: MemoryStrategy,
    },
    U8 {
        backing: int::LogicalU8Backing,
        total_bytes_read_from_disk: u64,
        strategy: MemoryStrategy,
    },
    U16 {
        backing: int::LogicalU16Backing,
        total_bytes_read_from_disk: u64,
        strategy: MemoryStrategy,
    },
    I16 {
        backing: int::LogicalI16Backing,
        total_bytes_read_from_disk: u64,
        strategy: MemoryStrategy,
    },
}

fn mmap_spill(path: &std::path::Path) -> Result<memmap2::Mmap, crate::query::types::TetError> {
    let file = std::fs::File::open(path).map_err(|e| {
        crate::query::types::TetError::Validation(format!("temp spill read failed: {e}"))
    })?;
    unsafe {
        memmap2::Mmap::map(&file).map_err(|e| {
            crate::query::types::TetError::Validation(format!("temp spill mmap failed: {e}"))
        })
    }
}

/// Load a materialized logical selection as `f64` for tier-C statistics.
pub(crate) fn materialized_logical_as_f64(
    materialized: &MaterializedLogical,
) -> Result<Vec<f64>, crate::query::types::TetError> {
    match materialized {
        MaterializedLogical::F32 { backing, .. } => match backing {
            super::types::LogicalF32Backing::InMemory(v) => {
                Ok(v.iter().map(|&x| f64::from(x)).collect())
            }
            super::types::LogicalF32Backing::TempSpill(temp) => {
                let mmap = mmap_spill(temp.path())?;
                Ok(bytemuck::cast_slice::<u8, f32>(&mmap)
                    .iter()
                    .map(|&x| f64::from(x))
                    .collect())
            }
        },
        MaterializedLogical::F64 { backing, .. } => match backing {
            super::types::LogicalF64Backing::InMemory(v) => Ok(v.clone()),
            super::types::LogicalF64Backing::TempSpill(temp) => {
                let mmap = mmap_spill(temp.path())?;
                Ok(bytemuck::cast_slice::<u8, f64>(&mmap).to_vec())
            }
        },
        MaterializedLogical::I32 { backing, .. } => int::materialized_logical_as_f64_i32(backing),
        MaterializedLogical::I64 { backing, .. } => int::materialized_logical_as_f64_i64(backing),
        MaterializedLogical::U8 { backing, .. } => int::materialized_logical_as_f64_u8(backing),
        MaterializedLogical::U16 { backing, .. } => int::materialized_logical_as_f64_u16(backing),
        MaterializedLogical::I16 { backing, .. } => int::materialized_logical_as_f64_i16(backing),
    }
}

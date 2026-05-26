//! Full logical selection (RAM vs temp spill) and preview helpers.

use std::path::Path;

use crate::query::engine::budget::{ExecutionBudget, MemoryStrategy};
use crate::query::engine::spill_policy::{SpillPathAllowlist, TempSpillFile};
use crate::query::types::{ReadPlan, TetError};
use crate::utils::dtype::ElementDtype;

use super::f32::{materialize_into_vec, preview_from_spill_file_f32};
use super::f64::{materialize_into_vec_f64, preview_from_spill_file_f64};
use super::int;
use super::logical::MaterializedLogical;
use super::types::{DecodePreviewBundle, LogicalF32Backing, LogicalF64Backing};

/// Decode the full logical selection, choosing RAM vs temp spill from the memory budget.
pub(crate) fn materialize_logical_selection(
    mmap: &[u8],
    plan: &ReadPlan,
    budget: &ExecutionBudget,
    allowlist: &SpillPathAllowlist,
    dtype: ElementDtype,
) -> Result<MaterializedLogical, TetError> {
    if budget.full_tensor_exceeds_budget(plan, dtype)? {
        let temp = TempSpillFile::create(allowlist)?;
        let bytes = crate::query::dispatch::spill_full_selection(mmap, plan, temp.path(), dtype)?;
        Ok(match dtype {
            ElementDtype::F32 => MaterializedLogical::F32 {
                backing: LogicalF32Backing::TempSpill(temp),
                total_bytes_read_from_disk: bytes,
                strategy: MemoryStrategy::TempSpillMaterialize,
            },
            ElementDtype::F64 => MaterializedLogical::F64 {
                backing: LogicalF64Backing::TempSpill(temp),
                total_bytes_read_from_disk: bytes,
                strategy: MemoryStrategy::TempSpillMaterialize,
            },
            ElementDtype::I32 => MaterializedLogical::I32 {
                backing: int::LogicalI32Backing::TempSpill(temp),
                total_bytes_read_from_disk: bytes,
                strategy: MemoryStrategy::TempSpillMaterialize,
            },
            ElementDtype::I64 => MaterializedLogical::I64 {
                backing: int::LogicalI64Backing::TempSpill(temp),
                total_bytes_read_from_disk: bytes,
                strategy: MemoryStrategy::TempSpillMaterialize,
            },
            ElementDtype::U8 => MaterializedLogical::U8 {
                backing: int::LogicalU8Backing::TempSpill(temp),
                total_bytes_read_from_disk: bytes,
                strategy: MemoryStrategy::TempSpillMaterialize,
            },
            ElementDtype::U16 => MaterializedLogical::U16 {
                backing: int::LogicalU16Backing::TempSpill(temp),
                total_bytes_read_from_disk: bytes,
                strategy: MemoryStrategy::TempSpillMaterialize,
            },
            ElementDtype::I16 => MaterializedLogical::I16 {
                backing: int::LogicalI16Backing::TempSpill(temp),
                total_bytes_read_from_disk: bytes,
                strategy: MemoryStrategy::TempSpillMaterialize,
            },
        })
    } else {
        match dtype {
            ElementDtype::F32 => {
                let (vec, bytes) = materialize_into_vec(mmap, plan)?;
                Ok(MaterializedLogical::F32 {
                    backing: LogicalF32Backing::InMemory(vec),
                    total_bytes_read_from_disk: bytes,
                    strategy: MemoryStrategy::InMemoryMaterialize,
                })
            }
            ElementDtype::F64 => {
                let (vec, bytes) = materialize_into_vec_f64(mmap, plan)?;
                Ok(MaterializedLogical::F64 {
                    backing: LogicalF64Backing::InMemory(vec),
                    total_bytes_read_from_disk: bytes,
                    strategy: MemoryStrategy::InMemoryMaterialize,
                })
            }
            ElementDtype::I32 => {
                let (vec, bytes) = int::materialize_into_vec_i32(mmap, plan)?;
                Ok(MaterializedLogical::I32 {
                    backing: int::LogicalI32Backing::InMemory(vec),
                    total_bytes_read_from_disk: bytes,
                    strategy: MemoryStrategy::InMemoryMaterialize,
                })
            }
            ElementDtype::I64 => {
                let (vec, bytes) = int::materialize_into_vec_i64(mmap, plan)?;
                Ok(MaterializedLogical::I64 {
                    backing: int::LogicalI64Backing::InMemory(vec),
                    total_bytes_read_from_disk: bytes,
                    strategy: MemoryStrategy::InMemoryMaterialize,
                })
            }
            ElementDtype::U8 => {
                let (vec, bytes) = int::materialize_into_vec_u8(mmap, plan)?;
                Ok(MaterializedLogical::U8 {
                    backing: int::LogicalU8Backing::InMemory(vec),
                    total_bytes_read_from_disk: bytes,
                    strategy: MemoryStrategy::InMemoryMaterialize,
                })
            }
            ElementDtype::U16 => {
                let (vec, bytes) = int::materialize_into_vec_u16(mmap, plan)?;
                Ok(MaterializedLogical::U16 {
                    backing: int::LogicalU16Backing::InMemory(vec),
                    total_bytes_read_from_disk: bytes,
                    strategy: MemoryStrategy::InMemoryMaterialize,
                })
            }
            ElementDtype::I16 => {
                let (vec, bytes) = int::materialize_into_vec_i16(mmap, plan)?;
                Ok(MaterializedLogical::I16 {
                    backing: int::LogicalI16Backing::InMemory(vec),
                    total_bytes_read_from_disk: bytes,
                    strategy: MemoryStrategy::InMemoryMaterialize,
                })
            }
        }
    }
}

/// First `max` logical values without a second full decode (from RAM or spill file).
pub(crate) fn preview_from_materialized(
    materialized: &MaterializedLogical,
    logical_len: usize,
    max: usize,
) -> Result<DecodePreviewBundle, TetError> {
    let cap = max.min(logical_len);
    if cap == 0 {
        return Ok(DecodePreviewBundle::all_truncated(logical_len > 0));
    }
    match materialized {
        MaterializedLogical::F32 { backing, .. } => {
            let (p, t) = preview_from_materialized_f32(backing, logical_len, max)?;
            Ok(DecodePreviewBundle {
                f32: p,
                f32_truncated: t,
                ..DecodePreviewBundle::empty()
            })
        }
        MaterializedLogical::F64 { backing, .. } => {
            let (p, t) = preview_from_materialized_f64(backing, logical_len, max)?;
            Ok(DecodePreviewBundle {
                f64: p,
                f64_truncated: t,
                ..DecodePreviewBundle::empty()
            })
        }
        MaterializedLogical::I32 { backing, .. } => {
            let (p, t) = int::preview_from_materialized_i32(backing, logical_len, max)?;
            Ok(DecodePreviewBundle {
                i32: p,
                i32_truncated: t,
                ..DecodePreviewBundle::empty()
            })
        }
        MaterializedLogical::I64 { backing, .. } => {
            let (p, t) = int::preview_from_materialized_i64(backing, logical_len, max)?;
            Ok(DecodePreviewBundle {
                i64: p,
                i64_truncated: t,
                ..DecodePreviewBundle::empty()
            })
        }
        MaterializedLogical::U8 { backing, .. } => {
            let (p, t) = int::preview_from_materialized_u8(backing, logical_len, max)?;
            Ok(DecodePreviewBundle {
                u8: p,
                u8_truncated: t,
                ..DecodePreviewBundle::empty()
            })
        }
        MaterializedLogical::U16 { backing, .. } => {
            let (p, t) = int::preview_from_materialized_u16(backing, logical_len, max)?;
            Ok(DecodePreviewBundle {
                u16: p,
                u16_truncated: t,
                ..DecodePreviewBundle::empty()
            })
        }
        MaterializedLogical::I16 { backing, .. } => {
            let (p, t) = int::preview_from_materialized_i16(backing, logical_len, max)?;
            Ok(DecodePreviewBundle {
                i16: p,
                i16_truncated: t,
                ..DecodePreviewBundle::empty()
            })
        }
    }
}

fn preview_from_materialized_f32(
    backing: &LogicalF32Backing,
    logical_len: usize,
    max_f32: usize,
) -> Result<(Vec<f32>, bool), TetError> {
    let cap = max_f32.min(logical_len);
    if cap == 0 {
        return Ok((Vec::new(), logical_len > 0));
    }
    match backing {
        LogicalF32Backing::InMemory(v) => Ok((v[..cap].to_vec(), logical_len > max_f32)),
        LogicalF32Backing::TempSpill(temp) => {
            preview_from_spill_file_f32(temp.path(), cap, logical_len)
        }
    }
}

fn preview_from_materialized_f64(
    backing: &LogicalF64Backing,
    logical_len: usize,
    max: usize,
) -> Result<(Vec<f64>, bool), TetError> {
    let cap = max.min(logical_len);
    if cap == 0 {
        return Ok((Vec::new(), logical_len > 0));
    }
    match backing {
        LogicalF64Backing::InMemory(v) => Ok((v[..cap].to_vec(), logical_len > max)),
        LogicalF64Backing::TempSpill(temp) => {
            preview_from_spill_file_f64(temp.path(), cap, logical_len)
        }
    }
}

/// Preview from an export spill file (single decode pass for spill + preview).
pub(crate) fn preview_from_spill_export_file(
    path: &Path,
    logical_len: usize,
    max: usize,
    dtype: ElementDtype,
) -> Result<DecodePreviewBundle, TetError> {
    let cap = max.min(logical_len);
    match dtype {
        ElementDtype::F32 => {
            let (p, t) = preview_from_spill_file_f32(path, cap, logical_len)?;
            Ok(DecodePreviewBundle {
                f32: p,
                f32_truncated: t,
                ..DecodePreviewBundle::empty()
            })
        }
        ElementDtype::F64 => {
            let (p, t) = preview_from_spill_file_f64(path, cap, logical_len)?;
            Ok(DecodePreviewBundle {
                f64: p,
                f64_truncated: t,
                ..DecodePreviewBundle::empty()
            })
        }
        ElementDtype::I32 => {
            let (p, t) = int::preview_from_spill_file_i32(path, cap, logical_len)?;
            Ok(DecodePreviewBundle {
                i32: p,
                i32_truncated: t,
                ..DecodePreviewBundle::empty()
            })
        }
        ElementDtype::I64 => {
            let (p, t) = int::preview_from_spill_file_i64(path, cap, logical_len)?;
            Ok(DecodePreviewBundle {
                i64: p,
                i64_truncated: t,
                ..DecodePreviewBundle::empty()
            })
        }
        ElementDtype::U8 => {
            let (p, t) = int::preview_from_spill_file_u8(path, cap, logical_len)?;
            Ok(DecodePreviewBundle {
                u8: p,
                u8_truncated: t,
                ..DecodePreviewBundle::empty()
            })
        }
        ElementDtype::U16 => {
            let (p, t) = int::preview_from_spill_file_u16(path, cap, logical_len)?;
            Ok(DecodePreviewBundle {
                u16: p,
                u16_truncated: t,
                ..DecodePreviewBundle::empty()
            })
        }
        ElementDtype::I16 => {
            let (p, t) = int::preview_from_spill_file_i16(path, cap, logical_len)?;
            Ok(DecodePreviewBundle {
                i16: p,
                i16_truncated: t,
                ..DecodePreviewBundle::empty()
            })
        }
    }
}

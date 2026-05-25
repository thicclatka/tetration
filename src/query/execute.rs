//! In-process query execution helpers for embedders (Phase 7).

use std::path::Path;

use crate::query::engine::spill_policy::SpillPathAllowlist;
use crate::query::engine::{plan_query_with_tet_mmap_ex};
use crate::query::types::{QueryDocument, QueryResponse, TetError};

/// Options for [`execute_query_document`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecuteQueryOptions {
    /// Preview element cap when the document has an `operation` (use `Some(0)` to skip preview bytes).
    pub preview_max: Option<usize>,
}

impl Default for ExecuteQueryOptions {
    fn default() -> Self {
        Self {
            preview_max: Some(0),
        }
    }
}

impl ExecuteQueryOptions {
    /// No preview floats; still runs aggregates when the document has an operation.
    #[must_use]
    pub const fn execute_no_preview() -> Self {
        Self {
            preview_max: Some(0),
        }
    }

    /// Plan + catalog only (same as `tet query` without `-x`).
    #[must_use]
    pub const fn plan_only() -> Self {
        Self {
            preview_max: None,
        }
    }
}

/// Parse JSON, validate, and run against a mmap'd `.tet` (parity with `tet query -t … -x`).
///
/// When `options.preview_max` is `None`, only planning metadata is attached (no execution block
/// unless the document requests spill export — see [`plan_query_with_tet_mmap_ex`]).
///
/// # Errors
///
/// Returns [`TetError`] for JSON, validation, catalog, or execution failures.
pub fn execute_query_json(
    json: &str,
    tet_path: &Path,
    mmap: &[u8],
    options: ExecuteQueryOptions,
    spill_allowlist: Option<&SpillPathAllowlist>,
) -> Result<QueryResponse, TetError> {
    let doc = crate::query::parse_query_json(json)?;
    crate::query::validate_query(&doc)?;
    execute_query_document(
        &doc,
        tet_path,
        mmap,
        options,
        spill_allowlist,
    )
}

/// Run a validated [`QueryDocument`] against `mmap` (path echoed as `tet_file` when set).
///
/// # Errors
///
/// Same as [`execute_query_json`].
pub fn execute_query_document(
    doc: &QueryDocument,
    tet_path: &Path,
    mmap: &[u8],
    options: ExecuteQueryOptions,
    spill_allowlist: Option<&SpillPathAllowlist>,
) -> Result<QueryResponse, TetError> {
    let tet_path = tet_path.display().to_string();
    plan_query_with_tet_mmap_ex(
        doc,
        Some(&tet_path),
        mmap,
        options.preview_max,
        spill_allowlist,
    )
}

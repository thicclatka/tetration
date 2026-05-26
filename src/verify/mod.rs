//! `.tet` file health verification (layout, chunk integrity, footer).
//!
//! Distinct from [`crate::catalog`] parsing: this module runs **checks**, emits
//! [`VerifyFinding`]s, and suggests manual fixes via [`VerifyRecommendation`].
//! By default this module is read-only; use [`crate::repair`] or `tet repair` to apply fixes.
//!
//! # Example
//!
//! ```no_run
//! use tetration::verify::verify_tet_file;
//!
//! let report = verify_tet_file("data.tet".as_ref()).unwrap();
//! assert!(report.ok, "{:?}", report.recommendations);
//! ```

mod chunks;
mod footer;
mod format;
mod recommend;
mod report;
mod run;

pub use chunks::DEEP_DECODE_MAX_CHUNKS;
pub use format::{format_verify_json, format_verify_quiet, format_verify_text};
pub use report::{
    TetVerifyReport, VerifyFinding, VerifyFixHint, VerifyRecommendation, VerifySeverity,
    VerifySummary,
};
pub use run::{verify_tet_bytes, verify_tet_file};

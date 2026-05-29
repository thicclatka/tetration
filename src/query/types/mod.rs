//! JSON query / response types.

mod document;
mod error;
mod plan;
mod response;

pub use document::{
    AxisSlice, ExecutionDeviceHint, ExecutionHints, Operation, OutputHint, OutputHints,
    QueryDocument, WriteHints, WriteTarget,
};
pub use error::TetError;
pub use plan::{CHUNK_TOUCH_POLICY, ChunkTouchPolicy, PlannedChunkIo, ReadPlan};
pub use response::{DatasetResolution, QueryExecutionPreview, QueryResponse};

pub(crate) use response::{ExecutionPreviewIo, OperationPreviewFields, QueryExecutionPreviewBuild};

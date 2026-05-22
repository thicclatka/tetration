use thiserror::Error;

#[derive(Debug, Error)]
pub enum TetError {
    #[error("invalid query JSON: {0}")]
    InvalidJson(#[from] serde_json::Error),
    #[error("validation: {0}")]
    Validation(String),
    #[error(transparent)]
    Catalog(#[from] crate::catalog::CatalogError),
}

//! Transform method tokens (`transform.method` on the query wire).

use super::TetError;

/// Shape-preserving element-wise transform (pass-1 fold stats, pass-2 rewrite).
///
/// Wire methods: `zscore`, `minmax`, `l1`, `l2`, `center`, `scale`, `log1p`, `sqrt`, `softmax`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TransformMethod {
    Zscore,
    Minmax,
    L1,
    L2,
    Center,
    Scale,
    Log1p,
    Sqrt,
    Softmax,
}

impl TransformMethod {
    /// Parse wire `method` string.
    ///
    /// # Errors
    ///
    /// [`TetError::Validation`] when the token is unknown.
    pub fn parse(token: &str) -> Result<Self, TetError> {
        match token {
            "zscore" => Ok(Self::Zscore),
            "minmax" => Ok(Self::Minmax),
            "l1" => Ok(Self::L1),
            "l2" => Ok(Self::L2),
            "center" => Ok(Self::Center),
            "scale" => Ok(Self::Scale),
            "log1p" => Ok(Self::Log1p),
            "sqrt" => Ok(Self::Sqrt),
            "softmax" => Ok(Self::Softmax),
            _ => Err(TetError::Validation(format!(
                "unknown transform method `{token}` (expected zscore, minmax, l1, l2, center, scale, log1p, sqrt, or softmax)"
            ))),
        }
    }

    /// Stable wire token for JSON `transform.method`.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Zscore => "zscore",
            Self::Minmax => "minmax",
            Self::L1 => "l1",
            Self::L2 => "l2",
            Self::Center => "center",
            Self::Scale => "scale",
            Self::Log1p => "log1p",
            Self::Sqrt => "sqrt",
            Self::Softmax => "softmax",
        }
    }
}

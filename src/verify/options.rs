//! Options for [`super::verify_tet_bytes`] / [`super::verify_tet_file`].

/// Controls optional verify depth (decode walk, etc.).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VerifyOptions {
    /// When `true`, decode every chunk payload. When `false` (CLI default), only the first
    /// [`super::chunks::DEEP_DECODE_MAX_CHUNKS`] chunks are decode-checked on large files (quick scan).
    pub deep_decode: bool,
}

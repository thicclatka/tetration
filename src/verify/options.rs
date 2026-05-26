//! Options for [`super::verify_tet_bytes`] / [`super::verify_tet_file`].

/// Controls optional verify depth (decode walk, etc.).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VerifyOptions {
    /// Decode every chunk payload; when `false`, only the first
    /// [`super::chunks::DEEP_DECODE_MAX_CHUNKS`] are decoded on large files.
    pub deep_decode: bool,
}

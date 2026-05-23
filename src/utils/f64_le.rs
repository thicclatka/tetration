//! Little-endian `f64` views into chunk payloads.

#![allow(dead_code)]

/// Decode one little-endian `f64` at logical index `i` (`byte_offset = i * 8`).
///
/// # Panics
///
/// Panics if `i * 8` overflows [`usize`].
#[inline]
#[must_use]
pub fn read_f64_le_at(bytes: &[u8], i: usize) -> f64 {
    let off = i
        .checked_mul(8)
        .expect("f64 index overflow in read_f64_le_at");
    debug_assert!(off + 8 <= bytes.len());
    bytemuck::pod_read_unaligned::<f64>(&bytes[off..off + 8])
}

/// Zero-copy `f64` slice when `bytes.len()` is a multiple of 8 and the slice is 8-byte aligned.
#[must_use]
pub fn try_cast_f64_le(bytes: &[u8]) -> Option<&[f64]> {
    if !bytes.len().is_multiple_of(8) {
        return None;
    }
    bytemuck::try_cast_slice(bytes).ok()
}

/// `f64` element count for a little-endian payload byte length (must be divisible by 8).
#[must_use]
pub const fn f64_count(bytes_len: usize) -> Option<usize> {
    if bytes_len.is_multiple_of(8) {
        Some(bytes_len / 8)
    } else {
        None
    }
}

/// Byte length for `count` little-endian `f64` values (`count * 8`), or `None` on overflow.
#[must_use]
pub const fn bytes_from_elem_count(count: u64) -> Option<u64> {
    count.checked_mul(8)
}

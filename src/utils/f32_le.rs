//! Little-endian `f32` views into chunk payloads.
//!
//! Layout v1 writers place the first payload at an **8-byte-aligned** file offset; each raw
//! `f32` tile has a byte length divisible by **4**. Mmap subslices are not always 4-byte
//! aligned (e.g. after variable-size zstd chunks), so per-element reads use
//! [`bytemuck::pod_read_unaligned`]. When alignment and length allow it, [`try_cast_f32_le`]
//! exposes a zero-copy `&[f32]` slice.

/// Decode one little-endian `f32` at logical index `i` (`byte_offset = i * 4`).
///
/// No alignment requirement on `bytes`. The caller must ensure `bytes.len() >= (i + 1) * 4`.
///
/// # Panics
///
/// Panics if `i * 4` overflows [`usize`].
#[inline]
#[must_use]
pub fn read_f32_le_at(bytes: &[u8], i: usize) -> f32 {
    let off = i
        .checked_mul(4)
        .expect("f32 index overflow in read_f32_le_at");
    debug_assert!(off + 4 <= bytes.len());
    bytemuck::pod_read_unaligned::<f32>(&bytes[off..off + 4])
}

/// Zero-copy `f32` slice when `bytes.len()` is a multiple of 4 and the slice is 4-byte aligned.
#[must_use]
pub fn try_cast_f32_le(bytes: &[u8]) -> Option<&[f32]> {
    if !bytes.len().is_multiple_of(4) {
        return None;
    }
    bytemuck::try_cast_slice(bytes).ok()
}

/// `f32` element count for a little-endian payload byte length (must be divisible by 4).
#[must_use]
pub const fn f32_count(bytes_len: usize) -> Option<usize> {
    if bytes_len.is_multiple_of(4) {
        Some(bytes_len / 4)
    } else {
        None
    }
}

/// Byte length for `count` little-endian `f32` values (`count * 4`), or `None` on overflow.
#[must_use]
pub const fn bytes_from_elem_count(count: u64) -> Option<u64> {
    count.checked_mul(4)
}

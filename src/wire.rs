//! Little-endian fixed-width primitives shared by layout and catalog.

#[inline]
pub(crate) fn u32_le_at(data: &[u8], i: usize) -> u32 {
    u32::from_le_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]])
}

#[inline]
pub(crate) fn u64_le_at(data: &[u8], i: usize) -> u64 {
    u64::from_le_bytes([
        data[i],
        data[i + 1],
        data[i + 2],
        data[i + 3],
        data[i + 4],
        data[i + 5],
        data[i + 6],
        data[i + 7],
    ])
}

#[inline]
pub(crate) fn take_u32_le(data: &[u8], cur: &mut usize) -> u32 {
    let v = u32_le_at(data, *cur);
    *cur += 4;
    v
}

#[inline]
pub(crate) fn take_u64_le(data: &[u8], cur: &mut usize) -> u64 {
    let v = u64_le_at(data, *cur);
    *cur += 8;
    v
}

#[inline]
pub(crate) fn put_u32_le(buf: &mut [u8], o: &mut usize, v: u32) {
    buf[*o..*o + 4].copy_from_slice(&v.to_le_bytes());
    *o += 4;
}

#[inline]
pub(crate) fn put_u64_le(buf: &mut [u8], o: &mut usize, v: u64) {
    buf[*o..*o + 8].copy_from_slice(&v.to_le_bytes());
    *o += 8;
}

/// Bytes to append so `(len + result) % 8 == 0`.
#[inline]
pub(crate) fn padding_to_align8(len: usize) -> usize {
    (8 - (len % 8)) % 8
}

#[inline]
pub(crate) fn align8_u64(n: u64) -> u64 {
    (n + 7) & !7
}

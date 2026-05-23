//! Little-endian [`bytemuck::Pod`] element views into chunk payloads.
//!
//! Each numeric dtype gets a small submodule (`f32_le`, `f64_le`, …) via [`define_le_pod_module`].

/// Defines `read_*_le_at`, `try_cast_*_le`, `*_count`, and `bytes_from_elem_count` for one `Pod` type.
macro_rules! define_le_pod_module {
    (
        $(#[$module_meta:meta])*
        $vis:vis mod $mod_name:ident;
        ty $elem:ty;
        elem_size: $elem_size:expr;
        read_fn: $read_fn:ident;
        cast_fn: $cast_fn:ident;
        count_fn: $count_fn:ident;
        type_name: $type_name:literal;
    ) => {
        $(#[$module_meta])*
        $vis mod $mod_name {
            /// Decode one little-endian element at logical index `i`.
            ///
            /// No alignment requirement on `bytes`. The caller must ensure
            /// `bytes.len() >= (i + 1) * elem_size`.
            ///
            /// # Panics
            ///
            /// Panics if `i * elem_size` overflows [`usize`].
            #[inline]
            #[must_use]
            pub fn $read_fn(bytes: &[u8], i: usize) -> $elem {
                let off = i
                    .checked_mul($elem_size)
                    .expect(concat!($type_name, " index overflow in ", stringify!($read_fn)));
                debug_assert!(off + $elem_size <= bytes.len());
                bytemuck::pod_read_unaligned::<$elem>(&bytes[off..off + $elem_size])
            }

            /// Zero-copy slice when `bytes.len()` is a multiple of `elem_size` and aligned.
            #[must_use]
            pub fn $cast_fn(bytes: &[u8]) -> Option<&[$elem]> {
                if !bytes.len().is_multiple_of($elem_size) {
                    return None;
                }
                bytemuck::try_cast_slice(bytes).ok()
            }

            /// Element count for a payload byte length (must be divisible by `elem_size`).
            #[must_use]
            pub const fn $count_fn(bytes_len: usize) -> Option<usize> {
                if bytes_len.is_multiple_of($elem_size) {
                    Some(bytes_len / $elem_size)
                } else {
                    None
                }
            }

            /// Byte length for `count` elements (`count * elem_size`), or `None` on overflow.
            #[must_use]
            pub const fn bytes_from_elem_count(count: u64) -> Option<u64> {
                count.checked_mul($elem_size as u64)
            }
        }
    };
}

define_le_pod_module! {
    #[doc = "Little-endian `f32` views into chunk payloads.\n\n\
        Layout v1 writers place the first payload at an **8-byte-aligned** file offset; each raw \
        `f32` tile has a byte length divisible by **4**. Mmap subslices are not always 4-byte \
        aligned (e.g. after variable-size zstd chunks), so per-element reads use \
        [`bytemuck::pod_read_unaligned`]. When alignment and length allow it, [`try_cast_f32_le`] \
        exposes a zero-copy `&[f32]` slice."]
    pub mod f32_le;
    ty f32;
    elem_size: 4;
    read_fn: read_f32_le_at;
    cast_fn: try_cast_f32_le;
    count_fn: f32_count;
    type_name: "f32";
}

define_le_pod_module! {
    #[allow(dead_code)]
    pub(crate) mod f64_le;
    ty f64;
    elem_size: 8;
    read_fn: read_f64_le_at;
    cast_fn: try_cast_f64_le;
    count_fn: f64_count;
    type_name: "f64";
}

define_le_pod_module! {
    #[allow(dead_code)]
    pub(crate) mod i32_le;
    ty i32;
    elem_size: 4;
    read_fn: read_i32_le_at;
    cast_fn: try_cast_i32_le;
    count_fn: i32_count;
    type_name: "i32";
}

define_le_pod_module! {
    #[allow(dead_code)]
    pub(crate) mod i64_le;
    ty i64;
    elem_size: 8;
    read_fn: read_i64_le_at;
    cast_fn: try_cast_i64_le;
    count_fn: i64_count;
    type_name: "i64";
}

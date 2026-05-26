//! Single-pass sum / sum-of-squares and min / max over numeric slabs (`f32` SIMD when available).

#[inline]
pub(crate) fn scalar_f32_sum_sumsq(vals: &[f32]) -> (f64, f64) {
    let mut sum = 0.0f64;
    let mut sumsq = 0.0f64;
    for &v in vals {
        let x = f64::from(v);
        sum += x;
        sumsq += x * x;
    }
    (sum, sumsq)
}

#[cfg(target_arch = "aarch64")]
mod arch {
    use std::arch::aarch64::{
        vaddvq_f64, vcvt_f64_f32, vget_high_f32, vget_low_f32, vld1q_f32, vmulq_f64,
    };

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn f32_sum_sumsq(vals: &[f32]) -> (f64, f64) {
        let ptr = vals.as_ptr();
        let len = vals.len();
        let mut sum = 0.0f64;
        let mut sumsq = 0.0f64;
        let mut i = 0usize;
        let simd_end = len - (len % 4);
        while i < simd_end {
            // SAFETY: `i` is chunk-aligned; caller ensures `vals.len()` matches the slice.
            let v = unsafe { vld1q_f32(ptr.add(i)) };
            let lo = vcvt_f64_f32(vget_low_f32(v));
            let hi = vcvt_f64_f32(vget_high_f32(v));
            sum += vaddvq_f64(lo) + vaddvq_f64(hi);
            let sq_lo = vmulq_f64(lo, lo);
            let sq_hi = vmulq_f64(hi, hi);
            sumsq += vaddvq_f64(sq_lo) + vaddvq_f64(sq_hi);
            i += 4;
        }
        while i < len {
            let x = f64::from(unsafe { *ptr.add(i) });
            sum += x;
            sumsq += x * x;
            i += 1;
        }
        (sum, sumsq)
    }
}

#[cfg(target_arch = "x86_64")]
mod arch {
    use std::arch::x86_64::*;

    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn f32_sum_sumsq_sse2(vals: &[f32]) -> (f64, f64) {
        let ptr = vals.as_ptr();
        let len = vals.len();
        let mut sum = 0.0f64;
        let mut sumsq = 0.0f64;
        let mut i = 0usize;
        let simd_end = len - (len % 4);
        while i < simd_end {
            // SAFETY: `i` is chunk-aligned; caller ensures `vals.len()` matches the slice.
            let (lo_arr, hi_arr) = unsafe {
                let v = _mm_loadu_ps(ptr.add(i));
                let lo = _mm_cvtps_pd(v);
                let hi = _mm_cvtps_pd(_mm_movehl_ps(v, v));
                let mut lo_arr = [0.0f64; 2];
                let mut hi_arr = [0.0f64; 2];
                _mm_storeu_pd(lo_arr.as_mut_ptr(), lo);
                _mm_storeu_pd(hi_arr.as_mut_ptr(), hi);
                (lo_arr, hi_arr)
            };
            sum += lo_arr[0] + lo_arr[1] + hi_arr[0] + hi_arr[1];
            sumsq += lo_arr[0] * lo_arr[0]
                + lo_arr[1] * lo_arr[1]
                + hi_arr[0] * hi_arr[0]
                + hi_arr[1] * hi_arr[1];
            i += 4;
        }
        while i < len {
            let x = f64::from(unsafe { *ptr.add(i) });
            sum += x;
            sumsq += x * x;
            i += 1;
        }
        (sum, sumsq)
    }
}

/// Sum and sum-of-squares for population variance (`f64` accumulators).
#[must_use]
pub(crate) fn f32_sum_sumsq(vals: &[f32]) -> (f64, f64) {
    if vals.is_empty() {
        return (0.0, 0.0);
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { arch::f32_sum_sumsq(vals) };
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("sse2") {
            return unsafe { arch::f32_sum_sumsq_sse2(vals) };
        }
    }
    scalar_f32_sum_sumsq(vals)
}

#[inline]
pub(crate) fn scalar_f32_min_max(vals: &[f32]) -> (f64, f64) {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for &v in vals {
        let vd = f64::from(v);
        min = min.min(vd);
        max = max.max(vd);
    }
    (min, max)
}

#[cfg(target_arch = "aarch64")]
mod minmax_arch {
    use std::arch::aarch64::{
        vdupq_n_f32, vget_high_f32, vget_lane_f32, vget_low_f32, vld1q_f32, vmaxq_f32, vminq_f32,
        vpmax_f32, vpmin_f32,
    };

    #[inline]
    unsafe fn horizontal_max_f32(v: std::arch::aarch64::float32x4_t) -> f32 {
        // SAFETY: caller enables NEON via `#[target_feature]`.
        unsafe {
            let pair = vpmax_f32(vget_low_f32(v), vget_high_f32(v));
            let one = vpmax_f32(pair, pair);
            vget_lane_f32(one, 0)
        }
    }

    #[inline]
    unsafe fn horizontal_min_f32(v: std::arch::aarch64::float32x4_t) -> f32 {
        // SAFETY: caller enables NEON via `#[target_feature]`.
        unsafe {
            let pair = vpmin_f32(vget_low_f32(v), vget_high_f32(v));
            let one = vpmin_f32(pair, pair);
            vget_lane_f32(one, 0)
        }
    }

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn f32_min_max(vals: &[f32]) -> (f64, f64) {
        let ptr = vals.as_ptr();
        let len = vals.len();
        let mut min_v = vdupq_n_f32(f32::INFINITY);
        let mut max_v = vdupq_n_f32(f32::NEG_INFINITY);
        let mut i = 0usize;
        let simd_end = len - (len % 4);
        while i < simd_end {
            // SAFETY: `i` is chunk-aligned; caller ensures `vals.len()` matches the slice.
            let v = unsafe { vld1q_f32(ptr.add(i)) };
            min_v = vminq_f32(min_v, v);
            max_v = vmaxq_f32(max_v, v);
            i += 4;
        }
        let mut min_f = unsafe { horizontal_min_f32(min_v) };
        let mut max_f = unsafe { horizontal_max_f32(max_v) };
        while i < len {
            let x = unsafe { *ptr.add(i) };
            min_f = min_f.min(x);
            max_f = max_f.max(x);
            i += 1;
        }
        (f64::from(min_f), f64::from(max_f))
    }
}

#[cfg(target_arch = "x86_64")]
mod minmax_arch {
    use std::arch::x86_64::*;

    #[inline]
    unsafe fn horizontal_min_f32(v: __m128) -> f32 {
        // SAFETY: caller enables SSE2 via `#[target_feature]`.
        unsafe {
            let shuf = _mm_movehl_ps(v, v);
            let mins = _mm_min_ps(v, shuf);
            let shuf2 = _mm_shuffle_ps(mins, mins, 0x01);
            _mm_cvtss_f32(_mm_min_ss(mins, shuf2))
        }
    }

    #[inline]
    unsafe fn horizontal_max_f32(v: __m128) -> f32 {
        // SAFETY: caller enables SSE2 via `#[target_feature]`.
        unsafe {
            let shuf = _mm_movehl_ps(v, v);
            let maxs = _mm_max_ps(v, shuf);
            let shuf2 = _mm_shuffle_ps(maxs, maxs, 0x01);
            _mm_cvtss_f32(_mm_max_ss(maxs, shuf2))
        }
    }

    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn f32_min_max_sse2(vals: &[f32]) -> (f64, f64) {
        let ptr = vals.as_ptr();
        let len = vals.len();
        let mut min_v = _mm_set1_ps(f32::INFINITY);
        let mut max_v = _mm_set1_ps(f32::NEG_INFINITY);
        let mut i = 0usize;
        let simd_end = len - (len % 4);
        while i < simd_end {
            // SAFETY: `i` is chunk-aligned; caller ensures `vals.len()` matches the slice.
            let v = unsafe { _mm_loadu_ps(ptr.add(i)) };
            min_v = _mm_min_ps(min_v, v);
            max_v = _mm_max_ps(max_v, v);
            i += 4;
        }
        let mut min_f = unsafe { horizontal_min_f32(min_v) };
        let mut max_f = unsafe { horizontal_max_f32(max_v) };
        while i < len {
            let x = unsafe { *ptr.add(i) };
            min_f = min_f.min(x);
            max_f = max_f.max(x);
            i += 1;
        }
        (f64::from(min_f), f64::from(max_f))
    }
}

/// Min and max over an `f32` slab (`f64` accumulators, matches scalar fold promotion).
#[must_use]
pub(crate) fn f32_min_max(vals: &[f32]) -> (f64, f64) {
    if vals.is_empty() {
        return (f64::INFINITY, f64::NEG_INFINITY);
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { minmax_arch::f32_min_max(vals) };
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("sse2") {
            return unsafe { minmax_arch::f32_min_max_sse2(vals) };
        }
    }
    scalar_f32_min_max(vals)
}

/// Sum and sum-of-squares for population variance over an `f64` slice.
#[must_use]
pub(crate) fn f64_sum_sumsq(vals: &[f64]) -> (f64, f64) {
    let mut sum = 0.0f64;
    let mut sumsq = 0.0f64;
    for &v in vals {
        sum += v;
        sumsq += v * v;
    }
    (sum, sumsq)
}

#[inline]
pub(crate) fn scalar_i32_sum_sumsq(vals: &[i32]) -> (f64, f64) {
    let mut sum = 0.0f64;
    let mut sumsq = 0.0f64;
    for &v in vals {
        let x = f64::from(v);
        sum += x;
        sumsq += x * x;
    }
    (sum, sumsq)
}

#[cfg(target_arch = "x86_64")]
mod i32_arch {
    use std::arch::x86_64::*;

    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn i32_sum_sumsq_sse2(vals: &[i32]) -> (f64, f64) {
        let ptr = vals.as_ptr();
        let len = vals.len();
        let mut sum = 0.0f64;
        let mut sumsq = 0.0f64;
        let mut i = 0usize;
        let simd_end = len - (len % 4);
        while i < simd_end {
            let (lo_arr, hi_arr) = unsafe {
                let v = _mm_loadu_si128(ptr.add(i) as *const __m128i);
                let lo = _mm_cvtepi32_pd(v);
                let hi = _mm_cvtepi32_pd(_mm_shuffle_epi32(v, 0xEE));
                let mut lo_arr = [0.0f64; 2];
                let mut hi_arr = [0.0f64; 2];
                _mm_storeu_pd(lo_arr.as_mut_ptr(), lo);
                _mm_storeu_pd(hi_arr.as_mut_ptr(), hi);
                (lo_arr, hi_arr)
            };
            sum += lo_arr[0] + lo_arr[1] + hi_arr[0] + hi_arr[1];
            sumsq += lo_arr[0] * lo_arr[0]
                + lo_arr[1] * lo_arr[1]
                + hi_arr[0] * hi_arr[0]
                + hi_arr[1] * hi_arr[1];
            i += 4;
        }
        while i < len {
            let x = f64::from(unsafe { *ptr.add(i) });
            sum += x;
            sumsq += x * x;
            i += 1;
        }
        (sum, sumsq)
    }
}

#[cfg(target_arch = "aarch64")]
mod i32_aarch64 {
    use std::arch::aarch64::{
        vaddvq_f64, vcvt_f64_f32, vcvtq_f32_s32, vget_high_f32, vget_low_f32, vld1q_s32, vmulq_f64,
    };

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn i32_sum_sumsq_neon(vals: &[i32]) -> (f64, f64) {
        let ptr = vals.as_ptr();
        let len = vals.len();
        let mut sum = 0.0f64;
        let mut sumsq = 0.0f64;
        let mut i = 0usize;
        let simd_end = len - (len % 4);
        while i < simd_end {
            // SAFETY: `i` is chunk-aligned; caller ensures `vals.len()` matches the slice.
            let v = unsafe { vld1q_s32(ptr.add(i)) };
            let vf = vcvtq_f32_s32(v);
            let lo = vcvt_f64_f32(vget_low_f32(vf));
            let hi = vcvt_f64_f32(vget_high_f32(vf));
            sum += vaddvq_f64(lo) + vaddvq_f64(hi);
            let sq_lo = vmulq_f64(lo, lo);
            let sq_hi = vmulq_f64(hi, hi);
            sumsq += vaddvq_f64(sq_lo) + vaddvq_f64(sq_hi);
            i += 4;
        }
        while i < len {
            let x = f64::from(unsafe { *ptr.add(i) });
            sum += x;
            sumsq += x * x;
            i += 1;
        }
        (sum, sumsq)
    }
}

/// Sum and sum-of-squares for `i32` slabs promoted to `f64`.
#[must_use]
pub(crate) fn i32_sum_sumsq(vals: &[i32]) -> (f64, f64) {
    if vals.is_empty() {
        return (0.0, 0.0);
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { i32_aarch64::i32_sum_sumsq_neon(vals) };
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("sse2") {
            return unsafe { i32_arch::i32_sum_sumsq_sse2(vals) };
        }
    }
    scalar_i32_sum_sumsq(vals)
}

#[inline]
pub(crate) fn scalar_u8_sum_sumsq(vals: &[u8]) -> (f64, f64) {
    let mut sum = 0.0f64;
    let mut sumsq = 0.0f64;
    for &v in vals {
        let x = f64::from(v);
        sum += x;
        sumsq += x * x;
    }
    (sum, sumsq)
}

#[cfg(target_arch = "x86_64")]
mod u8_arch {
    use std::arch::x86_64::*;

    #[inline]
    pub(super) unsafe fn accum_epi32_pd(sum: &mut f64, sumsq: &mut f64, reg: __m128i) {
        // SAFETY: caller enables SSE2 via `#[target_feature]`.
        unsafe {
            let lo_pd = _mm_cvtepi32_pd(reg);
            let hi_pd = _mm_cvtepi32_pd(_mm_shuffle_epi32(reg, 0xEE));
            let mut lo_arr = [0.0f64; 2];
            let mut hi_arr = [0.0f64; 2];
            _mm_storeu_pd(lo_arr.as_mut_ptr(), lo_pd);
            _mm_storeu_pd(hi_arr.as_mut_ptr(), hi_pd);
            *sum += lo_arr[0] + lo_arr[1] + hi_arr[0] + hi_arr[1];
            *sumsq += lo_arr[0] * lo_arr[0]
                + lo_arr[1] * lo_arr[1]
                + hi_arr[0] * hi_arr[0]
                + hi_arr[1] * hi_arr[1];
        }
    }

    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn u8_sum_sumsq_sse2(vals: &[u8]) -> (f64, f64) {
        let ptr = vals.as_ptr();
        let len = vals.len();
        let mut sum = 0.0f64;
        let mut sumsq = 0.0f64;
        let mut i = 0usize;
        let simd_end = len - (len % 16);
        while i < simd_end {
            // SAFETY: `i` is chunk-aligned; caller ensures `vals.len()` matches the slice.
            unsafe {
                let zero = _mm_setzero_si128();
                let v = _mm_loadu_si128(ptr.add(i) as *const __m128i);
                let lo16 = _mm_unpacklo_epi8(v, zero);
                let hi16 = _mm_unpackhi_epi8(v, zero);
                accum_epi32_pd(&mut sum, &mut sumsq, _mm_unpacklo_epi16(lo16, zero));
                accum_epi32_pd(&mut sum, &mut sumsq, _mm_unpackhi_epi16(lo16, zero));
                accum_epi32_pd(&mut sum, &mut sumsq, _mm_unpacklo_epi16(hi16, zero));
                accum_epi32_pd(&mut sum, &mut sumsq, _mm_unpackhi_epi16(hi16, zero));
            }
            i += 16;
        }
        while i < len {
            let x = f64::from(unsafe { *ptr.add(i) });
            sum += x;
            sumsq += x * x;
            i += 1;
        }
        (sum, sumsq)
    }
}

/// Sum and sum-of-squares for `u8` slabs promoted to `f64`.
#[must_use]
pub(crate) fn u8_sum_sumsq(vals: &[u8]) -> (f64, f64) {
    if vals.is_empty() {
        return (0.0, 0.0);
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("sse2") {
            return unsafe { u8_arch::u8_sum_sumsq_sse2(vals) };
        }
    }
    scalar_u8_sum_sumsq(vals)
}

#[inline]
pub(crate) fn i32_min_max(vals: &[i32]) -> (f64, f64) {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for &v in vals {
        let vd = f64::from(v);
        min = min.min(vd);
        max = max.max(vd);
    }
    (min, max)
}

#[inline]
pub(crate) fn u8_min_max(vals: &[u8]) -> (f64, f64) {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for &v in vals {
        let vd = f64::from(v);
        min = min.min(vd);
        max = max.max(vd);
    }
    (min, max)
}

#[inline]
pub(crate) fn scalar_i64_sum_sumsq(vals: &[i64]) -> (f64, f64) {
    let mut sum = 0.0f64;
    let mut sumsq = 0.0f64;
    for &v in vals {
        let x = v as f64;
        sum += x;
        sumsq += x * x;
    }
    (sum, sumsq)
}

#[cfg(target_arch = "x86_64")]
mod i64_arch {
    use std::arch::x86_64::*;

    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn i64_sum_sumsq_sse2(vals: &[i64]) -> (f64, f64) {
        let ptr = vals.as_ptr();
        let len = vals.len();
        let mut sum = 0.0f64;
        let mut sumsq = 0.0f64;
        let mut i = 0usize;
        let simd_end = len - (len % 2);
        while i < simd_end {
            // SAFETY: `i` is chunk-aligned; caller ensures `vals.len()` matches the slice.
            unsafe {
                let v = _mm_loadu_si128(ptr.add(i) as *const __m128i);
                let lo = _mm_cvtsi128_si64(v) as f64;
                let hi = _mm_cvtsi128_si64(_mm_unpackhi_epi64(v, v)) as f64;
                sum += lo + hi;
                sumsq += lo * lo + hi * hi;
            }
            i += 2;
        }
        while i < len {
            let x = unsafe { *ptr.add(i) } as f64;
            sum += x;
            sumsq += x * x;
            i += 1;
        }
        (sum, sumsq)
    }
}

/// Sum and sum-of-squares for `i64` slabs promoted to `f64`.
#[must_use]
pub(crate) fn i64_sum_sumsq(vals: &[i64]) -> (f64, f64) {
    if vals.is_empty() {
        return (0.0, 0.0);
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("sse2") {
            return unsafe { i64_arch::i64_sum_sumsq_sse2(vals) };
        }
    }
    scalar_i64_sum_sumsq(vals)
}

#[inline]
pub(crate) fn scalar_u32_sum_sumsq(vals: &[u32]) -> (f64, f64) {
    let mut sum = 0.0f64;
    let mut sumsq = 0.0f64;
    for &v in vals {
        let x = f64::from(v);
        sum += x;
        sumsq += x * x;
    }
    (sum, sumsq)
}

#[cfg(target_arch = "x86_64")]
mod u32_arch {
    use std::arch::x86_64::*;

    #[target_feature(enable = "sse4.1")]
    pub(super) unsafe fn u32_sum_sumsq_sse41(vals: &[u32]) -> (f64, f64) {
        let ptr = vals.as_ptr();
        let len = vals.len();
        let mut sum = 0.0f64;
        let mut sumsq = 0.0f64;
        let mut i = 0usize;
        let simd_end = len - (len % 4);
        while i < simd_end {
            let (lo_arr, hi_arr) = unsafe {
                let v = _mm_loadu_si128(ptr.add(i) as *const __m128i);
                let lo = _mm_cvtepu32_pd(v);
                let hi = _mm_cvtepu32_pd(_mm_shuffle_epi32(v, 0xEE));
                let mut lo_arr = [0.0f64; 2];
                let mut hi_arr = [0.0f64; 2];
                _mm_storeu_pd(lo_arr.as_mut_ptr(), lo);
                _mm_storeu_pd(hi_arr.as_mut_ptr(), hi);
                (lo_arr, hi_arr)
            };
            sum += lo_arr[0] + lo_arr[1] + hi_arr[0] + hi_arr[1];
            sumsq += lo_arr[0] * lo_arr[0]
                + lo_arr[1] * lo_arr[1]
                + hi_arr[0] * hi_arr[0]
                + hi_arr[1] * hi_arr[1];
            i += 4;
        }
        while i < len {
            let x = f64::from(unsafe { *ptr.add(i) });
            sum += x;
            sumsq += x * x;
            i += 1;
        }
        (sum, sumsq)
    }
}

/// Sum and sum-of-squares for `u32` slabs promoted to `f64`.
#[must_use]
pub(crate) fn u32_sum_sumsq(vals: &[u32]) -> (f64, f64) {
    if vals.is_empty() {
        return (0.0, 0.0);
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("sse4.1") {
            return unsafe { u32_arch::u32_sum_sumsq_sse41(vals) };
        }
    }
    scalar_u32_sum_sumsq(vals)
}

#[inline]
pub(crate) fn scalar_u64_sum_sumsq(vals: &[u64]) -> (f64, f64) {
    let mut sum = 0.0f64;
    let mut sumsq = 0.0f64;
    for &v in vals {
        let x = v as f64;
        sum += x;
        sumsq += x * x;
    }
    (sum, sumsq)
}

#[cfg(target_arch = "x86_64")]
mod u64_arch {
    use std::arch::x86_64::*;

    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn u64_sum_sumsq_sse2(vals: &[u64]) -> (f64, f64) {
        let ptr = vals.as_ptr();
        let len = vals.len();
        let mut sum = 0.0f64;
        let mut sumsq = 0.0f64;
        let mut i = 0usize;
        let simd_end = len - (len % 2);
        while i < simd_end {
            // SAFETY: `i` is chunk-aligned; caller ensures `vals.len()` matches the slice.
            unsafe {
                let v = _mm_loadu_si128(ptr.add(i) as *const __m128i);
                let lo = _mm_cvtsi128_si64(v) as f64;
                let hi = _mm_cvtsi128_si64(_mm_unpackhi_epi64(v, v)) as f64;
                sum += lo + hi;
                sumsq += lo * lo + hi * hi;
            }
            i += 2;
        }
        while i < len {
            let x = unsafe { *ptr.add(i) } as f64;
            sum += x;
            sumsq += x * x;
            i += 1;
        }
        (sum, sumsq)
    }
}

/// Sum and sum-of-squares for `u64` slabs promoted to `f64`.
#[must_use]
pub(crate) fn u64_sum_sumsq(vals: &[u64]) -> (f64, f64) {
    if vals.is_empty() {
        return (0.0, 0.0);
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("sse2") {
            return unsafe { u64_arch::u64_sum_sumsq_sse2(vals) };
        }
    }
    scalar_u64_sum_sumsq(vals)
}

#[inline]
pub(crate) fn scalar_u16_sum_sumsq(vals: &[u16]) -> (f64, f64) {
    let mut sum = 0.0f64;
    let mut sumsq = 0.0f64;
    for &v in vals {
        let x = f64::from(v);
        sum += x;
        sumsq += x * x;
    }
    (sum, sumsq)
}

#[cfg(target_arch = "x86_64")]
mod u16_arch {
    use super::u8_arch::accum_epi32_pd;
    use std::arch::x86_64::*;

    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn u16_sum_sumsq_sse2(vals: &[u16]) -> (f64, f64) {
        let ptr = vals.as_ptr();
        let len = vals.len();
        let mut sum = 0.0f64;
        let mut sumsq = 0.0f64;
        let mut i = 0usize;
        let simd_end = len - (len % 8);
        while i < simd_end {
            // SAFETY: `i` is chunk-aligned; caller ensures `vals.len()` matches the slice.
            unsafe {
                let zero = _mm_setzero_si128();
                let v = _mm_loadu_si128(ptr.add(i) as *const __m128i);
                accum_epi32_pd(&mut sum, &mut sumsq, _mm_unpacklo_epi16(v, zero));
                accum_epi32_pd(&mut sum, &mut sumsq, _mm_unpackhi_epi16(v, zero));
            }
            i += 8;
        }
        while i < len {
            let x = f64::from(unsafe { *ptr.add(i) });
            sum += x;
            sumsq += x * x;
            i += 1;
        }
        (sum, sumsq)
    }
}

/// Sum and sum-of-squares for `u16` slabs promoted to `f64`.
#[must_use]
pub(crate) fn u16_sum_sumsq(vals: &[u16]) -> (f64, f64) {
    if vals.is_empty() {
        return (0.0, 0.0);
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("sse2") {
            return unsafe { u16_arch::u16_sum_sumsq_sse2(vals) };
        }
    }
    scalar_u16_sum_sumsq(vals)
}

#[inline]
pub(crate) fn i64_min_max(vals: &[i64]) -> (f64, f64) {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for &v in vals {
        let vd = v as f64;
        min = min.min(vd);
        max = max.max(vd);
    }
    (min, max)
}

#[inline]
pub(crate) fn u32_min_max(vals: &[u32]) -> (f64, f64) {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for &v in vals {
        let vd = f64::from(v);
        min = min.min(vd);
        max = max.max(vd);
    }
    (min, max)
}

#[inline]
pub(crate) fn u64_min_max(vals: &[u64]) -> (f64, f64) {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for &v in vals {
        let vd = v as f64;
        min = min.min(vd);
        max = max.max(vd);
    }
    (min, max)
}

#[inline]
pub(crate) fn i16_sum_sumsq(vals: &[i16]) -> (f64, f64) {
    let mut sum = 0.0f64;
    let mut sumsq = 0.0f64;
    for &v in vals {
        let x = f64::from(v);
        sum += x;
        sumsq += x * x;
    }
    (sum, sumsq)
}

#[inline]
pub(crate) fn i16_min_max(vals: &[i16]) -> (f64, f64) {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for &v in vals {
        let vd = f64::from(v);
        min = min.min(vd);
        max = max.max(vd);
    }
    (min, max)
}

#[inline]
pub(crate) fn u16_min_max(vals: &[u16]) -> (f64, f64) {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for &v in vals {
        let vd = f64::from(v);
        min = min.min(vd);
        max = max.max(vd);
    }
    (min, max)
}

const F16_SIMD_CHUNK: usize = 8;

/// Sum and sum-of-squares for `f16` slabs (via `f32` SIMD chunks).
#[must_use]
pub(crate) fn f16_sum_sumsq(vals: &[half::f16]) -> (f64, f64) {
    if vals.is_empty() {
        return (0.0, 0.0);
    }
    let mut buf = [0.0f32; F16_SIMD_CHUNK];
    let mut sum = 0.0f64;
    let mut sumsq = 0.0f64;
    let mut i = 0usize;
    while i + F16_SIMD_CHUNK <= vals.len() {
        for (j, slot) in buf.iter_mut().enumerate() {
            *slot = f32::from(vals[i + j]);
        }
        let (s, sq) = f32_sum_sumsq(&buf);
        sum += s;
        sumsq += sq;
        i += F16_SIMD_CHUNK;
    }
    while i < vals.len() {
        let x = f64::from(f32::from(vals[i]));
        sum += x;
        sumsq += x * x;
        i += 1;
    }
    (sum, sumsq)
}

/// Min and max for `f16` slabs (via `f32` SIMD chunks).
#[must_use]
pub(crate) fn f16_min_max(vals: &[half::f16]) -> (f64, f64) {
    if vals.is_empty() {
        return (f64::INFINITY, f64::NEG_INFINITY);
    }
    let mut buf = [0.0f32; F16_SIMD_CHUNK];
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    let mut i = 0usize;
    while i + F16_SIMD_CHUNK <= vals.len() {
        for (j, slot) in buf.iter_mut().enumerate() {
            *slot = f32::from(vals[i + j]);
        }
        let (slab_min, slab_max) = f32_min_max(&buf);
        min = min.min(slab_min);
        max = max.max(slab_max);
        i += F16_SIMD_CHUNK;
    }
    while i < vals.len() {
        let vd = f64::from(f32::from(vals[i]));
        min = min.min(vd);
        max = max.max(vd);
        i += 1;
    }
    (min, max)
}

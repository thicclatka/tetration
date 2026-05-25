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

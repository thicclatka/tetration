//! Single-pass sum / sum-of-squares over numeric slabs (SIMD for `f32` when available).

#[inline]
fn scalar_f32_sum_sumsq(vals: &[f32]) -> (f64, f64) {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_slice() {
        assert_eq!(f32_sum_sumsq(&[]), (0.0, 0.0));
    }

    #[test]
    fn matches_scalar_reference() {
        let vals: Vec<f32> = (0..10_000).map(|i| (i as f32) * 0.001 - 5.0).collect();
        let scalar = scalar_f32_sum_sumsq(&vals);
        let fast = f32_sum_sumsq(&vals);
        assert!((scalar.0 - fast.0).abs() < 1e-6);
        assert!((scalar.1 - fast.1).abs() < 1e-3);
    }
}

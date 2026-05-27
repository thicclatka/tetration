//! Population covariance / Pearson correlation (rank-2).

use crate::query::materialize::covariance::run_covariance_correlation;

#[test]
fn cov_3x2_obs_axis_0() {
    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape = [3_u64, 2];
    let out = run_covariance_correlation(&values, &shape, 0, false).unwrap();
    let cov = out.covariance.unwrap();
    let p = 2;
    assert!((cov[0] - 8.0 / 3.0).abs() < 1e-9);
    assert!((cov[1] - 8.0 / 3.0).abs() < 1e-9);
    assert!((cov[p + 1] - 8.0 / 3.0).abs() < 1e-9);
}

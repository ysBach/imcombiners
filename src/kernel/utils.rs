//! Shared helpers: a tiny `Float` trait so combine/reject kernels can be generic
//! over `f32` / `f64` without pulling in `num_traits`.

use numpy::Element;
use std::env;
use std::sync::atomic::{AtomicUsize, Ordering};

const DEFAULT_PARALLEL_THRESHOLD: usize = 10000; // image > 100*100 will start to use parallelism by default
static PARALLEL_THRESHOLD: AtomicUsize = AtomicUsize::new(0);

pub fn parallel_threshold() -> usize {
    let current = PARALLEL_THRESHOLD.load(Ordering::Relaxed);
    if current != 0 {
        return current;
    }
    let threshold = env::var("IMCOMBINERS_PARALLEL_THRESHOLD")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(DEFAULT_PARALLEL_THRESHOLD);
    let _ = PARALLEL_THRESHOLD.compare_exchange(0, threshold, Ordering::Relaxed, Ordering::Relaxed);
    parallel_threshold()
}

pub fn set_parallel_threshold(threshold: usize) {
    PARALLEL_THRESHOLD.store(threshold, Ordering::Relaxed);
}

pub trait Float: Copy + PartialOrd + Element + Send + Sync + 'static {
    fn nan() -> Self;
    fn is_finite(self) -> bool;
    fn is_nan(self) -> bool;
    fn zero() -> Self;
    fn from_f64(x: f64) -> Self;
    fn to_f64(self) -> f64;
}

impl Float for f32 {
    #[inline]
    fn nan() -> Self {
        f32::NAN
    }
    #[inline]
    fn is_finite(self) -> bool {
        f32::is_finite(self)
    }
    #[inline]
    fn is_nan(self) -> bool {
        f32::is_nan(self)
    }
    #[inline]
    fn zero() -> Self {
        0.0
    }
    #[inline]
    fn from_f64(x: f64) -> Self {
        x as f32
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }
}

impl Float for f64 {
    #[inline]
    fn nan() -> Self {
        f64::NAN
    }
    #[inline]
    fn is_finite(self) -> bool {
        f64::is_finite(self)
    }
    #[inline]
    fn is_nan(self) -> bool {
        f64::is_nan(self)
    }
    #[inline]
    fn zero() -> Self {
        0.0
    }
    #[inline]
    fn from_f64(x: f64) -> Self {
        x
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self
    }
}

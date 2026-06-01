//! Shared helpers: a tiny `Float` trait so combine/reject kernels can be generic
//! over `f32` / `f64` without pulling in `num_traits`.

use numpy::Element;
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

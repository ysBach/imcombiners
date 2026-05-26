//! Along-axis-0 combine kernels. Each output pixel (h, w) is computed from the
//! length-N per-output stack `arr[:, h, w]`. NaN values are treated as masked.
//!
//! Parallelism: rayon over the flattened (H * W) output index.

use ndarray::{Array2, ArrayView3};
use numpy::Element;
use rayon::prelude::*;

use super::utils::{parallel_threshold, Float};

/// Combine method selector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CombineKind {
    Mean,
    Median,
    LMedian, // IRAF-style: lower of two middle values for even N
    Sum,
    Min,
    Max,
    Variance,
    WeightedAverage,
}

impl CombineKind {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "mean" | "average" | "avg" => Some(Self::Mean),
            "median" | "med" | "medi" => Some(Self::Median),
            "lmedian" | "lmed" | "lmd" => Some(Self::LMedian),
            "sum" => Some(Self::Sum),
            "min" => Some(Self::Min),
            "max" => Some(Self::Max),
            "variance" | "var" => Some(Self::Variance),
            "weighted_average" | "wvg" => Some(Self::WeightedAverage),
            _ => None,
        }
    }
}

// ---------- per-output-stack reductions ----------

#[inline]
fn flat_index(k: usize, pixel: usize, hw: usize) -> usize {
    k * hw + pixel
}

#[inline]
fn nanmean_strided<T: Float>(data: &[T], pixel: usize, n: usize, hw: usize) -> T {
    let mut s = 0.0_f64;
    let mut count = 0usize;
    for k in 0..n {
        let x = data[flat_index(k, pixel, hw)];
        if x.is_finite() {
            s += x.to_f64();
            count += 1;
        }
    }
    if count == 0 {
        T::nan()
    } else {
        T::from_f64(s / count as f64)
    }
}

#[inline]
fn nansum_strided<T: Float>(data: &[T], pixel: usize, n: usize, hw: usize) -> T {
    let mut s = 0.0_f64;
    let mut count = 0usize;
    for k in 0..n {
        let x = data[flat_index(k, pixel, hw)];
        if x.is_finite() {
            s += x.to_f64();
            count += 1;
        }
    }
    if count == 0 {
        T::nan()
    } else {
        T::from_f64(s)
    }
}

#[inline]
fn nanmin_strided<T: Float>(data: &[T], pixel: usize, n: usize, hw: usize) -> T {
    let mut out = T::nan();
    for k in 0..n {
        let x = data[flat_index(k, pixel, hw)];
        if x.is_finite() && (out.is_nan() || x < out) {
            out = x;
        }
    }
    out
}

#[inline]
fn nanmax_strided<T: Float>(data: &[T], pixel: usize, n: usize, hw: usize) -> T {
    let mut out = T::nan();
    for k in 0..n {
        let x = data[flat_index(k, pixel, hw)];
        if x.is_finite() && (out.is_nan() || x > out) {
            out = x;
        }
    }
    out
}

#[inline]
fn weighted_average_strided<T: Float>(
    data: &[T],
    pixel: usize,
    n: usize,
    hw: usize,
    weights: &[f64],
) -> T {
    debug_assert_eq!(n, weights.len());
    let mut s = 0.0_f64;
    let mut wsum = 0.0_f64;
    for k in 0..n {
        let x = data[flat_index(k, pixel, hw)];
        if x.is_finite() {
            let w = weights[k];
            s += x.to_f64() * w;
            wsum += w;
        }
    }
    if wsum == 0.0 {
        T::nan()
    } else {
        T::from_f64(s / wsum)
    }
}

#[inline]
fn nanvariance_strided<T: Float>(data: &[T], pixel: usize, n: usize, hw: usize, ddof: usize) -> T {
    let mut sum = 0.0_f64;
    let mut sumsq = 0.0_f64;
    let mut count = 0usize;
    for k in 0..n {
        let x = data[flat_index(k, pixel, hw)];
        if x.is_finite() {
            let xf = x.to_f64();
            sum += xf;
            sumsq += xf * xf;
            count += 1;
        }
    }
    if count <= ddof {
        return T::nan();
    }
    let mean = sum / count as f64;
    let numerator = (sumsq - sum * mean).max(0.0);
    T::from_f64(numerator / (count - ddof) as f64)
}

/// Extract finite values into `buf`, returns the finite count.
#[inline]
fn compact_finite_strided<T: Float>(
    data: &[T],
    pixel: usize,
    n: usize,
    hw: usize,
    buf: &mut [T],
) -> usize {
    let mut count = 0;
    for k in 0..n {
        let x = data[flat_index(k, pixel, hw)];
        if x.is_finite() {
            buf[count] = x;
            count += 1;
        }
    }
    count
}

/// In-place partial sort; we just sort fully (stack length is small, O(N log N) is fine).
#[inline]
fn sort_floats<T: Float>(buf: &mut [T]) {
    buf.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
}

#[inline]
fn nanmedian_strided<T: Float>(data: &[T], pixel: usize, n: usize, hw: usize, buf: &mut [T]) -> T {
    let count = compact_finite_strided(data, pixel, n, hw, buf);
    if count == 0 {
        return T::nan();
    }
    let s = &mut buf[..count];
    sort_floats(s);
    let mid = count / 2;
    if count % 2 == 1 {
        s[mid]
    } else {
        // Average of two middle values.
        T::from_f64((s[mid - 1].to_f64() + s[mid].to_f64()) / 2.0)
    }
}

#[inline]
fn nanlmedian_strided<T: Float>(data: &[T], pixel: usize, n: usize, hw: usize, buf: &mut [T]) -> T {
    // Lower median: for even N, return s[mid-1].
    let count = compact_finite_strided(data, pixel, n, hw, buf);
    if count == 0 {
        return T::nan();
    }
    let s = &mut buf[..count];
    sort_floats(s);
    let mid = count / 2;
    if count % 2 == 1 {
        s[mid]
    } else {
        s[mid - 1]
    }
}

// ---------- driver ----------

/// Combine `arr` (N, H, W) along axis 0. NaN-aware.
///
/// For `WeightedAverage`, `weights` must be `Some(slice of length N)`. Otherwise it is ignored.
pub fn combine_axis0<T: Float>(
    arr: &ArrayView3<T>,
    kind: CombineKind,
    weights: Option<&[f64]>,
    ddof: usize,
) -> Array2<T> {
    let (n, h, w) = (arr.shape()[0], arr.shape()[1], arr.shape()[2]);
    let hw = h * w;
    let mut out = Array2::<T>::from_elem((h, w), T::nan());

    // Take a contiguous reshape so par_iter_mut works over (h*w) cleanly.
    // Output is owned + freshly allocated, so it's contiguous: this is safe.
    let out_slice = out.as_slice_mut().expect("output is contiguous");

    // For weighted average, panic-up-front guard.
    if matches!(kind, CombineKind::WeightedAverage) {
        let w_ref = weights.expect("weighted average requires weights");
        assert_eq!(w_ref.len(), n, "weights length must equal stack size N");
    }

    let data = arr
        .as_slice_memory_order()
        .expect("combine kernels require contiguous C-order arrays");

    let compute_pixel = |pixel: usize, out_px: &mut T| {
        *out_px = match kind {
            CombineKind::Mean => nanmean_strided(data, pixel, n, hw),
            CombineKind::Sum => nansum_strided(data, pixel, n, hw),
            CombineKind::Min => nanmin_strided(data, pixel, n, hw),
            CombineKind::Max => nanmax_strided(data, pixel, n, hw),
            CombineKind::Variance => nanvariance_strided(data, pixel, n, hw, ddof),
            CombineKind::Median => {
                let mut tmp: Vec<T> = vec![T::zero(); n];
                nanmedian_strided(data, pixel, n, hw, &mut tmp)
            }
            CombineKind::LMedian => {
                let mut tmp: Vec<T> = vec![T::zero(); n];
                nanlmedian_strided(data, pixel, n, hw, &mut tmp)
            }
            CombineKind::WeightedAverage => {
                let w_ref = weights.expect("weights checked above");
                weighted_average_strided(data, pixel, n, hw, w_ref)
            }
        };
    };

    if hw >= parallel_threshold() {
        out_slice
            .par_iter_mut()
            .enumerate()
            .for_each(|(pixel, out_px)| compute_pixel(pixel, out_px));
    } else {
        out_slice
            .iter_mut()
            .enumerate()
            .for_each(|(pixel, out_px)| compute_pixel(pixel, out_px));
    }

    out
}

pub fn lmedian_axis0_ord<T>(arr: &ArrayView3<T>) -> Array2<T>
where
    T: Copy + Default + Element + Ord + Send + Sync + 'static,
{
    let (n, h, w) = (arr.shape()[0], arr.shape()[1], arr.shape()[2]);
    let hw = h * w;
    let mut out = Array2::<T>::from_elem((h, w), T::default());
    let out_slice = out.as_slice_mut().expect("Array2 contiguous");
    let kth = (n - 1) / 2;
    let data = arr
        .as_slice_memory_order()
        .expect("lmedian kernel requires contiguous C-order arrays");

    let compute_pixel = |pixel: usize, out_px: &mut T| {
        let mut col: Vec<T> = (0..n).map(|k| data[flat_index(k, pixel, hw)]).collect();
        col.sort_unstable();
        *out_px = col[kth];
    };

    if hw >= parallel_threshold() {
        out_slice
            .par_iter_mut()
            .enumerate()
            .for_each(|(pixel, out_px)| compute_pixel(pixel, out_px));
    } else {
        out_slice
            .iter_mut()
            .enumerate()
            .for_each(|(pixel, out_px)| compute_pixel(pixel, out_px));
    }

    out
}

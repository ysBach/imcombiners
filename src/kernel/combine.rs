//! Along-axis-0 combine adapters.
//!
//! The generic reduction algorithms live in the companion `reducers` crate.
//! This module keeps imcombiners' small API-specific layer: method parsing,
//! ndarray shape conversion, finite-only policy, and weighted output casting.

use ndarray::{Array2, Array3, ArrayView3};
use numpy::Element;
use rayon::prelude::*;
use reducers::axis;
use reducers::reducers_1d::{
    lmedian_valid_in_place, median_valid_in_place, percentiles_valid_in_place, var_mean_valid,
    Kind, Number,
};
use reducers::{Float, ScanPolicy};

const DEFAULT_STACK_PARALLEL_HW_THRESHOLD: usize = 10_000;

/// Combine method selector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CombineKind {
    Mean,
    Median,
    LMedian,
    Sum,
    Min,
    Max,
    Variance,
    NanAverage,
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
            _ => None,
        }
    }

    fn reducers_kind(self) -> Option<Kind> {
        match self {
            Self::Mean => Some(Kind::Mean),
            Self::Median => Some(Kind::Median),
            Self::LMedian => Some(Kind::LMedian),
            Self::Sum => Some(Kind::Sum),
            Self::Min => Some(Kind::Min),
            Self::Max => Some(Kind::Max),
            Self::Variance => Some(Kind::Var),
            Self::NanAverage => None,
        }
    }
}

fn axis0_shape<T>(arr: &ArrayView3<T>) -> (usize, usize, usize, usize) {
    let (n, h, w) = (arr.shape()[0], arr.shape()[1], arr.shape()[2]);
    (n, h, w, h * w)
}

fn contiguous_data<'a, T>(arr: &'a ArrayView3<T>, name: &str) -> &'a [T] {
    arr.as_slice_memory_order()
        .unwrap_or_else(|| panic!("{name} requires contiguous C-order arrays"))
}

fn array2_from_vec<T>(h: usize, w: usize, values: Vec<T>) -> Array2<T> {
    Array2::from_shape_vec((h, w), values).expect("reducers axis output has H*W elements")
}

#[inline]
fn should_parallel_stack(hw: usize, n: usize) -> bool {
    n > 0 && hw >= DEFAULT_STACK_PARALLEL_HW_THRESHOLD
}

#[inline]
fn compact_finite_strided<T: Float>(
    data: &[T],
    pixel: usize,
    n: usize,
    hw: usize,
    buf: &mut [T],
) -> usize {
    let mut count = 0;
    let mut idx = pixel;
    for _ in 0..n {
        let x = data[idx];
        if x.is_finite() {
            buf[count] = x;
            count += 1;
        }
        idx += hw;
    }
    count
}

#[inline]
fn median_strided<T: Float>(data: &[T], pixel: usize, n: usize, hw: usize, buf: &mut [T]) -> T {
    let count = compact_finite_strided(data, pixel, n, hw, buf);
    T::from_f64(median_valid_in_place(&mut buf[..count]))
}

#[inline]
fn lmedian_strided<T: Float>(data: &[T], pixel: usize, n: usize, hw: usize, buf: &mut [T]) -> T {
    let count = compact_finite_strided(data, pixel, n, hw, buf);
    T::from_f64(lmedian_valid_in_place(&mut buf[..count]))
}

#[inline]
fn var_mean_strided<T: Float>(
    data: &[T],
    pixel: usize,
    n: usize,
    hw: usize,
    ddof: usize,
    buf: &mut [T],
) -> (T, T) {
    let count = compact_finite_strided(data, pixel, n, hw, buf);
    let (var, mean) = var_mean_valid(&buf[..count], ddof);
    (T::from_f64(var), T::from_f64(mean))
}

fn median_axis0_stack<T: Float>(arr: &ArrayView3<T>, lower: bool) -> Array2<T> {
    let (n, h, w, outer) = axis0_shape(arr);
    let data = contiguous_data(arr, "median kernel");
    let mut values = vec![T::nan(); outer];

    let compute_pixel = |tmp: &mut Vec<T>, (pixel, out_px): (usize, &mut T)| {
        *out_px = if lower {
            lmedian_strided(data, pixel, n, outer, tmp)
        } else {
            median_strided(data, pixel, n, outer, tmp)
        };
    };

    if should_parallel_stack(outer, n) {
        values
            .par_iter_mut()
            .enumerate()
            .map_init(|| vec![T::zero(); n], compute_pixel)
            .count();
    } else {
        let mut tmp = vec![T::zero(); n];
        values
            .iter_mut()
            .enumerate()
            .for_each(|item| compute_pixel(&mut tmp, item));
    }

    array2_from_vec(h, w, values)
}

/// Combine `arr` (N, H, W) along axis 0 with imcombiners finite-only semantics.
///
/// For `NanAverage`, `weights` must be `Some(slice of length N)`. Otherwise it is ignored.
pub fn combine_axis0<T: Float>(
    arr: &ArrayView3<T>,
    kind: CombineKind,
    weights: Option<&[f64]>,
    ddof: usize,
) -> Array2<T> {
    let (n, h, w, outer) = axis0_shape(arr);

    if matches!(kind, CombineKind::Median | CombineKind::LMedian) {
        return median_axis0_stack(arr, matches!(kind, CombineKind::LMedian));
    }

    let data = contiguous_data(arr, "combine kernel");

    if matches!(kind, CombineKind::NanAverage) {
        let weights = weights.expect("nanaverage requires weights");
        assert_eq!(weights.len(), n, "weights length must equal stack size N");
        let weighted =
            axis::weighted_axis0(data, weights, true, n, outer, ScanPolicy::SkipNonFinite);
        let values = weighted.values.into_iter().map(T::from_f64).collect();
        return array2_from_vec(h, w, values);
    }

    let reducers_kind = kind
        .reducers_kind()
        .expect("non-weighted combine kind has reducers mapping");
    let values = axis::reduce_axis0(
        data,
        n,
        outer,
        reducers_kind,
        ddof,
        ScanPolicy::SkipNonFinite,
    );
    array2_from_vec(h, w, values)
}

/// Return `(variance, mean)` along axis 0.
pub fn variance_mean_axis0<T: Float>(arr: &ArrayView3<T>, ddof: usize) -> (Array2<T>, Array2<T>) {
    let (n, h, w, outer) = axis0_shape(arr);
    let data = contiguous_data(arr, "variance kernel");
    let mut vars = vec![T::nan(); outer];
    let mut means = vec![T::nan(); outer];

    let compute_pixel =
        |tmp: &mut Vec<T>, ((pixel, out_var), out_mean): ((usize, &mut T), &mut T)| {
            let (var, mean) = var_mean_strided(data, pixel, n, outer, ddof, tmp);
            *out_var = var;
            *out_mean = mean;
        };

    if should_parallel_stack(outer, n) {
        vars.par_iter_mut()
            .enumerate()
            .zip(means.par_iter_mut())
            .map_init(|| vec![T::zero(); n], compute_pixel)
            .count();
    } else {
        let mut tmp = vec![T::zero(); n];
        vars.iter_mut()
            .enumerate()
            .zip(means.iter_mut())
            .for_each(|item| compute_pixel(&mut tmp, item));
    }

    (array2_from_vec(h, w, vars), array2_from_vec(h, w, means))
}

/// Return NaN-aware percentiles along axis 0.
///
/// The result shape is `(H, W, Q)` so each output pixel owns one contiguous
/// percentile vector. Python moves the percentile axis to the front for the
/// public NumPy-compatible shape.
pub fn percentiles_axis0<T: Float>(arr: &ArrayView3<T>, qs: &[f64]) -> Array3<T> {
    let (n, h, w, outer) = axis0_shape(arr);
    let nq = qs.len();
    let data = contiguous_data(arr, "percentile kernel");

    let mut by_pixel = vec![f64::NAN; outer * nq];
    if nq > 0 {
        let compute_pixel = |tmp: &mut Vec<T>, (pixel, out_q): (usize, &mut [f64])| {
            let count = compact_finite_strided(data, pixel, n, outer, tmp);
            percentiles_valid_in_place(&mut tmp[..count], qs, out_q);
        };

        if should_parallel_stack(outer, n) {
            by_pixel
                .par_chunks_mut(nq)
                .enumerate()
                .map_init(|| vec![T::zero(); n], compute_pixel)
                .count();
        } else {
            let mut tmp = vec![T::zero(); n];
            by_pixel
                .chunks_mut(nq)
                .enumerate()
                .for_each(|item| compute_pixel(&mut tmp, item));
        }
    }

    let values = by_pixel.into_iter().map(T::from_f64).collect();
    Array3::from_shape_vec((h, w, nq), values).expect("percentile output has H*W*Q elements")
}

pub fn lmedian_axis0_ord<T>(arr: &ArrayView3<T>) -> Array2<T>
where
    T: Copy + Element + Number,
{
    let (n, h, w, outer) = axis0_shape(arr);
    let data = contiguous_data(arr, "lmedian kernel");
    let values = axis::reduce_axis0_number_exact(data, n, outer, Kind::LMedian);
    array2_from_vec(h, w, values)
}

#[cfg(test)]
mod tests {
    use super::{combine_axis0, CombineKind};
    use ndarray::Array3;

    #[test]
    fn median_preserves_finite_extremes_after_filtering_nonfinite_samples() {
        for value in [f64::MAX, -f64::MAX] {
            let stack =
                Array3::from_shape_vec((4, 1, 1), vec![value, f64::NAN, value, f64::INFINITY])
                    .unwrap();
            let result = combine_axis0(&stack.view(), CombineKind::Median, None, 0);
            assert_eq!(result[[0, 0]], value);
        }
    }

    #[test]
    fn lower_median_selects_negative_zero_independent_of_input_order() {
        for values in [vec![0.0_f64, -0.0], vec![-0.0_f64, 0.0]] {
            let stack = Array3::from_shape_vec((2, 1, 1), values).unwrap();
            let result = combine_axis0(&stack.view(), CombineKind::LMedian, None, 0);
            assert_eq!(result[[0, 0]].to_bits(), (-0.0_f64).to_bits());
        }
    }
}

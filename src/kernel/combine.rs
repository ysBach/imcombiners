//! Along-axis-0 combine kernels. Each output pixel (h, w) is computed from the
//! length-N per-output stack `arr[:, h, w]`. NaN values are treated as masked.
//!
//! Parallelism: rayon over the flattened (H * W) output index.

use ndarray::{Array1, Array2, Array3, ArrayView3};
use numpy::Element;
use rayon::prelude::*;

use super::utils::{minmax_1d_parallel_threshold, parallel_threshold, Float};

const MINMAX_1D_CHUNK: usize = 16_384;

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
fn nanmean_strided<T: Float>(data: &[T], pixel: usize, n: usize, hw: usize) -> T {
    let mut sums = [0.0_f64; 4];
    let mut counts = [0usize; 4];
    let mut idx = pixel;
    let mut k = 0usize;
    while k + 4 <= n {
        let x0 = data[idx];
        let x1 = data[idx + hw];
        let x2 = data[idx + 2 * hw];
        let x3 = data[idx + 3 * hw];
        if x0.is_finite() {
            sums[0] += x0.to_f64();
            counts[0] += 1;
        }
        if x1.is_finite() {
            sums[1] += x1.to_f64();
            counts[1] += 1;
        }
        if x2.is_finite() {
            sums[2] += x2.to_f64();
            counts[2] += 1;
        }
        if x3.is_finite() {
            sums[3] += x3.to_f64();
            counts[3] += 1;
        }
        idx += 4 * hw;
        k += 4;
    }
    while k < n {
        let x = data[idx];
        if x.is_finite() {
            sums[0] += x.to_f64();
            counts[0] += 1;
        }
        idx += hw;
        k += 1;
    }
    let s = sums.iter().sum::<f64>();
    let count = counts.iter().sum::<usize>();
    if count == 0 {
        T::nan()
    } else {
        T::from_f64(s / count as f64)
    }
}

#[inline]
fn nansum_strided<T: Float>(data: &[T], pixel: usize, n: usize, hw: usize) -> T {
    let mut sums = [0.0_f64; 4];
    let mut counts = [0usize; 4];
    let mut idx = pixel;
    let mut k = 0usize;
    while k + 4 <= n {
        let x0 = data[idx];
        let x1 = data[idx + hw];
        let x2 = data[idx + 2 * hw];
        let x3 = data[idx + 3 * hw];
        if x0.is_finite() {
            sums[0] += x0.to_f64();
            counts[0] += 1;
        }
        if x1.is_finite() {
            sums[1] += x1.to_f64();
            counts[1] += 1;
        }
        if x2.is_finite() {
            sums[2] += x2.to_f64();
            counts[2] += 1;
        }
        if x3.is_finite() {
            sums[3] += x3.to_f64();
            counts[3] += 1;
        }
        idx += 4 * hw;
        k += 4;
    }
    while k < n {
        let x = data[idx];
        if x.is_finite() {
            sums[0] += x.to_f64();
            counts[0] += 1;
        }
        idx += hw;
        k += 1;
    }
    let s = sums.iter().sum::<f64>();
    let count = counts.iter().sum::<usize>();
    if count == 0 {
        T::nan()
    } else {
        T::from_f64(s)
    }
}

#[inline]
fn nanmin_strided<T: Float>(data: &[T], pixel: usize, n: usize, hw: usize) -> T {
    let mut out = T::nan();
    let mut idx = pixel;
    for _ in 0..n {
        let x = data[idx];
        out = out.min_num(x);
        idx += hw;
    }
    out
}

#[inline]
fn nanmin_slice<T: Float>(values: &[T]) -> T {
    let mut outs = [T::nan(); 4];
    let mut chunks = values.chunks_exact(4);
    for chunk in &mut chunks {
        outs[0] = outs[0].min_num(chunk[0]);
        outs[1] = outs[1].min_num(chunk[1]);
        outs[2] = outs[2].min_num(chunk[2]);
        outs[3] = outs[3].min_num(chunk[3]);
    }
    let mut out = outs
        .into_iter()
        .reduce(|a, b| a.min_num(b))
        .expect("fixed accumulator length is non-empty");
    for &x in chunks.remainder() {
        out = out.min_num(x);
    }
    out
}

#[inline]
fn nanmax_strided<T: Float>(data: &[T], pixel: usize, n: usize, hw: usize) -> T {
    let mut out = T::nan();
    let mut idx = pixel;
    for _ in 0..n {
        let x = data[idx];
        out = out.max_num(x);
        idx += hw;
    }
    out
}

#[inline]
fn nanmax_slice<T: Float>(values: &[T]) -> T {
    let mut outs = [T::nan(); 4];
    let mut chunks = values.chunks_exact(4);
    for chunk in &mut chunks {
        outs[0] = outs[0].max_num(chunk[0]);
        outs[1] = outs[1].max_num(chunk[1]);
        outs[2] = outs[2].max_num(chunk[2]);
        outs[3] = outs[3].max_num(chunk[3]);
    }
    let mut out = outs
        .into_iter()
        .reduce(|a, b| a.max_num(b))
        .expect("fixed accumulator length is non-empty");
    for &x in chunks.remainder() {
        out = out.max_num(x);
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
    let mut idx = pixel;
    for &w in weights.iter().take(n) {
        let x = data[idx];
        if x.is_finite() {
            s += x.to_f64() * w;
            wsum += w;
        }
        idx += hw;
    }
    if wsum == 0.0 {
        T::nan()
    } else {
        T::from_f64(s / wsum)
    }
}

#[inline]
fn nanvariance_mean_strided<T: Float>(
    data: &[T],
    pixel: usize,
    n: usize,
    hw: usize,
    ddof: usize,
) -> (T, T) {
    let mut sums = [0.0_f64; 4];
    let mut sumsqs = [0.0_f64; 4];
    let mut counts = [0usize; 4];
    let mut idx = pixel;
    let mut k = 0usize;
    while k + 4 <= n {
        let x0 = data[idx];
        let x1 = data[idx + hw];
        let x2 = data[idx + 2 * hw];
        let x3 = data[idx + 3 * hw];
        if x0.is_finite() {
            let xf = x0.to_f64();
            sums[0] += xf;
            sumsqs[0] += xf * xf;
            counts[0] += 1;
        }
        if x1.is_finite() {
            let xf = x1.to_f64();
            sums[1] += xf;
            sumsqs[1] += xf * xf;
            counts[1] += 1;
        }
        if x2.is_finite() {
            let xf = x2.to_f64();
            sums[2] += xf;
            sumsqs[2] += xf * xf;
            counts[2] += 1;
        }
        if x3.is_finite() {
            let xf = x3.to_f64();
            sums[3] += xf;
            sumsqs[3] += xf * xf;
            counts[3] += 1;
        }
        idx += 4 * hw;
        k += 4;
    }
    while k < n {
        let x = data[idx];
        if x.is_finite() {
            let xf = x.to_f64();
            sums[0] += xf;
            sumsqs[0] += xf * xf;
            counts[0] += 1;
        }
        idx += hw;
        k += 1;
    }
    let sum = sums.iter().sum::<f64>();
    let sumsq = sumsqs.iter().sum::<f64>();
    let count = counts.iter().sum::<usize>();
    if count == 0 {
        return (T::nan(), T::nan());
    }
    let mean = sum / count as f64;
    if count <= ddof {
        return (T::nan(), T::from_f64(mean));
    }
    let numerator = (sumsq - sum * mean).max(0.0);
    (
        T::from_f64(numerator / (count - ddof) as f64),
        T::from_f64(mean),
    )
}

#[inline]
fn nanvariance_strided<T: Float>(data: &[T], pixel: usize, n: usize, hw: usize, ddof: usize) -> T {
    nanvariance_mean_strided(data, pixel, n, hw, ddof).0
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
fn cmp_float<T: Float>(a: &T, b: &T) -> std::cmp::Ordering {
    a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
}

#[inline]
fn nanmedian_strided<T: Float>(data: &[T], pixel: usize, n: usize, hw: usize, buf: &mut [T]) -> T {
    let count = compact_finite_strided(data, pixel, n, hw, buf);
    if count == 0 {
        return T::nan();
    }
    let mid = count / 2;
    let s = &mut buf[..count];
    if count % 2 == 1 {
        let (_, value, _) = s.select_nth_unstable_by(mid, cmp_float);
        *value
    } else {
        let (_, upper, _) = s.select_nth_unstable_by(mid, cmp_float);
        let upper = *upper;
        let lower = s[..mid]
            .iter()
            .copied()
            .max_by(cmp_float)
            .expect("even median lower partition is non-empty");
        T::from_f64((lower.to_f64() + upper.to_f64()) / 2.0)
    }
}

#[inline]
fn nanlmedian_strided<T: Float>(data: &[T], pixel: usize, n: usize, hw: usize, buf: &mut [T]) -> T {
    let count = compact_finite_strided(data, pixel, n, hw, buf);
    if count == 0 {
        return T::nan();
    }
    let idx = (count - 1) / 2;
    let (_, value, _) = buf[..count].select_nth_unstable_by(idx, cmp_float);
    *value
}

#[inline]
fn percentile_from_sorted<T: Float>(sorted: &[T], q: f64) -> T {
    if sorted.is_empty() {
        return T::nan();
    }
    let rank = (q / 100.0) * (sorted.len() - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    if lower == upper {
        return sorted[lower];
    }
    let fraction = rank - lower as f64;
    let lo = sorted[lower].to_f64();
    let hi = sorted[upper].to_f64();
    T::from_f64(lo + (hi - lo) * fraction)
}

#[derive(Clone, Copy)]
struct PercentileRank {
    lower: usize,
    upper: usize,
    fraction: f64,
}

#[inline]
fn percentile_rank(count: usize, q: f64) -> PercentileRank {
    let rank = (q / 100.0) * (count - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    PercentileRank {
        lower,
        upper,
        fraction: rank - lower as f64,
    }
}

#[inline]
fn unique_percentile_indices(ranks: &[PercentileRank]) -> Vec<usize> {
    let mut indices = Vec::<usize>::with_capacity(ranks.len() * 2);
    for rank in ranks {
        indices.push(rank.lower);
        indices.push(rank.upper);
    }
    indices.sort_unstable();
    indices.dedup();
    indices
}

#[inline]
fn selection_rank_budget(count: usize) -> usize {
    usize::BITS as usize - count.leading_zeros() as usize - 1
}

fn percentile_values<T: Float>(buf: &mut [T], qs: &[f64], out: &mut [T]) {
    debug_assert_eq!(qs.len(), out.len());
    if buf.is_empty() {
        out.fill(T::nan());
        return;
    }

    let ranks: Vec<PercentileRank> = qs.iter().map(|&q| percentile_rank(buf.len(), q)).collect();
    let needed_indices = unique_percentile_indices(&ranks);

    if needed_indices.len() <= selection_rank_budget(buf.len()) {
        let mut selected = Vec::<T>::with_capacity(needed_indices.len());
        let mut start = 0usize;
        for &idx in &needed_indices {
            let (_, value, _) = buf[start..].select_nth_unstable_by(idx - start, cmp_float);
            selected.push(*value);
            start = idx + 1;
        }
        for (out_px, rank) in out.iter_mut().zip(ranks.iter()) {
            let lo = selected[needed_indices.binary_search(&rank.lower).unwrap()].to_f64();
            let hi = selected[needed_indices.binary_search(&rank.upper).unwrap()].to_f64();
            *out_px = T::from_f64(lo + (hi - lo) * rank.fraction);
        }
    } else {
        buf.sort_unstable_by(cmp_float);
        for (out_px, &q) in out.iter_mut().zip(qs) {
            *out_px = percentile_from_sorted(buf, q);
        }
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

    match kind {
        CombineKind::Median | CombineKind::LMedian => {
            let compute_with_tmp = |tmp: &mut Vec<T>, (pixel, out_px): (usize, &mut T)| {
                tmp.resize(n, T::zero());
                *out_px = if matches!(kind, CombineKind::Median) {
                    nanmedian_strided(data, pixel, n, hw, tmp)
                } else {
                    nanlmedian_strided(data, pixel, n, hw, tmp)
                };
            };
            if hw >= parallel_threshold() {
                out_slice
                    .par_iter_mut()
                    .enumerate()
                    .map_init(|| vec![T::zero(); n], compute_with_tmp)
                    .count();
            } else {
                let mut tmp = vec![T::zero(); n];
                out_slice
                    .iter_mut()
                    .enumerate()
                    .for_each(|item| compute_with_tmp(&mut tmp, item));
            }
        }
        _ => {
            let compute_pixel = |pixel: usize, out_px: &mut T| {
                *out_px = match kind {
                    CombineKind::Mean => nanmean_strided(data, pixel, n, hw),
                    CombineKind::Sum => nansum_strided(data, pixel, n, hw),
                    CombineKind::Min => nanmin_strided(data, pixel, n, hw),
                    CombineKind::Max => nanmax_strided(data, pixel, n, hw),
                    CombineKind::Variance => nanvariance_strided(data, pixel, n, hw, ddof),
                    CombineKind::WeightedAverage => {
                        let w_ref = weights.expect("weights checked above");
                        weighted_average_strided(data, pixel, n, hw, w_ref)
                    }
                    CombineKind::Median | CombineKind::LMedian => unreachable!(),
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
        }
    }

    out
}

/// Return `(variance, mean)` along axis 0, sharing the variance accumulation pass.
pub fn variance_mean_axis0<T: Float>(arr: &ArrayView3<T>, ddof: usize) -> (Array2<T>, Array2<T>) {
    let (n, h, w) = (arr.shape()[0], arr.shape()[1], arr.shape()[2]);
    let hw = h * w;
    let mut var = Array2::<T>::from_elem((h, w), T::nan());
    let mut mean = Array2::<T>::from_elem((h, w), T::nan());
    let var_slice = var.as_slice_mut().expect("variance output is contiguous");
    let mean_slice = mean.as_slice_mut().expect("mean output is contiguous");
    let data = arr
        .as_slice_memory_order()
        .expect("variance_mean kernel requires contiguous C-order arrays");

    let compute_pixel = |pixel: usize, var_px: &mut T, mean_px: &mut T| {
        let (v, m) = nanvariance_mean_strided(data, pixel, n, hw, ddof);
        *var_px = v;
        *mean_px = m;
    };
    if hw >= parallel_threshold() {
        var_slice
            .par_iter_mut()
            .zip(mean_slice.par_iter_mut())
            .enumerate()
            .for_each(|(pixel, (var_px, mean_px))| compute_pixel(pixel, var_px, mean_px));
    } else {
        var_slice
            .iter_mut()
            .zip(mean_slice.iter_mut())
            .enumerate()
            .for_each(|(pixel, (var_px, mean_px))| compute_pixel(pixel, var_px, mean_px));
    }

    (var, mean)
}

/// Return NaN-aware percentiles along axis 0.
///
/// The result shape is `(H, W, Q)` so each output pixel owns one contiguous
/// percentile vector. Python moves the percentile axis to the front for the
/// public NumPy-compatible shape.
pub fn percentiles_axis0<T: Float>(arr: &ArrayView3<T>, qs: &[f64]) -> Array3<T> {
    let (n, h, w) = (arr.shape()[0], arr.shape()[1], arr.shape()[2]);
    let hw = h * w;
    let nq = qs.len();
    let mut out = Array3::<T>::from_elem((h, w, nq), T::nan());
    if nq == 0 {
        return out;
    }
    let out_slice = out.as_slice_mut().expect("percentile output is contiguous");
    let data = arr
        .as_slice_memory_order()
        .expect("percentile kernel requires contiguous C-order arrays");

    let compute_pixel = |buf: &mut Vec<T>, (pixel, out_q): (usize, &mut [T])| {
        buf.clear();
        let mut idx = pixel;
        for _ in 0..n {
            let x = data[idx];
            if x.is_finite() {
                buf.push(x);
            }
            idx += hw;
        }
        percentile_values(buf, qs, out_q);
    };

    if hw >= parallel_threshold() {
        out_slice
            .par_chunks_mut(nq)
            .enumerate()
            .map_init(|| Vec::<T>::with_capacity(n), compute_pixel)
            .count();
    } else {
        let mut buf = Vec::<T>::with_capacity(n);
        out_slice
            .chunks_mut(nq)
            .enumerate()
            .for_each(|item| compute_pixel(&mut buf, item));
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

    let compute_pixel = |col: &mut Vec<T>, (pixel, out_px): (usize, &mut T)| {
        col.clear();
        let mut idx = pixel;
        for _ in 0..n {
            col.push(data[idx]);
            idx += hw;
        }
        let (_, value, _) = col.select_nth_unstable(kth);
        *out_px = *value;
    };

    if hw >= parallel_threshold() {
        out_slice
            .par_iter_mut()
            .enumerate()
            .map_init(|| Vec::<T>::with_capacity(n), compute_pixel)
            .count();
    } else {
        let mut col = Vec::<T>::with_capacity(n);
        out_slice
            .iter_mut()
            .enumerate()
            .for_each(|item| compute_pixel(&mut col, item));
    }

    out
}

#[inline]
fn compact_finite_1d<T: Float>(values: &[T], buf: &mut Vec<T>) {
    buf.clear();
    for &x in values {
        if x.is_finite() {
            buf.push(x);
        }
    }
}

pub fn mean_1d<T: Float>(values: &[T]) -> T {
    let mut sums = [0.0_f64; 4];
    let mut counts = [0usize; 4];
    let mut chunks = values.chunks_exact(4);
    for chunk in &mut chunks {
        let x0 = chunk[0];
        let x1 = chunk[1];
        let x2 = chunk[2];
        let x3 = chunk[3];
        if x0.is_finite() {
            sums[0] += x0.to_f64();
            counts[0] += 1;
        }
        if x1.is_finite() {
            sums[1] += x1.to_f64();
            counts[1] += 1;
        }
        if x2.is_finite() {
            sums[2] += x2.to_f64();
            counts[2] += 1;
        }
        if x3.is_finite() {
            sums[3] += x3.to_f64();
            counts[3] += 1;
        }
    }
    for &x in chunks.remainder() {
        if x.is_finite() {
            sums[0] += x.to_f64();
            counts[0] += 1;
        }
    }
    let sum = sums.iter().sum::<f64>();
    let count = counts.iter().sum::<usize>();
    if count == 0 {
        T::nan()
    } else {
        T::from_f64(sum / count as f64)
    }
}

pub fn sum_1d<T: Float>(values: &[T]) -> T {
    let mut sums = [0.0_f64; 4];
    let mut counts = [0usize; 4];
    let mut chunks = values.chunks_exact(4);
    for chunk in &mut chunks {
        let x0 = chunk[0];
        let x1 = chunk[1];
        let x2 = chunk[2];
        let x3 = chunk[3];
        if x0.is_finite() {
            sums[0] += x0.to_f64();
            counts[0] += 1;
        }
        if x1.is_finite() {
            sums[1] += x1.to_f64();
            counts[1] += 1;
        }
        if x2.is_finite() {
            sums[2] += x2.to_f64();
            counts[2] += 1;
        }
        if x3.is_finite() {
            sums[3] += x3.to_f64();
            counts[3] += 1;
        }
    }
    for &x in chunks.remainder() {
        if x.is_finite() {
            sums[0] += x.to_f64();
            counts[0] += 1;
        }
    }
    let sum = sums.iter().sum::<f64>();
    let count = counts.iter().sum::<usize>();
    if count == 0 {
        T::nan()
    } else {
        T::from_f64(sum)
    }
}

pub fn min_1d<T: Float>(values: &[T]) -> T {
    if values.len() >= minmax_1d_parallel_threshold() {
        return values
            .par_chunks(MINMAX_1D_CHUNK)
            .map(nanmin_slice)
            .reduce(T::nan, |a, b| a.min_num(b));
    }
    nanmin_slice(values)
}

pub fn max_1d<T: Float>(values: &[T]) -> T {
    if values.len() >= minmax_1d_parallel_threshold() {
        return values
            .par_chunks(MINMAX_1D_CHUNK)
            .map(nanmax_slice)
            .reduce(T::nan, |a, b| a.max_num(b));
    }
    nanmax_slice(values)
}

pub fn variance_mean_1d<T: Float>(values: &[T], ddof: usize) -> (T, T) {
    let mut sums = [0.0_f64; 4];
    let mut sumsqs = [0.0_f64; 4];
    let mut counts = [0usize; 4];
    let mut chunks = values.chunks_exact(4);
    for chunk in &mut chunks {
        let x0 = chunk[0];
        let x1 = chunk[1];
        let x2 = chunk[2];
        let x3 = chunk[3];
        if x0.is_finite() {
            let xf = x0.to_f64();
            sums[0] += xf;
            sumsqs[0] += xf * xf;
            counts[0] += 1;
        }
        if x1.is_finite() {
            let xf = x1.to_f64();
            sums[1] += xf;
            sumsqs[1] += xf * xf;
            counts[1] += 1;
        }
        if x2.is_finite() {
            let xf = x2.to_f64();
            sums[2] += xf;
            sumsqs[2] += xf * xf;
            counts[2] += 1;
        }
        if x3.is_finite() {
            let xf = x3.to_f64();
            sums[3] += xf;
            sumsqs[3] += xf * xf;
            counts[3] += 1;
        }
    }
    for &x in chunks.remainder() {
        if x.is_finite() {
            let xf = x.to_f64();
            sums[0] += xf;
            sumsqs[0] += xf * xf;
            counts[0] += 1;
        }
    }
    let sum = sums.iter().sum::<f64>();
    let sumsq = sumsqs.iter().sum::<f64>();
    let count = counts.iter().sum::<usize>();
    if count == 0 {
        return (T::nan(), T::nan());
    }
    let mean = sum / count as f64;
    if count <= ddof {
        return (T::nan(), T::from_f64(mean));
    }
    let numerator = (sumsq - sum * mean).max(0.0);
    (
        T::from_f64(numerator / (count - ddof) as f64),
        T::from_f64(mean),
    )
}

pub fn variance_1d<T: Float>(values: &[T], ddof: usize) -> T {
    variance_mean_1d(values, ddof).0
}

pub fn median_1d<T: Float>(values: &[T]) -> T {
    let mut buf = Vec::<T>::with_capacity(values.len());
    compact_finite_1d(values, &mut buf);
    if buf.is_empty() {
        return T::nan();
    }
    let mid = buf.len() / 2;
    if buf.len() % 2 == 1 {
        let (_, value, _) = buf.select_nth_unstable_by(mid, cmp_float);
        *value
    } else {
        let (_, upper, _) = buf.select_nth_unstable_by(mid, cmp_float);
        let upper = *upper;
        let lower = buf[..mid]
            .iter()
            .copied()
            .max_by(cmp_float)
            .expect("even median lower partition is non-empty");
        T::from_f64((lower.to_f64() + upper.to_f64()) / 2.0)
    }
}

pub fn lmedian_1d<T: Float>(values: &[T]) -> T {
    let mut buf = Vec::<T>::with_capacity(values.len());
    compact_finite_1d(values, &mut buf);
    if buf.is_empty() {
        return T::nan();
    }
    let idx = (buf.len() - 1) / 2;
    let (_, value, _) = buf.select_nth_unstable_by(idx, cmp_float);
    *value
}

pub fn percentiles_1d<T: Float>(values: &[T], qs: &[f64]) -> Array1<f64> {
    let mut buf = Vec::<T>::with_capacity(values.len());
    compact_finite_1d(values, &mut buf);
    let mut out = vec![f64::NAN; qs.len()];
    if !buf.is_empty() {
        let mut out_t = vec![T::nan(); qs.len()];
        percentile_values(&mut buf, qs, &mut out_t);
        for (dst, src) in out.iter_mut().zip(out_t) {
            *dst = src.to_f64();
        }
    }
    Array1::from_vec(out)
}

pub fn weighted_average_1d<T: Float>(values: &[T], weights: &[f64]) -> T {
    debug_assert_eq!(values.len(), weights.len());
    let mut sum = 0.0_f64;
    let mut wsum = 0.0_f64;
    for (&x, &w) in values.iter().zip(weights) {
        if x.is_finite() {
            sum += x.to_f64() * w;
            wsum += w;
        }
    }
    if wsum == 0.0 {
        T::nan()
    } else {
        T::from_f64(sum / wsum)
    }
}

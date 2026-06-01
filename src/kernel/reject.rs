//! Per-pixel rejection algorithms: sigclip, ccdclip, linearclip, minmax, pclip.
//!
//! Each operates on a (N, H, W) stack and produces a boolean mask of the same
//! shape (`true` = rejected). The IRAF-style mode uses threshold restoration;
//! plain sigma clipping uses cumulative iterative rejection.

use ndarray::{Array2, Array3, ArrayView3};
use rayon::prelude::*;
use reducers::reducers_1d::{lmedian_valid_in_place, median_valid_in_place};
use std::any::TypeId;
use std::cmp::Ordering;

use super::utils::Float;

const SAMPLE_RESTORED_NKEEP: u8 = 64;
const SAMPLE_RESTORED_MAXREJ: u8 = 128;
const DEFAULT_REJECT_PARALLEL_HW_THRESHOLD: usize = 10_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RejectKind {
    LinearClip,
    SigClip,
    CcdClip,
    MinMax,
    PClip,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RejectDType {
    F32,
    F64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RejectOutputMode {
    Diagnostics,
    MaskOnly,
    RestoredFlags,
    FinalMean,
    FinalMedian,
}

#[inline]
fn reject_dtype<T: Float>() -> RejectDType {
    if TypeId::of::<T>() == TypeId::of::<f32>() {
        RejectDType::F32
    } else {
        RejectDType::F64
    }
}

#[inline]
fn final_output_mode(kind: FinalCombineKind) -> RejectOutputMode {
    match kind {
        FinalCombineKind::Mean => RejectOutputMode::FinalMean,
        FinalCombineKind::Median => RejectOutputMode::FinalMedian,
    }
}

#[inline]
fn should_parallel_reject(
    kind: RejectKind,
    hw: usize,
    n: usize,
    dtype: RejectDType,
    output_mode: RejectOutputMode,
) -> bool {
    // Rejection kernels own their Rayon crossover policy because iterative
    // clipping, rank-pair masks, diagnostics, and fused final combines do not
    // have the same cost model as reducers' generic order-statistic kernels.
    // This is the old imc conservative baseline; benchmark data should tune the
    // match arms below by kind/dtype/output mode without making reducers carry
    // imc-specific knobs.
    let threshold = match (kind, dtype, output_mode) {
        (RejectKind::LinearClip, _, _) => DEFAULT_REJECT_PARALLEL_HW_THRESHOLD,
        (RejectKind::SigClip, _, _) => DEFAULT_REJECT_PARALLEL_HW_THRESHOLD,
        (RejectKind::CcdClip, _, _) => DEFAULT_REJECT_PARALLEL_HW_THRESHOLD,
        (RejectKind::MinMax, _, _) => DEFAULT_REJECT_PARALLEL_HW_THRESHOLD,
        (RejectKind::PClip, _, _) => DEFAULT_REJECT_PARALLEL_HW_THRESHOLD,
    };
    n > 0 && hw >= threshold
}

#[inline]
fn should_parallel_reject_for<T: Float>(
    kind: RejectKind,
    hw: usize,
    n: usize,
    output_mode: RejectOutputMode,
) -> bool {
    should_parallel_reject(kind, hw, n, reject_dtype::<T>(), output_mode)
}

#[inline]
fn cmp_pair_t<T: Float>(a: &(T, usize), b: &(T, usize)) -> Ordering {
    a.0.to_f64().total_cmp(&b.0.to_f64())
}

#[inline]
fn debug_assert_mask_covers_nonfinite<T: Float>(vals: &[T], mask: &[bool]) {
    debug_assert_eq!(vals.len(), mask.len());
    debug_assert!(
        vals.iter().zip(mask).all(|(&x, &m)| m || x.is_finite()),
        "mask must mark every non-finite value"
    );
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CenFunc {
    Mean,
    Median,
    LowerMedian,
}

impl CenFunc {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "mean" | "average" | "avg" => Some(Self::Mean),
            "median" | "med" => Some(Self::Median),
            "lmedian" | "lmed" | "lower_median" | "lower-median" | "lower median" => {
                Some(Self::LowerMedian)
            }
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClipCenter {
    Mean,
    Median,
    LowerMedian,
    ClippingCenter,
}

impl ClipCenter {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "mean" | "average" | "avg" => Some(Self::Mean),
            "median" | "med" => Some(Self::Median),
            "lmedian" | "lmed" | "lower_median" | "lower-median" | "lower median" => {
                Some(Self::LowerMedian)
            }
            "center" | "cenfunc" | "clipping_center" | "clipcenter" => Some(Self::ClippingCenter),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StdFunc {
    Std,
    Mad,
}

impl StdFunc {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "std" | "standard_deviation" | "standard-deviation" => Some(Self::Std),
            "mad" | "median_absolute_deviation" | "median-absolute-deviation" => Some(Self::Mad),
            _ => None,
        }
    }
}

pub struct SigClipParams {
    pub sigma_lower: f64,
    pub sigma_upper: f64,
    pub maxiters: usize,
    pub ddof: usize,
    pub nkeep: usize,
    pub maxrej: usize,
    pub cenfunc: CenFunc,
    pub clip_cen: ClipCenter,
    pub stdfunc: StdFunc,
    pub revert_on_nkeep: bool,
    // ccdclip fields (used only when ccdclip == true)
    pub ccdclip: bool,
    pub rdnoise_ref: f64,
    pub snoise_ref: f64,
    pub scale_ref: f64,
    pub zero_ref: f64,
    pub rejection_gain: f64,
}

pub struct LinearClipParams {
    pub low_scale: f64,
    pub low: f64,
    pub upp_scale: f64,
    pub upp: f64,
    pub maxiters: usize,
    pub nkeep: usize,
    pub maxrej: usize,
    pub cenfunc: CenFunc,
    pub revert_on_nkeep: bool,
}

pub struct RejectOutput<T: Float> {
    pub mask: Array3<bool>, // (N, H, W); true = rejected (incl. input non-finite)
    pub low: Array2<T>,     // (H, W)
    pub upp: Array2<T>,     // (H, W)
    pub nit: Array2<u8>,    // (H, W)
    pub output_flags: Array2<u8>, // (H, W)
    pub std: Array2<T>,     // (H, W)
}

pub struct RejectOutput1d<T: Float> {
    pub mask: Vec<bool>,
    pub low: T,
    pub upp: T,
    pub nit: u8,
    pub output_flags: u8,
    pub std: T,
}

struct RejectCompute<T: Float> {
    mask: Array3<bool>,
    low: Option<Array2<T>>,
    upp: Option<Array2<T>>,
    nit: Option<Array2<u8>>,
    output_flags: Option<Array2<u8>>,
    std: Option<Array2<T>>,
}

impl<T: Float> RejectCompute<T> {
    fn into_output(self) -> RejectOutput<T> {
        RejectOutput {
            mask: self.mask,
            low: self.low.expect("diagnostic low array missing"),
            upp: self.upp.expect("diagnostic upp array missing"),
            nit: self.nit.expect("diagnostic nit array missing"),
            output_flags: self
                .output_flags
                .expect("diagnostic output_flags array missing"),
            std: self.std.expect("diagnostic std array missing"),
        }
    }
}

// ---------- linearclip ----------

struct LinearClipPixelScratch {
    mask: Vec<bool>,
    newly_masked: Vec<usize>,
}

impl LinearClipPixelScratch {
    fn new() -> Self {
        Self {
            mask: Vec::new(),
            newly_masked: Vec::new(),
        }
    }
}

struct LinearClipPixelResult<T: Float> {
    low: T,
    upp: T,
    nit: u8,
    output_flags: u8,
}

fn linear_center<T: Float>(vals: &[T], mask: &[bool], cenfunc: CenFunc, buf: &mut Vec<f64>) -> T {
    match cenfunc {
        CenFunc::Mean => col_nanmean(vals, mask),
        CenFunc::Median => col_nanmedian(vals, mask, buf),
        CenFunc::LowerMedian => col_nanlmedian(vals, mask, buf),
    }
}

fn linearclip_pixel<T: Float>(
    col: &[T],
    mask_in: &[bool],
    p: &LinearClipParams,
    buf: &mut Vec<f64>,
    out_mask: &mut [bool],
    restored_flags: Option<&mut [u8]>,
    scratch: &mut LinearClipPixelScratch,
) -> LinearClipPixelResult<T> {
    let n = col.len();
    let pre_masked = mask_in.iter().any(|&m| m);
    let mut restored_flags = restored_flags;
    if let Some(flags) = restored_flags.as_deref_mut() {
        flags.fill(0);
    }

    scratch.mask.resize(n, false);
    let mask = &mut scratch.mask[..n];
    let mut n_finite_initial = 0usize;
    for i in 0..n {
        if mask_in[i] || !col[i].is_finite() {
            mask[i] = true;
        } else {
            mask[i] = false;
            n_finite_initial += 1;
        }
    }

    let mut low_out = T::nan();
    let mut upp_out = T::nan();
    let mut nit = 0u8;
    let mut stopped_nkeep = false;
    let mut stopped_maxrej = false;
    let mut stopped_maxiters = false;

    for iter in 0..p.maxiters {
        let center = linear_center(col, mask, p.cenfunc, buf);
        if center.is_nan() {
            break;
        }
        let center = center.to_f64();
        let lo_try = p.low_scale * center - p.low;
        let up_try = p.upp_scale * center + p.upp;

        scratch.newly_masked.clear();
        let mut n_finite_new = 0usize;
        for i in 0..n {
            if !mask[i] {
                let x = col[i].to_f64();
                if x < lo_try || x > up_try {
                    mask[i] = true;
                    scratch.newly_masked.push(i);
                } else {
                    n_finite_new += 1;
                }
            }
        }

        nit = nit.saturating_add(1);
        if scratch.newly_masked.is_empty() {
            low_out = T::from_f64(lo_try);
            upp_out = T::from_f64(up_try);
            break;
        }

        let nrej = n_finite_initial.saturating_sub(n_finite_new);
        let mask_nkeep = p.revert_on_nkeep && n_finite_new < p.nkeep;
        let mask_maxrej = nrej > p.maxrej;
        if mask_nkeep || mask_maxrej {
            stopped_nkeep = mask_nkeep;
            stopped_maxrej = mask_maxrej;
            if let Some(flags) = restored_flags.as_deref_mut() {
                let restored_flag = ((mask_nkeep as u8) * SAMPLE_RESTORED_NKEEP)
                    | ((mask_maxrej as u8) * SAMPLE_RESTORED_MAXREJ);
                for &i in &scratch.newly_masked {
                    flags[i] |= restored_flag;
                }
            }
            for &i in &scratch.newly_masked {
                mask[i] = false;
            }
            break;
        }

        low_out = T::from_f64(lo_try);
        upp_out = T::from_f64(up_try);
        if iter + 1 == p.maxiters {
            stopped_maxiters = p.maxiters > 1;
        }
    }

    for i in 0..n {
        out_mask[i] = mask[i] && !mask_in[i];
    }
    let output_flags: u8 = (pre_masked as u8)
        | ((stopped_maxiters as u8) << 1)
        | ((stopped_nkeep as u8) << 2)
        | ((stopped_maxrej as u8) << 3);
    LinearClipPixelResult {
        low: low_out,
        upp: upp_out,
        nit,
        output_flags,
    }
}

pub fn linearclip<T: Float>(
    arr: &ArrayView3<T>,
    mask_in: Option<&ArrayView3<bool>>,
    p: &LinearClipParams,
) -> RejectOutput<T> {
    let (n, h, w) = (arr.shape()[0], arr.shape()[1], arr.shape()[2]);
    let hw = h * w;
    let mut mask_out = Array3::<bool>::from_elem((n, h, w), false);
    let mut low_out = Array2::<T>::from_elem((h, w), T::nan());
    let mut upp_out = Array2::<T>::from_elem((h, w), T::nan());
    let mut nit = Array2::<u8>::zeros((h, w));
    let mut output_flags = Array2::<u8>::zeros((h, w));
    let std = Array2::<T>::from_elem((h, w), T::nan());

    let data = arr
        .as_slice_memory_order()
        .expect("linearclip kernel requires contiguous C-order arrays");
    let mask_data = mask_in.and_then(|m| m.as_slice_memory_order());

    struct Px<T> {
        mask_col: Vec<bool>,
        low: T,
        upp: T,
        nit: u8,
        output_flags: u8,
    }

    let compute_pixel = |idx: usize| {
        let mut source_col = vec![T::zero(); n];
        let mut mask_in_col = vec![false; n];
        let mut out_col = vec![false; n];
        let mut raw_idx = idx;
        for k in 0..n {
            let x = data[raw_idx];
            source_col[k] = x;
            mask_in_col[k] = mask_data.is_some_and(|m| m[raw_idx]);
            raw_idx += hw;
        }
        let mut buf = Vec::<f64>::with_capacity(n);
        let mut scratch = LinearClipPixelScratch::new();
        let px = linearclip_pixel(
            &source_col,
            &mask_in_col,
            p,
            &mut buf,
            &mut out_col,
            None,
            &mut scratch,
        );
        Px {
            mask_col: out_col,
            low: px.low,
            upp: px.upp,
            nit: px.nit,
            output_flags: px.output_flags,
        }
    };

    let results: Vec<Px<T>> = if should_parallel_reject_for::<T>(
        RejectKind::LinearClip,
        hw,
        n,
        RejectOutputMode::Diagnostics,
    ) {
        (0..hw).into_par_iter().map(compute_pixel).collect()
    } else {
        (0..hw).map(compute_pixel).collect()
    };

    let mask_slice = mask_out
        .as_slice_memory_order_mut()
        .expect("linearclip output mask must be contiguous");
    for k in 0..n {
        let plane_offset = k * hw;
        for (idx, r) in results.iter().enumerate() {
            mask_slice[plane_offset + idx] = r.mask_col[k];
        }
    }
    let low_slice = low_out
        .as_slice_memory_order_mut()
        .expect("linearclip low output must be contiguous");
    let upp_slice = upp_out
        .as_slice_memory_order_mut()
        .expect("linearclip upp output must be contiguous");
    let nit_slice = nit
        .as_slice_memory_order_mut()
        .expect("linearclip nit output must be contiguous");
    let output_flags_slice = output_flags
        .as_slice_memory_order_mut()
        .expect("linearclip output_flags output must be contiguous");
    for (idx, r) in results.iter().enumerate() {
        low_slice[idx] = r.low;
        upp_slice[idx] = r.upp;
        nit_slice[idx] = r.nit;
        output_flags_slice[idx] = r.output_flags;
    }

    RejectOutput {
        mask: mask_out,
        low: low_out,
        upp: upp_out,
        nit,
        output_flags,
        std,
    }
}

pub fn linearclip_1d<T: Float>(
    values: &[T],
    mask_in: Option<&[bool]>,
    p: &LinearClipParams,
) -> RejectOutput1d<T> {
    let n = values.len();
    let no_mask;
    let mask = if let Some(mask) = mask_in {
        mask
    } else {
        no_mask = vec![false; n];
        &no_mask
    };
    let mut out_mask = vec![false; n];
    let mut buf = Vec::<f64>::with_capacity(n);
    let mut scratch = LinearClipPixelScratch::new();
    let px = linearclip_pixel(values, mask, p, &mut buf, &mut out_mask, None, &mut scratch);
    RejectOutput1d {
        mask: out_mask,
        low: px.low,
        upp: px.upp,
        nit: px.nit,
        output_flags: px.output_flags,
        std: T::nan(),
    }
}

pub fn linearclip_restored_flags<T: Float>(
    arr: &ArrayView3<T>,
    mask_in: Option<&ArrayView3<bool>>,
    p: &LinearClipParams,
) -> Array3<u8> {
    let (n, h, w) = (arr.shape()[0], arr.shape()[1], arr.shape()[2]);
    let hw = h * w;
    let mut flags_out = Array3::<u8>::zeros((n, h, w));
    let data = arr
        .as_slice_memory_order()
        .expect("linearclip restored-flags kernel requires contiguous C-order arrays");
    let mask_data = mask_in.and_then(|m| m.as_slice_memory_order());

    struct Px {
        flags_col: Vec<u8>,
    }

    let compute_pixel = |idx: usize| {
        let mut source_col = vec![T::zero(); n];
        let mut mask_in_col = vec![false; n];
        let mut out_col = vec![false; n];
        let mut restored_col = vec![0_u8; n];
        let mut raw_idx = idx;
        for k in 0..n {
            source_col[k] = data[raw_idx];
            mask_in_col[k] = mask_data.is_some_and(|m| m[raw_idx]);
            raw_idx += hw;
        }
        let mut buf = Vec::<f64>::with_capacity(n);
        let mut scratch = LinearClipPixelScratch::new();
        linearclip_pixel(
            &source_col,
            &mask_in_col,
            p,
            &mut buf,
            &mut out_col,
            Some(&mut restored_col),
            &mut scratch,
        );
        Px {
            flags_col: restored_col,
        }
    };

    let results: Vec<Px> = if should_parallel_reject_for::<T>(
        RejectKind::LinearClip,
        hw,
        n,
        RejectOutputMode::RestoredFlags,
    ) {
        (0..hw).into_par_iter().map(compute_pixel).collect()
    } else {
        (0..hw).map(compute_pixel).collect()
    };

    let flags_slice = flags_out
        .as_slice_memory_order_mut()
        .expect("linearclip restored-flags output must be contiguous");
    for k in 0..n {
        let plane_offset = k * hw;
        for (idx, r) in results.iter().enumerate() {
            flags_slice[plane_offset + idx] = r.flags_col[k];
        }
    }

    flags_out
}

struct SigClipPixelResult<T: Float> {
    low: T,
    upp: T,
    std: T,
    nit: u8,
    output_flags: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FinalCombineKind {
    Mean,
    Median,
}

struct SigClipPixelScratch {
    mask: Vec<bool>,
    newly_masked: Vec<usize>,
}

impl SigClipPixelScratch {
    fn new() -> Self {
        Self {
            mask: Vec::new(),
            newly_masked: Vec::new(),
        }
    }
}

// ---------- per-output-stack reductions reused here ----------
// All helpers in this section assume the caller has already folded input
// masks and non-finite values into `mask`.

fn col_nanmean_var<T: Float>(vals: &[T], mask: &[bool], ddof: usize) -> (T, f64) {
    debug_assert_mask_covers_nonfinite(vals, mask);
    let mut s = 0.0_f64;
    let mut ss = 0.0_f64;
    let mut n = 0usize;
    for (i, &x) in vals.iter().enumerate() {
        if !mask[i] {
            let xf = x.to_f64();
            s += xf;
            ss += xf * xf;
            n += 1;
        }
    }
    if n == 0 {
        return (T::nan(), f64::NAN);
    }
    let mean = s / n as f64;
    let var = if n <= ddof {
        f64::NAN
    } else {
        let var_num = (ss - s * mean).max(0.0);
        var_num / (n - ddof) as f64
    };
    (T::from_f64(mean), var)
}

#[inline]
fn col_nanmean<T: Float>(vals: &[T], mask: &[bool]) -> T {
    debug_assert_mask_covers_nonfinite(vals, mask);
    let mut s = 0.0_f64;
    let mut n = 0usize;
    for (i, &x) in vals.iter().enumerate() {
        if !mask[i] {
            s += x.to_f64();
            n += 1;
        }
    }
    if n == 0 {
        T::nan()
    } else {
        T::from_f64(s / n as f64)
    }
}

#[inline]
fn col_nanmedian<T: Float>(vals: &[T], mask: &[bool], buf: &mut Vec<f64>) -> T {
    debug_assert_mask_covers_nonfinite(vals, mask);
    buf.clear();
    for (i, &x) in vals.iter().enumerate() {
        if !mask[i] {
            buf.push(x.to_f64());
        }
    }
    let n = buf.len();
    if n == 0 {
        return T::nan();
    }
    T::from_f64(median_valid_in_place(buf))
}

#[inline]
fn col_nanlmedian<T: Float>(vals: &[T], mask: &[bool], buf: &mut Vec<f64>) -> T {
    debug_assert_mask_covers_nonfinite(vals, mask);
    buf.clear();
    for (i, &x) in vals.iter().enumerate() {
        if !mask[i] {
            buf.push(x.to_f64());
        }
    }
    let n = buf.len();
    if n == 0 {
        return T::nan();
    }
    T::from_f64(lmedian_valid_in_place(buf))
}

#[inline]
fn col_nanvariance_about<T: Float>(vals: &[T], mask: &[bool], center: T, ddof: usize) -> f64 {
    debug_assert_mask_covers_nonfinite(vals, mask);
    if center.is_nan() {
        return f64::NAN;
    }
    let center_f = center.to_f64();
    let mut s = 0.0_f64;
    let mut n = 0usize;
    for (i, &x) in vals.iter().enumerate() {
        if !mask[i] {
            let d = x.to_f64() - center_f;
            s += d * d;
            n += 1;
        }
    }
    if n <= ddof {
        return f64::NAN;
    }
    s / (n - ddof) as f64
}

#[inline]
fn col_nanmad_sigma_about<T: Float>(
    vals: &[T],
    mask: &[bool],
    center: T,
    ddof: usize,
    buf: &mut Vec<f64>,
) -> T {
    debug_assert_mask_covers_nonfinite(vals, mask);
    if center.is_nan() {
        return T::nan();
    }
    let center_f = center.to_f64();
    buf.clear();
    for (i, &x) in vals.iter().enumerate() {
        if !mask[i] {
            buf.push((x.to_f64() - center_f).abs());
        }
    }
    let n = buf.len();
    if n <= ddof {
        return T::nan();
    }
    let mut sigma = 1.4826 * median_valid_in_place(buf);
    if ddof > 0 {
        sigma *= (n as f64 / (n - ddof) as f64).sqrt();
    }
    T::from_f64(sigma)
}

#[inline]
fn col_minmax<T: Float>(vals: &[T], mask: &[bool]) -> (T, T) {
    debug_assert_mask_covers_nonfinite(vals, mask);
    let mut lo = T::nan();
    let mut up = T::nan();
    for (i, &x) in vals.iter().enumerate() {
        if !mask[i] {
            if lo.is_nan() || x < lo {
                lo = x;
            }
            if up.is_nan() || x > up {
                up = x;
            }
        }
    }
    (lo, up)
}

#[inline]
fn final_mean<T: Float>(vals: &[T], mask: &[bool]) -> T {
    debug_assert_mask_covers_nonfinite(vals, mask);
    let mut sum = 0.0_f64;
    let mut count = 0usize;
    for (i, &x) in vals.iter().enumerate() {
        if !mask[i] {
            sum += x.to_f64();
            count += 1;
        }
    }
    if count == 0 {
        T::nan()
    } else {
        T::from_f64(sum / count as f64)
    }
}

#[inline]
fn final_median<T: Float>(vals: &[T], mask: &[bool], buf: &mut Vec<f64>) -> T {
    debug_assert_mask_covers_nonfinite(vals, mask);
    buf.clear();
    for (i, &x) in vals.iter().enumerate() {
        if !mask[i] {
            buf.push(x.to_f64());
        }
    }
    let n = buf.len();
    if n == 0 {
        return T::nan();
    }
    T::from_f64(median_valid_in_place(buf))
}

#[inline]
fn final_combine<T: Float>(
    vals: &[T],
    mask: &[bool],
    kind: FinalCombineKind,
    buf: &mut Vec<f64>,
) -> T {
    match kind {
        FinalCombineKind::Mean => final_mean(vals, mask),
        FinalCombineKind::Median => final_median(vals, mask, buf),
    }
}

#[inline]
fn final_combine_from_sorted_pairs_t<T: Float>(
    pairs: &[(T, usize)],
    mask: &[bool],
    kind: FinalCombineKind,
    buf: &mut Vec<f64>,
) -> T {
    match kind {
        FinalCombineKind::Mean => {
            let mut sum = 0.0_f64;
            let mut count = 0usize;
            for &(value, k) in pairs {
                if !mask[k] {
                    sum += value.to_f64();
                    count += 1;
                }
            }
            if count == 0 {
                T::nan()
            } else {
                T::from_f64(sum / count as f64)
            }
        }
        FinalCombineKind::Median => {
            buf.clear();
            for &(value, k) in pairs {
                if !mask[k] {
                    buf.push(value.to_f64());
                }
            }
            if buf.is_empty() {
                T::nan()
            } else {
                T::from_f64(median_valid_in_place(buf))
            }
        }
    }
}

#[inline]
fn final_combine_from_sorted_pairs_f64<T: Float>(
    pairs: &[(f64, usize)],
    mask: &[bool],
    kind: FinalCombineKind,
    buf: &mut Vec<f64>,
) -> T {
    match kind {
        FinalCombineKind::Mean => {
            let mut sum = 0.0_f64;
            let mut count = 0usize;
            for &(value, k) in pairs {
                if !mask[k] {
                    sum += value;
                    count += 1;
                }
            }
            if count == 0 {
                T::nan()
            } else {
                T::from_f64(sum / count as f64)
            }
        }
        FinalCombineKind::Median => {
            buf.clear();
            for &(value, k) in pairs {
                if !mask[k] {
                    buf.push(value);
                }
            }
            if buf.is_empty() {
                T::nan()
            } else {
                T::from_f64(median_valid_in_place(buf))
            }
        }
    }
}

// ---------- per-pixel sigclip / ccdclip ----------

/// Returns (mask_out, low, upp, nit, output_flags).
///
/// Behaviour is selected by `SigClipParams`: `nkeep` may either revert a full
/// violating iteration or allow strict cumulative sigma clipping.
#[allow(clippy::too_many_arguments)]
fn sigclip_pixel<T: Float>(
    col: &[T],        // length N
    mask_in: &[bool], // length N
    p: &SigClipParams,
    diagnostics: bool,
    buf: &mut Vec<f64>,    // scratch for median
    out_mask: &mut [bool], // length N
    restored_flags: Option<&mut [u8]>,
    scratch: &mut SigClipPixelScratch,
) -> SigClipPixelResult<T> {
    let n = col.len();
    let pre_masked = mask_in.iter().any(|&m| m);
    let mut restored_flags = restored_flags;
    if let Some(flags) = restored_flags.as_deref_mut() {
        flags.fill(0);
    }
    // Working mask: true = currently rejected / non-finite.
    scratch.mask.resize(n, false);
    let mask = &mut scratch.mask[..n];
    let mut n_finite: usize = 0;
    for i in 0..n {
        if mask_in[i] || !col[i].is_finite() {
            mask[i] = true;
        } else {
            mask[i] = false;
            n_finite += 1;
        }
    }

    let nkeep = p.nkeep;
    let maxrej = p.maxrej;

    let mut nit: u8 = 1;
    let mut std_out = T::nan();
    let mut n_finite_old = n_finite;
    let mut stopped_nkeep = false;
    let mut stopped_maxrej = false;
    let mut stopped_maxiters = p.maxiters == 0;
    let unbounded = (!p.revert_on_nkeep || nkeep == 0) && maxrej == n;

    for k in 0..p.maxiters {
        stopped_maxiters = k + 1 == p.maxiters;
        let (cen, _std_center, std, spread_sq) =
            clipping_center_and_spread(col, mask, p, diagnostics, buf);

        if cen.is_nan() || !spread_sq.is_finite() {
            break;
        }
        std_out = std;

        let center = cen.to_f64();
        let lower_limit_sq = p.sigma_lower * p.sigma_lower * spread_sq;
        let upper_limit_sq = p.sigma_upper * p.sigma_upper * spread_sq;
        // Tentatively reject out-of-bounds; track new finite count.
        let mut n_finite_new = 0usize;
        // We'll need to roll back if nkeep/maxrej limits trip; so save current mask + bounds.
        scratch.newly_masked.clear();
        for i in 0..n {
            if !mask[i] {
                let delta = col[i].to_f64() - center;
                let delta_sq = delta * delta;
                if (delta < 0.0 && delta_sq > lower_limit_sq)
                    || (delta > 0.0 && delta_sq > upper_limit_sq)
                {
                    mask[i] = true;
                    scratch.newly_masked.push(i);
                } else {
                    n_finite_new += 1;
                }
            }
        }

        if unbounded {
            let n_change = n_finite_old - n_finite_new;
            if n_change == 0 {
                stopped_maxiters = false;
                break;
            }
            n_finite_old = n_finite_new;
        } else {
            let nrej = n - n_finite_new;
            let mask_nkeep = p.revert_on_nkeep && n_finite_new < nkeep;
            let mask_maxrej = nrej > maxrej;
            let mask_pix = mask_nkeep || mask_maxrej;

            if mask_pix {
                stopped_nkeep = mask_nkeep;
                stopped_maxrej = mask_maxrej;
                stopped_maxiters = false;
                if let Some(flags) = restored_flags.as_deref_mut() {
                    let restored_flag = ((mask_nkeep as u8) * SAMPLE_RESTORED_NKEEP)
                        | ((mask_maxrej as u8) * SAMPLE_RESTORED_MAXREJ);
                    for &i in &scratch.newly_masked {
                        flags[i] |= restored_flag;
                    }
                }
                // Revert this iteration's rejections; keep previous bounds.
                for &i in &scratch.newly_masked {
                    mask[i] = false;
                }
                // n_finite unchanged
                break;
            } else {
                let n_change = n_finite_old - n_finite_new;
                if n_change == 0 {
                    stopped_maxiters = false;
                    break;
                }
                n_finite_old = n_finite_new;
            }
        }

        nit = nit.saturating_add(1);
        let _ = k;
    }

    // Iterative sigma clipping is cumulative: a value rejected in an earlier
    // iteration remains rejected even if later thresholds move. If
    // `revert_on_nkeep` tripped above, only that final violating iteration has
    // already been undone.
    out_mask.copy_from_slice(mask);

    let (low_out, upp_out) = if diagnostics {
        col_minmax(col, out_mask)
    } else {
        (T::nan(), T::nan())
    };

    // Flag bits: 0 = normal completion/convergence. Bit 0 marks pre-masked
    // input values; higher bits mark stop/constraint conditions.
    let output_flags: u8 = (pre_masked as u8)
        | ((stopped_maxiters as u8) << 1)
        | ((stopped_nkeep as u8) << 2)
        | ((stopped_maxrej as u8) << 3);

    SigClipPixelResult {
        low: low_out,
        upp: upp_out,
        std: std_out,
        nit,
        output_flags,
    }
}

fn clipping_center_and_spread<T: Float>(
    col: &[T],
    mask: &[bool],
    p: &SigClipParams,
    diagnostics: bool,
    buf: &mut Vec<f64>,
) -> (T, T, T, f64) {
    let mut mean_std = None;
    let mut mean_value = None;
    let needs_mean_std = !p.ccdclip && p.clip_cen == ClipCenter::Mean;
    let cen = if needs_mean_std {
        let stats = col_nanmean_var(col, mask, p.ddof);
        mean_std = Some(stats);
        mean_value = Some(stats.0);
        match p.cenfunc {
            CenFunc::Mean => stats.0,
            CenFunc::Median => col_nanmedian(col, mask, buf),
            CenFunc::LowerMedian => col_nanlmedian(col, mask, buf),
        }
    } else if p.cenfunc == CenFunc::Mean || p.clip_cen == ClipCenter::Mean {
        let mean = col_nanmean(col, mask);
        mean_value = Some(mean);
        match p.cenfunc {
            CenFunc::Mean => mean,
            CenFunc::Median => col_nanmedian(col, mask, buf),
            CenFunc::LowerMedian => col_nanlmedian(col, mask, buf),
        }
    } else {
        match p.cenfunc {
            CenFunc::Mean => col_nanmean(col, mask),
            CenFunc::Median => col_nanmedian(col, mask, buf),
            CenFunc::LowerMedian => col_nanlmedian(col, mask, buf),
        }
    };
    let std_center = match p.clip_cen {
        ClipCenter::Mean => mean_value.unwrap_or_else(|| col_nanmean(col, mask)),
        ClipCenter::Median => {
            if p.cenfunc == CenFunc::Median {
                cen
            } else {
                col_nanmedian(col, mask, buf)
            }
        }
        ClipCenter::LowerMedian => {
            if p.cenfunc == CenFunc::LowerMedian {
                cen
            } else {
                col_nanlmedian(col, mask, buf)
            }
        }
        ClipCenter::ClippingCenter => cen,
    };
    let (std, spread_sq) = if p.ccdclip {
        // Variance from the CCD noise model. The square root is only needed
        // for diagnostics; clipping compares squared residuals to this value.
        if std_center.is_nan() {
            (T::nan(), f64::NAN)
        } else {
            let c = std_center.to_f64();
            let v = (1.0 + p.snoise_ref) * (c + p.zero_ref).abs() * p.scale_ref
                + p.rdnoise_ref * p.rdnoise_ref;
            let std = if diagnostics {
                T::from_f64(v.sqrt())
            } else {
                T::nan()
            };
            (std, v)
        }
    } else {
        match p.stdfunc {
            StdFunc::Std => {
                let var = match p.clip_cen {
                    ClipCenter::Mean => mean_std
                        .map_or_else(|| col_nanmean_var(col, mask, p.ddof).1, |stats| stats.1),
                    ClipCenter::Median | ClipCenter::LowerMedian => {
                        col_nanvariance_about(col, mask, std_center, p.ddof)
                    }
                    ClipCenter::ClippingCenter => col_nanvariance_about(col, mask, cen, p.ddof),
                };
                let std = if diagnostics && var.is_finite() {
                    T::from_f64(var.sqrt())
                } else {
                    T::nan()
                };
                (std, var)
            }
            StdFunc::Mad => {
                let spread = col_nanmad_sigma_about(col, mask, std_center, p.ddof, buf);
                let spread_sq = if spread.is_nan() {
                    f64::NAN
                } else {
                    let spread = spread.to_f64();
                    spread * spread
                };
                (spread, spread_sq)
            }
        }
    };
    (cen, std_center, std, spread_sq)
}

/// Sigclip / ccdclip driver (the only difference is how `std` is computed inside the loop,
/// controlled by `p.ccdclip`).
pub fn sigclip<T: Float>(
    arr: &ArrayView3<T>,
    mask_in: Option<&ArrayView3<bool>>,
    p: &SigClipParams,
) -> RejectOutput<T> {
    sigclip_impl(arr, mask_in, p, true).into_output()
}

pub fn sigclip_mask<T: Float>(
    arr: &ArrayView3<T>,
    mask_in: Option<&ArrayView3<bool>>,
    p: &SigClipParams,
) -> Array3<bool> {
    sigclip_impl(arr, mask_in, p, false).mask
}

pub fn sigclip_mask_1d<T: Float>(
    values: &[T],
    mask_in: Option<&[bool]>,
    p: &SigClipParams,
) -> Vec<bool> {
    let n = values.len();
    let no_mask;
    let mask = if let Some(mask) = mask_in {
        mask
    } else {
        no_mask = vec![false; n];
        &no_mask
    };
    let mut out_mask = vec![false; n];
    let mut buf = Vec::<f64>::with_capacity(n);
    let mut scratch = SigClipPixelScratch::new();
    sigclip_pixel(
        values,
        mask,
        p,
        false,
        &mut buf,
        &mut out_mask,
        None,
        &mut scratch,
    );
    out_mask
}

pub fn sigclip_1d<T: Float>(
    values: &[T],
    mask_in: Option<&[bool]>,
    p: &SigClipParams,
) -> RejectOutput1d<T> {
    let n = values.len();
    let reject_values;
    let col = if p.ccdclip && p.rejection_gain != 1.0 {
        reject_values = values
            .iter()
            .map(|&value| T::from_f64(value.to_f64() / p.rejection_gain))
            .collect::<Vec<_>>();
        &reject_values[..]
    } else {
        values
    };
    let no_mask;
    let mask = if let Some(mask) = mask_in {
        mask
    } else {
        no_mask = vec![false; n];
        &no_mask
    };
    let mut out_mask = vec![false; n];
    let mut buf = Vec::<f64>::with_capacity(n);
    let mut scratch = SigClipPixelScratch::new();
    let px = sigclip_pixel(
        col,
        mask,
        p,
        true,
        &mut buf,
        &mut out_mask,
        None,
        &mut scratch,
    );
    RejectOutput1d {
        mask: out_mask,
        low: px.low,
        upp: px.upp,
        nit: px.nit,
        output_flags: px.output_flags,
        std: px.std,
    }
}

struct SigClipCombineScratch<T: Float> {
    reject_col: Vec<T>,
    source_col: Vec<T>,
    mask_in_col: Vec<bool>,
    out_mask: Vec<bool>,
    stat_buf: Vec<f64>,
    pixel: SigClipPixelScratch,
}

impl<T: Float> SigClipCombineScratch<T> {
    fn with_capacity(n: usize) -> Self {
        Self {
            reject_col: vec![T::zero(); n],
            source_col: vec![T::zero(); n],
            mask_in_col: vec![false; n],
            out_mask: vec![false; n],
            stat_buf: Vec::with_capacity(n),
            pixel: SigClipPixelScratch::new(),
        }
    }

    fn ensure_len(&mut self, n: usize) {
        self.reject_col.resize(n, T::zero());
        self.source_col.resize(n, T::zero());
        self.mask_in_col.resize(n, false);
        self.out_mask.resize(n, false);
    }
}

struct SigClipResultScratch<T: Float> {
    col: Vec<T>,
    mask_in_col: Vec<bool>,
    out_mask: Vec<bool>,
    stat_buf: Vec<f64>,
    pixel: SigClipPixelScratch,
    restored_col: Vec<u8>,
}

impl<T: Float> SigClipResultScratch<T> {
    fn with_capacity(n: usize) -> Self {
        Self {
            col: vec![T::zero(); n],
            mask_in_col: vec![false; n],
            out_mask: vec![false; n],
            stat_buf: Vec::with_capacity(n),
            pixel: SigClipPixelScratch::new(),
            restored_col: vec![0; n],
        }
    }

    fn ensure_len(&mut self, n: usize) {
        self.col.resize(n, T::zero());
        self.mask_in_col.resize(n, false);
        self.out_mask.resize(n, false);
        self.restored_col.resize(n, 0);
    }
}

fn sigclip_final_combine<T: Float>(
    arr: &ArrayView3<T>,
    mask_in: Option<&ArrayView3<bool>>,
    p: &SigClipParams,
    kind: FinalCombineKind,
    rejection_gain: f64,
) -> Array2<T> {
    let (n, h, w) = (arr.shape()[0], arr.shape()[1], arr.shape()[2]);
    let hw = h * w;
    let data = arr
        .as_slice_memory_order()
        .expect("sigclip combine kernel requires contiguous C-order arrays");
    let mask_data = mask_in.and_then(|m| m.as_slice_memory_order());

    let compute_pixel = |scratch: &mut SigClipCombineScratch<T>, idx: usize| {
        scratch.ensure_len(n);
        let mut raw_idx = idx;
        for k in 0..n {
            let source = data[raw_idx];
            scratch.source_col[k] = source;
            scratch.reject_col[k] = if rejection_gain == 1.0 {
                source
            } else {
                T::from_f64(source.to_f64() / rejection_gain)
            };
            scratch.mask_in_col[k] = mask_data.is_some_and(|m| m[raw_idx]);
            raw_idx += hw;
        }
        sigclip_pixel(
            &scratch.reject_col,
            &scratch.mask_in_col,
            p,
            false,
            &mut scratch.stat_buf,
            &mut scratch.out_mask,
            None,
            &mut scratch.pixel,
        );
        final_combine(
            &scratch.source_col,
            &scratch.out_mask,
            kind,
            &mut scratch.stat_buf,
        )
    };

    let reject_kind = if p.ccdclip || rejection_gain != 1.0 {
        RejectKind::CcdClip
    } else {
        RejectKind::SigClip
    };
    let vals: Vec<T> =
        if should_parallel_reject_for::<T>(reject_kind, hw, n, final_output_mode(kind)) {
            (0..hw)
                .into_par_iter()
                .map_init(
                    || SigClipCombineScratch::<T>::with_capacity(n),
                    compute_pixel,
                )
                .collect()
        } else {
            let mut scratch = SigClipCombineScratch::<T>::with_capacity(n);
            (0..hw)
                .map(|idx| compute_pixel(&mut scratch, idx))
                .collect()
        };

    Array2::from_shape_vec((h, w), vals).expect("sigclip final-combine result shape mismatch")
}

pub fn sigclip_mean<T: Float>(
    arr: &ArrayView3<T>,
    mask_in: Option<&ArrayView3<bool>>,
    p: &SigClipParams,
) -> Array2<T> {
    sigclip_final_combine(arr, mask_in, p, FinalCombineKind::Mean, 1.0)
}

pub fn sigclip_median<T: Float>(
    arr: &ArrayView3<T>,
    mask_in: Option<&ArrayView3<bool>>,
    p: &SigClipParams,
) -> Array2<T> {
    sigclip_final_combine(arr, mask_in, p, FinalCombineKind::Median, 1.0)
}

pub fn ccdclip_mean<T: Float>(
    arr: &ArrayView3<T>,
    mask_in: Option<&ArrayView3<bool>>,
    p: &SigClipParams,
    gain: f64,
) -> Array2<T> {
    sigclip_final_combine(arr, mask_in, p, FinalCombineKind::Mean, gain)
}

pub fn ccdclip_median<T: Float>(
    arr: &ArrayView3<T>,
    mask_in: Option<&ArrayView3<bool>>,
    p: &SigClipParams,
    gain: f64,
) -> Array2<T> {
    sigclip_final_combine(arr, mask_in, p, FinalCombineKind::Median, gain)
}

fn sigclip_final_combine_1d<T: Float>(
    values: &[T],
    mask_in: Option<&[bool]>,
    p: &SigClipParams,
    kind: FinalCombineKind,
    rejection_gain: f64,
) -> T {
    let n = values.len();
    let no_mask;
    let mask = if let Some(mask) = mask_in {
        mask
    } else {
        no_mask = vec![false; n];
        &no_mask
    };

    let mut reject_col = vec![T::zero(); n];
    let mut out_mask = vec![false; n];
    for (i, &source) in values.iter().enumerate() {
        reject_col[i] = if rejection_gain == 1.0 {
            source
        } else {
            T::from_f64(source.to_f64() / rejection_gain)
        };
    }

    let mut stat_buf = Vec::<f64>::with_capacity(n);
    let mut pixel_scratch = SigClipPixelScratch::new();
    sigclip_pixel(
        &reject_col,
        mask,
        p,
        false,
        &mut stat_buf,
        &mut out_mask,
        None,
        &mut pixel_scratch,
    );
    final_combine(values, &out_mask, kind, &mut stat_buf)
}

pub fn sigclip_mean_1d<T: Float>(values: &[T], mask_in: Option<&[bool]>, p: &SigClipParams) -> T {
    sigclip_final_combine_1d(values, mask_in, p, FinalCombineKind::Mean, 1.0)
}

pub fn sigclip_median_1d<T: Float>(values: &[T], mask_in: Option<&[bool]>, p: &SigClipParams) -> T {
    sigclip_final_combine_1d(values, mask_in, p, FinalCombineKind::Median, 1.0)
}

pub fn ccdclip_mean_1d<T: Float>(
    values: &[T],
    mask_in: Option<&[bool]>,
    p: &SigClipParams,
    gain: f64,
) -> T {
    sigclip_final_combine_1d(values, mask_in, p, FinalCombineKind::Mean, gain)
}

pub fn ccdclip_median_1d<T: Float>(
    values: &[T],
    mask_in: Option<&[bool]>,
    p: &SigClipParams,
    gain: f64,
) -> T {
    sigclip_final_combine_1d(values, mask_in, p, FinalCombineKind::Median, gain)
}

pub fn sigclip_restored_flags<T: Float>(
    arr: &ArrayView3<T>,
    mask_in: Option<&ArrayView3<bool>>,
    p: &SigClipParams,
) -> Array3<u8> {
    let (n, h, w) = (arr.shape()[0], arr.shape()[1], arr.shape()[2]);
    let hw = h * w;
    let mut flags_out = Array3::<u8>::zeros((n, h, w));
    let data = arr
        .as_slice_memory_order()
        .expect("sigclip restored-flags kernel requires contiguous C-order arrays");
    let mask_data = mask_in.and_then(|m| m.as_slice_memory_order());

    let compute_pixel = |scratch: &mut SigClipResultScratch<T>, idx: usize| {
        scratch.ensure_len(n);
        let mut raw_idx = idx;
        for k in 0..n {
            let value = data[raw_idx];
            scratch.col[k] = if p.ccdclip && p.rejection_gain != 1.0 {
                T::from_f64(value.to_f64() / p.rejection_gain)
            } else {
                value
            };
            scratch.mask_in_col[k] = mask_data.is_some_and(|m| m[raw_idx]);
            raw_idx += hw;
        }
        sigclip_pixel(
            &scratch.col,
            &scratch.mask_in_col,
            p,
            false,
            &mut scratch.stat_buf,
            &mut scratch.out_mask,
            Some(&mut scratch.restored_col),
            &mut scratch.pixel,
        );
        scratch.restored_col.clone()
    };

    let reject_kind = if p.ccdclip {
        RejectKind::CcdClip
    } else {
        RejectKind::SigClip
    };
    let results: Vec<Vec<u8>> =
        if should_parallel_reject_for::<T>(reject_kind, hw, n, RejectOutputMode::RestoredFlags) {
            (0..hw)
                .into_par_iter()
                .map_init(
                    || SigClipResultScratch::<T>::with_capacity(n),
                    compute_pixel,
                )
                .collect()
        } else {
            let mut scratch = SigClipResultScratch::<T>::with_capacity(n);
            (0..hw)
                .map(|idx| compute_pixel(&mut scratch, idx))
                .collect()
        };

    let flags_slice = flags_out
        .as_slice_memory_order_mut()
        .expect("sigclip restored-flags output must be contiguous");
    for k in 0..n {
        let plane_offset = k * hw;
        for (idx, flags_col) in results.iter().enumerate() {
            flags_slice[plane_offset + idx] = flags_col[k];
        }
    }

    flags_out
}

fn sigclip_impl<T: Float>(
    arr: &ArrayView3<T>,
    mask_in: Option<&ArrayView3<bool>>,
    p: &SigClipParams,
    diagnostics: bool,
) -> RejectCompute<T> {
    let (n, h, w) = (arr.shape()[0], arr.shape()[1], arr.shape()[2]);

    let mut mask_out = Array3::<bool>::from_elem((n, h, w), false);
    let mut low = diagnostics.then(|| Array2::<T>::from_elem((h, w), T::nan()));
    let mut upp = diagnostics.then(|| Array2::<T>::from_elem((h, w), T::nan()));
    let mut nit = diagnostics.then(|| Array2::<u8>::zeros((h, w)));
    let mut output_flags = diagnostics.then(|| Array2::<u8>::zeros((h, w)));
    let mut std = diagnostics.then(|| Array2::<T>::from_elem((h, w), T::nan()));

    // Build per-pixel work units sequentially indexed; rayon over (h*w).
    let hw = h * w;
    let data = arr
        .as_slice_memory_order()
        .expect("sigclip kernel requires contiguous C-order arrays");
    let mask_data = mask_in.and_then(|m| m.as_slice_memory_order());

    // We need mutable access to disjoint output positions of mask_out: walk by (i, j) and
    // produce results, then write them into output arrays sequentially. To keep
    // it simple and correct, collect per-pixel results in parallel then assign.
    struct Px<T> {
        mask_col: Vec<bool>,
        low: T,
        upp: T,
        std: T,
        nit: u8,
        output_flags: u8,
    }

    let compute_pixel = |scratch: &mut SigClipResultScratch<T>, idx: usize| {
        scratch.ensure_len(n);
        let mut raw_idx = idx;
        for k in 0..n {
            let value = data[raw_idx];
            scratch.col[k] = if p.ccdclip && p.rejection_gain != 1.0 {
                T::from_f64(value.to_f64() / p.rejection_gain)
            } else {
                value
            };
            scratch.mask_in_col[k] = mask_data.is_some_and(|m| m[raw_idx]);
            raw_idx += hw;
        }
        let px = sigclip_pixel(
            &scratch.col,
            &scratch.mask_in_col,
            p,
            diagnostics,
            &mut scratch.stat_buf,
            &mut scratch.out_mask,
            None,
            &mut scratch.pixel,
        );
        Px {
            mask_col: scratch.out_mask.clone(),
            low: px.low,
            upp: px.upp,
            std: px.std,
            nit: px.nit,
            output_flags: px.output_flags,
        }
    };

    let reject_kind = if p.ccdclip {
        RejectKind::CcdClip
    } else {
        RejectKind::SigClip
    };
    let results: Vec<Px<T>> = if should_parallel_reject_for::<T>(
        reject_kind,
        hw,
        n,
        if diagnostics {
            RejectOutputMode::Diagnostics
        } else {
            RejectOutputMode::MaskOnly
        },
    ) {
        (0..hw)
            .into_par_iter()
            .map_init(
                || SigClipResultScratch::<T>::with_capacity(n),
                compute_pixel,
            )
            .collect()
    } else {
        let mut scratch = SigClipResultScratch::<T>::with_capacity(n);
        (0..hw)
            .map(|idx| compute_pixel(&mut scratch, idx))
            .collect()
    };

    let mask_slice = mask_out
        .as_slice_memory_order_mut()
        .expect("sigclip output mask must be contiguous");
    for k in 0..n {
        let plane_offset = k * hw;
        for (idx, r) in results.iter().enumerate() {
            mask_slice[plane_offset + idx] = r.mask_col[k];
        }
    }
    if diagnostics {
        let low_slice = low
            .as_mut()
            .expect("low array missing")
            .as_slice_memory_order_mut()
            .expect("sigclip low output must be contiguous");
        let upp_slice = upp
            .as_mut()
            .expect("upp array missing")
            .as_slice_memory_order_mut()
            .expect("sigclip upp output must be contiguous");
        let nit_slice = nit
            .as_mut()
            .expect("nit array missing")
            .as_slice_memory_order_mut()
            .expect("sigclip nit output must be contiguous");
        let output_flags_slice = output_flags
            .as_mut()
            .expect("output_flags array missing")
            .as_slice_memory_order_mut()
            .expect("sigclip output_flags output must be contiguous");
        let std_slice = std
            .as_mut()
            .expect("std array missing")
            .as_slice_memory_order_mut()
            .expect("sigclip std output must be contiguous");
        for (idx, r) in results.iter().enumerate() {
            low_slice[idx] = r.low;
            upp_slice[idx] = r.upp;
            nit_slice[idx] = r.nit;
            output_flags_slice[idx] = r.output_flags;
            std_slice[idx] = r.std;
        }
    }

    RejectCompute {
        mask: mask_out,
        low,
        upp,
        nit,
        output_flags,
        std,
    }
}

// ---------- minmax ----------

fn apply_minmax_pairs<T: Float>(
    finite_pairs: &mut [(T, usize)],
    mask: &mut [bool],
    q_low: f64,
    q_upp: f64,
    diagnostics: bool,
) -> (T, T) {
    let n_finite = finite_pairs.len();
    let n_rej_low = (n_finite as f64 * q_low + 0.001) as usize;
    let n_rej_upp = (n_finite as f64 * q_upp + 0.001) as usize;

    finite_pairs.sort_by(cmp_pair_t);
    for &(_, k) in finite_pairs.iter().take(n_rej_low.min(n_finite)) {
        mask[k] = true;
    }
    let high_count = n_rej_upp.min(n_finite.saturating_sub(n_rej_low));
    for &(_, k) in finite_pairs.iter().rev().take(high_count) {
        mask[k] = true;
    }

    if diagnostics && n_rej_low + high_count < n_finite {
        (
            finite_pairs[n_rej_low].0,
            finite_pairs[n_finite - high_count - 1].0,
        )
    } else {
        (T::nan(), T::nan())
    }
}

/// Per-pixel minmax rejection: reject `n_rej_low` smallest and `n_rej_upp` largest
/// of finite, unmasked values. Counts are derived per-pixel from the IRAF formula
/// `int(n_finite * q + 0.001)` so that `q_low`/`q_upp` express *fractions*; pass
/// `q_low = n_min / N`, `q_upp = n_max / N` from Python.
pub fn minmax<T: Float>(
    arr: &ArrayView3<T>,
    mask_in: Option<&ArrayView3<bool>>,
    q_low: f64,
    q_upp: f64,
) -> RejectOutput<T> {
    let (n, h, w) = (arr.shape()[0], arr.shape()[1], arr.shape()[2]);
    let mut mask_out = Array3::<bool>::from_elem((n, h, w), false);
    let mut low = Array2::<T>::from_elem((h, w), T::nan());
    let mut upp = Array2::<T>::from_elem((h, w), T::nan());
    let nit = Array2::<u8>::from_elem((h, w), 1);
    let mut output_flags = Array2::<u8>::zeros((h, w));
    let std = Array2::<T>::from_elem((h, w), T::nan());

    let hw = h * w;
    let data = arr
        .as_slice_memory_order()
        .expect("minmax kernel requires contiguous C-order arrays");
    let mask_data = mask_in.and_then(|m| m.as_slice_memory_order());

    struct Px<T> {
        mask_col: Vec<bool>,
        low: T,
        upp: T,
        output_flags: u8,
    }

    let compute_pixel = |scratch: &mut RankCombineScratch<T>, idx: usize| {
        scratch.ensure_len(n);
        scratch.pairs_t.clear();
        let mut pre_masked = false;
        let mut raw_idx = idx;
        for k in 0..n {
            let x = data[raw_idx];
            let masked_by_input = mask_data.is_some_and(|m| m[raw_idx]);
            pre_masked |= masked_by_input;
            let masked = masked_by_input || !x.is_finite();
            scratch.mask[k] = masked;
            if !masked {
                scratch.pairs_t.push((x, k));
            }
            raw_idx += hw;
        }
        let (lo, up) =
            apply_minmax_pairs(&mut scratch.pairs_t, &mut scratch.mask, q_low, q_upp, true);
        let c: u8 = pre_masked as u8;
        Px {
            mask_col: scratch.mask.clone(),
            low: lo,
            upp: up,
            output_flags: c,
        }
    };

    let results: Vec<Px<T>> = if should_parallel_reject_for::<T>(
        RejectKind::MinMax,
        hw,
        n,
        RejectOutputMode::Diagnostics,
    ) {
        (0..hw)
            .into_par_iter()
            .map_init(|| RankCombineScratch::<T>::with_capacity(n), compute_pixel)
            .collect()
    } else {
        let mut scratch = RankCombineScratch::<T>::with_capacity(n);
        (0..hw)
            .map(|idx| compute_pixel(&mut scratch, idx))
            .collect()
    };

    let mask_slice = mask_out
        .as_slice_memory_order_mut()
        .expect("minmax output mask must be contiguous");
    for k in 0..n {
        let plane_offset = k * hw;
        for (idx, r) in results.iter().enumerate() {
            mask_slice[plane_offset + idx] = r.mask_col[k];
        }
    }
    let low_slice = low
        .as_slice_memory_order_mut()
        .expect("minmax low output must be contiguous");
    let upp_slice = upp
        .as_slice_memory_order_mut()
        .expect("minmax upp output must be contiguous");
    let output_flags_slice = output_flags
        .as_slice_memory_order_mut()
        .expect("minmax output_flags output must be contiguous");
    for (idx, r) in results.iter().enumerate() {
        low_slice[idx] = r.low;
        upp_slice[idx] = r.upp;
        output_flags_slice[idx] = r.output_flags;
    }

    RejectOutput {
        mask: mask_out,
        low,
        upp,
        nit,
        output_flags,
        std,
    }
}

pub fn minmax_mask<T: Float>(
    arr: &ArrayView3<T>,
    mask_in: Option<&ArrayView3<bool>>,
    q_low: f64,
    q_upp: f64,
) -> Array3<bool> {
    let (n, h, w) = (arr.shape()[0], arr.shape()[1], arr.shape()[2]);
    let mut mask_out = Array3::<bool>::from_elem((n, h, w), false);

    let hw = h * w;
    let data = arr
        .as_slice_memory_order()
        .expect("minmax kernel requires contiguous C-order arrays");
    let mask_data = mask_in.and_then(|m| m.as_slice_memory_order());

    let compute_pixel = |scratch: &mut RankCombineScratch<T>, idx: usize| {
        scratch.ensure_len(n);
        scratch.pairs_t.clear();
        let mut raw_idx = idx;
        for k in 0..n {
            let x = data[raw_idx];
            let masked_by_input = mask_data.is_some_and(|m| m[raw_idx]);
            let masked = masked_by_input || !x.is_finite();
            scratch.mask[k] = masked;
            if !masked {
                scratch.pairs_t.push((x, k));
            }
            raw_idx += hw;
        }
        apply_minmax_pairs(&mut scratch.pairs_t, &mut scratch.mask, q_low, q_upp, false);
        scratch.mask.clone()
    };

    let results: Vec<Vec<bool>> =
        if should_parallel_reject_for::<T>(RejectKind::MinMax, hw, n, RejectOutputMode::MaskOnly) {
            (0..hw)
                .into_par_iter()
                .map_init(|| RankCombineScratch::<T>::with_capacity(n), compute_pixel)
                .collect()
        } else {
            let mut scratch = RankCombineScratch::<T>::with_capacity(n);
            (0..hw)
                .map(|idx| compute_pixel(&mut scratch, idx))
                .collect()
        };

    let mask_slice = mask_out
        .as_slice_memory_order_mut()
        .expect("minmax mask output must be contiguous");
    for k in 0..n {
        let plane_offset = k * hw;
        for (idx, mask_col) in results.iter().enumerate() {
            mask_slice[plane_offset + idx] = mask_col[k];
        }
    }

    mask_out
}

fn minmax_mask_and_bounds_1d<T: Float>(
    values: &[T],
    mask_in: Option<&[bool]>,
    q_low: f64,
    q_upp: f64,
    diagnostics: bool,
) -> (Vec<bool>, T, T, u8) {
    let n = values.len();
    let mut mask = vec![false; n];
    let mut finite_pairs: Vec<(T, usize)> = Vec::with_capacity(n);
    let mut pre_masked = false;
    for (k, &x) in values.iter().enumerate() {
        let masked_by_input = mask_in.is_some_and(|m| m[k]);
        pre_masked |= masked_by_input;
        let masked = masked_by_input || !x.is_finite();
        if masked {
            mask[k] = true;
        } else {
            finite_pairs.push((x, k));
        }
    }

    let (low, upp) = apply_minmax_pairs(&mut finite_pairs, &mut mask, q_low, q_upp, diagnostics);
    (mask, low, upp, pre_masked as u8)
}

pub fn minmax_1d<T: Float>(
    values: &[T],
    mask_in: Option<&[bool]>,
    q_low: f64,
    q_upp: f64,
) -> RejectOutput1d<T> {
    let (mask, low, upp, output_flags) =
        minmax_mask_and_bounds_1d(values, mask_in, q_low, q_upp, true);
    RejectOutput1d {
        mask,
        low,
        upp,
        nit: 1,
        output_flags,
        std: T::nan(),
    }
}

pub fn minmax_mask_1d<T: Float>(
    values: &[T],
    mask_in: Option<&[bool]>,
    q_low: f64,
    q_upp: f64,
) -> Vec<bool> {
    minmax_mask_and_bounds_1d(values, mask_in, q_low, q_upp, false).0
}

struct RankCombineScratch<T: Float> {
    mask: Vec<bool>,
    pairs_t: Vec<(T, usize)>,
    pairs_f64: Vec<(f64, usize)>,
    stat_buf: Vec<f64>,
    resid: Vec<f64>,
}

impl<T: Float> RankCombineScratch<T> {
    fn with_capacity(n: usize) -> Self {
        Self {
            mask: vec![false; n],
            pairs_t: Vec::with_capacity(n),
            pairs_f64: Vec::with_capacity(n),
            stat_buf: Vec::with_capacity(n),
            resid: Vec::with_capacity(n),
        }
    }

    fn ensure_len(&mut self, n: usize) {
        self.mask.resize(n, false);
    }
}

fn minmax_final_combine<T: Float>(
    arr: &ArrayView3<T>,
    mask_in: Option<&ArrayView3<bool>>,
    q_low: f64,
    q_upp: f64,
    kind: FinalCombineKind,
) -> Array2<T> {
    let (n, h, w) = (arr.shape()[0], arr.shape()[1], arr.shape()[2]);
    let hw = h * w;
    let data = arr
        .as_slice_memory_order()
        .expect("minmax combine kernel requires contiguous C-order arrays");
    let mask_data = mask_in.and_then(|m| m.as_slice_memory_order());

    let compute_pixel = |scratch: &mut RankCombineScratch<T>, idx: usize| {
        scratch.ensure_len(n);
        scratch.pairs_t.clear();
        let mut raw_idx = idx;
        for k in 0..n {
            let x = data[raw_idx];
            let masked_by_input = mask_data.is_some_and(|m| m[raw_idx]);
            let masked = masked_by_input || !x.is_finite();
            scratch.mask[k] = masked;
            if !masked {
                scratch.pairs_t.push((x, k));
            }
            raw_idx += hw;
        }
        apply_minmax_pairs(&mut scratch.pairs_t, &mut scratch.mask, q_low, q_upp, false);

        final_combine_from_sorted_pairs_t(
            &scratch.pairs_t,
            &scratch.mask,
            kind,
            &mut scratch.stat_buf,
        )
    };

    let vals: Vec<T> =
        if should_parallel_reject_for::<T>(RejectKind::MinMax, hw, n, final_output_mode(kind)) {
            (0..hw)
                .into_par_iter()
                .map_init(|| RankCombineScratch::<T>::with_capacity(n), compute_pixel)
                .collect()
        } else {
            let mut scratch = RankCombineScratch::<T>::with_capacity(n);
            (0..hw)
                .map(|idx| compute_pixel(&mut scratch, idx))
                .collect()
        };

    Array2::from_shape_vec((h, w), vals).expect("minmax final-combine result shape mismatch")
}

pub fn minmax_mean<T: Float>(
    arr: &ArrayView3<T>,
    mask_in: Option<&ArrayView3<bool>>,
    q_low: f64,
    q_upp: f64,
) -> Array2<T> {
    minmax_final_combine(arr, mask_in, q_low, q_upp, FinalCombineKind::Mean)
}

pub fn minmax_median<T: Float>(
    arr: &ArrayView3<T>,
    mask_in: Option<&ArrayView3<bool>>,
    q_low: f64,
    q_upp: f64,
) -> Array2<T> {
    minmax_final_combine(arr, mask_in, q_low, q_upp, FinalCombineKind::Median)
}

fn minmax_final_combine_1d<T: Float>(
    values: &[T],
    mask_in: Option<&[bool]>,
    q_low: f64,
    q_upp: f64,
    kind: FinalCombineKind,
) -> T {
    let n = values.len();
    let mut mask = vec![false; n];
    let mut pairs: Vec<(T, usize)> = Vec::with_capacity(n);
    for (k, &x) in values.iter().enumerate() {
        let masked = mask_in.is_some_and(|m| m[k]) || !x.is_finite();
        mask[k] = masked;
        if !masked {
            pairs.push((x, k));
        }
    }

    apply_minmax_pairs(&mut pairs, &mut mask, q_low, q_upp, false);
    let mut stat_buf = Vec::<f64>::with_capacity(n);
    final_combine_from_sorted_pairs_t(&pairs, &mask, kind, &mut stat_buf)
}

pub fn minmax_mean_1d<T: Float>(
    values: &[T],
    mask_in: Option<&[bool]>,
    q_low: f64,
    q_upp: f64,
) -> T {
    minmax_final_combine_1d(values, mask_in, q_low, q_upp, FinalCombineKind::Mean)
}

pub fn minmax_median_1d<T: Float>(
    values: &[T],
    mask_in: Option<&[bool]>,
    q_low: f64,
    q_upp: f64,
) -> T {
    minmax_final_combine_1d(values, mask_in, q_low, q_upp, FinalCombineKind::Median)
}

// ---------- pclip ----------

const MINCLIP: usize = 3;
const IRAF_RESID_TOL: f64 = 0.001;

#[inline]
fn iraf_pclip_offset(pclip: f64, nimages: usize) -> isize {
    let half = (nimages / 2).max(1);
    let mut offset = pclip;
    if offset.abs() < 1.0 {
        offset *= half as f64;
    }
    let offset = offset.trunc() as isize;
    if pclip < 0.0 {
        offset.max(-(half as isize)).min(-1)
    } else {
        offset.max(1).min(half as isize)
    }
}

#[inline]
fn iraf_nint(x: f64) -> isize {
    if x >= 0.0 {
        (x + 0.5).floor() as isize
    } else {
        (x - 0.5).ceil() as isize
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_pclip_pairs<T: Float>(
    pairs: &mut Vec<(f64, usize)>,
    mask: &mut [bool],
    resid: &mut Vec<f64>,
    pclip_offset: isize,
    sigma_lower: f64,
    sigma_upper: f64,
    nkeep: usize,
    diagnostics: bool,
) -> (T, T) {
    let n_total = pairs.len();
    if n_total == 0 || n_total < MINCLIP || n_total < nkeep.saturating_add(1) {
        return if diagnostics {
            col_minmax_from_masked_pairs::<T>(pairs, mask)
        } else {
            (T::nan(), T::nan())
        };
    }

    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));

    let n2_1based = 1 + n_total / 2;
    let even = n_total % 2 == 0;
    let med = if even {
        (pairs[n2_1based - 2].0 + pairs[n2_1based - 1].0) / 2.0
    } else {
        pairs[n2_1based - 1].0
    };

    let n3_1based = if pclip_offset < 0 {
        let base = if even { n2_1based - 1 } else { n2_1based };
        iraf_nint(base as f64 + pclip_offset as f64)
            .max(1)
            .min(n_total as isize) as usize
    } else {
        iraf_nint(n2_1based as f64 + pclip_offset as f64)
            .max(1)
            .min(n_total as isize) as usize
    };
    let sign = if pclip_offset < 0 { -1.0 } else { 1.0 };
    let sigma = sign * (pairs[n3_1based - 1].0 - med);
    if sigma == 0.0 {
        return if diagnostics {
            col_minmax_from_masked_pairs::<T>(pairs, mask)
        } else {
            (T::nan(), T::nan())
        };
    }

    resid.clear();
    resid.resize(n_total, 0.0);
    let mut nl = 0usize;
    while nl < n_total {
        let r = (med - pairs[nl].0) / sigma;
        if r < sigma_lower {
            break;
        }
        resid[nl] = r;
        nl += 1;
    }
    let mut nh = n_total - 1;
    loop {
        if nh < nl {
            break;
        }
        let r = (pairs[nh].0 - med) / sigma;
        if r < sigma_upper {
            break;
        }
        resid[nh] = r;
        if nh == 0 {
            break;
        }
        nh -= 1;
    }

    let maxkeep = nkeep.min(n_total);
    let mut n_kept = if nh >= nl { nh - nl + 1 } else { 0 };
    while n_kept < maxkeep {
        if nl == 0 {
            nh += 1;
        } else if nh + 1 == n_total {
            nl -= 1;
        } else {
            let r = resid[nl - 1];
            let s = resid[nh + 1];
            if r < s {
                nl -= 1;
                let r_tol = r + IRAF_RESID_TOL;
                if s <= r_tol {
                    nh += 1;
                }
                if nl > 0 && resid[nl - 1] <= r_tol {
                    nl -= 1;
                }
            } else {
                nh += 1;
                let s_tol = s + IRAF_RESID_TOL;
                if r <= s_tol {
                    nl -= 1;
                }
                if nh + 1 < n_total && resid[nh + 1] <= s_tol {
                    nh += 1;
                }
            }
        }
        n_kept = nh - nl + 1;
    }

    for (rank, &(_, k)) in pairs.iter().enumerate() {
        if rank < nl || rank > nh {
            mask[k] = true;
        }
    }

    if diagnostics {
        col_minmax_from_masked_pairs::<T>(pairs, mask)
    } else {
        (T::nan(), T::nan())
    }
}

#[allow(clippy::too_many_arguments)]
fn pclip_values<T: Float>(
    values: &[T],
    mask_in: Option<&[bool]>,
    pclip_offset: isize,
    sigma_lower: f64,
    sigma_upper: f64,
    nkeep: usize,
    diagnostics: bool,
) -> (Vec<bool>, T, T, u8) {
    let nimages = values.len();
    let mut mask = vec![false; nimages];
    let mut pairs: Vec<(f64, usize)> = Vec::with_capacity(nimages);
    let mut resid = Vec::<f64>::with_capacity(nimages);
    let mut pre_masked = false;
    for (k, &x) in values.iter().enumerate() {
        let masked_by_input = mask_in.is_some_and(|m| m[k]);
        pre_masked |= masked_by_input;
        if masked_by_input || !x.is_finite() {
            mask[k] = true;
        } else {
            pairs.push((x.to_f64(), k));
        }
    }

    let (low, upp) = apply_pclip_pairs(
        &mut pairs,
        &mut mask,
        &mut resid,
        pclip_offset,
        sigma_lower,
        sigma_upper,
        nkeep,
        diagnostics,
    );
    (mask, low, upp, pre_masked as u8)
}

fn col_minmax_from_masked_pairs<T: Float>(pairs: &[(f64, usize)], mask: &[bool]) -> (T, T) {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &(value, k) in pairs {
        if !mask[k] {
            lo = lo.min(value);
            hi = hi.max(value);
        }
    }
    if lo.is_infinite() {
        (T::nan(), T::nan())
    } else {
        (T::from_f64(lo), T::from_f64(hi))
    }
}

/// IRAF-style percentile clip: estimate sigma from a sorted rank offset around
/// the median, then apply lower/upper sigma thresholds in one pass.
pub fn pclip<T: Float>(
    arr: &ArrayView3<T>,
    mask_in: Option<&ArrayView3<bool>>,
    pclip: f64,
    sigma_lower: f64,
    sigma_upper: f64,
    nkeep: usize,
) -> RejectOutput<T> {
    let (n, h, w) = (arr.shape()[0], arr.shape()[1], arr.shape()[2]);
    let mut mask_out = Array3::<bool>::from_elem((n, h, w), false);
    let mut low = Array2::<T>::from_elem((h, w), T::nan());
    let mut upp = Array2::<T>::from_elem((h, w), T::nan());
    let nit = Array2::<u8>::from_elem((h, w), 1);
    let mut output_flags = Array2::<u8>::zeros((h, w));
    let std = Array2::<T>::from_elem((h, w), T::nan());

    let hw = h * w;
    let data = arr
        .as_slice_memory_order()
        .expect("pclip kernel requires contiguous C-order arrays");
    let mask_data = mask_in.and_then(|m| m.as_slice_memory_order());
    let pclip_offset = iraf_pclip_offset(pclip, n);

    struct Px<T> {
        mask_col: Vec<bool>,
        low: T,
        upp: T,
        output_flags: u8,
    }

    let compute_pixel = |scratch: &mut RankCombineScratch<T>, idx: usize| {
        scratch.ensure_len(n);
        scratch.pairs_f64.clear();
        let mut pre_masked = false;
        let mut raw_idx = idx;
        for k in 0..n {
            let x = data[raw_idx];
            let masked_by_input = mask_data.is_some_and(|m| m[raw_idx]);
            pre_masked |= masked_by_input;
            let masked = masked_by_input || !x.is_finite();
            scratch.mask[k] = masked;
            if !masked {
                scratch.pairs_f64.push((x.to_f64(), k));
            }
            raw_idx += hw;
        }
        let (lo, up) = apply_pclip_pairs(
            &mut scratch.pairs_f64,
            &mut scratch.mask,
            &mut scratch.resid,
            pclip_offset,
            sigma_lower,
            sigma_upper,
            nkeep,
            true,
        );
        Px {
            mask_col: scratch.mask.clone(),
            low: lo,
            upp: up,
            output_flags: pre_masked as u8,
        }
    };

    let results: Vec<Px<T>> =
        if should_parallel_reject_for::<T>(RejectKind::PClip, hw, n, RejectOutputMode::Diagnostics)
        {
            (0..hw)
                .into_par_iter()
                .map_init(|| RankCombineScratch::<T>::with_capacity(n), compute_pixel)
                .collect()
        } else {
            let mut scratch = RankCombineScratch::<T>::with_capacity(n);
            (0..hw)
                .map(|idx| compute_pixel(&mut scratch, idx))
                .collect()
        };

    let mask_slice = mask_out
        .as_slice_memory_order_mut()
        .expect("pclip output mask must be contiguous");
    for k in 0..n {
        let plane_offset = k * hw;
        for (idx, r) in results.iter().enumerate() {
            mask_slice[plane_offset + idx] = r.mask_col[k];
        }
    }
    let low_slice = low
        .as_slice_memory_order_mut()
        .expect("pclip low output must be contiguous");
    let upp_slice = upp
        .as_slice_memory_order_mut()
        .expect("pclip upp output must be contiguous");
    let output_flags_slice = output_flags
        .as_slice_memory_order_mut()
        .expect("pclip output_flags output must be contiguous");
    for (idx, r) in results.iter().enumerate() {
        low_slice[idx] = r.low;
        upp_slice[idx] = r.upp;
        output_flags_slice[idx] = r.output_flags;
    }

    RejectOutput {
        mask: mask_out,
        low,
        upp,
        nit,
        output_flags,
        std,
    }
}

pub fn pclip_mask<T: Float>(
    arr: &ArrayView3<T>,
    mask_in: Option<&ArrayView3<bool>>,
    pclip: f64,
    sigma_lower: f64,
    sigma_upper: f64,
    nkeep: usize,
) -> Array3<bool> {
    let (n, h, w) = (arr.shape()[0], arr.shape()[1], arr.shape()[2]);
    let mut mask_out = Array3::<bool>::from_elem((n, h, w), false);

    let hw = h * w;
    let data = arr
        .as_slice_memory_order()
        .expect("pclip kernel requires contiguous C-order arrays");
    let mask_data = mask_in.and_then(|m| m.as_slice_memory_order());
    let pclip_offset = iraf_pclip_offset(pclip, n);

    let compute_pixel = |scratch: &mut RankCombineScratch<T>, idx: usize| {
        scratch.ensure_len(n);
        scratch.pairs_f64.clear();
        let mut raw_idx = idx;
        for k in 0..n {
            let x = data[raw_idx];
            let masked_by_input = mask_data.is_some_and(|m| m[raw_idx]);
            let masked = masked_by_input || !x.is_finite();
            scratch.mask[k] = masked;
            if !masked {
                scratch.pairs_f64.push((x.to_f64(), k));
            }
            raw_idx += hw;
        }
        apply_pclip_pairs::<T>(
            &mut scratch.pairs_f64,
            &mut scratch.mask,
            &mut scratch.resid,
            pclip_offset,
            sigma_lower,
            sigma_upper,
            nkeep,
            false,
        );
        scratch.mask.clone()
    };

    let results: Vec<Vec<bool>> =
        if should_parallel_reject_for::<T>(RejectKind::PClip, hw, n, RejectOutputMode::MaskOnly) {
            (0..hw)
                .into_par_iter()
                .map_init(|| RankCombineScratch::<T>::with_capacity(n), compute_pixel)
                .collect()
        } else {
            let mut scratch = RankCombineScratch::<T>::with_capacity(n);
            (0..hw)
                .map(|idx| compute_pixel(&mut scratch, idx))
                .collect()
        };

    let mask_slice = mask_out
        .as_slice_memory_order_mut()
        .expect("pclip mask output must be contiguous");
    for k in 0..n {
        let plane_offset = k * hw;
        for (idx, mask_col) in results.iter().enumerate() {
            mask_slice[plane_offset + idx] = mask_col[k];
        }
    }

    mask_out
}

pub fn pclip_1d<T: Float>(
    values: &[T],
    mask_in: Option<&[bool]>,
    pclip: f64,
    sigma_lower: f64,
    sigma_upper: f64,
    nkeep: usize,
) -> RejectOutput1d<T> {
    let pclip_offset = iraf_pclip_offset(pclip, values.len());
    let (mask, low, upp, output_flags) = pclip_values(
        values,
        mask_in,
        pclip_offset,
        sigma_lower,
        sigma_upper,
        nkeep,
        true,
    );
    RejectOutput1d {
        mask,
        low,
        upp,
        nit: 1,
        output_flags,
        std: T::nan(),
    }
}

pub fn pclip_mask_1d<T: Float>(
    values: &[T],
    mask_in: Option<&[bool]>,
    pclip: f64,
    sigma_lower: f64,
    sigma_upper: f64,
    nkeep: usize,
) -> Vec<bool> {
    let pclip_offset = iraf_pclip_offset(pclip, values.len());
    pclip_values(
        values,
        mask_in,
        pclip_offset,
        sigma_lower,
        sigma_upper,
        nkeep,
        false,
    )
    .0
}

#[allow(clippy::too_many_arguments)]
fn pclip_final_combine<T: Float>(
    arr: &ArrayView3<T>,
    mask_in: Option<&ArrayView3<bool>>,
    pclip: f64,
    sigma_lower: f64,
    sigma_upper: f64,
    nkeep: usize,
    kind: FinalCombineKind,
) -> Array2<T> {
    let (n, h, w) = (arr.shape()[0], arr.shape()[1], arr.shape()[2]);
    let hw = h * w;
    let data = arr
        .as_slice_memory_order()
        .expect("pclip combine kernel requires contiguous C-order arrays");
    let mask_data = mask_in.and_then(|m| m.as_slice_memory_order());
    let pclip_offset = iraf_pclip_offset(pclip, n);

    let compute_pixel = |scratch: &mut RankCombineScratch<T>, idx: usize| {
        scratch.ensure_len(n);
        scratch.pairs_f64.clear();
        let mut raw_idx = idx;
        for k in 0..n {
            let x = data[raw_idx];
            let masked_by_input = mask_data.is_some_and(|m| m[raw_idx]);
            let masked = masked_by_input || !x.is_finite();
            scratch.mask[k] = masked;
            if !masked {
                scratch.pairs_f64.push((x.to_f64(), k));
            }
            raw_idx += hw;
        }

        apply_pclip_pairs::<T>(
            &mut scratch.pairs_f64,
            &mut scratch.mask,
            &mut scratch.resid,
            pclip_offset,
            sigma_lower,
            sigma_upper,
            nkeep,
            false,
        );

        final_combine_from_sorted_pairs_f64(
            &scratch.pairs_f64,
            &scratch.mask,
            kind,
            &mut scratch.stat_buf,
        )
    };

    let vals: Vec<T> =
        if should_parallel_reject_for::<T>(RejectKind::PClip, hw, n, final_output_mode(kind)) {
            (0..hw)
                .into_par_iter()
                .map_init(|| RankCombineScratch::<T>::with_capacity(n), compute_pixel)
                .collect()
        } else {
            let mut scratch = RankCombineScratch::<T>::with_capacity(n);
            (0..hw)
                .map(|idx| compute_pixel(&mut scratch, idx))
                .collect()
        };

    Array2::from_shape_vec((h, w), vals).expect("pclip final-combine result shape mismatch")
}

pub fn pclip_mean<T: Float>(
    arr: &ArrayView3<T>,
    mask_in: Option<&ArrayView3<bool>>,
    pclip: f64,
    sigma_lower: f64,
    sigma_upper: f64,
    nkeep: usize,
) -> Array2<T> {
    pclip_final_combine(
        arr,
        mask_in,
        pclip,
        sigma_lower,
        sigma_upper,
        nkeep,
        FinalCombineKind::Mean,
    )
}

pub fn pclip_median<T: Float>(
    arr: &ArrayView3<T>,
    mask_in: Option<&ArrayView3<bool>>,
    pclip: f64,
    sigma_lower: f64,
    sigma_upper: f64,
    nkeep: usize,
) -> Array2<T> {
    pclip_final_combine(
        arr,
        mask_in,
        pclip,
        sigma_lower,
        sigma_upper,
        nkeep,
        FinalCombineKind::Median,
    )
}

fn pclip_final_combine_1d<T: Float>(
    values: &[T],
    mask_in: Option<&[bool]>,
    pclip: f64,
    sigma_lower: f64,
    sigma_upper: f64,
    nkeep: usize,
    kind: FinalCombineKind,
) -> T {
    let n = values.len();
    let pclip_offset = iraf_pclip_offset(pclip, n);
    let mut mask = vec![false; n];
    let mut pairs: Vec<(f64, usize)> = Vec::with_capacity(n);
    for (k, &x) in values.iter().enumerate() {
        let masked = mask_in.is_some_and(|m| m[k]) || !x.is_finite();
        mask[k] = masked;
        if !masked {
            pairs.push((x.to_f64(), k));
        }
    }
    let mut resid = Vec::<f64>::with_capacity(n);
    apply_pclip_pairs::<T>(
        &mut pairs,
        &mut mask,
        &mut resid,
        pclip_offset,
        sigma_lower,
        sigma_upper,
        nkeep,
        false,
    );
    let mut stat_buf = Vec::<f64>::with_capacity(n);
    final_combine_from_sorted_pairs_f64(&pairs, &mask, kind, &mut stat_buf)
}

pub fn pclip_mean_1d<T: Float>(
    values: &[T],
    mask_in: Option<&[bool]>,
    pclip: f64,
    sigma_lower: f64,
    sigma_upper: f64,
    nkeep: usize,
) -> T {
    pclip_final_combine_1d(
        values,
        mask_in,
        pclip,
        sigma_lower,
        sigma_upper,
        nkeep,
        FinalCombineKind::Mean,
    )
}

pub fn pclip_median_1d<T: Float>(
    values: &[T],
    mask_in: Option<&[bool]>,
    pclip: f64,
    sigma_lower: f64,
    sigma_upper: f64,
    nkeep: usize,
) -> T {
    pclip_final_combine_1d(
        values,
        mask_in,
        pclip,
        sigma_lower,
        sigma_upper,
        nkeep,
        FinalCombineKind::Median,
    )
}

#[cfg(test)]
mod tests {
    use super::{
        should_parallel_reject, RejectDType, RejectKind, RejectOutputMode,
        DEFAULT_REJECT_PARALLEL_HW_THRESHOLD,
    };

    #[test]
    fn rejection_parallel_policy_is_owned_by_imc() {
        assert!(!should_parallel_reject(
            RejectKind::SigClip,
            DEFAULT_REJECT_PARALLEL_HW_THRESHOLD - 1,
            31,
            RejectDType::F32,
            RejectOutputMode::FinalMedian,
        ));
        assert!(should_parallel_reject(
            RejectKind::SigClip,
            DEFAULT_REJECT_PARALLEL_HW_THRESHOLD,
            31,
            RejectDType::F32,
            RejectOutputMode::FinalMedian,
        ));
    }

    #[test]
    fn rejection_parallel_policy_keeps_empty_work_serial() {
        assert!(!should_parallel_reject(
            RejectKind::MinMax,
            DEFAULT_REJECT_PARALLEL_HW_THRESHOLD,
            0,
            RejectDType::F64,
            RejectOutputMode::MaskOnly,
        ));
        assert!(!should_parallel_reject(
            RejectKind::PClip,
            0,
            31,
            RejectDType::F32,
            RejectOutputMode::Diagnostics,
        ));
    }
}

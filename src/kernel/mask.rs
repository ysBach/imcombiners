//! Mask operations shared by rejection pipelines.

use ndarray::{ArrayD, ArrayViewD, IxDyn};
use rayon::prelude::*;

fn spatial_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

fn unravel_index(mut idx: usize, strides: &[usize], shape: &[usize], coords: &mut [usize]) {
    for ((coord, &stride), &dim) in coords.iter_mut().zip(strides).zip(shape) {
        *coord = idx / stride;
        idx %= stride;
        debug_assert!(*coord < dim);
    }
}

fn build_offsets(ndim: usize, radius: f64) -> Vec<Vec<isize>> {
    let max_delta = radius.floor() as isize;
    let radius2 = radius * radius;
    let mut offsets = Vec::new();
    let mut current = vec![0isize; ndim];

    fn visit(
        axis: usize,
        max_delta: isize,
        radius2: f64,
        current: &mut [isize],
        offsets: &mut Vec<Vec<isize>>,
    ) {
        if axis == current.len() {
            let dist2 = current
                .iter()
                .map(|&v| {
                    let vf = v as f64;
                    vf * vf
                })
                .sum::<f64>();
            if dist2 <= radius2 {
                offsets.push(current.to_vec());
            }
            return;
        }
        for delta in -max_delta..=max_delta {
            current[axis] = delta;
            visit(axis + 1, max_delta, radius2, current, offsets);
        }
    }

    visit(0, max_delta, radius2, &mut current, &mut offsets);
    offsets
}

/// Dilate a stack mask over its spatial axes with an exact Euclidean radius.
///
/// Axis 0 is the stack axis and is never grown across. For each input plane,
/// every `true` sample marks all spatial samples whose integer-grid Euclidean
/// distance is `<= grow`.
pub fn grow_mask(mask: &ArrayViewD<bool>, grow: f64) -> Result<ArrayD<bool>, String> {
    if mask.ndim() < 2 {
        return Err(format!(
            "grow mask must have shape (N, *spatial); got {:?}",
            mask.shape()
        ));
    }
    if !grow.is_finite() || grow < 0.0 {
        return Err("grow must be a finite non-negative radius".to_string());
    }

    let shape = mask.shape().to_vec();
    let spatial_shape = &shape[1..];
    let n = shape[0];
    let spatial_size = spatial_shape.iter().product::<usize>();
    let data = mask
        .as_slice_memory_order()
        .ok_or_else(|| "grow mask must be C-contiguous".to_string())?;

    if spatial_size == 0 || grow == 0.0 || !data.iter().any(|&v| v) {
        return Ok(mask.to_owned());
    }

    let max_dist2 = spatial_shape
        .iter()
        .map(|&dim| {
            let d = dim.saturating_sub(1) as f64;
            d * d
        })
        .sum::<f64>();

    let mut out = vec![false; data.len()];
    if grow * grow >= max_dist2 {
        out.par_chunks_mut(spatial_size)
            .enumerate()
            .for_each(|(frame, out_frame)| {
                let start = frame * spatial_size;
                let input_frame = &data[start..start + spatial_size];
                if input_frame.iter().any(|&v| v) {
                    out_frame.fill(true);
                }
            });
        return ArrayD::from_shape_vec(IxDyn(&shape), out).map_err(|err| err.to_string());
    }

    let strides = spatial_strides(spatial_shape);
    let offsets = build_offsets(spatial_shape.len(), grow);

    out.par_chunks_mut(spatial_size)
        .enumerate()
        .take(n)
        .for_each(|(frame, out_frame)| {
            let start = frame * spatial_size;
            let input_frame = &data[start..start + spatial_size];
            let mut coords = vec![0usize; spatial_shape.len()];

            for (src_idx, &is_rejected) in input_frame.iter().enumerate() {
                if !is_rejected {
                    continue;
                }
                unravel_index(src_idx, &strides, spatial_shape, &mut coords);
                for offset in &offsets {
                    let mut dst_idx = 0usize;
                    let mut in_bounds = true;
                    for axis in 0..spatial_shape.len() {
                        let coord = coords[axis] as isize + offset[axis];
                        if coord < 0 || coord >= spatial_shape[axis] as isize {
                            in_bounds = false;
                            break;
                        }
                        dst_idx += coord as usize * strides[axis];
                    }
                    if in_bounds {
                        out_frame[dst_idx] = true;
                    }
                }
            }
        });

    ArrayD::from_shape_vec(IxDyn(&shape), out).map_err(|err| err.to_string())
}

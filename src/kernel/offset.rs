//! Offset/padding helpers for array stacks.

use ndarray::{ArrayD, ArrayViewD, Dimension, IxDyn};

use super::utils::Float;

#[inline]
fn axis_label(axis: usize) -> String {
    match axis {
        0 => "y".to_string(),
        1 => "x".to_string(),
        _ => format!("axis {axis}"),
    }
}

fn normalized_offsets_and_shape(
    shapes: &[Vec<usize>],
    offsets: &[Vec<i64>],
) -> Result<(Vec<Vec<usize>>, Vec<usize>), String> {
    if shapes.is_empty() {
        return Err("images must contain at least one array".to_string());
    }
    if shapes.len() != offsets.len() {
        return Err(format!(
            "offsets length {} does not match image count {}",
            offsets.len(),
            shapes.len()
        ));
    }
    let ndim = shapes[0].len();
    if ndim == 0 {
        return Err("images must have at least 1 dimension".to_string());
    }
    for shape in shapes {
        if shape.len() != ndim {
            return Err("images must all have the same number of dimensions".to_string());
        }
    }
    for offset in offsets {
        if offset.len() != ndim {
            return Err(format!(
                "offsets shape must be ({}, {}); got width {}",
                shapes.len(),
                ndim,
                offset.len()
            ));
        }
    }

    let mins: Vec<i64> = (0..ndim)
        .map(|axis| offsets.iter().map(|offset| offset[axis]).min().unwrap())
        .collect();
    let mut starts = Vec::with_capacity(offsets.len());
    let mut out_shape = vec![0usize; ndim];

    for (shape, offset) in shapes.iter().zip(offsets.iter()) {
        let mut start = Vec::with_capacity(ndim);
        for axis in 0..ndim {
            let label = axis_label(axis);
            let normalized = offset[axis]
                .checked_sub(mins[axis])
                .ok_or_else(|| format!("normalized {label} offset overflow"))?;
            let pos = usize::try_from(normalized)
                .map_err(|_| format!("normalized {label} offset does not fit usize"))?;
            let stop = pos
                .checked_add(shape[axis])
                .ok_or_else(|| format!("padded output {label} overflow"))?;
            out_shape[axis] = out_shape[axis].max(stop);
            start.push(pos);
        }
        starts.push(start);
    }

    Ok((starts, out_shape))
}

pub fn place_into_padded<T: Float>(
    images: &[ArrayViewD<'_, T>],
    offsets: &[Vec<i64>],
    fill: T,
) -> Result<ArrayD<T>, String> {
    let shapes: Vec<Vec<usize>> = images.iter().map(|image| image.shape().to_vec()).collect();
    let (starts, outer_shape) = normalized_offsets_and_shape(&shapes, offsets)?;
    let mut out_shape = Vec::with_capacity(outer_shape.len() + 1);
    out_shape.push(images.len());
    out_shape.extend(outer_shape);
    let mut out = ArrayD::<T>::from_elem(IxDyn(&out_shape), fill);

    for (k, image) in images.iter().enumerate() {
        let start = &starts[k];
        for (idx, value) in image.indexed_iter() {
            let mut out_idx = Vec::with_capacity(idx.ndim() + 1);
            out_idx.push(k);
            for axis in 0..idx.ndim() {
                out_idx.push(start[axis] + idx[axis]);
            }
            out[IxDyn(&out_idx)] = *value;
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::place_into_padded;

    #[test]
    fn normalizes_negative_offsets_and_computes_outer_shape() {
        let img0 = array![[1.0_f32, 2.0], [3.0, 4.0]];
        let img1 = array![[10.0_f32, 11.0, 12.0]];
        let images = vec![img0.view().into_dyn(), img1.view().into_dyn()];
        let offsets = vec![vec![2_i64, -1_i64], vec![-1_i64, 1_i64]];

        let out = place_into_padded(&images, &offsets, f32::NAN).unwrap();

        assert_eq!(out.shape(), &[2, 5, 5]);
        assert_eq!(out[[0, 3, 0]], 1.0);
        assert_eq!(out[[0, 4, 1]], 4.0);
        assert_eq!(out[[1, 0, 2]], 10.0);
        assert_eq!(out[[1, 0, 4]], 12.0);
        assert!(out[[0, 0, 0]].is_nan());
    }
}

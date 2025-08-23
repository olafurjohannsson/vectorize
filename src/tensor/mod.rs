use std::sync::Arc;
use std::ops::{Range, Div, Mul, Add, Sub};

#[derive(Clone, Debug)]
pub struct Tensor {
    pub data: Arc<Vec<f32>>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
}

impl Tensor {
    pub fn from_vec(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let strides = Self::compute_strides(&shape);
        Self {
            data: Arc::new(data),
            shape,
            strides,
        }
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let size = shape.iter().product();
        Self::from_vec(vec![0.0; size], shape)
    }

    pub fn ones(shape: Vec<usize>) -> Self {
        let size = shape.iter().product();
        Self::from_vec(vec![1.0; size], shape)
    }

    pub fn zeros_like(other: &Tensor) -> Tensor {
        Self::zeros(other.shape.clone())
    }

    pub fn ones_like(other: &Tensor) -> Tensor {
        Self::ones(other.shape.clone())
    }

    pub fn arange(start: i64, end: i64) -> Tensor {
        let data: Vec<f32> = (start..end).map(|i| i as f32).collect();
        let shape = vec![(end - start) as usize];
        Self::from_vec(data, shape)
    }

    pub fn full(shape: Vec<usize>, value: f32) -> Self {
        let size = shape.iter().product();
        Self::from_vec(vec![value; size], shape)
    }

    pub fn reshape(&self, new_shape: &[usize]) -> Tensor {
        // Verify the total size matches
        let total_size: usize = self.shape.iter().product();
        let new_size: usize = new_shape.iter().product();
        assert_eq!(total_size, new_size, "Cannot reshape tensor of size {} to size {}", total_size, new_size);

        Self::from_vec(self.data.to_vec(), new_shape.to_vec())
    }

    pub fn transpose(&self, dim0: i32, dim1: i32) -> Tensor {
        let ndim = self.shape.len() as i32;
        let dim0 = if dim0 < 0 { ndim + dim0 } else { dim0 } as usize;
        let dim1 = if dim1 < 0 { ndim + dim1 } else { dim1 } as usize;

        let mut new_shape = self.shape.clone();
        new_shape.swap(dim0, dim1);

        let mut new_strides = self.strides.clone();
        new_strides.swap(dim0, dim1);

        // 2D tensors
        if self.shape.len() == 2 {
            let rows = self.shape[0];
            let cols = self.shape[1];
            let mut result = Vec::with_capacity(rows * cols);

            for j in 0..cols {
                for i in 0..rows {
                    result.push(self.data[i * cols + j]);
                }
            }
            return Self::from_vec(result, vec![cols, rows]);
        }

        let mut result = vec![0.0; self.data.len()];
        for i in 0..self.data.len() {
            let old_idx = self.unravel_index(i);
            let mut new_idx = old_idx.clone();
            new_idx.swap(dim0, dim1);
            let new_flat_idx = self.ravel_index(&new_idx, &new_shape, &Self::compute_strides(&new_shape));
            result[new_flat_idx] = self.data[i];
        }

        Self::from_vec(result, new_shape)
    }

    pub fn unsqueeze(&self, dim: i32) -> Tensor {
        let mut new_shape = self.shape.clone();
        let ndim = self.shape.len() as i32;
        let dim = if dim < 0 { ndim + dim + 1 } else { dim } as usize;
        new_shape.insert(dim, 1);
        self.reshape(&new_shape)
    }

    pub fn squeeze(&self, dim: Option<i32>) -> Tensor {
        let new_shape = if let Some(d) = dim {
            let d = if d < 0 { self.shape.len() as i32 + d } else { d } as usize;
            let mut shape = self.shape.clone();
            if shape[d] == 1 {
                shape.remove(d);
            }
            shape
        } else {
            self.shape.iter().filter(|&&s| s != 1).copied().collect()
        };
        self.reshape(&new_shape)
    }

    pub fn expand_as(&self, other: &Tensor) -> Tensor {
        self.broadcast_to(&other.shape)
    }

    pub fn broadcast_to(&self, target_shape: &[usize]) -> Tensor {
        assert_eq!(self.shape.len(), target_shape.len(), "Shapes must have same number of dimensions");

        let mut result = self.clone();
        for (i, (&src_dim, &tgt_dim)) in self.shape.iter().zip(target_shape.iter()).enumerate() {
            if src_dim != tgt_dim {
                assert_eq!(src_dim, 1, "Can only broadcast dimension of size 1");
                // Repeat along this dimension
                let mut new_data = Vec::with_capacity(result.data.len() * tgt_dim);
                let outer_size: usize = result.shape[..i].iter().product();
                let inner_size: usize = result.shape[i+1..].iter().product();

                for outer in 0..outer_size {
                    for _ in 0..tgt_dim {
                        for inner in 0..inner_size {
                            let idx = outer * inner_size + inner;
                            new_data.push(result.data[idx]);
                        }
                    }
                }

                let mut new_shape = result.shape.clone();
                new_shape[i] = tgt_dim;
                result = Self::from_vec(new_data, new_shape);
            }
        }
        result
    }

    pub fn matmul(&self, other: &Tensor) -> Tensor {
        // different dimensionalities
        match (self.shape.len(), other.shape.len()) {
            (2, 2) => self.matmul_2d(other),
            (3, 2) => {
                // Batch matmul: [batch, m, k] @ [k, n] -> [batch, m, n]
                let batch = self.shape[0];
                let m = self.shape[1];
                let k = self.shape[2];
                let n = other.shape[1];
                assert_eq!(k, other.shape[0]);

                let mut result = Vec::with_capacity(batch * m * n);
                for b in 0..batch {
                    for i in 0..m {
                        for j in 0..n {
                            let mut sum = 0.0;
                            for l in 0..k {
                                sum += self.data[b * m * k + i * k + l] * other.data[l * n + j];
                            }
                            result.push(sum);
                        }
                    }
                }
                Self::from_vec(result, vec![batch, m, n])
            },
            (4, 4) => {
                // 4D matmul for attention: [batch, heads, seq, dim] @ [batch, heads, dim, seq]
                assert_eq!(self.shape[0], other.shape[0]); // batch
                assert_eq!(self.shape[1], other.shape[1]); // heads
                assert_eq!(self.shape[3], other.shape[2]); // dim

                let batch = self.shape[0];
                let heads = self.shape[1];
                let seq1 = self.shape[2];
                let dim = self.shape[3];
                let seq2 = other.shape[3];

                let mut result = Vec::with_capacity(batch * heads * seq1 * seq2);
                for b in 0..batch {
                    for h in 0..heads {
                        for i in 0..seq1 {
                            for j in 0..seq2 {
                                let mut sum = 0.0;
                                for k in 0..dim {
                                    let idx1 = b * heads * seq1 * dim + h * seq1 * dim + i * dim + k;
                                    let idx2 = b * heads * dim * seq2 + h * dim * seq2 + k * seq2 + j;
                                    sum += self.data[idx1] * other.data[idx2];
                                }
                                result.push(sum);
                            }
                        }
                    }
                }
                Self::from_vec(result, vec![batch, heads, seq1, seq2])
            },
            _ => panic!("Unsupported matmul dimensions: {:?} @ {:?}", self.shape, other.shape)
        }
    }

    fn matmul_2d(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape.len(), 2);
        assert_eq!(other.shape.len(), 2);
        assert_eq!(self.shape[1], other.shape[0]);

        let m = self.shape[0];
        let n = other.shape[1];
        let k = self.shape[1];

        let mut result = vec![0.0; m * n];

        // matrix multiplication
        for i in 0..m {
            for k_idx in 0..k {
                let a_val = self.data[i * k + k_idx];
                for j in 0..n {
                    result[i * n + j] += a_val * other.data[k_idx * n + j];
                }
            }
        }

        Self::from_vec(result, vec![m, n])
    }

    pub fn add(&self, other: &Tensor) -> Tensor {
        // Handle broadcasting
        if self.shape == other.shape {
            let data: Vec<f32> = self.data.iter()
                .zip(other.data.iter())
                .map(|(&a, &b)| a + b)
                .collect();
            Self::from_vec(data, self.shape.clone())
        } else {
            // broadcasting for bias addition
            if other.shape.len() == 1 && other.shape[0] == self.shape[self.shape.len() - 1] {
                let mut result = Vec::with_capacity(self.data.len());
                let last_dim = self.shape[self.shape.len() - 1];
                for (i, &val) in self.data.iter().enumerate() {
                    result.push(val + other.data[i % last_dim]);
                }
                Self::from_vec(result, self.shape.clone())
            } else {
                let broadcasted = other.broadcast_to(&self.shape);
                self.add(&broadcasted)
            }
        }
    }

    pub fn sub(&self, other: &Tensor) -> Tensor {
        if self.shape == other.shape {
            let data: Vec<f32> = self.data.iter()
                .zip(other.data.iter())
                .map(|(&a, &b)| a - b)
                .collect();
            Self::from_vec(data, self.shape.clone())
        } else {
            let broadcasted = other.broadcast_to(&self.shape);
            self.sub(&broadcasted)
        }
    }

    pub fn mul(&self, other: &Tensor) -> Tensor {
        if self.shape == other.shape {
            let data: Vec<f32> = self.data.iter()
                .zip(other.data.iter())
                .map(|(&a, &b)| a * b)
                .collect();
            Self::from_vec(data, self.shape.clone())
        } else {
            let broadcasted = other.broadcast_to(&self.shape);
            self.mul(&broadcasted)
        }
    }

    pub fn div(&self, other: &Tensor) -> Tensor {
        if self.shape == other.shape {
            let data: Vec<f32> = self.data.iter()
                .zip(other.data.iter())
                .map(|(&a, &b)| a / b)
                .collect();
            Self::from_vec(data, self.shape.clone())
        } else {
            let broadcasted = other.broadcast_to(&self.shape);
            self.div(&broadcasted)
        }
    }

    pub fn add_scalar(&self, scalar: f32) -> Tensor {
        let data: Vec<f32> = self.data.iter().map(|&x| x + scalar).collect();
        Self::from_vec(data, self.shape.clone())
    }

    pub fn sub_scalar(&self, scalar: f32) -> Tensor {
        let data: Vec<f32> = self.data.iter().map(|&x| x - scalar).collect();
        Self::from_vec(data, self.shape.clone())
    }

    pub fn mul_scalar(&self, scalar: f32) -> Tensor {
        let data: Vec<f32> = self.data.iter().map(|&x| x * scalar).collect();
        Self::from_vec(data, self.shape.clone())
    }

    pub fn div_scalar(&self, scalar: f32) -> Tensor {
        let data: Vec<f32> = self.data.iter().map(|&x| x / scalar).collect();
        Self::from_vec(data, self.shape.clone())
    }

    pub fn relu(&self) -> Tensor {
        let data: Vec<f32> = self.data.iter().map(|&x| x.max(0.0)).collect();
        Self::from_vec(data, self.shape.clone())
    }

    pub fn gelu(&self) -> Tensor {
        // GELU approximation: x * 0.5 * (1.0 + tanh(sqrt(2.0 / π) * (x + 0.044715 * x^3)))
        let data: Vec<f32> = self.data.iter().map(|&x| {
            let inner = (2.0_f32 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x.powi(3));
            x * 0.5 * (1.0 + inner.tanh())
        }).collect();
        Self::from_vec(data, self.shape.clone())
    }

    pub fn tanh(&self) -> Tensor {
        let data: Vec<f32> = self.data.iter().map(|&x| x.tanh()).collect();
        Self::from_vec(data, self.shape.clone())
    }

    pub fn sigmoid(&self) -> Tensor {
        let data: Vec<f32> = self.data.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();
        Self::from_vec(data, self.shape.clone())
    }

    pub fn softmax(&self, dim: i32) -> Tensor {
        let dim = if dim < 0 { self.shape.len() as i32 + dim } else { dim } as usize;

        // 2D tensors
        if self.shape.len() == 2 && dim == 1 {
            let rows = self.shape[0];
            let cols = self.shape[1];
            let mut result = Vec::with_capacity(self.data.len());

            for i in 0..rows {
                let row_start = i * cols;
                let row_end = row_start + cols;
                let row = &self.data[row_start..row_end];

                // Find max for numerical stability
                let max = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

                // Compute exp and sum
                let mut exp_vals: Vec<f32> = row.iter().map(|&x| (x - max).exp()).collect();
                let sum: f32 = exp_vals.iter().sum();

                // normalize
                exp_vals.iter_mut().for_each(|x| *x /= sum);
                result.extend(exp_vals);
            }

            return Self::from_vec(result, self.shape.clone());
        }

        let dim_size = self.shape[dim];
        let outer_size: usize = self.shape[..dim].iter().product();
        let inner_size: usize = self.shape[dim+1..].iter().product();

        let mut result = vec![0.0; self.data.len()];

        for outer in 0..outer_size {
            for inner in 0..inner_size {
                // max along dimension
                let mut max = f32::NEG_INFINITY;
                for i in 0..dim_size {
                    let idx = outer * dim_size * inner_size + i * inner_size + inner;
                    max = max.max(self.data[idx]);
                }

                // Compute exp and sum
                let mut sum = 0.0;
                for i in 0..dim_size {
                    let idx = outer * dim_size * inner_size + i * inner_size + inner;
                    let exp_val = (self.data[idx] - max).exp();
                    result[idx] = exp_val;
                    sum += exp_val;
                }

                // Normalize
                for i in 0..dim_size {
                    let idx = outer * dim_size * inner_size + i * inner_size + inner;
                    result[idx] /= sum;
                }
            }
        }

        Self::from_vec(result, self.shape.clone())
    }

    pub fn layer_norm(&self, normalized_shape: &[usize], eps: f32) -> Tensor {
        // Assumes normalization over last dimensions
        let norm_size: usize = normalized_shape.iter().product();
        let batch_size = self.data.len() / norm_size;

        let mut result = Vec::with_capacity(self.data.len());

        for b in 0..batch_size {
            let start = b * norm_size;
            let end = start + norm_size;
            let slice = &self.data[start..end];

            // compute mean and variance
            let mean: f32 = slice.iter().sum::<f32>() / norm_size as f32;
            let variance: f32 = slice.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f32>() / norm_size as f32;

            // Normalize
            let std = (variance + eps).sqrt();
            for &x in slice {
                result.push((x - mean) / std);
            }
        }

        Self::from_vec(result, self.shape.clone())
    }

    pub fn rms_norm(&self, eps: f32) -> Tensor {
        let norm_size = self.shape[self.shape.len() - 1];
        let batch_size = self.data.len() / norm_size;

        let mut result = Vec::with_capacity(self.data.len());

        for b in 0..batch_size {
            let start = b * norm_size;
            let end = start + norm_size;
            let slice = &self.data[start..end];

            // Compute RMS
            let rms: f32 = (slice.iter().map(|&x| x * x).sum::<f32>() / norm_size as f32).sqrt();

            // Normalize
            for &x in slice {
                result.push(x / (rms + eps));
            }
        }

        Self::from_vec(result, self.shape.clone())
    }

    pub fn sum(&self, dims: &[i32], keepdim: bool) -> Tensor {
        if dims.is_empty() {
            // Sum all elements
            let sum: f32 = self.data.iter().sum();
            return Self::from_vec(vec![sum], vec![1]);
        }

        let mut result = self.clone();
        for &dim in dims {
            result = result.sum_single_dim(dim, keepdim);
        }
        result
    }

    fn sum_single_dim(&self, dim: i32, keepdim: bool) -> Tensor {
        let dim = if dim < 0 { self.shape.len() as i32 + dim } else { dim } as usize;

        let dim_size = self.shape[dim];
        let outer_size: usize = self.shape[..dim].iter().product();
        let inner_size: usize = self.shape[dim+1..].iter().product();

        let mut result = Vec::with_capacity(outer_size * inner_size);

        for outer in 0..outer_size {
            for inner in 0..inner_size {
                let mut sum = 0.0;
                for i in 0..dim_size {
                    let idx = outer * dim_size * inner_size + i * inner_size + inner;
                    sum += self.data[idx];
                }
                result.push(sum);
            }
        }

        let mut new_shape = self.shape.clone();
        if keepdim {
            new_shape[dim] = 1;
        } else {
            new_shape.remove(dim);
        }

        Self::from_vec(result, new_shape)
    }

    pub fn mean(&self, dims: &[i32], keepdim: bool) -> Tensor {
        let sum = self.sum(dims, keepdim);
        let count = dims.iter().fold(1, |acc, &d| {
            let dim = if d < 0 { self.shape.len() as i32 + d } else { d } as usize;
            acc * self.shape[dim]
        });
        sum.div_scalar(count as f32)
    }

    pub fn max(&self, dim: Option<i32>, keepdim: bool) -> Tensor {
        if let Some(d) = dim {
            self.max_single_dim(d, keepdim)
        } else {
            // Global max
            let max = self.data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            Self::from_vec(vec![max], vec![1])
        }
    }

    fn max_single_dim(&self, dim: i32, keepdim: bool) -> Tensor {
        let dim = if dim < 0 { self.shape.len() as i32 + dim } else { dim } as usize;

        let dim_size = self.shape[dim];
        let outer_size: usize = self.shape[..dim].iter().product();
        let inner_size: usize = self.shape[dim+1..].iter().product();

        let mut result = Vec::with_capacity(outer_size * inner_size);

        for outer in 0..outer_size {
            for inner in 0..inner_size {
                let mut max = f32::NEG_INFINITY;
                for i in 0..dim_size {
                    let idx = outer * dim_size * inner_size + i * inner_size + inner;
                    max = max.max(self.data[idx]);
                }
                result.push(max);
            }
        }

        let mut new_shape = self.shape.clone();
        if keepdim {
            new_shape[dim] = 1;
        } else {
            new_shape.remove(dim);
        }

        Self::from_vec(result, new_shape)
    }

    pub fn pow(&self, exponent: f32) -> Tensor {
        let data: Vec<f32> = self.data.iter().map(|&x| x.powf(exponent)).collect();
        Self::from_vec(data, self.shape.clone())
    }

    pub fn sqrt(&self) -> Tensor {
        let data: Vec<f32> = self.data.iter().map(|&x| x.sqrt()).collect();
        Self::from_vec(data, self.shape.clone())
    }

    pub fn exp(&self) -> Tensor {
        let data: Vec<f32> = self.data.iter().map(|&x| x.exp()).collect();
        Self::from_vec(data, self.shape.clone())
    }

    pub fn log(&self) -> Tensor {
        let data: Vec<f32> = self.data.iter().map(|&x| x.ln()).collect();
        Self::from_vec(data, self.shape.clone())
    }

    pub fn abs(&self) -> Tensor {
        let data: Vec<f32> = self.data.iter().map(|&x| x.abs()).collect();
        Self::from_vec(data, self.shape.clone())
    }

    pub fn clamp(&self, min: f32, max: f32) -> Tensor {
        let data: Vec<f32> = self.data.iter().map(|&x| x.clamp(min, max)).collect();
        Self::from_vec(data, self.shape.clone())
    }

    pub fn clamp_min(&self, min: f32) -> Tensor {
        let data: Vec<f32> = self.data.iter().map(|&x| x.max(min)).collect();
        Self::from_vec(data, self.shape.clone())
    }

    pub fn gather(&self, dim: usize, indices: &Tensor) -> Tensor {
        // assumes dim=0 and 2D weight matrix
        if dim == 0 && self.shape.len() == 2 {
            let mut output_data = Vec::new();
            for &idx in indices.data.iter() {
                let idx = idx as usize;
                let start = idx * self.shape[1];
                let end = start + self.shape[1];
                output_data.extend_from_slice(&self.data[start..end]);
            }

            let mut output_shape = indices.shape.clone();
            output_shape.push(self.shape[1]);
            return Self::from_vec(output_data, output_shape);
        }

        panic!("General gather  implemented");
    }

    pub fn slice(&self, ranges: &[Range<usize>]) -> Tensor {
        assert_eq!(ranges.len(), self.shape.len());

        let mut new_shape = Vec::new();
        for (i, range) in ranges.iter().enumerate() {
            new_shape.push(range.end - range.start);
        }

        let mut result = Vec::new();
        self.slice_recursive(&mut result, &self.data, &self.shape, &self.strides, ranges, 0, 0);

        Self::from_vec(result, new_shape)
    }

    fn slice_recursive(&self, result: &mut Vec<f32>, data: &[f32], shape: &[usize],
                       strides: &[usize], ranges: &[Range<usize>], dim: usize, offset: usize) {
        if dim == shape.len() {
            result.push(data[offset]);
            return;
        }

        for i in ranges[dim].clone() {
            let new_offset = offset + i * strides[dim];
            self.slice_recursive(result, data, shape, strides, ranges, dim + 1, new_offset);
        }
    }

    ///
    /// Utils
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn to_vec_2d(&self) -> Vec<Vec<f32>> {
        assert_eq!(self.shape.len(), 2, "Tensor must be 2D");
        let rows = self.shape[0];
        let cols = self.shape[1];

        let mut result = Vec::with_capacity(rows);
        for i in 0..rows {
            let start = i * cols;
            let end = start + cols;
            result.push(self.data[start..end].to_vec());
        }
        result
    }

    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    fn unravel_index(&self, index: usize) -> Vec<usize> {
        let mut idx = index;
        let mut result = vec![0; self.shape.len()];
        for i in 0..self.shape.len() {
            result[i] = idx / self.strides[i];
            idx %= self.strides[i];
        }
        result
    }

    fn ravel_index(&self, indices: &[usize], shape: &[usize], strides: &[usize]) -> usize {
        indices.iter().zip(strides.iter()).map(|(i, s)| i * s).sum()
    }
}

/// Ops
impl Div<f32> for Tensor {
    type Output = Tensor;
    fn div(self, scalar: f32) -> Tensor {
        self.div_scalar(scalar)
    }
}

impl Div<f32> for &Tensor {
    type Output = Tensor;
    fn div(self, scalar: f32) -> Tensor {
        self.div_scalar(scalar)
    }
}

pub trait Attention {
    fn scaled_dot_product_attention(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>
    ) -> Tensor;
}
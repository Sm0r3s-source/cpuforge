//! A deliberately small row-major f32 tensor. No autograd: backward passes are
//! written by hand per op, which is simpler and faster for fixed architectures.

use crate::rng::Rng;

#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}

impl Tensor {
    pub fn zeros(shape: &[usize]) -> Self {
        let n = shape.iter().product();
        Self { data: vec![0.0; n], shape: shape.to_vec() }
    }

    pub fn from_vec(data: Vec<f32>, shape: &[usize]) -> Self {
        assert_eq!(
            data.len(),
            shape.iter().product::<usize>(),
            "data length {} does not match shape {:?}",
            data.len(),
            shape
        );
        Self { data, shape: shape.to_vec() }
    }

    pub fn randn(shape: &[usize], std: f32, rng: &mut Rng) -> Self {
        let mut t = Self::zeros(shape);
        rng.fill_normal(&mut t.data, std);
        t
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Rows of a 2-D tensor.
    pub fn rows(&self) -> usize {
        assert_eq!(self.shape.len(), 2, "expected 2-D tensor, got {:?}", self.shape);
        self.shape[0]
    }

    /// Columns of a 2-D tensor.
    pub fn cols(&self) -> usize {
        assert_eq!(self.shape.len(), 2, "expected 2-D tensor, got {:?}", self.shape);
        self.shape[1]
    }

    pub fn row(&self, r: usize) -> &[f32] {
        let c = self.cols();
        &self.data[r * c..(r + 1) * c]
    }

    pub fn row_mut(&mut self, r: usize) -> &mut [f32] {
        let c = self.cols();
        &mut self.data[r * c..(r + 1) * c]
    }

    pub fn fill(&mut self, v: f32) {
        self.data.fill(v);
    }

    /// self += other (same shape).
    pub fn add_assign(&mut self, other: &Tensor) {
        assert_eq!(self.shape, other.shape);
        for (a, b) in self.data.iter_mut().zip(&other.data) {
            *a += *b;
        }
    }

    /// self += alpha * other (same shape).
    pub fn axpy(&mut self, alpha: f32, other: &Tensor) {
        assert_eq!(self.shape, other.shape);
        for (a, b) in self.data.iter_mut().zip(&other.data) {
            *a += alpha * *b;
        }
    }

    pub fn scale(&mut self, alpha: f32) {
        for a in self.data.iter_mut() {
            *a *= alpha;
        }
    }

    pub fn max_abs_diff(&self, other: &Tensor) -> f32 {
        assert_eq!(self.shape, other.shape);
        self.data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shapes_and_rows() {
        let t = Tensor::from_vec((0..6).map(|x| x as f32).collect(), &[2, 3]);
        assert_eq!(t.rows(), 2);
        assert_eq!(t.cols(), 3);
        assert_eq!(t.row(1), &[3.0, 4.0, 5.0]);
    }

    #[test]
    #[should_panic]
    fn from_vec_checks_length() {
        let _ = Tensor::from_vec(vec![1.0, 2.0], &[3]);
    }

    #[test]
    fn axpy_works() {
        let mut a = Tensor::from_vec(vec![1.0, 2.0], &[2]);
        let b = Tensor::from_vec(vec![10.0, 20.0], &[2]);
        a.axpy(0.5, &b);
        assert_eq!(a.data(), &[6.0, 12.0]);
    }
}

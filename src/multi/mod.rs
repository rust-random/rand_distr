//! Contains Multi-dimensional distributions.
//!
//! We provide a trait `MultiDistribution` which allows to sample from a multi-dimensional distribution without extra allocations.
//! All multi-dimensional distributions implement `MultiDistribution` instead of the `Distribution` trait.

use alloc::vec::Vec;
use rand::Rng;

/// This trait allows to sample from a multi-dimensional distribution without extra allocations.
/// For convenience it also provides a `sample` method which returns the result as a `Vec`.
pub trait MultiDistribution<T> {
    /// returns the length of one sample (dimension of the distribution)
    fn sample_len(&self) -> usize;
    /// samples from the distribution and writes the result to `buf`
    fn sample_to_buf<R: Rng + ?Sized>(&self, rng: &mut R, buf: &mut [T]);
    /// samples from the distribution and returns the result as a `Vec`, to avoid extra allocations use `sample_to_buf`
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Vec<T>
    where
        T: Default,
    {
        let mut buf = Vec::new();
        buf.resize_with(self.sample_len(), || T::default());
        self.sample_to_buf(rng, &mut buf);
        buf
    }
}

pub use dirichlet::Dirichlet;

mod dirichlet;

//! Contains Multi-dimensional distributions.
//!
//! We provide a trait `MultiDistribution` which allows to sample from a multi-dimensional distribution without extra allocations.
//! All multi-dimensional distributions implement `MultiDistribution` instead of the `Distribution` trait.

use rand::Rng;

/// This trait allows to sample from a multi-dimensional distribution without extra allocations.
pub trait MultiDistribution<T> {
    /// returns the length of one sample (dimension of the distribution)
    fn sample_len(&self) -> usize;
    /// samples from the distribution and writes the result to `output`
    fn sample_to_slice<R: Rng + ?Sized>(&self, rng: &mut R, output: &mut [T]);
}


macro_rules! distribution_impl {
    ($scalar:ident) => {
        fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Vec<$scalar> {
                use crate::multi::MultiDistribution;
                let mut buf = vec![Default::default(); self.sample_len()];
                self.sample_to_slice(rng, &mut buf);
                buf
            }
    };
}

pub use dirichlet::Dirichlet;

mod dirichlet;

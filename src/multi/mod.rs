//! Contains Multi-dimensional distributions.
//!
//! We provide a trait `MultiDistribution` which allows to sample from a multi-dimensional distribution without extra allocations.
//! All multi-dimensional distributions should implement this trait addidionally to the `Distribution` trait returning a `Vec` of samples.

use rand::Rng;

/// This trait allows to sample from a multi-dimensional distribution without extra allocations.
/// Typically distributions will implement `MultiDistribution<[F]>` where `F` is the type of the samples.
pub trait MultiDistribution<S: ?Sized> {
    /// Sample from the distribution using the given random number generator and write the result to `output`.
    /// The method panics if the buffer is too small to hold the samples.
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R, output: &mut S);
}

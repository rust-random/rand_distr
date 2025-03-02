//! Contains Multi-dimensional distributions.
//! 
//! We provide a trait `MultiDistribution` which allows to sample from a multi-dimensional distribution without extra allocations.
//! All multi-dimensional distributions should implement this trait addidionally to the `Distribution` trait returning a `Vec` of samples.

/// This trait allows to sample from a multi-dimensional distribution without extra allocations.
/// Typically distributions will implement `MultiDistribution<[F]>` where `F` is the type of the samples.
pub trait MultiDistribution<S> {
    fn sample<R : Rng + ?Sized, S : ?Sized>(&self, rng: &mut R, output: &mut S);
}
#[allow(unused_imports)]
use num_traits::Float;
use rand::{Rng, RngExt, distr::Distribution};

/// The [truncated normal distribution](https://en.wikipedia.org/wiki/Truncated_normal_distribution).
///
/// # Current Implementation
/// We follow the approach described in
/// Robert, Christian P. (1995). "Simulation of truncated normal variables".
/// Statistics and Computing. 5 (2): 121–125.

#[derive(Debug)]
pub struct NormalTruncated(Method);

#[derive(Debug)]
enum Method {
    Rejection(NormalTruncatedRejection),
    OneSided(bool, NormalTruncatedOneSided), // bool indicates if lower bound is used
    TailInterval(bool, NormalTruncatedTailInterval), // bool indicates mirrored upper-tail proposal
    TwoSided(NormalTruncatedTwoSided),
}

#[derive(Debug)]
/// Errors that can occur when constructing a `NormalTruncated` distribution.
pub enum Error {
    /// The standard deviation was not positive.
    InvalidStdDev,
    /// The lower bound was not less than the upper bound.
    InvalidBounds,
}

impl NormalTruncated {
    /// Constructs a new `NormalTruncated` distribution with the given
    /// mean, standard deviation, lower bound, and upper bound.
    pub fn new(mean: f64, stddev: f64, lower: f64, upper: f64) -> Result<Self, Error> {
        if !(stddev > 0.0) {
            return Err(Error::InvalidStdDev);
        }
        if !(lower < upper) {
            return Err(Error::InvalidBounds);
        }

        let std_lower = (lower - mean) / stddev;
        let std_upper = (upper - mean) / stddev;

        if upper == f64::INFINITY {
            // This threshold depends on how fast normal vs exponential sampling is. This value was found empirically, but it can probably be tuned better.
            if std_lower > 0.3 {
                // One sided truncation, lower bound only
                Ok(NormalTruncated(Method::OneSided(
                    true,
                    NormalTruncatedOneSided::new(mean, stddev, std_lower),
                )))
            } else {
                // We use naive rejection sampling
                // Also catches the case where both bounds are infinite
                Ok(NormalTruncated(Method::Rejection(
                    NormalTruncatedRejection {
                        normal: crate::Normal::new(mean, stddev).unwrap(),
                        lower,
                        upper,
                    },
                )))
            }
        } else if lower == f64::NEG_INFINITY {
            // This threshold depends on how fast normal vs exponential sampling is. This value was found empirically, but it can probably be tuned better.
            if std_upper < -0.3 {
                // One sided truncation, upper bound only
                Ok(NormalTruncated(Method::OneSided(
                    false,
                    NormalTruncatedOneSided::new(-mean, stddev, -std_upper),
                )))
            } else {
                // We use naive rejection sampling
                Ok(NormalTruncated(Method::Rejection(
                    NormalTruncatedRejection {
                        normal: crate::Normal::new(mean, stddev).unwrap(),
                        lower,
                        upper,
                    },
                )))
            }
        } else {
            // Two sided truncation
            let diff = std_upper - std_lower;
            // Threshold can probably be tuned better for performance
            if diff >= 1.0 && std_lower <= 0.3 && std_upper >= -0.3 {
                // Naive rejection sampling
                Ok(NormalTruncated(Method::Rejection(
                    NormalTruncatedRejection {
                        normal: crate::Normal::new(mean, stddev).unwrap(),
                        lower,
                        upper,
                    },
                )))
            } else if std_lower >= 0.5 && diff >= 1.0 {
                // Two sided truncation in the upper tail.
                // Use the one-sided sampler as a proposal and reject past the upper bound.
                Ok(NormalTruncated(Method::TailInterval(
                    false,
                    NormalTruncatedTailInterval::new(
                        NormalTruncatedOneSided::new(mean, stddev, std_lower),
                        upper,
                    ),
                )))
            } else if std_upper <= -0.5 && diff >= 1.0 {
                // Mirror the lower-tail case to reuse the same one-sided sampler.
                Ok(NormalTruncated(Method::TailInterval(
                    true,
                    NormalTruncatedTailInterval::new(
                        NormalTruncatedOneSided::new(-mean, stddev, -std_upper),
                        -lower,
                    ),
                )))
            } else {
                // Two sided truncation
                Ok(NormalTruncated(Method::TwoSided(
                    NormalTruncatedTwoSided::new(mean, stddev, std_lower, std_upper),
                )))
            }
        }
    }
}

impl Distribution<f64> for NormalTruncated {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        match &self.0 {
            Method::Rejection(rej) => rej.sample(rng),
            Method::OneSided(true, one_sided) => one_sided.sample(rng),
            Method::OneSided(false, one_sided) => -one_sided.sample(rng),
            Method::TailInterval(false, tail_interval) => tail_interval.sample(rng),
            Method::TailInterval(true, tail_interval) => -tail_interval.sample(rng),
            Method::TwoSided(two_sided) => two_sided.sample(rng),
        }
    }
}

/// A truncated normal distribution using naive rejection sampling.
/// We use this when the acceptance rate is high enough.
#[derive(Debug)]
struct NormalTruncatedRejection {
    normal: crate::Normal<f64>,
    lower: f64,
    upper: f64,
}

impl Distribution<f64> for NormalTruncatedRejection {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        let mut sample;
        loop {
            sample = self.normal.sample(rng);
            if sample >= self.lower && sample <= self.upper {
                break;
            }
        }
        sample
    }
}

#[derive(Debug)]
struct NormalTruncatedOneSided {
    alpha_star: f64,
    lower_bound: f64,
    exp_distribution: crate::Exp<f64>,
    mu: f64,
    sigma: f64,
}

impl NormalTruncatedOneSided {
    fn new(mu: f64, sigma: f64, standard_lower_bound: f64) -> Self {
        let alpha_star = (standard_lower_bound + (standard_lower_bound.powi(2) + 4.0).sqrt()) / 2.0;
        let lambda = alpha_star;
        NormalTruncatedOneSided {
            alpha_star,
            lower_bound: standard_lower_bound,
            exp_distribution: crate::Exp::new(lambda).unwrap(),
            mu,
            sigma,
        }
    }
}

impl Distribution<f64> for NormalTruncatedOneSided {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        loop {
            let z = self.exp_distribution.sample(rng) + self.lower_bound;
            let u: f64 = rng.random();
            let rho = (-0.5 * (z - self.alpha_star).powi(2)).exp();
            if u <= rho {
                return self.mu + self.sigma * z;
            }
        }
    }
}

#[derive(Debug)]
struct NormalTruncatedTailInterval {
    proposal: NormalTruncatedOneSided,
    upper_bound: f64,
}

impl NormalTruncatedTailInterval {
    fn new(proposal: NormalTruncatedOneSided, upper_bound: f64) -> Self {
        NormalTruncatedTailInterval {
            proposal,
            upper_bound,
        }
    }
}

impl Distribution<f64> for NormalTruncatedTailInterval {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        loop {
            let sample = self.proposal.sample(rng);
            if sample <= self.upper_bound {
                return sample;
            }
        }
    }
}

#[derive(Debug)]
struct NormalTruncatedTwoSided {
    mu: f64,
    sigma: f64,
    // In standard normal coordinates
    standard_lower: f64,
    // In standard normal coordinates
    standard_upper: f64,
}

impl NormalTruncatedTwoSided {
    fn new(mu: f64, sigma: f64, standard_lower: f64, standard_upper: f64) -> Self {
        NormalTruncatedTwoSided {
            mu,
            sigma,
            standard_lower,
            standard_upper,
        }
    }
}

impl Distribution<f64> for NormalTruncatedTwoSided {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        loop {
            let z = rng.random_range(self.standard_lower..self.standard_upper);
            let u: f64 = rng.random();
            let rho = if self.standard_lower <= 0.0 && self.standard_upper >= 0.0 {
                (-0.5 * z.powi(2)).exp()
            } else if self.standard_upper < 0.0 {
                (0.5 * (self.standard_upper.powi(2) - z.powi(2))).exp()
            } else {
                (0.5 * (self.standard_lower.powi(2) - z.powi(2))).exp()
            };
            if u <= rho {
                return self.mu + self.sigma * z;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uses_tail_interval_method_for_positive_tail() {
        let distr = NormalTruncated::new(0.0, 1.0, 2.0, 3.0).unwrap();
        assert!(matches!(distr.0, Method::TailInterval(false, _)));
    }

    #[test]
    fn uses_tail_interval_method_for_negative_tail() {
        let distr = NormalTruncated::new(0.0, 1.0, -3.0, -2.0).unwrap();
        assert!(matches!(distr.0, Method::TailInterval(true, _)));
    }

    #[test]
    fn keeps_uniform_two_sided_method_for_narrow_positive_interval() {
        let distr = NormalTruncated::new(0.0, 1.0, 0.1, 0.2).unwrap();
        assert!(matches!(distr.0, Method::TwoSided(_)));
    }
}

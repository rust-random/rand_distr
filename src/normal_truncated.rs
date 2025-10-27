use rand::{distr::Distribution, Rng};

pub struct NormalTruncated(Method);
pub enum Method {
    Rejection(NormalTruncatedRejection),
    OneSided(),
}

pub enum Error {
    NonPosStdDev,
}

impl NormalTruncated {
    pub fn new(mean: f64, stddev: f64, lower: f64, upper: f64) -> Result<Self, Error> {
        if stddev <= 0.0 {
            return Err(Error::NonPosStdDev);
        }
        
        // When the lower bound is smaller than the mean, naive rejection sampling will have at least 
        if lower < mean {
            return Ok(NormalTruncated(Method::Rejection(NormalTruncatedRejection {
                normal: crate::Normal::new(mean, stddev).unwrap(),
                lower,
                upper,
            })));
        } 
        todo!()
    }
}

/// A truncated normal distribution using naive rejection sampling.
/// We use this when the acceptance rate is high enough.
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

struct NormalTruncatedOneSided {
    alpha_star: f64,
    lower_bound: f64,
    exp_distribution: crate::Exp<f64>,
    mu: f64,
    sigma: f64,
}

impl NormalTruncatedOneSided {
    fn new(mu: f64, sigma: f64, lower: f64) -> Self {
        let standart_lower_bound = (lower - mu) / sigma;
        let alpha_star = (standart_lower_bound + (standart_lower_bound.powi(2) + 4.0).sqrt()) / 2.0;
        let lambda = alpha_star;
        NormalTruncatedOneSided {
            alpha_star,
            lower_bound: lower - mu,
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

struct NormalTruncatedTwoSided {
    mu: f64,
    sigma: f64,
    // In standard normal coordinates
    lower: f64,
    // In standard normal coordinates
    upper: f64,
}

impl NormalTruncatedTwoSided {
    fn new(mu: f64, sigma: f64, lower: f64, upper: f64) -> Self {
        NormalTruncatedTwoSided {
            mu,
            sigma,
            lower: (lower - mu) / sigma,
            upper: (upper - mu) / sigma,
        }
    }
}

impl Distribution<f64> for NormalTruncatedTwoSided {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        loop {
            let z = rng.random_range(self.lower..self.upper);
            let u: f64 = rng.random();
            let rho = if self.lower <= 0.0 && self.upper >= 0.0 {
                (-0.5 * z.powi(2)).exp()
            } else if self.upper < 0.0 {
                (0.5 * (self.upper.powi(2) - z.powi(2))).exp()
            } else {
                (0.5 * (self.lower.powi(2) - z.powi(2))).exp()
            };
            if u <= rho {
                return self.mu + self.sigma * z;
            }
        }
    }
}
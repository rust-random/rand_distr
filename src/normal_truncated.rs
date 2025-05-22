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
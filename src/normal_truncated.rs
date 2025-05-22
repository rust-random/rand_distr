use rand::{distr::Distribution, Rng};

pub struct NormalTruncated {
    mean: f64,
    stddev: f64,
    lower: f64,
    upper: f64,
}

pub enum Error {
    NonPosStdDev,
}

impl NormalTruncated {
    pub fn new(mean: f64, stddev: f64, lower: f64, upper: f64) -> Result<Self, Error> {
        if stddev <= 0.0 {
            return Err(Error::NonPosStdDev);
        }
        Ok(NormalTruncated {
            mean,
            stddev,
            lower,
            upper,
        })
    }
}

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
#![no_main]

use std::convert::Infallible;

use arbitrary::Arbitrary;

use libfuzzer_sys::fuzz_target;

use rand::{SeedableRng, TryRng};
use rand::rngs::Xoshiro128PlusPlus;
use rand_distr::multi::Dirichlet;

/// Some distributions are parameterized by F: Float; various floating point
/// types behave similarly but precision and range differences make them not
/// perfectly interchangable.
type Float = f32;

/// Model all distribution parameters in one struct; this can be more awkward
/// to control compared to separate fuzzers, but is easier to use.
#[derive(Debug, Clone, Copy, Arbitrary)]
enum Parameters {
    BernouilliFloat {
        p: f64,
    },
    BernouilliRatio {
        num: u32,
        denom: u32,
    },
    Beta {
        alpha: Float,
        beta: Float,
    },
    Binomial {
        n: u64,
        p: f64,
    },
    Cauchy {
        median: Float,
        scale: Float,
    },
    ChiSquared {
        k: Float,
    },
    Dirichlet2 {
        alpha: [Float; 2],
    },
    Dirichlet3 {
        alpha: [Float; 3],
    },
    Dirichlet4 {
        alpha: [Float; 4],
    },
    Dirichlet5 {
        alpha: [Float; 5],
    },
    Exp {
        lambda: Float,
    },
    Exp1 {},
    FisherF {
        m: Float,
        n: Float,
    },
    Frechet {
        location: Float,
        scale: Float,
        shape: Float,
    },
    Gamma {
        shape: Float,
        scale: Float,
    },
    Geometric {
        p: f64,
    },
    Gumbel {
        location: Float,
        scale: Float,
    },
    Hypergeometric {
        nn: u64,
        kk: u64,
        n: u64,
    },
    InverseGaussian {
        mean: Float,
        shape: Float,
    },
    LogNormal {
        mu: Float,
        sigma: Float,
    },
    LogNormalMeanCV {
        mean: Float,
        cv: Float,
    },
    Normal {
        mean: Float,
        std_dev: Float,
    },
    Open01 {},
    OpenClosed01 {},
    NormalInverseGaussian {
        alpha: Float,
        beta: Float,
    },
    Pareto {
        scale: Float,
        shape: Float,
    },
    Pert {
        min: Float,
        max: Float,
        shape: Float,
        mode: Float,
    },
    Poisson {
        lambda: Float,
    },
    SkewNormal {
        location: Float,
        scale: Float,
        shape: Float,
    },
    StandardNormal {},
    StandardGeometric {},
    StandardUniform {},
    StudentT {
        nu: Float,
    },
    Triangular {
        min: Float,
        max: Float,
        mode: Float,
    },
    UniformFloat {
        lo: Float,
        hi: Float,
    },
    UniformU64 {
        lo: u64,
        hi: u64,
    },
    UnitBall {},
    UnitCircle {},
    UnitDisk {},
    UnitSphere {},
    Weibull {
        scale: Float,
        shape: Float,
    },
    Zeta {
        s: Float,
    },
    Zipf {
        n: Float,
        s: Float,
    },
}

#[derive(Debug, Clone, Copy, Arbitrary)]
struct Input {
    rng_initial: [u8; 16],
    rng_tail_seed: u64,
    params: Parameters,
}

/// This RNG makes it easier to for a fuzzer to find some rare events
/// (like two sampled f64s in a row being identical), while permitting
/// algorithms that require many samples.
struct TestRng<'a> {
    initial: &'a [u8],
    remaining: usize,
    tail_rng: Xoshiro128PlusPlus,
}

impl<'a> TestRng<'a> {
    fn new(initial: &'a [u8], tail_seed: u64) -> TestRng<'a> {
        TestRng {
            initial,
            remaining: initial.len(),
            tail_rng: Xoshiro128PlusPlus::seed_from_u64(tail_seed),
        }
    }
}
impl TryRng for TestRng<'_> {
    type Error = Infallible;
    fn try_next_u32(&mut self) -> Result<u32, Self::Error> {
        let mut b = [0u8; 4];
        self.try_fill_bytes(&mut b)?;
        Ok(u32::from_le_bytes(b))
    }
    fn try_next_u64(&mut self) -> Result<u64, Self::Error> {
        let mut b = [0u8; 8];
        self.try_fill_bytes(&mut b)?;
        Ok(u64::from_le_bytes(b))
    }
    fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Self::Error> {
        let tail = if self.remaining > 0 {
            let (head, tail) = dst.split_at_mut(self.remaining.min(dst.len()));
            let initial_tail = &self.initial[self.initial.len() - self.remaining..];
            head.copy_from_slice(&initial_tail[..head.len()]);
            self.remaining -= head.len();
            tail
        } else {
            dst
        };
        self.tail_rng.try_fill_bytes(tail)
    }
}

fn sample_one(rng: &mut TestRng, params: &Parameters) -> Option<()> {
    use Parameters as P;
    use rand_distr::*;
    use std::hint::black_box;

    match *params {
        P::BernouilliFloat { p } => {
            black_box(Bernoulli::new(p).ok()?.sample(rng));
        }
        P::BernouilliRatio { num, denom } => {
            black_box(Bernoulli::from_ratio(num, denom).ok()?.sample(rng));
        }
        P::Beta { alpha, beta } => {
            let v = black_box(Beta::new(alpha, beta).ok()?.sample(rng));
            assert!((0.0..=1.0).contains(&v), "{}", v);
        }
        P::Binomial { n, p } => {
            let v = black_box(Binomial::new(n, p).ok()?.sample(rng));
            assert!(v <= n, "{}", v);
        }
        P::Cauchy { median, scale } => {
            let v = black_box(Cauchy::new(median, scale).ok()?.sample(rng));
            // ideal range (-inf, inf) rounds to fp [-inf, inf]
            assert!(!v.is_nan(), "{}", v);
        }
        P::ChiSquared { k } => {
            // ideal range (0, inf) rounds to fp [0, inf]
            let v = black_box(ChiSquared::new(k).ok()?.sample(rng));
            assert!(v >= 0.0, "{}", v);
        }
        P::Dirichlet2 { alpha } => {
            // Warning: only a tiny fraction of parameters is OK, so this distribution
            // may not be fuzzed effectively.
            let v = black_box(Dirichlet::new(&alpha).ok()?.sample(rng));
            assert!(v.iter().all(|x| *x >= 0.0), "{:?}", v);
            assert!(
                (v.iter().copied().sum::<Float>() - 1.0) < 20.0 * Float::EPSILON,
                "{:?}",
                v
            );
        }
        P::Dirichlet3 { alpha } => {
            let v = black_box(Dirichlet::new(&alpha).ok()?.sample(rng));
            assert!(v.iter().all(|x| *x >= 0.0), "{:?}", v);
            assert!(
                (v.iter().copied().sum::<Float>() - 1.0) < 30.0 * Float::EPSILON,
                "{:?}",
                v
            );
        }
        P::Dirichlet4 { alpha } => {
            let v = black_box(Dirichlet::new(&alpha).ok()?.sample(rng));
            assert!(v.iter().all(|x| *x >= 0.0), "{:?}", v);
            assert!(
                (v.iter().copied().sum::<Float>() - 1.0) < 40.0 * Float::EPSILON,
                "{:?}",
                v
            );
        }
        P::Dirichlet5 { alpha } => {
            let v = black_box(Dirichlet::new(&alpha).ok()?.sample(rng));
            assert!(v.iter().all(|x| *x >= 0.0), "{:?}", v);
            assert!(
                (v.iter().copied().sum::<Float>() - 1.0) < 50.0 * Float::EPSILON,
                "{:?}",
                v
            );
        }
        P::Exp { lambda } => {
            let v = black_box(Exp::new(lambda).ok()?.sample(rng));
            assert!(v >= 0.0, "{}", v);
        }
        P::Exp1 {} => {
            let v = black_box(Distribution::<Float>::sample(&Exp1, rng));
            assert!(v >= 0.0, "{}", v);
        }
        P::FisherF { m, n } => {
            let v = black_box(FisherF::new(m, n).ok()?.sample(rng));
            assert!(v >= 0.0, "{}", v);
        }
        P::Frechet {
            location,
            scale,
            shape,
        } => {
            let v = black_box(Frechet::new(location, scale, shape).ok()?.sample(rng));
            assert!(v >= location, "{}", v);
        }
        P::Gamma { shape, scale } => {
            let v = black_box(Gamma::new(shape, scale).ok()?.sample(rng));
            assert!(v >= 0.0, "{}", v);
        }
        P::Geometric { p } => {
            black_box(Geometric::new(p).ok()?.sample(rng));
        }
        P::Gumbel { location, scale } => {
            let v = black_box(Gumbel::new(location, scale).ok()?.sample(rng));
            assert!(!v.is_nan(), "{}", v);
        }
        P::Hypergeometric { nn, kk, n } => {
            let v = black_box(Hypergeometric::new(nn, kk, n).ok()?.sample(rng));
            assert!(v <= n, "{}", v);
        }
        P::InverseGaussian { mean, shape } => {
            let v = black_box(InverseGaussian::new(mean, shape).ok()?.sample(rng));
            assert!(v >= 0.0, "{}", v);
        }
        P::LogNormal { mu, sigma } => {
            let v = black_box(LogNormal::new(mu, sigma).ok()?.sample(rng));
            assert!(v >= 0.0, "{}", v);
        }
        P::LogNormalMeanCV { mean, cv } => {
            let v = black_box(LogNormal::from_mean_cv(mean, cv).ok()?.sample(rng));
            assert!(v >= 0.0, "{}", v);
        }
        P::Normal { mean, std_dev } => {
            let v = black_box(Normal::new(mean, std_dev).ok()?.sample(rng));
            assert!(!v.is_nan(), "{}", v);
        }
        P::NormalInverseGaussian { alpha, beta } => {
            let v = black_box(NormalInverseGaussian::new(alpha, beta).ok()?.sample(rng));
            assert!(!v.is_nan(), "{}", v);
        }
        P::Open01 {} => {
            let v = black_box(Open01.sample(rng));
            assert!(0.0 < v && v < 1.0, "{}", v);
        }
        P::OpenClosed01 {} => {
            let v = black_box(OpenClosed01.sample(rng));
            assert!(0.0 < v && v <= 1.0, "{}", v);
        }
        P::Pareto { scale, shape } => {
            let v = black_box(Pareto::new(scale, shape).ok()?.sample(rng));
            assert!(v >= scale, "{}", v);
        }
        P::Pert {
            min,
            max,
            shape,
            mode,
        } => {
            let v = black_box(
                Pert::new(min, max)
                    .with_shape(shape)
                    .with_mode(mode)
                    .ok()?
                    .sample(rng),
            );
            assert!((min..=max).contains(&v), "{}", v);
        }
        P::Poisson { lambda } => {
            let v = black_box(Poisson::new(lambda).ok()?.sample(rng));
            assert!(
                v >= 0.0 && v <= 2.0_f32.powi(64) && v.fract() == 0.0,
                "{}",
                v
            );
        }
        P::SkewNormal {
            location,
            scale,
            shape,
        } => {
            let v = black_box(SkewNormal::new(location, scale, shape).ok()?.sample(rng));
            assert!(!v.is_nan(), "{}", v);
        }
        P::StandardNormal {} => {
            let v = black_box(Distribution::<Float>::sample(&StandardNormal, rng));
            assert!(!v.is_nan(), "{}", v);
        }
        P::StandardGeometric {} => {
            black_box(StandardGeometric.sample(rng));
        }
        P::StandardUniform {} => {
            let v = black_box(Distribution::<Float>::sample(&StandardUniform, rng));
            assert!((0.0..1.0).contains(&v), "{}", v);
        }
        P::StudentT { nu } => {
            let v = black_box(StudentT::new(nu).ok()?.sample(rng));
            assert!(!v.is_nan(), "{}", v);
        }
        P::Triangular { min, max, mode } => {
            let v = black_box(Triangular::new(min, max, mode).ok()?.sample(rng));
            assert!(min <= v && v <= max, "{}", v);
        }
        P::UniformFloat { lo, hi } => {
            let v = black_box(Uniform::new(lo, hi).ok()?.sample(rng));
            assert!(lo <= v && v <= hi, "{}", v);
        }
        P::UniformU64 { lo, hi } => {
            let v = black_box(Uniform::new(lo, hi).ok()?.sample(rng));
            assert!(lo <= v && v <= hi, "{}", v);
        }
        P::UnitBall {} => {
            let v = black_box(Distribution::<[Float; 3]>::sample(&UnitBall, rng));
            // Tighter bounds are possible, but this should catch extreme outliers
            assert!(
                v.iter().all(|x| x.abs() <= 1.0 + 10.0 * Float::EPSILON),
                "{:?}",
                v
            );
        }
        P::UnitCircle {} => {
            let v = black_box(Distribution::<[Float; 2]>::sample(&UnitCircle, rng));
            assert!(
                v.iter().all(|x| x.abs() <= 1.0 + 10.0 * Float::EPSILON),
                "{:?}",
                v
            );
        }
        P::UnitDisk {} => {
            let v = black_box(Distribution::<[Float; 2]>::sample(&UnitDisc, rng));
            assert!(
                v.iter().all(|x| x.abs() <= 1.0 + 10.0 * Float::EPSILON),
                "{:?}",
                v
            );
        }
        P::UnitSphere {} => {
            let v = black_box(Distribution::<[Float; 3]>::sample(&UnitSphere, rng));
            assert!(
                v.iter().all(|x| x.abs() <= 1.0 + 10.0 * Float::EPSILON),
                "{:?}",
                v
            );
        }
        P::Weibull { scale, shape } => {
            let v = black_box(Weibull::new(scale, shape).ok()?.sample(rng));
            assert!(v >= 0.0, "{}", v);
        }
        P::Zeta { s } => {
            let v = black_box(Zeta::new(s).ok()?.sample(rng));
            assert!(v >= 0.0 && (v.is_infinite() || v.fract() == 0.0), "{}", v);
        }
        P::Zipf { n, s } => {
            let v = black_box(Zipf::new(n, s).ok()?.sample(rng));
            assert!(v >= 0.0 && (v.is_infinite() || v.fract() == 0.0), "{}", v);
        }
    }
    Some(())
}

fuzz_target!(|input: Input| {
    let mut rng = TestRng::new(&input.rng_initial, input.rng_tail_seed);
    let _ = sample_one(&mut rng, &input.params);
});

// Copyright 2025 Developers of the Rand project.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand_distr::{Bernoulli, Binomial, Distribution, Geometric};

/// Say `n` samples are drawn independently from a Bernoulli distribution
/// with probability `q` in `[q_lb, q_ub]` of outputting 1. Let X be their sum.
///
/// If `k > q_ub n`, this function returns an estimate (some inaccuracy possible
/// due to floating point error) of an upper bound on the log-probability that X >= k;
/// if `k < q_lb n`, the function returns an estimate of an upper bound for the
/// log-probability that X <= k. Otherwise, `k` might be equal to `q n` and the
/// function returns log-probability 0.0 (probability 1.0).
///
/// Note: the value returned is the logarithm of the probability bound estimate.
fn bernouilli_ln_tail_prob_bound(q_lb: f64, q_ub: f64, k: u64, n: u64) -> f64 {
    fn kl_div_lb(r: f64, p: f64) -> f64 {
        // Note: this calculation can be inaccurate when r, p are tiny
        if p <= 0.0 {
            if r > p { f64::INFINITY } else { 0.0 }
        } else if 1.0 - p <= 0.0 {
            if r < p { f64::INFINITY } else { 0.0 }
        } else if r == 0.0 {
            (1.0 - r) * f64::ln((1.0 - r) / (1.0 - p))
        } else if 1.0 - r == 0.0 {
            r * f64::ln(r / p)
        } else {
            r * f64::ln(r / p) + (1.0 - r) * f64::ln((1.0 - r) / (1.0 - p))
        }
    }

    assert!(k <= n);
    assert!(0.0 <= q_lb && q_ub <= 1.0);

    let r = (k as f64) / (n as f64);
    if r < q_lb {
        -(n as f64) * kl_div_lb(r, q_lb)
    } else if r > q_ub {
        -(n as f64) * kl_div_lb(1.0 - r, 1.0 - q_ub)
    } else {
        0.0
    }
}

/// For y=e^x, z = min(2 y, 1), return ln(z)
fn min_2x_under_ln(x: f64) -> f64 {
    if x.is_nan() {
        x
    } else {
        0.0f64.min(x + f64::ln(2.0))
    }
}

// Threshold probability for the output of a test possibly indicating
// a discrepancy between the actual and ideal distribution.
// (The event could happen by chance on a 1e-3 fraction of seeds even
// if the distributions match.)
const POSSIBLE_DISCREPANCY_THRESHOLD: f64 = -3.0 * std::f64::consts::LN_10;

// Threshold probability for the output of a test certainly indicating
// a discrepancy between the actual and ideal distribution
// (Hardware failures are many orders of magnitude more likely
// than the entire system being correct.)
const CERTAIN_DISCREPANCY_THRESHOLD: f64 = -40.0 * std::f64::consts::LN_10;

#[derive(Debug)]
enum TestFailure {
    #[allow(unused)]
    Possible(f64),
    Certain,
}

fn test_binary(
    seed: u64,
    ideal_prob_lb: f64,
    ideal_prob_ub: f64,
    sample_size: u64,
    sample_fn: &dyn Fn(&mut SmallRng) -> bool,
) -> Result<(), TestFailure> {
    let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
    let mut ones: u64 = 0;
    for _ in 0..sample_size {
        ones += if sample_fn(&mut rng) { 1 } else { 0 };
    }

    let ln_single_tail_p =
        bernouilli_ln_tail_prob_bound(ideal_prob_lb, ideal_prob_ub, ones, sample_size);
    // Double the probability to correct for the fact that there are two tails
    let ln_p = min_2x_under_ln(ln_single_tail_p);

    if ln_p < CERTAIN_DISCREPANCY_THRESHOLD {
        Err(TestFailure::Certain)
    } else if ln_p < POSSIBLE_DISCREPANCY_THRESHOLD {
        Err(TestFailure::Possible(f64::exp(ln_p)))
    } else {
        Ok(())
    }
}

/// Verify that the re-exported Bernoulli sampler is
/// not clearly far from the correct distribution
#[test]
fn test_bernouilli() {
    let sample_size = 1000000;
    let seed = 0x1;

    // Check that the Bernouilli sampler is not far from correct
    for p_base in [0.0, 1e-9, 1e-3, 1.0 / 3.0, 0.5] {
        for p in [p_base, 1.0 - p_base] {
            test_binary(seed, p, p, sample_size, &|rng| {
                let b = Bernoulli::new(p).unwrap();
                b.sample(rng)
            })
            .unwrap();
        }
    }

    // Check that the test will actually catch clear discrepancies.
    assert!(matches!(
        test_binary(seed, 0.4, 0.4, sample_size, &|rng| {
            let b = Bernoulli::new(0.6).unwrap();
            b.sample(rng)
        }),
        Err(TestFailure::Certain)
    ));
}

/// For X ~ Binomial(n; p), returns Pr[X mod 2 = 1]
fn binomial_last_bit_probability(n: u64, p: f64) -> f64 {
    /* Since
     *
     * 1 = (p + (1-p))^n = ∑_k \binom{n}{k} p^k (1-p)^{n-k} ,
     *
     * and
     *
     * (-p + (1-p))^n = ∑_k (-1)^k \binom{n}{k} p^k (1-p)^{n-k} ,
     *
     * adding them together gives:
     *
     * 1 + (1 - 2p)^n = ∑_k (1 + (-1)^k) \binom{n}{k} p^k (1-p)^{n-k}
     *                = ∑_k 2 ⋅ 1_{k mod 2 = 0} \binom{n}{k} p^k (1-p)^{n-k}
     *                = 2 Pr[k mod 2 = 0] .
     *
     * So:
     *
     *      Pr[k mod 2 = 1] = 1 - ½ (1 + (1 - 2p)^n) = ½ (1 - (1 - 2p)^n)
     */

    0.5 * (1.0 - (1.0 - 2.0 * p).powi(n.try_into().unwrap_or(i32::MAX)))
}

/// Do samples from a binomial distribution, taken mod 2, match the expected distribution?
/// (This is a likely failure mode of samplers using floating point.)
#[test]
fn test_binomial_last_bit() {
    let sample_size = 100000;
    let seed = 0x1;

    let mut sizes = Vec::<u64>::new();
    // Binomial::new(n, p) currently panics for when a quantity derived from n is >= i64::MAX
    for i in 1..=62 {
        sizes.push((1 << i) - 1);
        sizes.push(1 << i);
        sizes.push((1 << i) + 1);
        sizes.push(3 * (1 << (i - 1)));
    }

    for s in sizes {
        for p in [0.1, 0.5, 0.9] {
            let t = binomial_last_bit_probability(s, p);

            let Ok(dist) = Binomial::new(s, p) else {
                println!("Failed to create Binomial with n={}, p={}", s, p);
                continue;
            };

            let res = test_binary(seed, t, t, sample_size, &|rng| dist.sample(rng) % 2 == 1);

            // Binomial::new()'s documentation only promises accuracy up to n=~2^53
            // Using `p` closer to 0 or 1 produces a narrower peak which is easier to sample correctly
            if s <= 1 << 53 {
                assert!(res.is_ok());
            } else {
                println!(
                    "Binomial distribution with n={}, p={:.4}, last bit prob {:.4}, log2(n)={:.3}: last bit result {:?}",
                    s,
                    p,
                    t,
                    (s as f64).log2(),
                    res
                );
            }
        }
    }
}

/// For X ~ Geometric(p), returns Pr[X mod 2 = 1]
fn geometric_last_bit_probability(p: f64) -> f64 {
    /* The geometric probabilities are
     * 0   1        2           3
     * p,  (1-p)p,  (1-p)^2 p,  (1-p)^3 p, ...
     *
     * As   Pr[X mod 2 = 1] = (1 - p) Pr[X mod 2 = 0],
     * and  Pr[X mod 2 = 1] = 1 - Pr[X mod 2 = 0],
     * it follows:
     *
     *  Pr[X mod 2 = 1] = 1 - 1/(2 - p)
     */
    (1.0 - p) / (2.0 - p)
}

#[test]
fn test_geometric_last_bit() {
    let sample_size = 100000;
    let seed = 0x1;

    // Test on distributions with values of p of various closeness to 0.0
    for i in 0..=128 {
        let p = 0.5_f64.powf(i as f64 / 2.0);

        if 1.0 - p == 1.0 {
            // The current implementation of Geometric always returns u64::MAX when 1.0 - p = 1.0
            continue;
        }

        let t = geometric_last_bit_probability(p);
        let clipped_prob = (special::Primitive::ln_1p(-p) * 2.0f64.powi(64)).exp();

        let Ok(dist) = Geometric::new(p) else {
            println!("Failed to create Geometric with p={}", p);
            continue;
        };

        let res = test_binary(
            seed,
            t - clipped_prob,
            t + clipped_prob,
            sample_size,
            &|rng| dist.sample(rng) % 2 == 1,
        );

        println!(
            "Geometric distribution with p={:e}, last bit prob {}-{}: last bit result {:?}",
            p,
            t - clipped_prob,
            t + clipped_prob,
            res
        );

        assert!(res.is_ok(), "{:?}", res);
    }
}

/// How accurate are the probabilities for outputting 0 and n for Binomial(n, p)?
///
/// Uncareful boundary handling can make the endpoint probabilities wrong by a
/// constant factor, although the tails have low enough probability that this
/// can be hard to detect.
#[test]
#[ignore]
fn test_binomial_endpoints() {
    let p = 0.5;
    // Note: conclusively indicating an error tends to require more than 1 / event probability
    // samples, so this test needs ~100M samples to find any issues in the below at n >= 20
    let sample_size = 400_000_000;
    let seed = 0x1;

    // With the current implementation, n <= 20 always uses BINV or Poisson sampling, so to check
    // the BTPE implementation larger `s` is needed.

    for s in 1..25u64 {
        let Ok(dist) = Binomial::new(s, p) else {
            println!("Failed to create Binomial with n={}, p={}", s, p);
            continue;
        };

        let t = p.powi(s as i32) + (1.0 - p).powi(s as i32);
        let res = test_binary(seed, t, t, sample_size, &|rng| {
            let v = dist.sample(rng);
            v == 0 || v == s
        });
        println!(
            "Binomial({}, {}) endpoint test, evt prob {:e}, {:?}",
            s, p, t, res
        );
        assert!(res.is_ok());
    }
}

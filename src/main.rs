use rand::Rng;
use core::array;
use std::sync::Arc;

mod precomp;
mod params;

use crate::{precomp::*, params::*};

/// An exponential distribution with parameter `a`.
struct ExpoDist {
    a: f64
}

impl ExpoDist {
    /// Create a new exponential distribution with parameter `a`.
    fn new(a: f64) -> Self {
        Self { a }
    }

    /// Sample from the distribution.
    fn sample(&self) -> f64 {
        let mut rng = rand::thread_rng();
        let ptile: f64 = rng.gen();
        -(1f64 - ptile).ln() / self.a
    }
}

/// A draw of the adversary's coins and rewards in a given sample.
#[derive(Debug)]
struct AdvDraw {
    coins: [f64; ADV_COINS+2],
    rewards: [f64; ADV_COINS+1]
}

impl AdvDraw {
    /// Given that exactly the first `i_star` coins beat c0, compute the best expected reward
    /// the adversary can obtain by broadcasting one of their coins.
    fn best_from(&self, i_star: usize) -> f64 {
        let mut best = 0f64;
        for i in 1..=i_star {
            let score = (1f64 + self.rewards[i]) * Self::beat_unseen_honest_prob(self.coins[i]);
            if score > best {
                best = score;
            }
        }
        best
    }

    /// The probability that coin `x` beats the unseen honest coin.
    fn beat_unseen_honest_prob(x: f64) -> f64 {
        if x == f64::MAX {
            0f64
        } else {
            (-x * (1f64 - BETA) * (1f64 - ALPHA)).exp()
        }
    }

    /// The probability that coin `x` beats the seen honest coin.
    fn beat_seen_honest_prob(x: f64) -> f64 {
        if x == f64::MAX {
            0f64
        } else {
            (-x * BETA * (1f64 - ALPHA)).exp()
        }
    }
    
    /// The probability that coin `x` beats both honest coins.
    fn beat_honest_prob(x: f64) -> f64 {
        if x == f64::MAX {
            0f64
        } else {
            (-x * (1f64 - ALPHA)).exp()
        }
    }
}

/// Given a distribution `advdist` draw coins and rewards for the adversary.
fn draw_adv(advdist: &Cdf) -> AdvDraw {
    // 2a: draw adversary rewards and coins
    let expodist = ExpoDist::new(ALPHA);
    let mut coins = [0f64; ADV_COINS+2];
    let mut cum = 0f64;
    for i in 1..=ADV_COINS {
        cum += expodist.sample(); // draw exp
        coins[i] = cum;
    }
    coins[ADV_COINS+1] = f64::MAX;
    AdvDraw {
        coins,
        rewards: array::from_fn(|_| advdist.sample())
    }
}

/// The precomputed data.
enum Precomp {
    /// Data needed if `BETA == 0`.
    None(Cdf, f64),
    /// Data needed if `BETA == 1`.
    Short(Cdf, Emax, f64),
    /// Data needed if `0 < BETA < 1`.
    Long(Cdf, Table, f64)
}

/// Given precomputed data `precomp` sample a single new adversary reward.
fn sample(precomp: Arc<Precomp>) -> f64 {
    match &*precomp {
        Precomp::None(cdf, lambda) => {
            let adv = draw_adv(&cdf);
            let best = adv.best_from(ADV_COINS);
            best - lambda
        }
        Precomp::Short(cdf, emax, lambda) => {
            let adv = draw_adv(&cdf);
            let mut cum = 0f64;
            for i_star in 1..=ADV_COINS {
                let best = adv.best_from(i_star);
                cum += emax.get(best) * (
                    AdvDraw::beat_honest_prob(adv.coins[i_star]) - 
                    AdvDraw::beat_honest_prob(adv.coins[i_star + 1])
                );
            }
            // println!("scored {}", cum);
            cum - lambda
        },
        Precomp::Long(cdf, table, lambda) => {
            let adv = draw_adv(&cdf);
            let mut cum = 0f64;
            for i_star in 1..=ADV_COINS {
                let best = adv.best_from(i_star);
                cum -= table.get(adv.coins[i_star], best);
                cum += table.get(adv.coins[i_star + 1], best);
            }
            cum - lambda
        }
    }
}

/// Given samples `samp` and `lambda` compute the distribution's cdf.
async fn precompute_cdf(samp: Samples, lambda: f64) -> Cdf {
    // 1a: compute adversary pdf
    let (min, max) = (samp.round as f64 * (0f64 - lambda), samp.round as f64 * (1f64 - lambda));
    let adv_pdf = samp.to_pdf(min, max).await;
    // 1b: convert into cdf
    let adv_cdf = adv_pdf.to_cdf();
    adv_cdf
} 

/// Given samples `samp` and `lambda` precompute the requisite data for efficient sampling.
async fn precompute(samp: Samples, lambda: f64) -> Precomp {
    let round = samp.round;
    let adv_cdf = precompute_cdf(samp, lambda).await;
    println!("round {:#?} expected win {:#?}", round, adv_cdf.exp());
    if BETA == 0f64 {
        return Precomp::None(adv_cdf, lambda);
    }
    // 1c and 1d: compute E_max thetas
    let emax = adv_cdf.to_emax();
    if BETA == 1f64 {
        return Precomp::Short(adv_cdf, emax, lambda);
    }
    // 1e: compute G(gamma, c)
    let table = emax.to_table(round + 1, lambda).await;
    Precomp::Long(adv_cdf, table, lambda)
}

/// Helper to compute new samples.
async fn add_layer_helper(precomp: Arc<Precomp>) -> Vec<f64> {
    let mut new_data = Vec::with_capacity(SAMPLES_DRAWN / PARALLELISM_FACTOR);
    for _ in 0..SAMPLES_DRAWN / PARALLELISM_FACTOR {
        new_data.push(sample(precomp.clone()));
    }
    new_data
}

/// Given samples `samp` and `lambda`, compute a new set of samples for the following round.
async fn finite_sample_add_layer(samp: Samples, lambda: f64) -> Samples {
    let mut new_samps = Samples { round: samp.round + 1, data: Default::default() };
    let mut handles = Vec::with_capacity(PARALLELISM_FACTOR);
    let precomp = Arc::new(precompute(samp, lambda).await);
    for _ in 0..PARALLELISM_FACTOR {
        handles.push(tokio::spawn(add_layer_helper(precomp.clone())));
    }
    for (i, handle) in handles.into_iter().enumerate() {
        new_samps.data[i] = handle.await.unwrap();
    }
    fooflate(new_samps, lambda).await
}

/// Inflate or deflate `samp` depending on the mode of the simulation.
async fn fooflate(samp: Samples, lambda: f64) -> Samples {
    let adv_cdf = precompute_cdf(samp.clone(), lambda).await;
    let shift_percent = ((1f64 / CHERNOFF_ERROR).ln() / (2 * SAMPLES_DRAWN) as f64).sqrt().ceil() as f64;
    let bounding_ptile = match MODE {
        Mode::LowerBounding => shift_percent,
        Mode::UpperBounding => 1f64 - shift_percent
    };
    let bounding_cutoff = adv_cdf.inv_get(bounding_ptile);
    let mut handles = Vec::with_capacity(PARALLELISM_FACTOR);
    for mut vec in samp.data {
        handles.push(tokio::spawn(async move {
            for entry in vec.iter_mut() {
                match MODE {
                    Mode::LowerBounding => {
                        if *entry > bounding_cutoff {
                            *entry = samp.round as f64 * (0f64 - lambda);
                        }
                    },
                    Mode::UpperBounding => {
                        if *entry < bounding_cutoff {
                            *entry = samp.round as f64 * (1f64 - lambda);
                        }
                    },
                }
            }
            vec
        }));
    }
    let mut new_samps = Samples { round: samp.round, data: Default::default() };
    for (i, handle) in handles.into_iter().enumerate() {
        new_samps.data[i] = handle.await.unwrap();
    }
    new_samps
}

/// Find the expected reward of the adversary facing cost `lambda` to play each round.
async fn simulate(lambda: f64) -> f64 {
    let mut dist = Samples { round: 0, data: array::from_fn(|_| Vec::from([0f64])) };
    for _ in 1..ROUND_DEPTH {
        dist = finite_sample_add_layer(dist, lambda).await;
    }
    let adv_cdf = precompute_cdf(dist, lambda).await;
    adv_cdf.exp()
}

/// TODO: normal binary search doesn't really make sense in context of upper/lower bounding
async fn binary_search() -> f64 {
    let mut lo = ALPHA;
    let mut hi = 1.0;
    loop {
        let lambda = (hi + lo) / 2.0;
        println!("trying lambda = {:?}", lambda);
        if hi - lo < 0.001 { break lambda; }
        let reward = simulate(lambda).await;
        if reward > 0.0 {
            lo = lambda
        } else {
            hi = lambda
        }
    }
}

#[tokio::main]
async fn main() {
    std::env::set_var("RUST_BACKTRACE", "1");
    binary_search().await;
}
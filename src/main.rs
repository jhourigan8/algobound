mod precomp;
mod params;

use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand::Rng;
use core::array;
use std::sync::Arc;
use std::time::{Instant, SystemTime};

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
    fn sample(&self, rng: &mut SmallRng) -> f64 {
        let ptile: f64 = rng.gen();
        -(1f64 - ptile).ln() / self.a
    }
}

/// A draw of the adversary's coins and rewards in a given sample.
#[derive(Debug)]
struct AdvDraw {
    coins: Vec<f64>, // first slot empty
    rewards: Vec<f64>
}

impl AdvDraw {
    /// Given that exactly the first `i_star` coins beat c0, compute the best expected reward
    /// the adversary can obtain by broadcasting one of their coins.
    fn best_from(&self, i_star: usize, alpha: f64, beta: f64) -> f64 {
        let mut best = 0f64;
        for i in 1..=i_star {
            let score = (1f64 + self.rewards[i]) * Self::beat_unseen_honest_prob(self.coins[i], alpha, beta);
            if score > best {
                best = score;
            }
        }
        best
    }

    /// The probability that coin `x` beats the unseen honest coin.
    fn beat_unseen_honest_prob(x: f64, alpha: f64, beta: f64) -> f64 {
        if x == f64::MAX {
            0f64
        } else {
            (-x * (1f64 - beta) * (1f64 - alpha)).exp()
        }
    }

    /// The probability that coin `x` beats the seen honest coin.
    fn beat_seen_honest_prob(x: f64, alpha: f64, beta: f64) -> f64 {
        if x == f64::MAX {
            0f64
        } else {
            (-x * beta * (1f64 - alpha)).exp()
        }
    }
    
    /// The probability that coin `x` beats both honest coins.
    fn beat_honest_prob(x: f64, alpha: f64) -> f64 {
        if x == f64::MAX {
            0f64
        } else {
            (-x * (1f64 - alpha)).exp()
        }
    }
}

/// Given a distribution `advdist` draw coins and rewards for the adversary.
fn draw_adv(advdist: &Cdf, adv_coins: usize, alpha: f64, rng: &mut SmallRng) -> AdvDraw {
    // 2a: draw adversary rewards and coins
    let expodist = ExpoDist::new(alpha);
    let mut coins = Vec::from([0.0]);
    let mut cum = 0f64;
    for _ in 1..=adv_coins {
        cum += expodist.sample(rng); // draw exp
        coins.push(cum);
    }
    coins.push(f64::MAX);
    let mut rewards = Vec::from([0.0]);
    for _ in 1..=adv_coins {
        rewards.push(advdist.sample(rng));
    }
    AdvDraw { coins, rewards }
}

/// The precomputed data.
enum Precomp {
    /// Data needed if `BETA == 0`.
    None(Cdf),
    /// Data needed if `BETA == 1`.
    Short(Cdf, Emax),
    /// Data needed if `0 < BETA < 1`.
    Long(Cdf, Table)
}

/// Given precomputed data `precomp` sample a single new adversary reward.
fn sample(
    precomp: Arc<Precomp>, adv_coins: usize, alpha: f64, beta: f64, lambda: f64, rounding: Rounding, rng: &mut SmallRng
) -> f64 {
    match &*precomp {
        Precomp::None(cdf) => {
            let adv = draw_adv(&cdf, adv_coins, alpha, rng);
            let best = adv.best_from(adv_coins, alpha, beta);
            best - lambda
        }
        Precomp::Short(cdf, emax) => {
            let adv = draw_adv(&cdf, adv_coins, alpha, rng);
            let mut cum = 0f64;
            for i_star in 1..=adv_coins {
                let best = adv.best_from(i_star, alpha, beta);
                cum += emax.get(best, &rounding) * (
                    AdvDraw::beat_honest_prob(adv.coins[i_star], alpha) - 
                    AdvDraw::beat_honest_prob(adv.coins[i_star + 1], alpha)
                );
            }
            cum - lambda
        },
        Precomp::Long(cdf, table) => {
            let adv = draw_adv(&cdf, adv_coins, alpha, rng);
            let mut cum = 0f64;
            let opposite = &!rounding.clone();
            for i_star in 1..=adv_coins {
                let best = adv.best_from(i_star, alpha, beta);
                cum -= table.get(AdvDraw::beat_seen_honest_prob(adv.coins[i_star], alpha, beta), best, opposite);
                cum += table.get(AdvDraw::beat_seen_honest_prob(adv.coins[i_star + 1], alpha, beta), best, &rounding);
            }
            cum - lambda
        }
    }
}

/// Given samples `samp` and `lambda` compute the distribution's cdf.
async fn precompute_cdf(samp: Samples, epsilon: f64, lambda: f64, rounding: Rounding) -> Cdf {
    // 1a: compute adversary pdf
    let (min, max) = (samp.round as f64 * (0.0 - lambda), samp.round as f64 * (1.0 - lambda));
    let adv_pdf = samp.to_pdf(min, max, epsilon, rounding.clone()).await;
    // 1b: convert into cdf
    let adv_cdf = adv_pdf.to_cdf();
    adv_cdf
} 

/// Given samples `samp` and `lambda` precompute the requisite data for efficient sampling.
async fn precompute(
    samp: Samples, epsilon: f64, eta: f64, alpha: f64, beta: f64, lambda: f64, rounding: Rounding
) -> Precomp {
    let round = samp.round;
    let adv_cdf = precompute_cdf(samp, epsilon, lambda, rounding.clone()).await;
    // println!("round {:#?} expected win {:#?}", round, adv_cdf.exp());
    if beta == 0.0 {
        return Precomp::None(adv_cdf);
    }
    // 1c and 1d: compute E_max thetas
    let emax = adv_cdf.to_emax();
    if beta == 1.0 {
        return Precomp::Short(adv_cdf, emax);
    }
    // 1e: compute G(gamma, c)
    let table = emax.to_table(eta, beta, rounding).await;
    Precomp::Long(adv_cdf, table)
}

/// A struct to carry around needed parameters.
#[derive(Clone, Copy)]
struct Parameters {
    adv_coins: usize,
    epsilon: f64,
    eta: f64,
    alpha: f64,
    beta: f64,
    rounding: Rounding,
    samples_drawn: usize, 
    parallelism_factor: usize,
    round_depth: usize,
    chernoff_error: f64
}

/// Helper to compute new samples.
async fn add_layer_helper(precomp: Arc<Precomp>, params: Parameters, lambda: f64) -> Vec<f64> {
    let mut new_data = Vec::with_capacity(params.samples_drawn / params.parallelism_factor);
    let mut rng = rand::rngs::SmallRng::from_entropy();
    for _ in 0..params.samples_drawn / params.parallelism_factor {
        new_data.push(sample(precomp.clone(), params.adv_coins, params.alpha, params.beta, lambda, params.rounding, &mut rng));
    }
    new_data
}

/// Given samples `samp` and `lambda`, compute a new set of samples for the following round.
async fn finite_sample_add_layer(samp: Samples, params: &Parameters, lambda: f64) -> Samples {
    let mut data = Vec::default();
    for _ in 0..params.parallelism_factor {
        data.push(Vec::default());
    }
    let mut new_samps = Samples { round: samp.round + 1, data };
    let mut handles = Vec::with_capacity(params.parallelism_factor);
    let precomp = Arc::new(precompute(samp, params.epsilon, params.eta, params.alpha, params.beta, lambda, params.rounding).await);
    for _ in 0..params.parallelism_factor {
        handles.push(tokio::spawn(add_layer_helper(precomp.clone(), params.clone(), lambda)));
    }
    for (i, handle) in handles.into_iter().enumerate() {
        new_samps.data[i] = handle.await.unwrap();
    }
    fooflate(new_samps, params, lambda).await
}

/// Inflate or deflate `samp` depending on the mode of the simulation.
async fn fooflate(samp: Samples, params: &Parameters, lambda: f64) -> Samples {
    let shift_percent = ((1.0 / params.chernoff_error).ln() / (2 * params.samples_drawn) as f64).sqrt() as f64;
    let rounding = params.rounding;
    let bounding_ptile = match rounding {
        Rounding::Down => 1.0 - shift_percent,
        Rounding::Up => shift_percent
    };
    let num_flate = match rounding {
        Rounding::Down => (bounding_ptile * params.samples_drawn as f64).floor() as usize,
        Rounding::Up => (bounding_ptile * params.samples_drawn as f64).ceil() as usize
    };
    // dirty binary search
    let mut lo = samp.round as f64 * (0.0 - lambda);
    let mut hi = samp.round as f64 * (1.0 - lambda);
    loop {
        let mut mid = (lo + hi) / 2.0;
        let mut handles = Vec::with_capacity(params.parallelism_factor);
        for vec in samp.data.clone() {
            handles.push(tokio::spawn(async move {
                let mut lo_count = 0;
                let mut hi_count = 0;
                for entry in vec.iter() {
                    if *entry <= mid {
                        hi_count += 1;
                    }
                    if *entry < mid {
                        lo_count += 1;
                    }
                }
                (lo_count, hi_count)
            }));
        }
        let mut lo_tot = 0;
        let mut hi_tot = 0;
        for (i, handle) in handles.into_iter().enumerate() {
            let (lo_x, hi_x) = handle.await.unwrap();
            lo_tot += lo_x;
            hi_tot += hi_x;
        }
        // println!("lo {:?} {:?} num {:?} hi {:?} {:?}", lo, lo_tot, num_flate, hi, hi_tot);
        if lo_tot > num_flate {
            hi = mid;
            continue;
        }
        if hi_tot < num_flate {
            lo = mid;
            continue;
        }
        // lo_tot <= num_flate <= hi_tot
        break;
    }
    // rounding
    let bounding_cutoff = (lo + hi) / 2.0;
    let mut handles = Vec::with_capacity(params.parallelism_factor);
    for mut vec in samp.data.clone() {
        handles.push(tokio::spawn(async move {
            let mut count = 0;
            for entry in vec.iter_mut() {
                match rounding {
                    Rounding::Down => {
                        if *entry >= bounding_cutoff {
                            *entry = samp.round as f64 * (0.0 - lambda);
                            count += 1;
                        }
                    },
                    Rounding::Up => {
                        if *entry <= bounding_cutoff {
                            *entry = samp.round as f64 * (1.0 - lambda);
                            count += 1;
                        }
                    },
                }
            }
            (vec, count)
        }));
    }
    let mut data = Vec::default();
    let mut count = 0;
    for (i, handle) in handles.into_iter().enumerate() {
        let (vec, count_partial) = handle.await.unwrap();
        data.push(vec);
        count += count_partial;
    }
    Samples { round: samp.round, data }
}

/// Find the expected reward of the adversary facing cost `lambda` to play each round.
async fn simulate(params: &Parameters, lambda: f64) -> f64 {
    let mut data = Vec::new();
    for _ in 0..params.parallelism_factor {
        data.push(Vec::from([0.0]));
    }
    let mut dist = Samples { round: 1, data };
    for i in 1..params.round_depth {
        // println!("round {:?} at {:?}", i, SystemTime::now());
        dist = finite_sample_add_layer(dist, params, lambda).await;
    }
    let adv_cdf = precompute_cdf(dist, params.epsilon, lambda, params.rounding).await;
    adv_cdf.exp()
}

/// TODO: desc
async fn binary_search(params: &Parameters) -> f64 {
    let mut lo = params.alpha;
    let mut lo_reward = simulate(&params, lo).await;
    let mut hi = 1.0;
    let mut hi_reward = lo_reward + -1.0 * (hi - lo);
    // println!("init lo re = {:?}, hi re = {:?}, time = {:?}", lo_reward, hi_reward, SystemTime::now());
    loop {
        let mut lambda = lo + (hi - lo) * (lo_reward / (lo_reward - hi_reward));
        lambda = lambda.max(0.0).min(1.0);
        // println!("lo {:?} re {:?}, hi {:?} re {:?}, lambda {:?}", lo, lo_reward, hi, hi_reward, lambda);
        let reward = simulate(&params, lambda).await;
        // println!("lambda = {:?}, reward = {:?}, time = {:?}", lambda, reward, SystemTime::now());
        if reward.abs() < 0.001 { return lambda; }
        if reward > 0.0 {
            lo = lambda;
            lo_reward = reward;
        }
        if reward < 0.0 {
            hi = lambda;
            hi_reward = reward;
        }
    }
}

#[derive(Debug, Clone)]
struct SimResult {
    alpha: f64,
    beta: f64,
    lower_bound: f64,
    upper_bound: f64
}

#[tokio::main]
async fn main() {
    let mut results = Vec::default();
    for beta in [0.0, 0.5, 1.0] {
        for alpha in [0.25, 0.375, 0.5] {
            let mut lower_bound = 0.0;
            let mut upper_bound = 0.0;
            for rounding in [Rounding::Down, Rounding::Up] {
                println!("{:?} {:?} {:?} at {:?}", rounding, alpha, beta, SystemTime::now());
                let params = Parameters {
                    adv_coins: 10,
                    epsilon: 0.001,
                    eta: 0.0001,
                    alpha,
                    beta,
                    rounding,
                    samples_drawn: 10_000_000, 
                    parallelism_factor: 100,
                    round_depth: 20,
                    chernoff_error: 0.0005
                };
                let lambda = binary_search(&params).await;
                match rounding {
                    Rounding::Down => lower_bound = lambda,
                    Rounding::Up => upper_bound = lambda
                };
            }
            results.push( SimResult { alpha, beta, lower_bound, upper_bound });
            println!("{:?} at {:?}", SimResult { alpha, beta, lower_bound, upper_bound }, SystemTime::now());
        }
    }
    println!("{:?}", results);
}

mod tests {
    use super::*;

    const DELTA: f64 = 0.000001;

    fn assert_approx_eq(a: &f64, b: &f64) {
        println!("{:#?}, {:#?}", a, b);
        assert!((a - b).abs() < DELTA);
    }

    // Tight approx uni dist over [0, 1]
    async fn uniform_dist() -> Pdf {
        let mut data = Vec::<f64>::new();
        for i in 0..10_000 {
            data.push(0.0005 + 0.0001 * (i as f64));
        }
        return Samples {
            round: 1,
            data: Vec::from([data])
        }.to_pdf(0.0, 1.0, 0.0001, Rounding::Down).await;
    }

    #[test]
    fn expo_dist() {
        // dist with parameter 2
        // so EV 0.5, stdev 0.5, e^{-2} frac passes 1
        let dist = ExpoDist::new(2.0);
        let mut sum = 0.0;
        let mut passed = 0;
        for _ in 0..10_000 {
            let y = dist.sample();
            sum += y;
            if y > 1.0 { passed += 1; }
        }
        // EV 0.5, stdev 0.005
        let avg = sum / 10_000.0;
        assert!(avg < 0.5 + 0.01);
        assert!(avg > 0.5 - 0.01);
        // EV e^{-2}, stdev less than 0.005 too
        let frac_passed = passed as f64 / 10_000.0;
        println!("{:?}", frac_passed);
        let e = std::f64::consts::E;
        assert!(frac_passed < e.powf(-2.0) + 0.01);
        assert!(frac_passed > e.powf(-2.0) - 0.01);
    }

    #[tokio::test]
    async fn adv_draw() {
        let uniform = uniform_dist().await.to_cdf();
        let mut tot = 0.0;
        for _ in 0..10_000 {
            // 9 coins, 0.5 expo param, uni 0-1 rewards each time
            let advdraw = draw_adv(&uniform, 9, 0.5);
            // beta = 1, 10 coins beat public => all guaranteed to win
            // EV is one plus max of 9 uniforms = 1.9
            tot += advdraw.best_from(9, 0.5, 1.0);
        }
        // expected 1.9, stdev < 0.005
        let avg = tot / 10_000.0;
        println!("{:?}", avg);
        assert!(avg < 1.9 + 0.01);
        assert!(avg > 1.9 - 0.01);
        // draw with zero reward on coin 1, nonzero reward on coin 2
        let advdraw = AdvDraw {
            coins: Vec::from([0.0, 1.0, 2.0]),
            rewards: Vec::from([0.0, 0.0, 1.0])
        };
        // unseen honest is 25%, so parameter .25 implies
        // e^{-.25} win probability
        let score1 = advdraw.best_from(1, 0.5, 0.5);
        let e = std::f64::consts::E;
        assert_approx_eq(&score1, &e.powf(-0.25));
        // now adding the second coin: e^{-.5} win probability and score 2
        // when win. this is bigger, so should get it as best:
        let score2 = advdraw.best_from(2, 0.5, 0.5);
        assert_approx_eq(&score2, &(2.0 * e.powf(-0.5)));
        // however, with alpha = 0.2, beta = 0, there's 80% honest, and coin
        // 1 is the better choice (e^{-.8} vs 2e^{-1.6})
        let score3 = advdraw.best_from(2, 0.2, 0.0);
        assert_approx_eq(&score3, &e.powf(-0.8));
        // finally, a few checks of the best_x_honest_prob functions.
        // unseen: 35% mass
        assert_approx_eq(&e.powf(-3.0 * 0.35), &AdvDraw::beat_unseen_honest_prob(3.0, 0.5, 0.3));
        // seen: 15% mass
        assert_approx_eq(&e.powf(-3.0 * 0.15), &AdvDraw::beat_seen_honest_prob(3.0, 0.5, 0.3));
        // either: 50% mass
        assert_approx_eq(&e.powf(-3.0 * 0.50), &AdvDraw::beat_honest_prob(3.0, 0.5));
    }

    #[tokio::test]
    async fn sample() {
        // lower 0.2628173828125, upper 0.2635498046875
        // todo: overleaf functions, finish tests, overleaf tests, benchmark 10^8, some trial simulations
    }

    #[tokio::test]
    async fn flate() {
        // 0 to 1 by 0.01 hops, 100 values
        let mut data = Vec::new();
        for i in 0..100 {
            data.push(i as f64 * 0.01);
        }
        let samps = Samples {
            round: 2,
            data: Vec::from([data])
        };
        // e^{-2} chernoff error => shift pct = sqrt(ln(1 / chernoff error) / 2n )
        // = sqrt(2 / (2 * 100)) = 1/10
        let e = std::f64::consts::E;
        let mut params = Parameters {
            adv_coins: 10,
            epsilon: 0.0001, 
            eta: 0.001,
            alpha: 0.25,
            beta: 1.0,
            rounding: Rounding::Down, // used
            samples_drawn: 100, // used
            parallelism_factor: 1, // used
            round_depth: 20,
            chernoff_error: e.powf(-2.0) // used
        };
        // -0.8 to 1.2
        // exp is .9 * (0.0 + 0.89)/2 + .1 * -0.8 = 0.4005 - 0.08 = 0.3205
        println!("{:?}", fooflate(samps.clone(), &params, 0.4).await.data[0]);
        let down = fooflate(samps.clone(), &params, 0.4).await;
        let mut sum = 0.0;
        for i in 0..100 {
            sum += down.data[0][i]
        }
        assert_approx_eq(&0.3205, &(sum/100.0));
        // exp is .9 * (0.1 + 0.99)/2 + .1 * 1.2 = 0.4905 + .12 = 0.6105
        params.rounding = Rounding::Up;
        let up = fooflate(samps.clone(), &params, 0.4).await;
        let mut sum = 0.0;
        for i in 0..100 {
            sum += up.data[0][i]
        }
        assert_approx_eq(&0.6105, &(sum/100.0));
        // sqrt(x) for x=0 to 49
        params.parallelism_factor = 5;
        params.samples_drawn = 50;
        let mut data = Vec::new();
        for i in 0..5 {
            data.push(Vec::new());
            for j in 0..10 {
                data[i].push((10.0 * i as f64 + j as f64).sqrt());
            }
        }
        let samps = Samples { round: 10, data };
        let up = fooflate(samps, &params, 0.1).await;
        // -1.0 to 9.0
        // shift pct = sqrt(2 / (2 * 50)) = sqrt(1/50) = 8 values
        let mut tgt = 0.0;
        for i in 0..50 {
            if i < 8 { 
                tgt += 9.0
            } else {
                tgt += (i as f64).sqrt();
            }
        }
        let mut sum = 0.0;
        for i in 0..5 {
            println!("{:?}", up.data[i]);
            for j in 0..10 {
                sum += up.data[i][j];
            }
        }
        assert_approx_eq(&tgt, &sum);
        // all same val => should round all
        params.parallelism_factor = 1;
        params.samples_drawn = 10;
        let mut data = Vec::new();
        for i in 0..10 {
            data.push(0.0);
        }
        let samps = Samples { round: 1, data: Vec::from([data]) };
        let up = fooflate(samps, &params, 0.3).await;
        for i in 0..10 {
            assert_approx_eq(&0.7, &up.data[0][i]);
        }
    }
}
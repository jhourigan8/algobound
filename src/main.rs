mod precomp;

use std::env;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand::Rng;
use rayon::prelude::*;
use core::array;
use std::sync::Arc;
use std::time::{Instant, SystemTime};
use std::ops::Not;
use std::fs::File;
use std::fs::OpenOptions;
use std::io::prelude::*;

use crate::precomp::*;

/// An enum specifying if the current simulation is to yield
/// a lower bound or an upper bound to the true `lambda`.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub enum Rounding {
    Down,
    Up
}

impl Not for Rounding {
    type Output = Self;

    fn not(self) -> Self::Output {
        match self {
            Rounding::Down => Rounding::Up,
            Rounding::Up => Rounding::Down
        }
    }
}

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
    Long(Cdf, Emax)
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
        Precomp::Long(cdf, emax) => {
            let adv = draw_adv(&cdf, adv_coins, alpha, rng);
            let mut cum = 0f64; 
            let opposite = &!rounding.clone();
            for i_star in 1..=adv_coins {
                let best = adv.best_from(i_star, alpha, beta);

                let win_old = AdvDraw::beat_seen_honest_prob(adv.coins[i_star], alpha, beta);
                let win_new = AdvDraw::beat_seen_honest_prob(adv.coins[i_star + 1], alpha, beta);

                // bigger theta => smaller inside => bigger result (more dist carry)
                let theta_old = win_old.powf((1.0 - beta) / beta);
                let theta_new = win_new.powf((1.0 - beta) / beta);
                let theta_max = theta_old.max(theta_new);
                let theta_min = theta_old.min(theta_new);

                // heuristic: linear loss gap
                // so try to uniform dot the thetas
                // 1x: 
                // 100x: 0.2538081466053813 to 0.2566943525664075, 20.467504791s
                let loss_heuristic = (theta_max - theta_min) * (win_old - win_new);
                let num_steps = (100.0 * loss_heuristic).ceil();
                let theta_step = (theta_max - theta_min) / num_steps;
                let init_step = match &rounding {
                    Rounding::Down => 0.0,
                    Rounding::Up => 1.0
                };
                let mut sum = 0.0;
                for i in 0..(num_steps as usize) {
                    let theta = theta_min + (i as f64 + init_step) * theta_step;
                    let integrand = if theta != 0.0 {
                        theta * emax.get(best / theta, &rounding)
                    } else {
                        best
                    };
                    sum += integrand;
                }
                sum /= num_steps;
                let bound_diff = (win_old - win_new) * sum;
                let diff = bound_diff;
                cum += diff;
            }
            cum - lambda
        }
    }
}

/// Given samples `samp` and `lambda` compute the distribution's cdf.
async fn precompute_cdf(samp: Samples, epsilon: f64, lambda: f64, rounding: Rounding) -> Cdf {
    // 1a: compute adversary pdf
    let (min, max) = (-lambda, samp.round as f64 * (1.0 - lambda));
    let adv_pdf = samp.to_pdf(min, max, epsilon, rounding.clone());
    // 1b: convert into cdf
    let adv_cdf = adv_pdf.to_cdf();
    adv_cdf
} 

/// Given samples `samp` and `lambda` precompute the requisite data for efficient sampling.
async fn precompute(
    adv_cdf: Cdf, eta: f64, beta: f64, rounding: Rounding
) -> Precomp {
    // println!("round {:#?} expected win {:#?}", round, adv_cdf.exp());
    if beta == 0.0 {
        return Precomp::None(adv_cdf);
    }
    // 1c and 1d: compute E_max thetas
    let emax = adv_cdf.to_emax();
    if beta == 1.0 {
        return Precomp::Short(adv_cdf, emax);
    }
    Precomp::Long(adv_cdf, emax)
}

/// A struct to carry around needed parameters.
#[derive(Clone, Copy, Debug)]
struct Parameters {
    /// The number of coins drawn for the adversary assuming their stake is spread
    /// across arbitrarily many accounts.
    adv_coins: usize,
    /// Discretization interval size for all functions of the adversary's rewards.
    /// Namely, used in `Pdf`, `Cdf`, `Emax`, and the second argument to `Table`.
    epsilon: f64,
    /// Discretization interval size for the first argument to `Table`.
    eta: f64,
    /// The fraction of stake controlled by the adversary.
    alpha: f64,
    /// The fraction of honest stake credential broadcasts which the adversary
    /// is permitted to see before having to decide what to broadcast each round.
    beta: f64,
    /// The type of bound to be computed.
    rounding: Rounding,
    /// The number of samples drawn to compute the reward distribution on each round.
    samples_drawn: usize,
    /// The number of independent tasks we split any parallel work into.
    parallelism_factor: usize,
    /// The number of rounds simulated before returning the adversary's reward for a
    /// given value of `lambda`.
    round_depth: usize,
    /// Upper bound on the probability that a single `inflate` / `deflate` 
    /// procedure fails to produce a stochastic upper / lower bound.
    chernoff_error: f64,
    /// TODO
    flate_group: usize
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
async fn finite_sample_add_layer(adv_cdf: Cdf, last_round: usize, params: &Parameters, lambda: f64) -> Cdf {
    let mut new_samps = Samples { round: last_round + 1, data: Vec::default() };
    let mut handles = Vec::with_capacity(params.parallelism_factor);
    let now = Instant::now();
    let precomp = Arc::new(precompute(adv_cdf.clone(), params.eta, params.beta, params.rounding).await);
    println!("table time: {:?}", now.elapsed());
    let now = Instant::now();
    for _ in 0..params.parallelism_factor {
        handles.push(tokio::spawn(add_layer_helper(precomp.clone(), params.clone(), lambda)));
    }
    for (i, handle) in handles.into_iter().enumerate() {
        new_samps.data.append(&mut handle.await.unwrap());
    }
    println!("sampling time: {:?}", now.elapsed());
    // let cdf = precompute_cdf(new_samps.clone(), params.epsilon, lambda, params.rounding).await;
    // println!("unflated: exp {:?} var {:?} 4th {:?} 6th {:?}", cdf.exp(), cdf.moment(2), cdf.moment(4), cdf.moment(6));
    let now = Instant::now();
    fooflate(&mut new_samps, params, lambda).await;
    println!("fooflate time: {:?}", now.elapsed());
    // println!("fooflate {:?}", now.elapsed());
    let now = Instant::now();
    let cdf = precompute_cdf(new_samps, params.epsilon, lambda, params.rounding).await;
    println!("cdf time: {:?}", now.elapsed());
    // println!("tocdf {:?}", now.elapsed());
    // println!("flated: exp {:?} var {:?} 4th {:?} 6th {:?}", cdf.exp(), cdf.moment(2), cdf.moment(4), cdf.moment(6));
    cdf
}

/// Experimental.
async fn fooflate(new_samps: &mut Samples, params: &Parameters, lambda: f64) {
    let shift_percent = ((1.0 / params.chernoff_error).ln() / (2 * params.samples_drawn) as f64).sqrt() as f64;
    // println!("shift pct {:?}", shift_percent);
    let num_shift = (shift_percent * params.samples_drawn as f64).ceil() as usize;
    new_samps.data.par_sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    match params.rounding {
        Rounding::Up => {
            // println!("shifty {:?}", num_shift);
            let mut lo_idx = 0;
            let mut hi_idx = new_samps.data.len();
            'outer: loop {
                for _ in 0..params.flate_group {
                    new_samps.data[lo_idx] = if hi_idx == new_samps.data.len() {
                        new_samps.round as f64 * (1.0 - lambda)
                    } else {
                        new_samps.data[hi_idx]
                    };
                    lo_idx += 1;
                    if lo_idx >= num_shift {
                        break 'outer;
                    }
                }
                hi_idx -= 1;
            }
        },
        Rounding::Down => {
            for i in new_samps.data.len() - num_shift..new_samps.data.len() {
                new_samps.data[i] = -lambda;
            }
        }
    }
}

/// Find the expected reward of the adversary facing cost `lambda` to play each round.
async fn simulate(params: &Parameters, lambda: f64) -> f64 {
    let dist = Samples { round: 0, data: Vec::from([0.0]) };
    let mut cdf = precompute_cdf(dist, params.epsilon, lambda, params.rounding).await;
    for i in 0..params.round_depth {
        // println!("round {:?} at {:?}", i, SystemTime::now());
        cdf = finite_sample_add_layer(cdf, i, params, lambda).await;
    }
    cdf.exp()
}

/// Search for the value of `lambda` which yields zero expected adversary reward.
async fn search(params: &Parameters) -> f64 {
    let target = 0.25 / (params.samples_drawn as f64).sqrt();
    let mut lo = params.alpha;
    let mut lo_reward = simulate(&params, lo).await;
    let mut hi = params.alpha + params.alpha * params.alpha; // empirically always an upper bound
    let mut hi_reward = simulate(&params, hi).await;
    // println!("init lo re = {:?}, hi re = {:?}, time = {:?}", lo_reward, hi_reward, SystemTime::now());
    loop {
        let mut lambda = match params.rounding {
            Rounding::Down => lo + (hi - lo) * ((lo_reward - target) / (lo_reward - hi_reward)),
            Rounding::Up => lo + (hi - lo) * ((lo_reward + target) / (lo_reward - hi_reward))
        };
        lambda = lambda.max(0.0).min(1.0);
        // println!("lo {:?} re {:?}, hi {:?} re {:?}, lambda {:?}", lo, lo_reward, hi, hi_reward, lambda);
        let reward = simulate(&params, lambda).await;
        // println!("lambda = {:?}, reward = {:?}, time = {:?}", lambda, reward, SystemTime::now());
        match params.rounding {
            Rounding::Down => if reward > 0.0 && reward < 2.0 * target { println!("EXTRA {:?}", reward); return lambda; },
            Rounding::Up => if reward < 0.0 && reward > -2.0 * target { println!("EXTRA {:?}", -reward); return lambda; }
        }
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

async fn compute_interval(
    alpha: f64, beta: f64, chernoff_error: f64, target_width: f64, samp_scale: f64
) -> SimResult {
    let mut lower_bound = 0.0;
    let mut upper_bound = 0.0;
    for rounding in [Rounding::Down, Rounding::Up] {
        let params = compute_params(alpha, beta, chernoff_error, target_width, samp_scale, rounding);
        let lambda = search(&params).await;
        match rounding {
            Rounding::Down => lower_bound = lambda,
            Rounding::Up => upper_bound = lambda
        };
    }
    let res = SimResult { alpha, beta, lower_bound, upper_bound };
    res
}

fn check_vals(point: f64, target_width: f64) -> (f64, f64) {
    (point - 0.39 * target_width, point + 0.61 * target_width)
}

async fn check_point(
    lo: f64, hi: f64, alpha: f64, beta: f64, chernoff_error: f64, target_width: f64, samp_scale: f64
) -> bool {
    println!("checking {:?} to {:?}", lo, hi);
    let mut success = true;
    for rounding in [Rounding::Down, Rounding::Up] {
        let params = compute_params(alpha, beta, chernoff_error, target_width, samp_scale, rounding);
        let lambda = match rounding {
            Rounding::Down => lo,
            Rounding::Up => hi
        };
        let reward = simulate(&params, lambda).await;
        match rounding {
            Rounding::Down => if reward < 0.0 { success = false; },
            Rounding::Up => if reward > 0.0 { success = false; },
        }
        let mut bounds = OpenOptions::new()
            .read(true)
            .append(true)
            .create(true)
            .open("bounds.txt")
            .unwrap();
        bounds.write_all(
            &format!(
                "reward on {:?} is {:?} round {:?}\n", lambda, reward, rounding
            ).into_bytes()
        ).unwrap();
        bounds.flush().unwrap();
        println!("reward on {:?} is {:?}", lambda, reward);
    }
    success
}

fn compute_params(
    alpha: f64, beta: f64, chernoff_error: f64, mut target_width: f64, samp_scale: f64, rounding: Rounding
) -> Parameters {
    if alpha > 0.29 { panic!("alpha too large"); }
    let round_coin_map = [
        (0.02, 4, 5), (0.05, 5, 5), (0.08, 6, 6), (0.10, 6, 6), (0.12, 7, 6),
        (0.15, 8, 7), (0.17, 9, 7), (0.2, 10, 8), (0.23, 12, 9), (0.25, 13, 9),
        (0.26, 14, 9), (0.27, 16, 9), (0.28, 17, 9), (0.29, 18, 9)
    ];
    let mut i = 0;
    let (adv_coins, round_depth) = loop {
        if alpha <= round_coin_map[i].0 {
            break (round_coin_map[i].1, round_coin_map[i].2)
        }
        i += 1;
    };
    let foo_size = (1.0 / chernoff_error).ln().sqrt();
    if chernoff_error != 1.0 { 
        target_width /= 1.0 + 0.3 * foo_size * (round_depth as f64);
    }
    let (epsilon, eta) = (0.02 * target_width * target_width, 0.0);
    let samples_drawn = (50.0 * 0.125 / (target_width * target_width)).ceil() as usize;
    let p = Parameters {
        adv_coins,
        epsilon,
        eta,
        alpha,
        beta,
        rounding,
        samples_drawn, 
        parallelism_factor: 100,
        round_depth,
        chernoff_error,
        flate_group: 20
    };
    println!("{:?}", p);
    p
}

static HELP_MSG : &str =
"usage: cargo run fixed-parameter fixed-value step-size target-width chernoff-error
\tfixed-parameter: string literal `alpha` or `beta`
\tfixed-value: value of fixed parameter
\tstep-size: step size of other parameter
\ttarget-width: desired output interval width
\tchernoff-error: upper bound on fooflate error probability each round
\tsamp-scale?: optional parameter to scale number of samples taken";

#[tokio::main]
async fn main() {
    // bounding with chebyshev
    // get samples for X^2
    // run inflate procedure on those
    // this gives high prob bound on Var[X]
    // (napkin: 1 in 1000 or less up to r^2 = 100, so maybe +0.1)
    // bound should be at most 1/4, so 1/2 stdev bound
    // with 1 milli samples, 1/2k stdev bound
    // then 1/200 loss.. not great
    // we could try this with barry-essen
    // distance at most rho/sigma^3sqrt(n) to normal cdf
    // this is at most r/sigmasqrt(n)
    // again if bound sigma can get smth like 4/sqrt(n)
    let args: Vec<String> = env::args().collect();
    if args.len() != 6 && args.len() != 7 {
        panic!("{}", HELP_MSG);
    }
    let target_width: f64 = args[4].parse().expect(HELP_MSG);
    let chernoff_error = args[5].parse().expect(HELP_MSG);
    let mut results = OpenOptions::new()
        .read(true)
        .append(true)
        .create(true)
        .open("results.txt")
        .unwrap();
    let start = Instant::now();
    let (alpha_vals, beta_vals) = {
        let val: f64 = args[2].parse().expect(HELP_MSG);
        let step: f64 = args[3].parse().expect(HELP_MSG);
        let mut vec = Vec::new();
        match args[1].as_str() {
            "alpha" => {
                vec.push(0.0);
                while vec[vec.len() - 1] != 1.0 {
                    let mut val = vec[vec.len() - 1] + step;
                    if val > 1.0 - 0.001 { val = 1.0; }
                    vec.push(val);
                }
                (Vec::from([val]), vec)
            },
            "beta" => {
                vec.push(0.01);
                while vec[vec.len() - 1] != 0.29 { 
                    let mut val = vec[vec.len() - 1] + step;
                    if val > 0.29 - 0.001 { val = 0.29; }
                    vec.push(val);
                }
                (vec, Vec::from([val]))
            },
            "pair" => {
                let (alpha, beta) = (val, step);
                let point_est = if args.len() == 6 { 
                    let interval = compute_interval(alpha, beta, 1.0, target_width, 1.0).await;
                    println!("Unflated guess computed {:?}!", start.elapsed());
                    println!("Unflated guess was {:?} to {:?}", interval.lower_bound, interval.upper_bound);
                    println!("Elapsed time was {:?}", start.elapsed());
                    (interval.upper_bound + interval.lower_bound) / 2.0
                } else { 
                    args[6].parse().expect(HELP_MSG) 
                };
                let (lo, hi) = check_vals(point_est, target_width);
                let success = check_point(lo, hi, alpha, beta, chernoff_error, target_width, 1.0).await;
                if success {
                    results.write_all(
                        &format!(
                            "{:?}, {:?}, {:?}, {:?}, {:?}\n", 
                            alpha, 
                            beta, 
                            chernoff_error,
                            lo,
                            hi
                        ).into_bytes()
                    ).unwrap();
                    println!("Check success!");
                    println!("Runtime was {:?}", start.elapsed());
                } else {
                    println!("fail!");
                }
                return;
            }
            _ => { panic!("{}", HELP_MSG); }
        }
    };
    let samp_scale = if args.len() == 6 { 1.0 } else { args[6].parse().expect(HELP_MSG) };
    let mut ctr = 1;
    let num = alpha_vals.len() * beta_vals.len();
    for alpha in alpha_vals {
        for beta in &beta_vals {
            println!("Running simulation {:?} of {:?}.", ctr, num);
            let interval = compute_interval(alpha, *beta, chernoff_error, target_width, samp_scale).await;
            println!("{:?}", interval);
            results.write_all(
                &format!(
                    "{:?}, {:?}, {:?}, {:?}, {:?}\n", 
                    alpha, 
                    beta, 
                    chernoff_error,
                    interval.lower_bound,
                    interval.upper_bound
                ).into_bytes()
            ).unwrap();
            results.flush().unwrap();
            let elapsed = start.elapsed();
            let est = elapsed
                .checked_mul((num - ctr) as u32).expect("")
                .checked_div(ctr as u32).expect("");
            ctr += 1;
            println!("Estimated remaining time: {:?}", est);
        }
    }
    println!("Simulations complete!");
    println!("Runtime was {:?}", start.elapsed());
    
    // let checks = check_point(tight, 0.01, 0.25, 0.0, 0.0000001, 0.0, 1.0, 7, 10, 1_000_000, 20, 100).await;
    // todo: 
    // 1 get rid of last round inflate deflate
    // 2 function to set number of rounds
    // 3 running inflate deflate
    // alpha = 0.01, 0.07, 0.15, 0.25 across all beta in 0.04 increments
    // beta = 0.00, 0.50, 1.0 across all alpha in 0.01 increments

    // hoeffding inequality: pr t dev <= exp(-2t^2/nr^2), take t=omega(rsqrt(n)), e.g. 5rsqrt(n)
    // 

    /*
    // res SimResult { alpha: 0.25, beta: 0.0, lower_bound: 0.2523253688367698, upper_bound: 0.25374179234262856 }
    compute_interval(0.25, 0.0, 0.0000001, 0.0, 1.0, 7, 10, 500_000, 20, 100).await; -> 0.2530
    // res SimResult { alpha: 0.25, beta: 0.01, lower_bound: 0.2515111846113837, upper_bound: 0.25501946390208863 }
    compute_interval(0.25, 0.01, 0.0010, 0.0001, 1.0, 7, 10, 500_000, 20, 100).await; -> 0.2532
    // res SimResult { alpha: 0.25, beta: 0.33, lower_bound: 0.2529557762463644, upper_bound: 0.25613922998564437 }
    compute_interval(0.25, 0.33, 0.0010, 0.0001, 1.0, 7, 10, 500_000, 20, 100).await; -> 0.2545
    // res SimResult { alpha: 0.25, beta: 0.67, lower_bound: 0.2548422152727763, upper_bound: 0.25855874210641727 }
    compute_interval(0.25, 0.67, 0.0010, 0.0001, 1.0, 7, 10, 500_000, 20, 100).await; -> 0.2567
    // res SimResult { alpha: 0.25, beta: 0.99, lower_bound: 0.26110185650623824, upper_bound: 0.26559568812136763 }
    compute_interval(0.25, 0.99, 0.0010, 0.0001, 1.0, 7, 10, 500_000, 20, 100).await; -> 0.2633
    // res SimResult { alpha: 0.25, beta: 1.0, lower_bound: 0.2627840073683429, upper_bound: 0.26393765814823444 }
    compute_interval(0.25, 1.0, 0.0000001, 0.0, 1.0, 7, 10, 500_000, 20, 100).await; -> 0.2634
    */
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
            data
        }.to_pdf(0.0, 1.0, 0.0001, Rounding::Down);
    }

    #[test]
    fn expo_dist() {
        // dist with parameter 2
        // so EV 0.5, stdev 0.5, e^{-2} frac passes 1
        let dist = ExpoDist::new(2.0);
        let mut sum = 0.0;
        let mut passed = 0;
        let mut rng = rand::rngs::SmallRng::from_entropy();
        for _ in 0..10_000 {
            let y = dist.sample(&mut rng);
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
        let mut rng = rand::rngs::SmallRng::from_entropy();
        for _ in 0..10_000 {
            // 9 coins, 0.5 expo param, uni 0-1 rewards each time
            let advdraw = draw_adv(&uniform, 9, 0.5, &mut rng);
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
    async fn flate() {
        // 0 to 1 by 0.01 hops, 100 values
        let mut data = Vec::new();
        for i in 0..100 {
            data.push(i as f64 * 0.01);
        }
        let mut samps = Samples {
            round: 2,
            data
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
            chernoff_error: e.powf(-2.0), // used
            flate_group: 1 // used
        };
        // round down: 10 sent to -lambda = -0.4,
        // exp is .9 * (0.0 + 0.89)/2 + .1 * -0.4 = 0.3605
        fooflate(&mut samps, &params, 0.4).await;
        let mut sum = 0.0;
        for i in 0..100 {
            sum += samps.data[i];
            println!("{:?}", samps.data[i]);
        }
        assert_approx_eq(&0.3605, &(sum/100.0));
        // round up: 1 sample sent to 2*(1-lambda) = 1.2
        // other 9 sent to 0.91 thru 0.99
        // exp is .9 * (0.1 + 0.99)/2 + .01 * 1.2 + 0.09 * 0.95 = 0.5880
        let mut data = Vec::new();
        for i in 0..100 {
            data.push(i as f64 * 0.01);
        }
        let mut samps = Samples {
            round: 2,
            data
        };
        params.rounding = Rounding::Up;
        fooflate(&mut samps, &params, 0.4).await;
        let mut sum = 0.0;
        for i in 0..100 {
            sum += samps.data[i]
        }
        assert_approx_eq(&0.5880, &(sum/100.0));
        // sqrt(x) for x=0 to 49
        params.samples_drawn = 50;
        let mut data = Vec::new();
        for i in 0..50 {
            data.push((i as f64).sqrt());
        }
        let mut up = Samples { round: 10, data };
        fooflate(&mut up, &params, 0.1).await;
        // shift pct = sqrt(2 / (2 * 50)) = sqrt(1/50) = 8 values
        let mut tgt = 0.0;
        for i in 0..50 {
            if i < 8 { 
                // 10 * (1 - lambda) = 9
                if i == 0 {
                    tgt += 9.0;
                }
                // i^th smallest samp
                else {
                    tgt += ((50 - i) as f64).sqrt();
                }
            } else {
                tgt += (i as f64).sqrt();
            }
        }
        let mut sum = 0.0;
        for i in 0..50 {
            sum += up.data[i];
        }
        assert_approx_eq(&tgt, &sum);
        // shift pct = sqrt(2 / (2 * 10)) = sqrt(1/10) = 4 values
        // max is 1 * (1 - lambda) = 0.7
        params.samples_drawn = 10;
        let mut data = Vec::new();
        for i in 0..10 {
            data.push(0.0);
        }
        let mut up = Samples { round: 1, data };
        fooflate(&mut up, &params, 0.3).await;
        let mut sum = 0.0;
        for i in 0..10 {
            sum += up.data[i];
        }
        assert_approx_eq(&0.7, &sum);
    }
}
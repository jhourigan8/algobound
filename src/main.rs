use rand::Rng;
use core::array;
use std::{mem, sync::Arc};

#[derive(Clone, Default, Debug)]
struct ApproxFunc<T> {
    buckets: usize, // Number of buckets
    min: f64, // Min value in range
    max: f64, // Max value of range
    vals: Vec<T>, // Values in buckets
}

impl<T: Clone + Default> ApproxFunc<T> {
    pub fn new(min: f64, max: f64, error: f64) -> Self {
        let mut buckets = ((max - min) / error).ceil() as usize;
        if buckets == 0 { buckets = 1; }
        let vals = vec![T::default(); buckets];
        Self {
            buckets,
            min,
            max,
            vals,
        }
    }
}

impl<T: std::fmt::Debug> ApproxFunc<T> {
    pub fn idx(&self, x: f64) -> usize {
        if x > self.max || x < self.min { println!("{:?}, {:?}, {:?}", self.min, self.max, x); panic!("outside of function domain"); }
        let step = (self.max - self.min) / self.buckets as f64;
        let mut bucket = ((x - self.min) / step).floor() as usize;
        if bucket == self.buckets { bucket -= 1; }
        bucket
    }

    pub fn set(&mut self, x: f64, y: T) -> Option<T> {
        let idx = self.idx(x);
        Some(mem::replace(&mut self.vals[idx], y))
    }

    pub fn get_mut(&mut self, x: f64) -> &mut T {
        let idx = self.idx(x);
        &mut self.vals[idx]
    }

    pub fn get(&self, x: f64) -> &T {
        &self.vals[self.idx(x)]
    }
}

impl ApproxFunc<f64> {
    pub async fn samples_to_pdf(min: f64, max: f64, error: f64, samples: Vec<f64>) -> Self {
        let mut pdf = Self::new(min, max, error);
        let delta = 1f64 / samples.len() as f64;
        for sample in samples {
            *pdf.get_mut(sample) += delta;
        }
        pdf
    }

    pub async fn para_samples_to_pdf(min: f64, max: f64, error: f64, samples: Samples) -> Self {
        let mut pdf = Self::new(min, max, error);
        let mut handles = Vec::with_capacity(PARALLELISM_FACTOR);
        for vec in samples.data {
            handles.push(tokio::spawn(Self::samples_to_pdf(min, max, error, vec)));
        }
        for handle in handles {
            pdf.add(&handle.await.unwrap())
        }
        pdf.mul_const(0.1);
        pdf
    }

    pub fn pdf_to_cdf(&self) -> Self {
        let mut cdf = vec![0f64; self.buckets];
        let mut cum: f64 = 0f64;
        for (idx, p) in self.vals.iter().enumerate() {
            cum += p;
            cdf[idx] = cum;
        }
        ApproxFunc::<f64> {
            buckets: self.buckets,
            min: self.min,
            max: self.max,
            vals: cdf,
        }
    }

    pub fn exp(&self) -> f64 {
        let step = (self.max - self.min) / self.buckets as f64;
        let min = match MODE {
            Mode::LowerBounding => self.min,
            Mode::UpperBounding => self.min + step
        };
        let mut tot: f64 = 0f64;
        let mut prev = 0f64;
        for i in 0..self.buckets {
            tot += (self.vals[i] - prev) * (min + step * i as f64);
            prev = self.vals[i];
        }
        tot
    }

    fn cdf_to_exp_max(&self) -> ApproxFunc<f64> {
        let step = (self.max - self.min) / self.buckets as f64;
        let mut exp = self.exp();
        let mut vals = vec![0f64; self.buckets];
        for i in 0..self.buckets {
            vals[i] = exp;
            exp += step * self.vals[i];
        }
        ApproxFunc::<f64> {
            buckets: self.buckets,
            min: self.min,
            max: self.max, 
            vals,
        }
    }

    fn emax_to_table(&self) -> ApproxFunc<ApproxFunc<f64>> {
        let mut table = ApproxFunc::<>::new(self.min, self.max, EPSILON);
        let mut gamma = self.min + EPSILON / 2f64;
        for _ in 0..self.buckets {
            table.set(gamma, self.partial_compute_table(gamma));
            gamma += EPSILON;
        }
        table
    }

    fn partial_compute_table(&self, gamma: f64) -> ApproxFunc<f64> {
        // TODO: assuming emax is g/e^... for big c. and just check fn in general
        let max = (1f64 / INTEGRAL_MASS_CUTOFF).ln() / (BETA * (1f64 - ALPHA));
        let mut cum = 0f64;
        let coeff = ETA * BETA * (1f64 - ALPHA);
        let mut vec = Vec::default();
        let mut c = 0f64;
        while c < max {
            let bcast_score = gamma / (-c * (1f64 - ALPHA) * (1f64 - BETA)).exp();
            let emax_score = if bcast_score < self.min { 
                self.get(self.min) 
            } else if bcast_score > self.max {
                &bcast_score
            } else {
                self.get(bcast_score)
            };
            cum += coeff * (-c * (1f64 - ALPHA)).exp() * *emax_score;
            vec.push(cum);
            c += ETA;
        }
        ApproxFunc::<f64> {
            buckets: vec.len(),
            min: 0f64,
            max: c,
            vals: vec
        }
    }

    pub fn mul(&mut self, other: &ApproxFunc<f64>) {
        let step = (self.max - self.min) / self.buckets as f64;
        for i in 0..self.buckets {
            self.vals[i] *= other.get(self.min + step * i as f64);
        }
    }

    pub fn mul_const(&mut self, c: f64) {
        for i in 0..self.buckets {
            self.vals[i] *= c;
        }
    }

    pub fn add(&mut self, other: &ApproxFunc<f64>) {
        let step = (self.max - self.min) / self.buckets as f64;
        for i in 0..self.buckets {
            self.vals[i] += other.get(self.min + step * i as f64);
        }
    }

    pub fn cdf_to_sample(&self) -> f64 {
        let mut rng = rand::thread_rng();
        let ptile: f64 = rng.gen();
        // find bucket where BUCK val <= ptile < BUCK + 1 val
        // lower: round down, upper: round up
        // invariant: BUCK in [lo..hi)
        let mut lo = 0;
        let mut hi = self.buckets;
        while lo + 1 != hi {
            let mid = (lo + hi) / 2;
            if self.vals[mid] <= ptile { lo = mid } else { hi = mid };
        }
        // this is only used to sample adversary rewards so just round down/up
        let buck = match MODE {
            Mode::LowerBounding => lo,
            Mode::UpperBounding => hi
        };
        self.min + (buck as f64 / self.buckets as f64) * (self.max - self.min)
    }
}

struct ExpoDist {
    a: f64
}

impl ExpoDist {
    fn new(a: f64) -> Self {
        Self { a }
    }

    fn sample(&self) -> f64 {
        let mut rng = rand::thread_rng();
        let ptile: f64 = rng.gen();
        -(1f64 - ptile).ln() / self.a
    }
}

#[derive(Debug)]
struct AdvDraw {
    coins: [f64; G+2],
    rewards: [f64; G+1]
}

impl AdvDraw {
    fn best_from(&self, i_star: usize) -> f64 {
        let mut best = 0f64;
        for i in 1..=i_star {
            let score = (1f64 + self.rewards[i]) * win_prob(self.coins[i]);
            if score > best {
                best = score;
            }
        }
        best
    }
}

fn win_prob(x: f64) -> f64 {
    if x == f64::MAX {
        0f64
    } else {
        (-x * (1f64 - BETA) * (1f64 - ALPHA)).exp()
    }
}

fn foo(x: f64) -> f64 {
    if x == f64::MAX {
        0f64
    } else {
        (-x * (1f64 - ALPHA)).exp()
    }
}

fn draw_adv(advdist: &ApproxFunc<f64>) -> AdvDraw {
    // 2a: draw adversary rewards and coins
    let expodist = ExpoDist::new(ALPHA);
    let mut coins = [0f64; G+2];
    let mut cum = 0f64;
    for i in 1..=G {
        cum += expodist.sample(); // draw exp
        coins[i] = cum;
    }
    coins[G+1] = f64::MAX;
    AdvDraw {
        coins,
        rewards: array::from_fn(|_| advdist.cdf_to_sample())
    }
}

fn sample_short(adv: &AdvDraw, emax: &ApproxFunc<f64>, lambda: f64) -> f64 {
    let mut cum = 0f64;
    for i_star in 1..=G {
        let best = adv.best_from(i_star);
        // println!("best is {:?}", best);
        cum += emax.get(best) * (foo(adv.coins[i_star]) - foo(adv.coins[i_star + 1]));
    }
    cum - lambda
}

fn sample_long(adv: &AdvDraw, table: &ApproxFunc<ApproxFunc<f64>>, lambda: f64) -> f64 {
    let mut cum = 0f64;
    for i_star in 1..=G {
        let best = adv.best_from(i_star);
        let partial = table.get(best);
        if adv.coins[i_star] > partial.max {
            break;
        }
        cum -= partial.get(adv.coins[i_star]);
        if adv.coins[i_star + 1] > partial.max { 
            cum += partial.get(partial.max);
            break;
        }
        cum += partial.get(adv.coins[i_star + 1]);
    }
    cum - lambda
}

const GAMMA: f64 = 0.001;
const EPSILON: f64 = 0.001;
const ETA: f64 = 0.004;
const ALPHA: f64 = 0.5;
const BETA: f64 = 1.0;
const G: usize = 20;
const N: usize = 200_000;
const T: usize = 30;
const INTEGRAL_MASS_CUTOFF: f64 = 0.00001;
const MODE: Mode = Mode::LowerBounding;
const PARALLELISM_FACTOR: usize = 10;

#[derive(Debug)]
enum Mode {
    LowerBounding,
    UpperBounding
}

#[derive(Debug)]
struct Samples {
    round: usize,
    data: [Vec<f64>; PARALLELISM_FACTOR]
}

async fn precompute_cdf(samp: Samples, lambda: f64) -> ApproxFunc<f64> {
    // 1a: compute adversary pdf
    let (min, max) = ((samp.round + 1) as f64 * (0f64 - lambda), 1f64 + (samp.round + 1) as f64 * (1f64 - lambda));
    let adv_pdf = ApproxFunc::<f64>::para_samples_to_pdf(min, max, EPSILON, samp).await;
    // 1b: convert into cdf
    let adv_cdf = adv_pdf.pdf_to_cdf();
    adv_cdf
} 

async fn precompute_short(samp: Samples, lambda: f64) -> (ApproxFunc<f64>, ApproxFunc<f64>) {
    let round = samp.round;
    let adv_cdf = precompute_cdf(samp, lambda).await;
    // 1c and 1d: compute E_max thetas
    let emax = adv_cdf.cdf_to_exp_max();
    println!("round {:#?} expected win {:#?}", round, adv_cdf.exp());
    (adv_cdf, emax)
}

async fn precompute_long(samp: Samples, lambda: f64) -> (ApproxFunc<f64>, ApproxFunc<ApproxFunc<f64>>) {
    // 1a thru 1d: call above funciton
    let (adv_cdf, emax) = precompute_short(samp, lambda).await;
    // 1e: compute G(gamma, c)
    let table = emax.emax_to_table();
    (adv_cdf, table)
}

async fn add_layer_helper_short(cdf: Arc<ApproxFunc<f64>>, emax: Arc<ApproxFunc<f64>>, lambda: f64) -> Vec<f64> {
    let mut new_data = Vec::with_capacity(N / PARALLELISM_FACTOR);
    for _ in 0..N / PARALLELISM_FACTOR {
        let adv = draw_adv(&*cdf);
        new_data.push(sample_short(&adv, &*emax, lambda));
    }
    new_data
}

async fn add_layer_helper_long(cdf: Arc<ApproxFunc<f64>>, table: Arc<ApproxFunc<ApproxFunc<f64>>>, lambda: f64) -> Vec<f64> {
    let mut new_data = Vec::with_capacity(N / PARALLELISM_FACTOR);
    for _ in 0..N / PARALLELISM_FACTOR {
        let adv = draw_adv(&*cdf);
        new_data.push(sample_long(&adv, &*table, lambda));
    }
    new_data
}

async fn add_layer(samp: Samples, lambda: f64) -> Samples {
    let mut new_samps = Samples { round: samp.round + 1, data: Default::default() };
    let mut handles = Vec::with_capacity(PARALLELISM_FACTOR);
    let bounding_ptile = match MODE {
        Mode::LowerBounding => 1.0 - (bound_size() as f64 / N as f64),
        Mode::UpperBounding => bound_size() as f64 / N as f64
    };
    let mut bounding_cutoff = f64::default();
    if BETA == 1f64 {
        let (cdf, emax) = precompute_short(samp, lambda).await;
        bounding_cutoff = *cdf.get(bounding_ptile);
        let (cdf, emax) = (Arc::new(cdf), Arc::new(emax));
        for _ in 0..PARALLELISM_FACTOR {
            handles.push(tokio::spawn(add_layer_helper_short(
                cdf.clone(), emax.clone(), lambda
            )));
        }
        for (i, handle) in handles.into_iter().enumerate() {
            new_samps.data[i] = handle.await.unwrap();
        }
    } else {
        let (cdf, table) = precompute_long(samp, lambda).await;
        bounding_cutoff = *cdf.get(bounding_ptile);
        let (cdf, table) = (Arc::new(cdf), Arc::new(table));
        for _ in 0..PARALLELISM_FACTOR {
            handles.push(tokio::spawn(add_layer_helper_long(
                cdf.clone(), table.clone(), lambda
            )));
        }
        for (i, handle) in handles.into_iter().enumerate() {
            new_samps.data[i] = handle.await.unwrap();
        }
    }
    match MODE {
        Mode::LowerBounding => deflate(new_samps, bounding_cutoff, lambda).await,
        Mode::UpperBounding => inflate(new_samps, bounding_cutoff, lambda).await
    }
}

fn bound_size() -> usize {
    (N as f64 * ((1f64 / GAMMA).ln() / (2 * N) as f64).sqrt()).ceil() as usize
}

async fn inflate(samp: Samples, bounding_cutoff: f64, lambda: f64) -> Samples {
    let mut handles = Vec::with_capacity(PARALLELISM_FACTOR);
    for mut vec in samp.data {
        handles.push(tokio::spawn(async move {
            for entry in vec.iter_mut() {
                if *entry < bounding_cutoff {
                    *entry = samp.round as f64 * (1f64 - lambda);
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

async fn deflate(samp: Samples, bounding_cutoff: f64, lambda: f64) -> Samples {
    let mut handles = Vec::with_capacity(PARALLELISM_FACTOR);
    for mut vec in samp.data {
        handles.push(tokio::spawn(async move {
            for entry in vec.iter_mut() {
                if *entry > bounding_cutoff {
                    *entry = samp.round as f64 * (0f64 - lambda);
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

/// Return expected reward
async fn simulate(lambda: f64) -> f64 {
    let mut dist = Samples { round: 0, data: array::from_fn(|_| Vec::from([0f64])) };
    for _ in 1..T {
        dist = add_layer(dist, lambda).await;
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

// start on binary search
// parallelization
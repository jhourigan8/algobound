use rand::Rng;
use std::{mem, sync::Arc};
use crate::Rounding;
use rand::rngs::SmallRng;
use rand::SeedableRng;

/// A struct defining a function from [`min`, `max`] to T.
/// The domain is discretized into `buckets` equally sized buckets.
#[derive(Clone, Default, Debug)]
struct ApproxFunc<T> {
    /// Bucket width
    error: f64, 
    /// Min value of domain
    min: f64, 
    /// Max value of domain
    max: f64, 
    /// Values in each bucket
    vals: Vec<T>,
}

impl<T: Clone + Default> ApproxFunc<T> {
    /// Construct a new `ApproxFunc<T>` with discretization error at most `error`.
    /// Values are initially `T::default()`.
    pub fn new(min: f64, max: f64, error: f64) -> Self {
        let buckets = ((max - min) / error).ceil() as usize;
        let new_max = min + buckets as f64 * error;
        let vals = vec![T::default(); buckets + 1];
        Self {
            error,
            min,
            max: new_max,
            vals,
        }
    }
}

impl<T: std::fmt::Debug> ApproxFunc<T> {
    /// Compute the index of the bucket which `x` lies in.
    /// If `x` is out of range instead return the index of the first / last bucket.
    pub fn idx(&self, mut x: f64, rounding: &Rounding) -> usize {
        // Deal with float arithmetic errors.
        if x > self.max {
            x = self.max;
        } else if x < self.min {
            x = self.min;
        }
        // Round based on rounding.
        let mut bucket = match rounding {
            Rounding::Down => ((x - self.min) / self.error).floor() as usize,
            Rounding::Up => ((x - self.min) / self.error).ceil() as usize
        };
        if bucket == self.vals.len() { bucket -= 1; }
        bucket
    }

    /// Set the value in bucket `i` to `y`.
    pub fn set_idx(&mut self, i: usize, y: T) -> Option<T> {
        Some(mem::replace(&mut self.vals[i], y))
    }

    /// Get a mutable reference to the value `x` is mapped to.
    pub fn get_mut(&mut self, x: f64, rounding: &Rounding) -> &mut T {
        let idx = self.idx(x, rounding);
        &mut self.vals[idx]
    }

    /// Get a reference to the value `x` is mapped to.
    pub fn get(&self, x: f64, rounding: &Rounding) -> &T {
        &self.vals[self.idx(x, rounding)]
    }
}

impl ApproxFunc<f64> {
    /// Scale the outputs of `self` by `c`.
    pub fn mul_const(&mut self, c: f64) {
        for i in 0..self.vals.len() {
            self.vals[i] *= c;
        }
    }

    /// Approximate function addition. Assumes `self` and `other` have same domain and error.
    pub fn add(&mut self, other: &ApproxFunc<f64>) {
        for i in 0..self.vals.len() {
            self.vals[i] += other.vals[i];
        }
    }
}

/// A collection of samples and the round on which they were collected.
#[derive(Clone, Default, Debug)]
pub struct Samples {
    /// The round on which the samples were collected.
    pub round: usize,
    /// The samples.
    pub data: Vec<f64>
}

impl Samples {
    /// Given a collection of samples, construct an equivalent discretized pdf
    /// with domain [`min`, `max`] and error `error`.
    pub fn to_pdf(self, min: f64, max: f64, error: f64, rounding: Rounding) -> Pdf {
        let mut pdf = Pdf {
            f: ApproxFunc::<f64>::new(min, max, error)
        };
        let delta = 1f64 / self.data.len() as f64;
        for sample in self.data {
            // Lower tick on LB, upper tick on UB
            *pdf.f.get_mut(sample, &rounding) += delta;
        }
        pdf
    }
}

/// A wrapper struct representing a function which is a pdf.
#[derive(Clone, Default, Debug)]
pub struct Pdf {
    f: ApproxFunc<f64>
}

impl Pdf {
    /// Convert pdf to an equivalent cdf with the same domain and number of buckets.
    pub fn to_cdf(&self) -> Cdf {
        let mut cdf = vec![0f64; self.f.vals.len()];
        let mut cum: f64 = 0f64;
        for (idx, p) in self.f.vals.iter().enumerate() {
            cum += p;
            cdf[idx] = cum;
        }
        Cdf {
            f: ApproxFunc::<f64> {
                error: self.f.error,
                min: self.f.min,
                max: self.f.max,
                vals: cdf,
            }
        }
    }
}

/// A wrapper struct representing a function which is a cdf.
#[derive(Clone, Default, Debug)]
pub struct Cdf {
    f: ApproxFunc<f64>
}

impl Cdf {
    /// Compute the expectation of a random variable drawn according to this cdf.
    pub fn exp(&self) -> f64 {
        let min = self.f.min;
        let mut tot = 0f64;
        let mut prev = 0f64;
        for i in 0..self.f.vals.len() {
            tot += (self.f.vals[i] - prev) * (min + i as f64 * self.f.error);
            prev = self.f.vals[i];
        }
        tot
    }

    /// Compute Emax(x) = E[max(x, r)] where r ~ cdf.
    /// Emax has same domain and discretization error as the cdf.
    pub fn to_emax(&self) -> Emax {
        let mut exp = self.exp();
        let mut vals = vec![0f64; self.f.vals.len()];
        for i in 0..self.f.vals.len() {
            vals[i] = exp;
            exp += self.f.error * self.f.vals[i];
        }
        Emax {
            f: ApproxFunc::<f64> {
                error: self.f.error,
                min: self.f.min,
                max: self.f.max, 
                vals,
            }
        }
    }

    /// Sample a value according to the cdf.
    pub fn sample(&self, rng: &mut SmallRng) -> f64 {
        let ptile: f64 = rng.gen();
        self.inv_get(ptile)
    }

    /// Given a percentile `ptile`, get the smallest value x for which `ptile` <= cdf(x).
    pub fn inv_get(&self, ptile: f64) -> f64 {
        let mut lo = 0;
        let mut hi = self.f.vals.len();
        while lo + 1 != hi {
            let mid = (lo + hi) / 2;
            // println!("{:?} {:?} {:?} {:?} {:?}", lo, mid, hi, ptile, self.f.vals[mid]);
            if self.f.vals[mid] < ptile { lo = mid } else { hi = mid };
        }
        self.f.min + self.f.error * hi as f64
    }
}

/// A wrapper struct representing a function which is an emax of 
/// the adversary's rewards.
#[derive(Clone, Default, Debug)]
pub struct Emax {
    f: ApproxFunc<f64>
}

impl Emax {
    /// Get a reference to the value `x` is mapped to.
    pub fn get(&self, x: f64, rounding: &Rounding) -> f64 {
        if x > self.f.max { return x; }
        // Check endpoint values in containing interval and return smallest / largest depending on rounding.
        let mut ret = match rounding {
            Rounding::Down => &f64::MAX,
            Rounding::Up => &f64::MIN
        };
        for m1 in [Rounding::Down, Rounding::Up] {
            let val = self.f.get(x, &m1);
            match rounding {
                Rounding::Down => if val < ret { ret = val; },
                Rounding::Up => if val > ret { ret = val; },
            }
        }
        *ret
    }

    /// Compute a table given this emax.
    /// The domain and error in `gamma` will be the same as emax while
    /// `theta` will have error `error`.
    pub async fn to_table(self, error: f64, beta: f64, rounding: Rounding) -> Table {
        let mut vals = Vec::new();
        let mut gamma = self.f.min;
        let emax = Arc::new(self);
        let mut handles = Vec::default();
        for _ in 0..emax.f.vals.len() {
            handles.push(tokio::spawn(Self::to_table_helper(emax.clone(), gamma, error, beta, rounding.clone())));
            gamma += emax.f.error;
        }
        for handle in handles.into_iter() {
            vals.push(handle.await.unwrap());
        }
        Table {
            f: ApproxFunc::<ApproxFunc::<f64>> {
                error: emax.f.error,
                min: emax.f.min,
                max: emax.f.max,
                vals
            }
        }
    }

    /// Helper function to compute table(_, `gamma`) for a given `gamma`.
    async fn to_table_helper(
        emax: Arc<Emax>, gamma: f64, error: f64, beta: f64, rounding: Rounding
    ) -> ApproxFunc<f64> {
        let mut cum = 0f64;
        let mut vec = Vec::default();
        vec.push(0.0);
        let mut zeta = 1.0;
        while zeta > 0.0 {
            // integrand is monotonically increasing, so this gives
            // lower and upper riemann sum respectively
            let rounded_zeta = match rounding {
                Rounding::Down => (zeta - error).max(0.0),
                Rounding::Up => zeta
            };
            let pow_zeta = rounded_zeta.powf((1.0 - beta) / beta);
            let emax_arg = gamma / pow_zeta;
            let score = if emax_arg > emax.f.max {
                // get back gamma
                gamma
            } else {
                // boost
                pow_zeta * emax.get(emax_arg, &rounding)
            };
            cum += error * score;
            vec.push(cum);
            zeta -= error;
        }
        ApproxFunc::<f64> {
            error,
            min: zeta,
            max: 1.0,
            vals: vec.iter().copied().rev().collect()
        }
    }
}

/// A wrapper struct representing a function which is an emax.
#[derive(Clone, Default, Debug)]
pub struct Table {
    f: ApproxFunc<ApproxFunc<f64>>
}

impl Table {
    /// Get a reference to the value (`theta`, `gamma`) is mapped to.
    pub fn get(&self, theta: f64, gamma: f64, rounding: &Rounding) -> &f64 {
        // println!("table get {:?} {:?}", theta, gamma);
        // Check all values in containing grid and return smallest / largest depending on rounding.
        let mut ret = match rounding {
            Rounding::Down => &f64::MAX,
            Rounding::Up => &f64::MIN
        };
        for m1 in [Rounding::Down, Rounding::Up] {
            for m2 in [Rounding::Down, Rounding::Up] {
                let val = self.f.get(gamma, &m1).get(theta, &m2);
                match rounding {
                    Rounding::Down => if val < ret { ret = val; },
                    Rounding::Up => if val > ret { ret = val; },
                }
            }
        }
        ret
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;

    const DELTA: f64 = 0.000001;

    fn assert_approx_eq(a: &f64, b: &f64) {
        println!("{:#?}, {:#?}", a, b);
        assert!((a - b).abs() < DELTA);
    }

    // Distribution 1: approx uniform over [0, 2], round 1
    // 0.0 to 2.0 with 0.01 error
    // 2000 samples evenly spaced every 0.001, no parallelization
    async fn uniform_dist() -> Pdf {
        let mut data = Vec::<f64>::new();
        for i in 0..2_000 {
            data.push(0.0005 + 0.001 * (i as f64));
        }
        return Samples {
            round: 1,
            data: Vec::from([data])
        }.to_pdf(0.0, 2.0, 0.01, Rounding::Down).await;
    }

    // Distribution 2: discrete distribution, round 10
    // 0.0 to 1.0 with 0.1 error
    // points at 0.05, 0.1, 0.2, 0.4; 4 parallel arrays
    async fn discrete_dist() -> Pdf {
        return Samples {
            round: 10,
            data: Vec::from([
                Vec::from([0.05]), Vec::from([0.1]), Vec::from([0.2]), Vec::from([0.4])
            ])
        }.to_pdf(0.0, 1.0, 0.1, Rounding::Up).await;
    }

    // Distribution 3: binomial distribution, round 5
    // 0.0 to 10.0 with 1.0 error
    // n = 10, p = 1/2; 8 parallel arrays
    async fn binomial_dist() -> Pdf {
        let mut data = Vec::<Vec<f64>>::new();
        for i in 0..8 {
            data.push(Vec::<f64>::new());
            for j in 0..128 {
                let mut num = 128 * i + j;
                let mut score = 0f64;
                while num > 0 {
                    if num % 2 == 1 { score += 1.0; }
                    num >>= 1;
                }
                data[i].push(score);
            }
        }
        return Samples {
            round: 5,
            data
        }.to_pdf(0.0, 10.0, 1.0, Rounding::Up).await;
    }
    
    #[test]
    fn approx_func() {
        // should have 8 buckets from 0.0 to 1.04
        let f = ApproxFunc::<f64>::new(0.0, 1.0, 0.13);
        assert_approx_eq(&f.error, &0.13);
        assert_approx_eq(&f.min, &0.0);
        assert_approx_eq(&f.max, &1.04);
        assert_eq!(f.vals.len(), 9);
        // check idx. (get, get_mut, set just use idx directly)
        // exactly on a cutoff => gives correctly
        assert_eq!(f.idx(0.26, &Rounding::Up), 2);
        assert_eq!(f.idx(0.52, &Rounding::Down), 4);
        // out of bounds => rounded in
        assert_eq!(f.idx(-125.0, &Rounding::Down), 0);
        assert_eq!(f.idx(1.05, &Rounding::Down), 8);
        // in bounds => rounds up or down correctly
        assert_eq!(f.idx(0.3, &Rounding::Down), 2);
        assert_eq!(f.idx(0.4, &Rounding::Up), 4);
    }

    #[tokio::test]
    async fn samples() {
        let pdf = uniform_dist().await;
        // Get value at 0, 1/200th of mass there
        assert_approx_eq(pdf.f.get(-5.0, &Rounding::Down), &0.005);
        // Get value at 1, no mass rounded there
        assert_approx_eq(pdf.f.get(2.1111, &Rounding::Down), &0.0);
        // Get value somewhere in middle, 1/200th of mass there
        assert_approx_eq(pdf.f.get(1.37568, &Rounding::Up), &0.005);

        let pdf = discrete_dist().await;
        // Get value at 0.1, 1/2 of mass there
        assert_approx_eq(pdf.f.get(0.1, &Rounding::Down), &0.5);
        assert_approx_eq(pdf.f.get(0.1, &Rounding::Up), &0.5);
        // Get value at 0.7, no mass there
        assert_approx_eq(pdf.f.get(0.711, &Rounding::Down), &0.0);
        // Get value at 0.2, 1/4 of mass there
        assert_approx_eq(pdf.f.get(0.199, &Rounding::Up), &0.25);

        let pdf = binomial_dist().await;
        // Get value at 3, 10C3/2^10 = 120/1024 = 15/128 mass there
        assert_approx_eq(pdf.f.get(3.0, &Rounding::Down), &(15.0/128.0));
    }
    
    #[tokio::test]
    async fn pdf() {
        let cdf = uniform_dist().await.to_cdf();
        // Get value at 0, 1/200th of mass there or below
        assert_approx_eq(cdf.f.get(-5.0, &Rounding::Down), &0.005);
        // Get value at 1, all mass there or below
        assert_approx_eq(cdf.f.get(2.1111, &Rounding::Down), &1.0);
        // Get values up to 1.38, has 139 out of 200 vals, 69.5$
        assert_approx_eq(cdf.f.get(1.37568, &Rounding::Up), &0.695);

        let cdf = discrete_dist().await.to_cdf();
        // Get value at 0.7, all mass below
        assert_approx_eq(cdf.f.get(0.711, &Rounding::Down), &1.0);

        let cdf = binomial_dist().await.to_cdf();
        // Get value at 5. 10C5 = 10*9*8*7*6/120 = 9*4*7 = 252,
        // so pdf is 252/1024, so cdf is 1/2 + 126/1024
        assert_approx_eq(cdf.f.get(5.9, &Rounding::Down), &(0.5 + 126.0/1024.0));
    }

    #[tokio::test]
    async fn cdf() {
        let cdf = uniform_dist().await.to_cdf();
        // Rounded down => expectation is 1 - 0.005
        assert_approx_eq(&cdf.exp(), &0.995);
        // Rounded down => hits 0.5 at 0.99
        assert_approx_eq(&cdf.inv_get(0.5000), &0.99);
        // Brief check sample works. Variance of 250k samples < 0.002
        let mut rng = rand::rngs::SmallRng::from_entropy();
        let mut sum = 0.0;
        for _ in 0..250_000 {
            sum += cdf.sample(&mut rng);
        }
        let avg = sum / 250_000.0;
        assert!(avg - 0.995 < 0.006);
        assert!(0.995 - avg < 0.006);
        let emax = cdf.to_emax();
        // Emax with min does nothing
        assert_approx_eq(&emax.f.get(0.0, &Rounding::Down), &0.995);
        // Emax with max gives max
        assert_approx_eq(&emax.f.get(2.0, &Rounding::Down), &2.0);

        let cdf = discrete_dist().await.to_cdf();
        // Rounded to 0.1, 0.1, 0.2, 0.4 so exp is 0.2
        assert_approx_eq(&cdf.exp(), &0.2);
        // Hits 0.8 at 0.4 (last point)
        println!("{:?}", cdf.f.get(0.31, &Rounding::Down));
        assert_approx_eq(&cdf.inv_get(0.8), &0.4);
        let emax = cdf.to_emax();
        // exp of 0.3, 0.3, 0.3, 0.4 is 0.325
        assert_approx_eq(&emax.f.get(0.31, &Rounding::Down), &0.325);
        // exp of 0.2, 0.2, 0.2, 0.4 is 0.25
        assert_approx_eq(&emax.f.get(0.21, &Rounding::Down), &0.25);

        let cdf = binomial_dist().await.to_cdf();
        // Exact so exp is 5
        assert_approx_eq(&cdf.exp(), &5.0);
        // Hits 0.999 at 9
        assert_approx_eq(&cdf.inv_get(0.999), &9.0);
    }

    #[tokio::test]
    async fn emax() {
        // emax should be parabolic from 1 up to 2 = 1 + x^2/4
        let emax = uniform_dist().await.to_cdf().to_emax();
        // Emax get rounds up and down correctly
        assert_approx_eq(&emax.get(1.985, &Rounding::Up), &1.99);
        assert_approx_eq(&emax.get(1.985, &Rounding::Down), &(0.995 * 1.98 + 0.005 * 1.99));
        // Emax above upper bound returns value back
        assert_eq!(&emax.get(3.0, &Rounding::Up), &3.0);
        // table with beta = 0.5 => exponent is 1, integrand is zeta * Emax(gamma / zeta)
        let table = emax.to_table(0.01, 0.5, Rounding::Down).await;
        // fix gamma = 1. then while zeta <= 0.5, this is just gamma
        // so diff between zeta = 0.0 and zeta = 0.5 should be 0.5
        assert_approx_eq(&(table.get(0.0, 1.0, &Rounding::Up) - table.get(0.50, 1.0, &Rounding::Up)), &0.50);
        // now, from 0.5 to 1.0 it's integrating zeta * (1 + 1/4zeta^2)
        // lower bounding => picks lower zeta each time
        // wolfram alpha gives .547043044. we should get something slightly below this 
        // due to rounded down emax gets. let's check:
        assert!(table.get(-0.00001, 1.0, &Rounding::Down) < &(0.50 + 0.547043044));
        assert!(table.get(-0.00001, 1.0, &Rounding::Down) > &(0.50 + 0.547043044 - 0.01));
    }
}
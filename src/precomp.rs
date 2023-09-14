use rand::Rng;
use std::{mem, sync::Arc};
use crate::{params::*, AdvDraw};

/// A struct defining a function from [`min`, `max`] to T.
/// The domain is discretized into `buckets` equally sized buckets.
#[derive(Clone, Default, Debug)]
struct ApproxFunc<T> {
    /// Number of buckets domain is partitioned into
    buckets: usize, 
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
    /// Compute the index of the bucket which `x` lies in.
    /// If `x` is out of range instead return the index of the first / last bucket.
    pub fn idx(&self, mut x: f64) -> usize {
        // Deal with float arithmetic errors.
        if x > self.max {
            x = self.max;
        } else if x < self.min {
            x = self.min;
        }
        let step = (self.max - self.min) / self.buckets as f64;
        // Always round bucket down.
        let mut bucket = ((x - self.min) / step).floor() as usize;
        if bucket == self.buckets { bucket -= 1; }
        bucket
    }

    /// Set the value in bucket `i` to `y`.
    pub fn set_idx(&mut self, i: usize, y: T) -> Option<T> {
        Some(mem::replace(&mut self.vals[i], y))
    }

    /// Get a mutable reference to the value `x` is mapped to.
    pub fn get_mut(&mut self, x: f64) -> &mut T {
        let idx = self.idx(x);
        &mut self.vals[idx]
    }

    /// Get a reference to the value `x` is mapped to.
    pub fn get(&self, x: f64) -> &T {
        &self.vals[self.idx(x)]
    }

    /// Get a reference to the bucket following `x`.
    pub fn get_next(&self, x: f64) -> &T {
        &self.vals[self.idx(x) + 1]
    }
}

impl ApproxFunc<f64> {
    /// Scale the outputs of `self` by `c`.
    pub fn mul_const(&mut self, c: f64) {
        for i in 0..self.buckets {
            self.vals[i] *= c;
        }
    }

    /// Approximate function addition. Preserves the domain of `self`.
    pub fn add(&mut self, other: &ApproxFunc<f64>) {
        let step = (self.max - self.min) / self.buckets as f64;
        for i in 0..self.buckets {
            self.vals[i] += other.get(self.min + step * i as f64);
        }
    }
}

/// A collection of samples and the round on which they were collected.
#[derive(Clone, Default, Debug)]
pub struct Samples {
    /// The round on which the samples were collected.
    pub round: usize,
    /// The samples.
    pub data: [Vec<f64>; PARALLELISM_FACTOR]
}

impl Samples {
    /// Given a collection of samples, construct an equivalent discretized pdf
    /// with domain [`min`, `max`]. Error of pdf is `EPSILON`.
    pub async fn to_pdf(self, min: f64, max: f64) -> Pdf {
        let mut pdf = Pdf {
            f: ApproxFunc::<f64>::new(min, max, EPSILON)
        };
        let mut handles = Vec::with_capacity(PARALLELISM_FACTOR);
        for vec in self.data {
            handles.push(tokio::spawn(Self::to_pdf_helper(min, max, vec)));
        }
        for handle in handles {
            pdf.f.add(&handle.await.unwrap().f)
        }
        pdf.f.mul_const(1f64 / PARALLELISM_FACTOR as f64);
        pdf
    }

    /// Helper processes a single batch of samples to produce a partial pdf.
    async fn to_pdf_helper(min: f64, max: f64, samples: Vec<f64>) -> Pdf {
        let mut pdf = Pdf {
            f: ApproxFunc::<f64>::new(min, max, EPSILON)
        };
        let delta = 1f64 / samples.len() as f64;
        for sample in samples {
            *pdf.f.get_mut(sample) += delta;
        }
        pdf
    }
}

/// A wrapper struct representing a function which is a pdf.
pub struct Pdf {
    f: ApproxFunc<f64>
}

impl Pdf {
    /// Convert pdf to an equivalent cdf with the same domain and number of buckets.
    pub fn to_cdf(&self) -> Cdf {
        let mut cdf = vec![0f64; self.f.buckets];
        let mut cum: f64 = 0f64;
        for (idx, p) in self.f.vals.iter().enumerate() {
            cum += p;
            cdf[idx] = cum;
        }
        Cdf {
            f: ApproxFunc::<f64> {
                buckets: self.f.buckets,
                min: self.f.min,
                max: self.f.max,
                vals: cdf,
            }
        }
    }
}

/// A wrapper struct representing a function which is a cdf.
pub struct Cdf {
    f: ApproxFunc<f64>
}

impl Cdf {
    /// Compute the expectation of a random variable drawn according to this cdf.
    pub fn exp(&self) -> f64 {
        let min = self.f.min + EPSILON * 0.5;
        let mut tot: f64 = 0f64;
        let mut prev = 0f64;
        for i in 0..self.f.buckets {
            tot += (self.f.vals[i] - prev) * (min + EPSILON * i as f64);
            prev = self.f.vals[i];
        }
        tot
    }

    /// Compute Emax(x) = E[max(x, r)] where r ~ cdf.
    /// Emax has same domain and discretization error as the cdf.
    pub fn to_emax(&self) -> Emax {
        let additional_buckets = ((1.0 / EPSILON) as usize) + 1;
        let buckets = self.f.buckets + additional_buckets;
        let mut exp = self.exp();
        let mut vals = vec![0f64; buckets];
        for i in 0..self.f.buckets {
            vals[i] = exp;
            exp += EPSILON * self.f.vals[i];
        }
        for i in self.f.buckets..buckets {
            vals[i] = exp;
            exp += EPSILON;
        }
        Emax {
            f: ApproxFunc::<f64> {
                buckets,
                min: self.f.min,
                max: self.f.max + additional_buckets as f64 * EPSILON, 
                vals,
            }
        }
    }

    /// Sample a value according to the cdf.
    pub fn sample(&self) -> f64 {
        let mut rng = rand::thread_rng();
        let ptile: f64 = rng.gen();
        self.inv_get(ptile)
    }

    /// Given a percentile `ptile`, get the value x for which cdf(x) = `ptile`.
    pub fn inv_get(&self, ptile: f64) -> f64 {
        // find bucket where BUCK val <= ptile < BUCK + 1 val
        // lower: round down, upper: round up
        // invariant: BUCK in [lo..hi)
        let mut lo = 0;
        let mut hi = self.f.buckets;
        while lo + 1 != hi {
            let mid = (lo + hi) / 2;
            if self.f.vals[mid] <= ptile { lo = mid } else { hi = mid };
        }
        // this is only used to sample adversary rewards so just round down/up (TODO check thiss)
        let buck = match MODE {
            Mode::LowerBounding => lo,
            Mode::UpperBounding => hi
        };
        self.f.min + (buck as f64 / self.f.buckets as f64) * (self.f.max - self.f.min)
    }
}

/// A wrapper struct representing a function which is an emax of 
/// the adversary's rewards.
pub struct Emax {
    f: ApproxFunc<f64>
}

impl Emax {
    /// Get a reference to the value `x` is mapped to.
    pub fn get(&self, x: f64) -> &f64 {
        // Check endpoint values in containing interval and return smallest / largest depending on mode.
        let mut ret = &f64::MIN;
        for val in [self.f.get(x), self.f.get_next(x)] {
            if val < ret {
                ret = val;
            }
        }
        ret
    }

    /// Compute a table given this emax.
    /// The domain and error in `gamma` will be the same as emax while
    /// `theta` will have error `ETA`.
    pub async fn to_table(self, round: usize, lambda: f64) -> Table {
        let mut vals = Vec::new();
        let mut gamma = self.f.min + match MODE {
            Mode::LowerBounding => 0f64,
            Mode::UpperBounding => EPSILON
        };
        let arc_f = Arc::new(self.f);
        let mut handles = Vec::default();
        for _ in 0..arc_f.buckets {
            handles.push(tokio::spawn(Self::to_table_helper(arc_f.clone(), gamma, round, lambda)));
            gamma += EPSILON;
        }
        for handle in handles.into_iter() {
            vals.push(handle.await.unwrap());
        }
        Table {
            f: ApproxFunc::<ApproxFunc::<f64>> {
                buckets: vals.len(),
                min: arc_f.min,
                max: arc_f.min + vals.len() as f64 * EPSILON,
                vals
            }
        }
    }

    /// Helper function to compute table(_, `gamma`) for a given `gamma`.
    async fn to_table_helper(
        f: Arc<ApproxFunc<f64>>, gamma: f64, round: usize, lambda: f64
    ) -> ApproxFunc<f64> {
        let direct = BETA < 0.5f64;
        let mut cum = 0f64;
        let coeff = if direct {
            ETA * BETA * (1f64 - ALPHA)
        } else {
            ETA * BETA / (1f64 - BETA)
        };
        let max = if direct {
            (round as f64 * (1f64 - lambda) / EPSILON).ln() / ((1f64 - BETA) * (1f64 - ALPHA))
        } else {
            1f64
        };
        let mut vec = Vec::default();
        let mut theta = 0f64 + ETA * 0.5;
        while theta < max {
            let bcast_score = gamma / if direct { AdvDraw::beat_unseen_honest_prob(theta) } else { theta };
            let emax_score = if bcast_score > f.max {
                &bcast_score
            } else if bcast_score < f.min {
                &f.min
            } else {
                f.get(bcast_score)
            };
            cum += coeff * *emax_score * if direct { AdvDraw::beat_honest_prob(theta) } else { 1f64 };
            vec.push(cum);
            theta += ETA;
        }
        ApproxFunc::<f64> {
            buckets: vec.len(),
            min: 0f64,
            max: theta,
            vals: vec
        }
    }
}

/// A wrapper struct representing a function which is an emax.
pub struct Table {
    f: ApproxFunc<ApproxFunc<f64>>
}

impl Table {
    /// Get a reference to the value (`theta`, `gamma`) is mapped to.
    pub fn get(&self, theta: f64, gamma: f64) -> &f64 {
        // Check all values in containing grid and return smallest / largest depending on mode.
        let mut ret = &f64::MIN;
        for partial in [self.f.get(gamma), self.f.get_next(gamma)] {
            for val in [partial.get(theta), partial.get_next(theta)] {
                if val < ret {
                    ret = val;
                }
            }
        }
        ret
    }
}
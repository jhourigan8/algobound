/// An enum specifying if the current simulation is to yield
/// a lower bound or an upper bound to the true `lambda`.
#[derive(Debug)]
#[allow(dead_code)]
pub enum Mode {
    LowerBounding,
    UpperBounding
}

/// Upper bound on the probability that a single `inflate` / `deflate` 
/// procedure fails to produce a stochastic upper / lower bound.
pub const CHERNOFF_ERROR: f64 = 0.001;

/// Discretization interval size for all functions of the adversary's rewards.
/// Namely, used in `Pdf`, `Cdf`, `Emax`, and the second argument to `Table`.
pub const EPSILON: f64 = 0.001;

/// Discretization interval size for the first argument to `Table`.
pub const ETA: f64 = 0.004;

/// The fraction of stake controlled by the adversary.
pub const ALPHA: f64 = 0.5;

/// The fraction of honest stake credential broadcasts which the adversary
/// is permitted to see before having to decide what to broadcast each round.
pub const BETA: f64 = 0.4;

/// The number of coins drawn for the adversary assuming their stake is spread
/// across arbitrarily many accounts.
pub const ADV_COINS: usize = 20;

/// The number of samples drawn to compute the reward distribution on each round.
pub const SAMPLES_DRAWN: usize = 200_000;

/// The number of rounds simulated before returning the adversary's reward for a
/// given value of `lambda`.
pub const ROUND_DEPTH: usize = 30;

/// The type of bound to be computed.
pub const MODE: Mode = Mode::LowerBounding;

/// The number of threads spawned for concurrent portions of the code.
pub const PARALLELISM_FACTOR: usize = 10;
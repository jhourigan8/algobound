Visit https://www.rust-lang.org/tools/install for information on installing rust.
To run simulations locally, clone this repo and run `cargo run <args>` in the directory.
Five arguments are expected:
- fixed-parameter: string literal `alpha` or `beta`
- fixed-value: value of fixed parameter
- step-size: step size of other parameter
- target-width: desired output interval width
- chernoff-error: upper bound on fooflate error probability each round

(old) docs available at https://jhourigan8.github.io/algobound/
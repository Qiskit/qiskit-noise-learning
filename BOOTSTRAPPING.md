# Bootstrapping Error Bars on Generator Error Rates

## Background

`qiskit-noise-learning` provides tools for learning Pauli Lindblad generator models of errors. As a consequence of how we perform these fittings, propagation of uncertainty from the experimental data to the final generator rates is not so straightforward. Therefore, we are interested in bootstrapping them.

Your task is to implement this bootstrapping in the analysis framework that `qiskit-noise-learning` provides.

There are different ways that we can perform this bootstrapping, but for now let's focus on this method: Each circuit that we execute is performed with different randomizations (twirls). We want to randomly sample subsets of these randomizations to probe the uncertainty in the resulting generator rates.

There are a few different mechanisms we can use for producing the bootstrap samples of increasing complexity:
1. Naively take random subsets of the randomizations to use (basic numpy sampling, etc).
2. Use scipy.stats.bootstrap.
3. Use the https://arch.readthedocs.io/ to perform the sampling.

If possible, it would be good to use (3) since it is the most rigorous and informative. But please leave room for us to try using the naive method to compare, sanity check, etc.

## Interacting with this repo

There is a venv set up at `./.venv` for you to use. Your starting place should be the demo notebook in `./docs/noise_learner_demo.ipynb`.

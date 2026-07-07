# Markov Chain Monte Carlo (MCMC) Sampling

A Python implementation of the Metropolis-Hastings algorithm for sampling from complex probability distributions using Markov Chain Monte Carlo methods.

## Overview

This project implements the **Metropolis-Hastings algorithm**, a fundamental MCMC technique used in Bayesian inference, statistical physics, and machine learning. The algorithm allows you to draw samples from any probability distribution, even when you only know its unnormalized form.

## Features

- **Metropolis-Hastings Algorithm**: Core MCMC sampler implementation
- **Flexible Distributions**: Support for any target and proposal distribution
- **Convergence Analysis**: Acceptance rate tracking and visualization
- **Visualization Tools**: Built-in plotting for trace plots and posterior distributions
- **Type Hints**: Fully typed code for better IDE support

## How It Works

The Metropolis-Hastings algorithm works by:

1. Starting at an initial state
2. Proposing a new state from a proposal distribution
3. Calculating an acceptance ratio based on the target distribution
4. Accepting or rejecting the proposal with probability equal to the acceptance ratio
5. Repeating until convergence

Over time, the samples drawn converge to the target distribution's stationary distribution.

## Installation

```bash
pip install numpy matplotlib
```

## Usage

```python
import numpy as np
from markov_chain_monte_carlo import MCMC

# Define your target distribution (unnormalized)
def target_dist(x):
    return np.exp(-0.5 * x**2)  # Standard normal

# Define a proposal distribution
def proposal_dist(current_state):
    return current_state + np.random.normal(0, 1)  # Random walk

# Initialize and run sampler
mcmc = MCMC(target_dist=target_dist,
            proposal_dist=proposal_dist,
            initial_state=0.0,
            seed=42)

samples, acceptance_rate = mcmc.metropolis_hastings(n_iterations=10000)

print(f"Acceptance rate: {acceptance_rate:.2%}")
print(f"Sample mean: {np.mean(samples):.4f}")
print(f"Sample std: {np.std(samples):.4f}")
```

## Key Parameters

| Parameter       | Description                                  |
| --------------- | -------------------------------------------- |
| `target_dist`   | Unnormalized target probability distribution |
| `proposal_dist` | Distribution for proposing new states        |
| `initial_state` | Starting point for the Markov chain          |
| `n_iterations`  | Number of samples to generate                |
| `seed`          | Random seed for reproducibility              |

## Output

The script generates:

1. **Acceptance Rate**: Percentage of proposals accepted (typically 20-50% is good)
2. **Trace Plot**: Shows how samples evolve across iterations
3. **Histogram**: Compares empirical samples to theoretical distribution
4. **Statistics**: Sample mean and standard deviation

## Mathematical Background

The acceptance ratio in Metropolis-Hastings is:

$$\alpha = \min\left(1, \frac{p(x_{proposal})}{p(x_{current})}\right)$$

Where `p(x)` is the unnormalized target distribution.

## Applications

- **Bayesian Inference**: Sample from posterior distributions
- **Model Fitting**: Parameter estimation in complex models
- **Statistical Physics**: Sampling from Gibbs distributions
- **Probabilistic Programming**: Core engine for inference

## Convergence Tips

- **Burn-in**: Discard first 1000-5000 samples to remove initialization bias
- **Proposal Scaling**: Adjust proposal variance to achieve ~25-45% acceptance rate
- **Thinning**: Use every k-th sample to reduce autocorrelation
- **Diagnostics**: Use trace plots to check for good mixing

## Example Output

```
Acceptance rate: 67.89%
Sample mean: -0.0234
Sample std: 1.0156
```

## Requirements

- Python 3.7+
- NumPy
- Matplotlib

![MCMC Results](figures\Figure_1.png)

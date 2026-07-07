import numpy as np
from typing import Callable, Tuple

class MCMC:
    """Markov Chain Monte Carlo sampler using Metropolis-Hastings algorithm."""
    
    def __init__(self, target_dist: Callable, proposal_dist: Callable, 
                 initial_state: float, seed: int = None):
        """
        Initialize MCMC sampler.
        
        Args:
            target_dist: Target probability distribution (unnormalized)
            proposal_dist: Proposal distribution for generating candidates
            initial_state: Starting point for the chain
            seed: Random seed for reproducibility
        """
        self.target_dist = target_dist
        self.proposal_dist = proposal_dist
        self.current_state = initial_state
        if seed is not None:
            np.random.seed(seed)
    
    def metropolis_hastings(self, n_iterations: int) -> Tuple[np.ndarray, float]:
        """
        Run Metropolis-Hastings algorithm.
        
        Args:
            n_iterations: Number of iterations to run
            
        Returns:
            samples: Array of samples from the chain
            acceptance_rate: Proportion of accepted proposals
        """
        samples = np.zeros(n_iterations)
        accepted = 0
        
        for i in range(n_iterations):
            # Generate proposal
            proposal = self.proposal_dist(self.current_state)
            
            # Calculate acceptance ratio
            alpha = self.target_dist(proposal) / self.target_dist(self.current_state)
            
            # Accept or reject
            if np.random.rand() < alpha:
                self.current_state = proposal
                accepted += 1
            
            samples[i] = self.current_state
        
        acceptance_rate = accepted / n_iterations
        return samples, acceptance_rate
    
import matplotlib.pyplot as plt

# Define a target distribution (e.g., standard normal)
def target_dist(x):
    """Unnormalized Gaussian distribution"""
    return np.exp(-0.5 * x**2)

# Define a proposal distribution (random walk)
def proposal_dist(current_state):
    """Propose next state by random walk"""
    return current_state + np.random.normal(0, 1)

# Initialize and run MCMC
mcmc = MCMC(target_dist=target_dist, 
            proposal_dist=proposal_dist, 
            initial_state=0.0, 
            seed=42)

# Sample from the distribution
n_iterations = 10000
samples, acceptance_rate = mcmc.metropolis_hastings(n_iterations)

print(f"Acceptance rate: {acceptance_rate:.2%}")
print(f"Sample mean: {np.mean(samples):.4f}")
print(f"Sample std: {np.std(samples):.4f}")

# Visualize results
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Trace plot
axes[0].plot(samples)
axes[0].set_xlabel('Iteration')
axes[0].set_ylabel('Value')
axes[0].set_title('Trace Plot')

# Histogram with theoretical distribution
axes[1].hist(samples[1000:], bins=50, density=True, alpha=0.7, label='MCMC samples')
x = np.linspace(-4, 4, 100)
axes[1].plot(x, np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi), 'r-', label='True distribution')
axes[1].set_xlabel('Value')
axes[1].set_ylabel('Density')
axes[1].set_title('Posterior Distribution')
axes[1].legend()

plt.tight_layout()
plt.show()
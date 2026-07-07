import numpy as np
from typing import Dict, List

class MarkovChain:
    def __init__(self, transition_matrix: np.ndarray, states: List[str]):
        """
        Initialize a Markov Chain.
        
        Args:
            transition_matrix: Square matrix of transition probabilities
            states: List of state names
        """
        self.transition_matrix = transition_matrix
        self.states = states
        self.n_states = len(states)
    
    def next_state(self, current_state: str) -> str:
        """Generate the next state based on current state."""
        state_idx = self.states.index(current_state)
        probabilities = self.transition_matrix[state_idx]
        next_idx = np.random.choice(self.n_states, p=probabilities)
        return self.states[next_idx]
    
    def simulate(self, start_state: str, steps: int) -> List[str]:
        """Simulate a sequence of states."""
        sequence = [start_state]
        current = start_state
        for _ in range(steps):
            current = self.next_state(current)
            sequence.append(current)
        return sequence


# Example usage
if __name__ == "__main__":
    # Define states and transition probabilities
    states = ["Sunny", "Rainy"]
    # Transition matrix: rows are current states, columns are next states
    transition = np.array([
        [0.8, 0.2],  # From Sunny
        [0.3, 0.7]   # From Rainy
    ])
    
    mc = MarkovChain(transition, states)
    sequence = mc.simulate("Sunny", 10)
    print(sequence)
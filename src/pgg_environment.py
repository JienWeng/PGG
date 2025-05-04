import numpy as np
from typing import List, Tuple, Optional

class PublicGoodsGame:
    def __init__(
        self, 
        num_agents: int, 
        endowments: List[float], 
        multiplication_factor: float, 
        action_space: List[float], 
        state_bins: List[Tuple[float, float]], 
        noise_std: float = 0.05  # σᵣ = 0.05 for multiplication factor noise
    ) -> None:
        """
        Initialize PGG with noisy multiplication factor.
        
        Args:
            num_agents: Number of agents in the game
            endowments: Initial endowments for each agent
            multiplication_factor: Base multiplication factor r_t
            action_space: Discrete action space as fractions of endowment
            state_bins: Discretized state bins as (endowment, contribution) tuples
            noise_std: Standard deviation for Gaussian noise added to multiplication factor
        """
        self.num_agents = num_agents
        self.endowments = endowments
        self.r = multiplication_factor  # Store as self.r for consistency
        self.action_space = action_space
        self.state_bins = state_bins
        self.noise_std = noise_std
        
        # Initialize states
        self.states = [(e, 0.0) for e in self.endowments]
        
    def step(
        self, 
        actions: List[float], 
        active_agents: List[int]
    ) -> Tuple[List[Tuple[float, float]], List[float], bool]:
        """
        Execute one step with noisy multiplication factor.
        """
        if len(actions) != len(active_agents):
            raise ValueError(f"Expected {len(active_agents)} actions, got {len(actions)}")
            
        # Calculate contributions without noise
        contributions = np.zeros(self.num_agents)
        for i, agent_idx in enumerate(active_agents):
            contributions[agent_idx] = actions[i] * self.endowments[agent_idx]
        
        # Calculate total contribution
        total_contribution = np.sum(contributions)
        
        # Apply Gaussian noise to multiplication factor: r̃ᵢ,ₜ ~ N(rₜ, σᵣ)
        noisy_multiplication_factor = self.r + np.random.normal(0, self.noise_std)
        
        # Ensure multiplication factor remains positive
        noisy_multiplication_factor = max(0, noisy_multiplication_factor)
        
        # Calculate public good with noisy multiplication factor
        public_good = noisy_multiplication_factor * total_contribution / len(active_agents) if active_agents else 0
        
        # Initialize rewards
        rewards = np.zeros(self.num_agents)
        
        # Calculate rewards using noisy multiplication factor
        for i, agent_idx in enumerate(active_agents):
            rewards[agent_idx] = self.endowments[agent_idx] - contributions[agent_idx] + public_good
        
        # Update states
        next_states = self.states.copy()
        
        # Update states for active agents
        for i, agent_idx in enumerate(active_agents):
            others_contribution = (total_contribution - contributions[agent_idx]) / (len(active_agents) - 1) if len(active_agents) > 1 else 0
            next_states[agent_idx] = self._discretize_state(
                self.endowments[agent_idx], 
                others_contribution
            )
        
        return next_states, rewards.tolist(), False
        
    def reset(self) -> List[Tuple[float, float]]:
        """
        Reset environment to initial states.
        
        Returns:
            Initial states for all agents
        """
        self.states = [(e, 0.0) for e in self.endowments]
        return self.states
        
    def _discretize_state(self, endowment: float, avg_contribution: float) -> Tuple[float, float]:
        """
        Discretize state to nearest bin.
        
        Args:
            endowment: The agent's endowment
            avg_contribution: Average contribution of other agents
            
        Returns:
            The discretized state as (endowment, contribution) tuple
        """
        # Find the closest endowment bin
        endowment_bins = sorted(set(e for e, _ in self.state_bins))
        closest_endowment = min(endowment_bins, key=lambda e: abs(e - endowment))
        
        # Find the closest contribution bin for this endowment
        contribution_bins = sorted(set(c for e, c in self.state_bins if e == closest_endowment))
        closest_contribution = min(contribution_bins, key=lambda c: abs(c - avg_contribution))
        
        return (closest_endowment, closest_contribution)

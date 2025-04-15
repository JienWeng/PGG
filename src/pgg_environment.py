import numpy as np
from typing import List, Tuple, Optional

class PublicGoodsGame:
    """
    A modular, complex multi-agent Public Goods Game (MARL) environment with a large action space.
    
    Supports heterogeneous endowments, noisy contributions, and dynamic group sizes.
    """
    
    def __init__(
        self, 
        num_agents: int, 
        endowments: List[float], 
        multiplication_factor: float, 
        action_space: List[float], 
        state_bins: List[Tuple[float, float]], 
        noise_std: float
    ) -> None:
        """
        Initialize PGG with MARL settings.
        
        Args:
            num_agents: Number of agents in the game (e.g., 4)
            endowments: Initial endowments for each agent (e.g., [0.5, 1.0, 1.5, 2.0])
            multiplication_factor: The public goods multiplication factor (e.g., 2.0)
            action_space: Discrete action space as fractions of endowment (e.g., [0, 0.04, ..., 1.0])
            state_bins: Discretized state bins as (endowment, contribution) tuples
            noise_std: Standard deviation for Gaussian noise added to actions (e.g., 0.1)
        """
        if len(endowments) != num_agents:
            raise ValueError(f"Expected {num_agents} endowment values, got {len(endowments)}")
            
        self.num_agents = num_agents
        self.endowments = np.array(endowments, dtype=float)
        self.multiplication_factor = multiplication_factor
        self.action_space = np.array(action_space, dtype=float)
        self.state_bins = state_bins
        self.noise_std = noise_std
        
        # Initialize states as (endowment_i, 0) for each agent
        self.states = [(e, 0.0) for e in self.endowments]
        
    def step(
        self, 
        actions: List[float], 
        active_agents: List[int]
    ) -> Tuple[List[Tuple[float, float]], List[float], bool]:
        """
        Execute one step in the PGG, return next states, rewards, and done flag.
        
        Args:
            actions: List of actions (contribution fractions) for each active agent
            active_agents: Indices of agents participating in this round
            
        Returns:
            next_states: List of next states for all agents
            rewards: List of rewards for all agents
            done: Whether the episode is finished (always False for PGG)
        """
        if len(actions) != len(active_agents):
            raise ValueError(f"Expected {len(active_agents)} actions, got {len(actions)}")
            
        # Add Gaussian noise to actions and clip to [0, 1]
        noisy_actions = np.clip(
            np.array(actions) + np.random.normal(0, self.noise_std, len(actions)), 
            0, 
            1
        )
        
        # Initialize rewards for all agents (inactive agents get 0)
        rewards = np.zeros(self.num_agents)
        
        # Calculate contributions: action_i * endowment_i for each active agent
        contributions = np.zeros(self.num_agents)
        for i, agent_idx in enumerate(active_agents):
            contributions[agent_idx] = noisy_actions[i] * self.endowments[agent_idx]
        
        # Calculate the public good value
        total_contribution = np.sum(contributions)
        public_good = self.multiplication_factor * total_contribution / len(active_agents) if active_agents else 0
        
        # Calculate rewards for active agents
        for i, agent_idx in enumerate(active_agents):
            # Reward = endowment - contribution + public good share
            rewards[agent_idx] = self.endowments[agent_idx] - contributions[agent_idx] + public_good
        
        # Update states for all agents
        next_states = self.states.copy()
        
        # For active agents, update state with average contribution of others
        for i, agent_idx in enumerate(active_agents):
            # Calculate average contribution of others (excluding self)
            others_contribution = (total_contribution - contributions[agent_idx]) / (len(active_agents) - 1) if len(active_agents) > 1 else 0
            
            # Discretize state
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

import numpy as np
from typing import List, Tuple, Optional

class PublicGoodsGame:
    def __init__(
        self, 
        num_agents: int, 
        endowments: List[float], 
        multiplication_factor: float, 
        action_space: List[float], 
        state_bins: List[Tuple[float, float]]
    ) -> None:
        """
        Initialize PGG.
        
        Args:
            num_agents: Number of agents in the game
            endowments: Initial endowments for each agent
            multiplication_factor: Base multiplication factor r
            action_space: Discrete action space as fractions of endowment
            state_bins: Discretized state bins as (endowment, contribution) tuples
        """
        self.num_agents = num_agents
        self.endowments = endowments
        self.r = multiplication_factor
        self.action_space = action_space
        self.state_bins = state_bins
        
        # Initialize states
        self.states = [(e, 0.0) for e in self.endowments]
        
    def step(
        self, 
        actions: List[float]
    ) -> Tuple[List[Tuple[float, float]], List[float], bool]:
        """
        Execute one step of the Public Goods Game.
        Assumes all agents are active and actions are provided for all agents.
        """
        if len(actions) != self.num_agents:
            raise ValueError(f"Expected {self.num_agents} actions, got {len(actions)}")
            
        contributions = np.zeros(self.num_agents)
        for agent_idx in range(self.num_agents):
            contributions[agent_idx] = actions[agent_idx] * self.endowments[agent_idx]
        
        total_contribution = np.sum(contributions)
        
        # Calculate public good using the base multiplication factor
        public_good = self.r * total_contribution / self.num_agents if self.num_agents > 0 else 0
        
        rewards = np.zeros(self.num_agents)
        
        # Calculate rewards
        for agent_idx in range(self.num_agents):
            rewards[agent_idx] = self.endowments[agent_idx] - contributions[agent_idx] + public_good
        
        next_states = self.states.copy()
        
        # Update states for all agents
        for agent_idx in range(self.num_agents):
            if self.num_agents > 1:
                others_contribution = (total_contribution - contributions[agent_idx]) / (self.num_agents - 1)
            else:
                others_contribution = 0.0
            
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

import numpy as np
from typing import List, Tuple, Optional

class PublicGoodsGame:
    def __init__(
        self, 
        num_agents: int, 
        endowments: List[float], 
        multiplication_factor: float, 
        action_space: List[float], 
        state_bins: List[Tuple[float]] # Changed: state_bins are now List[Tuple[float]] e.g. [(0.5,), (1.0,)]
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
        # self.state_bins stores the possible states, e.g., [(0.5,), (1.0,), (1.5,), (2.0,)]
        self.state_bins = state_bins 
        self.states: List[Tuple[float]] = [] # Stores current state for each agent, e.g., [(0.5,), (1.0,)...]
        self.reset()
        
    def step(
        self, 
        actions: List[float]
    ) -> Tuple[List[Tuple[float]], List[float], bool]: # Return type for states changed
        """
        Execute one step of the Public Goods Game.
        All agents are active and actions are provided for all agents.
        """
        if len(actions) != self.num_agents:
            raise ValueError(f"Expected {self.num_agents} actions, got {len(actions)}")
            
        contributions = np.zeros(self.num_agents)
        for agent_idx in range(self.num_agents):
            # Ensure action is valid (e.g., within [0, 1] if it's a fraction)
            # Assuming actions are fractions of endowment to contribute
            actual_contribution = actions[agent_idx] * self.endowments[agent_idx]
            # Clip contribution to be between 0 and agent's endowment
            contributions[agent_idx] = np.clip(actual_contribution, 0, self.endowments[agent_idx])
        
        total_contribution = np.sum(contributions)
        
        # Calculate public good share for each agent
        # Standard PGG: public_good_share = r * total_contribution / num_agents
        public_good_share = self.r * total_contribution / self.num_agents if self.num_agents > 0 else 0
        
        rewards = np.zeros(self.num_agents)
        
        # Calculate rewards
        for agent_idx in range(self.num_agents):
            rewards[agent_idx] = (self.endowments[agent_idx] - contributions[agent_idx]) + public_good_share
        
        next_agent_states: List[Tuple[float]] = [] # Changed type
        
        # Update states for all agents - state is now just their own endowment
        for agent_idx in range(self.num_agents):
            # The 'avg_contribution_others' is no longer part of the state
            # The state is solely determined by the agent's own fixed endowment
            agent_state = self._discretize_state(self.endowments[agent_idx])
            next_agent_states.append(agent_state)
        
        self.states = next_agent_states # Update current environment states
        
        return self.states, rewards.tolist(), False # game never ends in this setup
        
    def reset(self) -> List[Tuple[float]]: # Return type changed
        """
        Reset environment to initial states.
        Initial state for each agent is its own endowment.
        """
        self.states = []
        for i in range(self.num_agents):
            # State is the agent's endowment, represented as a tuple
            self.states.append(self._discretize_state(self.endowments[i]))
        return self.states
        
    def _discretize_state(self, endowment: float) -> Tuple[float]: # avg_contribution removed
        """
        Discretize the agent's state. Now, the state is just its own endowment.
        The 'state_bins' passed to __init__ should be like [(0.5,), (1.0,) ...].
        This function ensures the endowment matches one of these exact state tuples.
        """
        # Find the closest endowment bin (which should be the endowment itself if endowments are bin values)
        # The state is represented as a tuple, e.g., (0.5,)
        
        # Assuming self.state_bins contains tuples like (0.5,), (1.0,)
        # And endowment is a float like 0.5
        # We need to find the tuple in self.state_bins that matches the endowment.
        target_state_tuple = (round(endowment, 2),) # Ensure consistent formatting if endowments have decimals
        
        # Validate that this endowment state exists in the defined state_bins
        if target_state_tuple not in self.state_bins:
            # This case should ideally not happen if endowments align with state_bins
            # Fallback: find the numerically closest state bin tuple
            closest_state_tuple = min(self.state_bins, key=lambda s_tuple: abs(s_tuple[0] - endowment))
            # print(f"Warning: Endowment {endowment} not directly in state_bins. Using closest: {closest_state_tuple}")
            return closest_state_tuple
            
        return target_state_tuple

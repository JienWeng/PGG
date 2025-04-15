import numpy as np
from typing import Dict, List, Tuple, Optional, Union

class QAgent:
    """
    Q-Learning agent for multi-agent reinforcement learning (MARL) settings.
    
    Implements epsilon-greedy exploration and standard Q-Learning update rule.
    Suitable for Public Goods Game with large action spaces.
    """
    
    def __init__(
        self,
        state_space: List[Tuple[float, float]],
        action_space: List[float],
        learning_rate: float,
        discount_factor: float,
        epsilon: float,
        epsilon_decay: float,
        epsilon_min: float
    ) -> None:
        """
        Initialize Q-Learning agent.
        
        Args:
            state_space: List of possible states as (endowment, contribution) tuples
            action_space: List of possible actions (contribution fractions)
            learning_rate: Alpha parameter for Q-learning update (e.g., 0.1)
            discount_factor: Gamma parameter for future reward discounting (e.g., 0.9)
            epsilon: Initial exploration probability (e.g., 1.0)
            epsilon_decay: Decay rate for epsilon after each update (e.g., 0.995)
            epsilon_min: Minimum value for epsilon (e.g., 0.01)
        """
        self.state_space = state_space
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Initialize Q-table with zeros
        self.q_table: Dict[Tuple[Tuple[float, float], float], float] = {}
        for state in self.state_space:
            for action in self.action_space:
                self.q_table[(state, action)] = 0.0
    
    def choose_action(self, state: Tuple[float, float]) -> float:
        """
        Choose action via epsilon-greedy policy.
        
        Args:
            state: Current state as (endowment, contribution) tuple
            
        Returns:
            Selected action from action_space
        """
        # Exploration: random action
        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_space)
        
        # Exploitation: best action based on Q-values
        else:
            # Get all Q-values for current state
            q_values = {action: self.q_table.get((state, action), 0.0) 
                        for action in self.action_space}
            
            # Find action with highest Q-value (breaking ties randomly)
            max_q_value = max(q_values.values())
            best_actions = [action for action, q_value in q_values.items() 
                           if q_value == max_q_value]
            
            return np.random.choice(best_actions)
    
    def update(
        self,
        state: Tuple[float, float],
        action: float,
        reward: float,
        next_state: Tuple[float, float]
    ) -> None:
        """
        Update Q-table with Q-Learning rule.
        
        Args:
            state: Current state as (endowment, contribution) tuple
            action: Action taken (contribution fraction)
            reward: Reward received
            next_state: Next state as (endowment, contribution) tuple
        """
        # Calculate max Q-value for next state
        next_max_q_value = max(
            [self.q_table.get((next_state, next_action), 0.0) 
             for next_action in self.action_space]
        )
        
        # Current Q-value
        current_q = self.q_table.get((state, action), 0.0)
        
        # Q-Learning update rule
        self.q_table[(state, action)] = current_q + self.learning_rate * (
            reward + self.discount_factor * next_max_q_value - current_q
        )
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def get_q_values(self) -> Dict[Tuple[Tuple[float, float], float], float]:
        """
        Return current Q-table.
        
        Returns:
            Dictionary mapping (state, action) pairs to Q-values
        """
        return self.q_table

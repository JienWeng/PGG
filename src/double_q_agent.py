import numpy as np
from typing import Dict, List, Tuple, Optional, Union

class DoubleQAgent:
    """
    Double Q-Learning agent for multi-agent reinforcement learning (MARL) settings.
    
    Implements epsilon-greedy exploration and double Q-learning update rule.
    Uses two separate Q-tables to reduce overestimation bias.
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
        Initialize Double Q-Learning agent.
        
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
        
        # Initialize two Q-tables with zeros for Double Q-Learning
        self.q_table_A: Dict[Tuple[Tuple[float, float], float], float] = {}
        self.q_table_B: Dict[Tuple[Tuple[float, float], float], float] = {}
        
        # Initialize both Q-tables
        for state in self.state_space:
            for action in self.action_space:
                self.q_table_A[(state, action)] = 0.0
                self.q_table_B[(state, action)] = 0.0
    
    def choose_action(self, state: Tuple[float, float]) -> float:
        """
        Choose action via epsilon-greedy on average Q-values.
        
        Args:
            state: Current state as (endowment, contribution) tuple
            
        Returns:
            Selected action from action_space
        """
        # Exploration: random action
        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_space)
        
        # Exploitation: best action based on average Q-values
        else:
            # Calculate average Q-values from both tables
            avg_q_values = {
                action: (
                    self.q_table_A.get((state, action), 0.0) + 
                    self.q_table_B.get((state, action), 0.0)
                ) / 2
                for action in self.action_space
            }
            
            # Find action with highest average Q-value (breaking ties randomly)
            max_q_value = max(avg_q_values.values())
            best_actions = [
                action for action, q_value in avg_q_values.items() 
                if q_value == max_q_value
            ]
            
            return np.random.choice(best_actions)
    
    def update(
        self,
        state: Tuple[float, float],
        action: float,
        reward: float,
        next_state: Tuple[float, float]
    ) -> None:
        """
        Update Q_A or Q_B with Double Q-Learning rule.
        
        Randomly selects which Q-table to update (50% chance each).
        For updating Q_A, uses Q_B for next state evaluation and vice versa.
        
        Args:
            state: Current state as (endowment, contribution) tuple
            action: Action taken (contribution fraction)
            reward: Reward received
            next_state: Next state as (endowment, contribution) tuple
        """
        # Randomly decide which Q-table to update (50% chance for each)
        if np.random.random() < 0.5:
            # Update Q_A using Q_B for next state evaluation
            
            # Find best action according to Q_A
            best_next_action = max(
                self.action_space, 
                key=lambda a: self.q_table_A.get((next_state, a), 0.0)
            )
            
            # Get Q-value from Q_B for this action
            next_q_value = self.q_table_B.get((next_state, best_next_action), 0.0)
            
            # Current Q-value from Q_A
            current_q = self.q_table_A.get((state, action), 0.0)
            
            # Update Q_A
            self.q_table_A[(state, action)] = current_q + self.learning_rate * (
                reward + self.discount_factor * next_q_value - current_q
            )
        else:
            # Update Q_B using Q_A for next state evaluation
            
            # Find best action according to Q_B
            best_next_action = max(
                self.action_space, 
                key=lambda a: self.q_table_B.get((next_state, a), 0.0)
            )
            
            # Get Q-value from Q_A for this action
            next_q_value = self.q_table_A.get((next_state, best_next_action), 0.0)
            
            # Current Q-value from Q_B
            current_q = self.q_table_B.get((state, action), 0.0)
            
            # Update Q_B
            self.q_table_B[(state, action)] = current_q + self.learning_rate * (
                reward + self.discount_factor * next_q_value - current_q
            )
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def get_q_values(self) -> Tuple[
        Dict[Tuple[Tuple[float, float], float], float],
        Dict[Tuple[Tuple[float, float], float], float]
    ]:
        """
        Return Q_A and Q_B tables.
        
        Returns:
            Tuple containing both Q-tables as dictionaries
        """
        return (self.q_table_A, self.q_table_B)

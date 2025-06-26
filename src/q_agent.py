import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Union

class QAgent:
    """
    Q-Learning agent. State is now just its own endowment (as a tuple).
    """
    
    def __init__(
        self,
        agent_endowment: float, 
        state_space: List[Tuple[float]], # Changed: e.g., [(0.5,), (1.0,)]
        action_space: List[float],
        learning_rate: float,
        discount_factor: float,
        epsilon: float,
        epsilon_decay: float,
        epsilon_min: float
    ) -> None:
        self.agent_endowment_value = agent_endowment # The float value of the endowment
        # The state for this agent will always be (self.agent_endowment_value,)
        self.my_state: Tuple[float] = (round(agent_endowment, 2),) 

        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table: keys are (state_tuple, action_float)
        # State_tuple is like (0.5,)
        self.q_table: Dict[Tuple[Tuple[float], float], float] = {}
        
        # Initialize Q-table only for the agent's own fixed state
        # All other states in the global state_space are irrelevant to this agent
        if self.my_state in state_space: # Ensure the agent's state is valid
            for action in self.action_space:
                self.q_table[(self.my_state, action)] = 0.0
        else:
            # This should not happen if main.py and pgg_environment.py are set up correctly
            print(f"Warning: Agent's state {self.my_state} not in provided state_space {state_space} for agent with endowment {self.agent_endowment_value}")
            # Fallback: initialize for the raw endowment tuple anyway
            for action in self.action_space:
                 self.q_table[(self.my_state, action)] = 0.0

    def choose_action(self, state: Tuple[float]) -> float: # State type changed
        """Choose action using epsilon-greedy policy."""
        # The state passed should always be self.my_state for this agent
        if state != self.my_state:
            # This might indicate a mismatch or an unexpected state being passed.
            # For this simplified model, we assume the agent only acts based on its fixed state.
            # print(f"Warning: Agent received state {state} but expected {self.my_state}. Using self.my_state.")
            current_state_key = self.my_state
        else:
            current_state_key = state

        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.action_space)  # Explore
        else:
            # Exploit: choose the action with the highest Q-value for the current state
            q_values_for_state = {action: self.q_table.get((current_state_key, action), -np.inf) 
                                  for action in self.action_space}
            if not q_values_for_state: # Should not happen if q_table initialized
                 return random.choice(self.action_space)
            return max(q_values_for_state, key=q_values_for_state.get)

    def update(
        self,
        state: Tuple[float], # State type changed
        action: float,
        reward: float,
        next_state: Tuple[float] # State type changed
    ) -> None:
        """Update Q-value using the Q-Learning rule."""
        # For this agent, state and next_state should always be self.my_state
        # We use self.my_state as the key for Q-table entries.
        
        current_q_value = self.q_table.get((self.my_state, action), 0.0)
        
        # Find max Q-value for the next state (which is also self.my_state)
        max_next_q_value = -np.inf
        if (self.my_state, self.action_space[0]) in self.q_table: # Check if q_table has entries for this state
            max_next_q_value = max(self.q_table.get((self.my_state, next_action), -np.inf) 
                                   for next_action in self.action_space)
        else: # Should not happen if initialized correctly
            max_next_q_value = 0.0


        new_q_value = current_q_value + self.learning_rate * \
                      (reward + self.discount_factor * max_next_q_value - current_q_value)
        self.q_table[(self.my_state, action)] = new_q_value
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def get_q_values(self) -> Dict[Tuple[Tuple[float], float], float]: # Q-table key type changed
        return self.q_table

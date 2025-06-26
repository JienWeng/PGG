import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Union

class DoubleQAgent:
    """
    Double Q-Learning agent. State is now just its own endowment (as a tuple).
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
        self.agent_endowment_value = agent_endowment
        self.my_state: Tuple[float] = (round(agent_endowment, 2),)

        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.q_table_A: Dict[Tuple[Tuple[float], float], float] = {}
        self.q_table_B: Dict[Tuple[Tuple[float], float], float] = {}

        if self.my_state in state_space:
            for action in self.action_space:
                self.q_table_A[(self.my_state, action)] = 0.0
                self.q_table_B[(self.my_state, action)] = 0.0
        else:
            print(f"Warning: Agent's state {self.my_state} not in provided state_space {state_space} for agent with endowment {self.agent_endowment_value}")
            for action in self.action_space:
                self.q_table_A[(self.my_state, action)] = 0.0
                self.q_table_B[(self.my_state, action)] = 0.0
                
    def choose_action(self, state: Tuple[float]) -> float: # State type changed
        """Choose action using epsilon-greedy policy based on the sum of Q_A and Q_B."""
        if state != self.my_state:
            # print(f"Warning: Agent received state {state} but expected {self.my_state}. Using self.my_state.")
            current_state_key = self.my_state
        else:
            current_state_key = state

        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.action_space)  # Explore
        else:
            # Exploit: choose action with max(Q_A(s,a) + Q_B(s,a))
            sum_q_values = {
                action: self.q_table_A.get((current_state_key, action), -np.inf) + \
                          self.q_table_B.get((current_state_key, action), -np.inf)
                for action in self.action_space
            }
            if not sum_q_values:
                return random.choice(self.action_space)
            return max(sum_q_values, key=sum_q_values.get)

    def update(
        self,
        state: Tuple[float], # State type changed
        action: float,
        reward: float,
        next_state: Tuple[float] # State type changed
    ) -> None:
        """Update Q-tables using Double Q-Learning rule."""
        # State and next_state are always self.my_state for this agent.
        
        if random.uniform(0, 1) < 0.5:
            # Update Q_A using Q_B to select the best next action
            q_table_to_update = self.q_table_A
            q_table_for_next_action_selection = self.q_table_B
        else:
            # Update Q_B using Q_A to select the best next action
            q_table_to_update = self.q_table_B
            q_table_for_next_action_selection = self.q_table_A

        current_q_value = q_table_to_update.get((self.my_state, action), 0.0)
        
        # Find best next action using one Q-table
        best_next_action = None
        if (self.my_state, self.action_space[0]) in q_table_for_next_action_selection:
            max_q_for_next_action_selection = -np.inf
            for next_act in self.action_space:
                q_val = q_table_for_next_action_selection.get((self.my_state, next_act), -np.inf)
                if q_val > max_q_for_next_action_selection:
                    max_q_for_next_action_selection = q_val
                    best_next_action = next_act
        
        if best_next_action is None: # Fallback if no actions found or q-table empty for state
            best_next_action = random.choice(self.action_space) # Or handle as 0 if appropriate

        # Get Q-value of that best_next_action from the *other* Q-table (the one being updated)
        # No, this is wrong for standard Double Q.
        # Get Q-value of best_next_action from the Q-table NOT used for selection (i.e., the one being updated for its Q(s',a*) part)
        # Actually, for Double Q, the Q(s', a*) uses the *other* table.
        # Q_A(s,a) = Q_A(s,a) + alpha * (r + gamma * Q_B(s', argmax_a' Q_A(s',a')) - Q_A(s,a))
        
        # Simpler: Q_A(s,a) = Q_A(s,a) + alpha * (r + gamma * Q_B(s', best_action_from_A_for_s') - Q_A(s,a))
        # Let's re-evaluate the Double Q-learning update logic carefully.
        # Standard Double Q update:
        # If updating Q_A:
        #   best_action_in_next_state_from_A = argmax_a' Q_A(next_state, a')
        #   target = reward + discount_factor * Q_B(next_state, best_action_in_next_state_from_A)
        #   Q_A(state, action) += learning_rate * (target - Q_A(state, action))

        # Find best action in next_state using the Q-table that is *not* being updated for Q(s',a*) selection
        # (i.e., if updating Q_A, use Q_A to find best_next_action_for_target)
        
        # Let's use the primary Q table (the one selected by coin flip for update) to find best action for s'
        # And the secondary Q table to get the value of that action.
        
        # Corrected Double Q-Learning update:
        if random.uniform(0, 1) < 0.5: # Update Q_A
            # Select best action for next_state using Q_A
            best_next_action_q_a = max(self.action_space, key=lambda act: self.q_table_A.get((self.my_state, act), -np.inf))
            # Get value of this action from Q_B
            td_target_value = self.q_table_B.get((self.my_state, best_next_action_q_a), 0.0)
            
            current_val = self.q_table_A.get((self.my_state, action), 0.0)
            self.q_table_A[(self.my_state, action)] = current_val + self.learning_rate * \
                (reward + self.discount_factor * td_target_value - current_val)
        else: # Update Q_B
            # Select best action for next_state using Q_B
            best_next_action_q_b = max(self.action_space, key=lambda act: self.q_table_B.get((self.my_state, act), -np.inf))
            # Get value of this action from Q_A
            td_target_value = self.q_table_A.get((self.my_state, best_next_action_q_b), 0.0)

            current_val = self.q_table_B.get((self.my_state, action), 0.0)
            self.q_table_B[(self.my_state, action)] = current_val + self.learning_rate * \
                (reward + self.discount_factor * td_target_value - current_val)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_q_values(self) -> Tuple[
        Dict[Tuple[Tuple[float], float], float],
        Dict[Tuple[Tuple[float], float], float]
    ]: # Q-table key type changed
        return self.q_table_A, self.q_table_B

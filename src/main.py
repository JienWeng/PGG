import os
import sys
from typing import List, Tuple, Dict, Any
from pathlib import Path
import pandas as pd
import numpy as np
from ast import literal_eval

# Import custom modules
from pgg_environment import PublicGoodsGame
from q_agent import QAgent
from double_q_agent import DoubleQAgent
from simulation import run_simulation

def main() -> None:
    """Run MARL experiments for multiple r and seeds."""
    # Environment parameters
    num_agents = 4
    endowments = [0.5 * (i+1) for i in range(num_agents)] 
    multiplication_factors = [3.5] 
    action_space = [0, 0.5, 1]
    # action_space = [round(i * 0.04, 2) for i in range(26)]  # 0.0 to 1.0 in steps of 0.04
    
    # The 'state_bins' for the environment will be the unique endowment values, represented as tuples.
    unique_endowment_values = sorted(list(set(endowments)))
    environment_state_bins = [(e,) for e in unique_endowment_values] 
    
    # Agent parameters
    learning_rate = 0.05
    discount_factor = 0.95
    epsilon = 1.0
    epsilon_decay = 0.9995
    epsilon_min = 0.05
    
    # Simulation parameters
    num_episodes = 200000
    seeds = [1,2,3,4,5,6,7,8,9,10] 
    output_dir = "results_control/"
    
    try:
        # Create base output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Run experiments for each multiplication factor
        for r in multiplication_factors:
            print(f"\nRunning experiments for r = {r}")
            r_output_dir = f"{output_dir}r{r}/"
            Path(r_output_dir).mkdir(parents=True, exist_ok=True)
            
            # Create environment
            env = PublicGoodsGame(
                num_agents=num_agents,
                endowments=endowments,
                multiplication_factor=r,
                action_space=action_space,
                state_bins=environment_state_bins # Pass the simplified state bins
            )
            
            # Run experiments for each seed
            for seed in seeds:
                print(f"\nRunning experiments for seed = {seed}")
                seed_output_dir = f"{r_output_dir}seed{seed}/"
                Path(seed_output_dir).mkdir(parents=True, exist_ok=True)
                
                # Set seed for reproducibility
                np.random.seed(seed)
                
                # Create and run Q-Learning agents
                q_agents = [QAgent(
                    agent_endowment=endowments[i], 
                    state_space=environment_state_bins, # Pass simplified state space
                    action_space=action_space,
                    learning_rate=learning_rate,
                    discount_factor=discount_factor,
                    epsilon=epsilon,
                    epsilon_decay=epsilon_decay,
                    epsilon_min=epsilon_min
                ) for i in range(num_agents)]
                
                print("Running Q-Learning simulation...")
                run_simulation(
                    algorithm=f"q_r{r}_seed{seed}",
                    num_episodes=num_episodes,
                    env=env,
                    agents=q_agents,
                    output_dir=seed_output_dir
                )
                
                # Create and run Double Q-Learning agents
                double_q_agents = [DoubleQAgent(
                    agent_endowment=endowments[i], 
                    state_space=environment_state_bins, # Pass simplified state space
                    action_space=action_space,
                    learning_rate=learning_rate,
                    discount_factor=discount_factor,
                    epsilon=epsilon,
                    epsilon_decay=epsilon_decay,
                    epsilon_min=epsilon_min
                ) for i in range(num_agents)]
                
                print("Running Double Q-Learning simulation...")
                run_simulation(
                    algorithm=f"double_q_r{r}_seed{seed}",
                    num_episodes=num_episodes,
                    env=env,
                    agents=double_q_agents,
                    output_dir=seed_output_dir
                )
                
            print(f"Completed experiments for r = {r}")
    
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

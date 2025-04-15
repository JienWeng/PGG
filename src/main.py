import os
from typing import List, Tuple, Dict
from pathlib import Path
import pandas as pd
import numpy as np
from ast import literal_eval

# Import custom modules
from pgg_environment import PublicGoodsGame
from q_agent import QAgent
from double_q_agent import DoubleQAgent
from simulation import run_simulation
from analysis import plot_metrics, plot_heatmap, save_summary
from evaluation import evaluate_q_tables

def aggregate_metrics(seed_dirs: List[str], output_file: str) -> pd.DataFrame:
    """Aggregate metrics across seeds using mean."""
    dfs = [pd.read_csv(os.path.join(seed_dir, os.path.basename(output_file))) 
           for seed_dir in seed_dirs]
    return pd.DataFrame({col: np.mean([df[col] for df in dfs], axis=0) 
                        for col in dfs[0].columns})

def aggregate_q_tables(seed_dirs: List[str], output_file: str, is_double_q: bool) -> pd.DataFrame:
    """Aggregate Q-tables across seeds using mean."""
    dfs = [pd.read_csv(os.path.join(seed_dir, os.path.basename(output_file))) 
           for seed_dir in seed_dirs]
    
    if is_double_q:
        return pd.DataFrame({
            'agent': dfs[0]['agent'],
            'state': dfs[0]['state'],
            'action': dfs[0]['action'],
            'q_a_value': np.mean([df['q_a_value'] for df in dfs], axis=0),
            'q_b_value': np.mean([df['q_b_value'] for df in dfs], axis=0)
        })
    else:
        return pd.DataFrame({
            'agent': dfs[0]['agent'],
            'state': dfs[0]['state'],
            'action': dfs[0]['action'],
            'q_value': np.mean([df['q_value'] for df in dfs], axis=0)
        })

def main() -> None:
    """Run MARL experiments for multiple r and seeds."""
    # Set experiment parameters
    num_agents = 4
    endowments = [0.5, 1.0, 1.5, 2.0]
    multiplication_factors = [
        1.5, 
        2.0, 
        2.5,
        3.0,
        3.5,
        4.0
        ]
    action_space = [i * 0.04 for i in range(25)]  # 0, 0.10, ..., 1.0
    state_bins = [(e, c) for e in [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5] 
                 for c in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]
    
    # Agent parameters
    noise_std = 0.05
    learning_rate = 0.05
    discount_factor = 0.95
    epsilon = 1.0
    epsilon_decay = 0.9995
    epsilon_min = 0.05
    
    # Simulation parameters
    num_episodes = 10000
    steps_per_episode = 200
    seeds = [42, 123, 456] 
    output_dir = "results/"
    
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
            state_bins=state_bins,
            noise_std=noise_std
        )
        
        # Run experiments for each seed
        seed_dirs = []
        for seed in seeds:
            print(f"\nRunning experiments for seed = {seed}")
            seed_output_dir = f"{r_output_dir}seed{seed}/"
            Path(seed_output_dir).mkdir(parents=True, exist_ok=True)
            seed_dirs.append(seed_output_dir)
            
            # Set random seed
            np.random.seed(seed)
            
            # Create Q-Learning agents
            q_agents = [
                QAgent(
                    state_space=state_bins,
                    action_space=action_space,
                    learning_rate=learning_rate,
                    discount_factor=discount_factor,
                    epsilon=epsilon,
                    epsilon_decay=epsilon_decay,
                    epsilon_min=epsilon_min
                )
                for _ in range(num_agents)
            ]
            
            # Create Double Q-Learning agents
            double_q_agents = [
                DoubleQAgent(
                    state_space=state_bins,
                    action_space=action_space,
                    learning_rate=learning_rate,
                    discount_factor=discount_factor,
                    epsilon=epsilon,
                    epsilon_decay=epsilon_decay,
                    epsilon_min=epsilon_min
                )
                for _ in range(num_agents)
            ]
            
            # Run simulations for this seed
            print("Running Q-Learning simulation...")
            run_simulation(
                algorithm=f"q_r{r}_seed{seed}",
                num_episodes=num_episodes,
                steps_per_episode=steps_per_episode,
                env=env,
                agents=q_agents,
                output_dir=seed_output_dir
            )
            
            print("Running Double Q-Learning simulation...")
            run_simulation(
                algorithm=f"double_q_r{r}_seed{seed}",
                num_episodes=num_episodes,
                steps_per_episode=steps_per_episode,
                env=env,
                agents=double_q_agents,
                output_dir=seed_output_dir
            )
        
        # Aggregate results across seeds
        print("\nAggregating results across seeds...")
        
        # Aggregate metrics
        q_metrics = aggregate_metrics(
            seed_dirs,
            f"q_r{r}_seed{seeds[0]}_metrics.csv"
        )
        double_q_metrics = aggregate_metrics(
            seed_dirs,
            f"double_q_r{r}_seed{seeds[0]}_metrics.csv"
        )
        
        # Save aggregated metrics
        q_metrics.to_csv(f"{r_output_dir}q_r{r}_metrics.csv", index=False)
        double_q_metrics.to_csv(f"{r_output_dir}double_q_r{r}_metrics.csv", index=False)
        
        # Aggregate Q-tables
        q_qvalues = aggregate_q_tables(
            seed_dirs,
            f"q_r{r}_seed{seeds[0]}_qvalues.csv",
            False
        )
        double_q_qvalues = aggregate_q_tables(
            seed_dirs,
            f"double_q_r{r}_seed{seeds[0]}_qvalues.csv",
            True
        )
        
        # Save aggregated Q-tables
        q_qvalues.to_csv(f"{r_output_dir}q_r{r}_qvalues.csv", index=False)
        double_q_qvalues.to_csv(f"{r_output_dir}double_q_r{r}_qvalues.csv", index=False)
        
        # Process Q-values for evaluation
        q_dict = {}
        double_q_dict = {}
        
        # Process Q-Learning Q-values
        for agent in q_qvalues['agent'].unique():
            agent_data = q_qvalues[q_qvalues['agent'] == agent]
            q_dict[agent] = {}
            for _, row in agent_data.iterrows():
                state = literal_eval(row['state'])
                q_dict[agent][(state, row['action'])] = row['q_value']
        
        # Process Double Q-Learning Q-values
        for agent in double_q_qvalues['agent'].unique():
            agent_data = double_q_qvalues[double_q_qvalues['agent'] == agent]
            double_q_dict[agent] = {}
            for _, row in agent_data.iterrows():
                state = literal_eval(row['state'])
                double_q_dict[agent][(state, row['action'])] = (row['q_a_value'] + row['q_b_value']) / 2
        
        # # Generate analysis plots and summaries
        # print("Analyzing aggregated results...")
        # plot_metrics(q_metrics, double_q_metrics, r_output_dir, r)
        # plot_heatmap(q_qvalues, double_q_qvalues, r_output_dir, r)
        # save_summary(q_metrics, double_q_metrics, r_output_dir, r)
        # evaluate_q_tables(q_dict, double_q_dict, r_output_dir, r)
        
        print(f"Completed experiments for r = {r}")

if __name__ == "__main__":
    main()

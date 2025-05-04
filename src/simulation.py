import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Any
import random
from pathlib import Path

# Import environment and agents
from pgg_environment import PublicGoodsGame
from q_agent import QAgent
from double_q_agent import DoubleQAgent

def calculate_shapley_value(contributions: List[float], r: float) -> List[float]:
    """
    Calculate Shapley values for each player in the public goods game.
    
    Args:
        contributions: List of contributions from each player
        r: Multiplier for public goods
        
    Returns:
        List of Shapley values for each player
    """
    n = len(contributions)
    shapley_values = [0.0] * n
    
    def v(coalition: List[int]) -> float:
        """Calculate value of a coalition"""
        if not coalition:
            return 0.0
        coalition_sum = sum(contributions[i] for i in coalition)
        return (r * coalition_sum) / len(coalition)
    
    # Calculate Shapley value for each player
    for i in range(n):
        for subset in range(2 ** (n-1)):
            coalition = []
            pos = 0
            for j in range(n):
                if j != i:
                    if subset & (1 << pos):
                        coalition.append(j)
                    pos += 1
            
            # Calculate marginal contribution
            s = len(coalition)
            weight = float(s * (n-s-1)) / float(n)
            marginal = v(coalition + [i]) - v(coalition)
            shapley_values[i] += weight * marginal
            
    return shapley_values

def calculate_metrics(episode_contributions: List[float], episode_rewards: List[float], r: float) -> Dict[str, float]:
    """Calculate metrics for the current episode."""
    n_agents = len(episode_contributions)
    
    # Basic metrics
    avg_contribution = np.mean(episode_contributions)
    total_contribution = np.sum(episode_contributions)
    social_welfare = np.sum(episode_rewards)
    
    # Calculate Shapley values
    shapley_values = calculate_shapley_value(episode_contributions, r)
    
    # Normalized contributions (as percentages of endowments)
    norm_contributions = [c/(0.5*(i+1))*100 for i, c in enumerate(episode_contributions)]
    
    metrics = {
        'avg_contribution': avg_contribution,
        'total_contribution': total_contribution,
        'social_welfare': social_welfare,
        'fairness': 1.0 - (max(shapley_values) - min(shapley_values)) / (max(shapley_values) + 1e-10),
        'action_diversity': len(set(norm_contributions)) / n_agents,
        **{f'agent_{i}_contrib': contrib for i, contrib in enumerate(episode_contributions)},
        **{f'agent_{i}_norm_contrib': norm_contrib for i, norm_contrib in enumerate(norm_contributions)},
        **{f'agent_{i}_shapley': sv for i, sv in enumerate(shapley_values)}
    }
    
    return metrics

def run_simulation(
    algorithm: str,
    num_episodes: int,
    steps_per_episode: int,
    env: PublicGoodsGame,
    agents: List[Union[QAgent, DoubleQAgent]],
    output_dir: str
) -> Dict[str, List[float]]:
    """
    Run MARL simulation, return and save metrics.
    
    Executes a multi-agent reinforcement learning simulation for a Public Goods Game
    with either Q-Learning or Double Q-Learning agents. Collects various metrics on
    cooperation, contribution patterns, and social welfare.
    
    Args:
        algorithm: Algorithm identifier (e.g., "q_r1.5" or "double_q_r2.0")
        num_episodes: Number of episodes to run
        steps_per_episode: Number of steps per episode
        env: PublicGoodsGame environment instance
        agents: List of QAgent or DoubleQAgent instances
        output_dir: Directory to save output files
        
    Returns:
        Dictionary containing collected metrics across episodes
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize metrics dictionaries
    metrics_history = {
        "avg_contribution": [],
        "total_contribution": [],
        "social_welfare": [],
        "fairness": [],
        "action_diversity": []
    }
    
    # Add individual contributions, normalized contributions, and Shapley values for each agent
    for i in range(len(agents)):
        metrics_history[f"agent_{i}_contrib"] = []
        metrics_history[f"agent_{i}_norm_contrib"] = []
        metrics_history[f"agent_{i}_shapley"] = []
    
    # Simulation loop
    for episode in range(num_episodes):
        # Reset environment
        states = env.reset()
        
        # Track metrics for this episode
        episode_contributions = [0.0] * len(agents)
        episode_rewards = [0.0] * len(agents)
        episode_actions = set()
        
        for step in range(steps_per_episode):
            # Determine active agents (75% chance for all agents, 25% chance for all but one)
            if random.random() < 0.75:
                active_agents = list(range(len(agents)))
            else:
                # Randomly exclude one agent
                excluded_agent = random.randint(0, len(agents) - 1)
                active_agents = [i for i in range(len(agents)) if i != excluded_agent]
            
            # Choose actions for active agents
            actions = []
            for idx in active_agents:
                action = agents[idx].choose_action(states[idx])
                actions.append(action)
                episode_actions.add(action)
            
            # Take step in environment
            next_states, rewards, done = env.step(actions, active_agents)
            
            # Update agents
            for i, agent_idx in enumerate(active_agents):
                agents[agent_idx].update(states[agent_idx], actions[i], rewards[agent_idx], next_states[agent_idx])
                
                # Track contributions and rewards
                contribution = actions[i] * env.endowments[agent_idx]
                episode_contributions[agent_idx] += contribution
                episode_rewards[agent_idx] += rewards[agent_idx]
            
            # Update states
            states = next_states
        
        # Calculate metrics
        episode_metrics = calculate_metrics(episode_contributions, episode_rewards, env.r)
        
        # Store metrics
        for key, value in episode_metrics.items():
            metrics_history[key].append(value)
        
        # Print progress every 100 episodes
        if episode % 100 == 0:
            print(f"Episode {episode}/{num_episodes} - Avg Contribution: {episode_metrics['avg_contribution']:.4f}, Social Welfare: {episode_metrics['social_welfare']:.4f}")
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'episode': range(num_episodes),
        'avg_contribution': metrics_history['avg_contribution'],
        'total_contribution': metrics_history['total_contribution'],
        'social_welfare': metrics_history['social_welfare'],
        'fairness': metrics_history['fairness'],
        'action_diversity': metrics_history['action_diversity']
    })
    
    # Add individual agent contributions, normalized contributions, and Shapley values
    for i in range(len(agents)):
        metrics_df[f'agent_{i}_contrib'] = metrics_history[f'agent_{i}_contrib']
        metrics_df[f'agent_{i}_norm_contrib'] = metrics_history[f'agent_{i}_norm_contrib']
        metrics_df[f'agent_{i}_shapley'] = metrics_history[f'agent_{i}_shapley']
    
    # Save metrics to CSV
    metrics_file = os.path.join(output_dir, f"{algorithm}_metrics.csv")
    metrics_df.to_csv(metrics_file, index=False)
    print(f"Metrics saved to {metrics_file}")
    
    # Save Q-values
    q_data = []
    is_double_q = isinstance(agents[0], DoubleQAgent)
    
    for agent_idx, agent in enumerate(agents):
        if is_double_q:
            # For Double Q-Learning agents
            q_table_a, q_table_b = agent.get_q_values()
            for state in env.state_bins:  # Use state bins from environment
                for action in env.action_space:
                    q_data.append({
                        'agent': agent_idx,
                        'state': str(state),
                        'action': action,
                        'q_a_value': q_table_a.get((state, action), 0.0),
                        'q_b_value': q_table_b.get((state, action), 0.0)
                    })
        else:
            # For standard Q-Learning agents
            q_table = agent.get_q_values()
            for state in env.state_bins:  # Use state bins from environment
                for action in env.action_space:
                    q_data.append({
                        'agent': agent_idx,
                        'state': str(state),
                        'action': action,
                        'q_value': q_table.get((state, action), 0.0)
                    })
    
    # Save Q-values to CSV
    q_values_file = os.path.join(output_dir, f"{algorithm}_qvalues.csv")
    pd.DataFrame(q_data).to_csv(q_values_file, index=False)
    print(f"Q-values saved to {q_values_file}")
    
    return metrics_history

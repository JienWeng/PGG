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

def gini_coefficient(contributions: List[float]) -> float:
    """
    Calculate Gini coefficient as a measure of inequality.
    
    Args:
        contributions: List of contribution values
        
    Returns:
        Gini coefficient (0 = perfect equality, 1 = perfect inequality)
    """
    # Handle case with no contributions
    if sum(contributions) == 0:
        return 0.0
        
    # Sort contributions in ascending order
    sorted_contributions = sorted(contributions)
    n = len(sorted_contributions)
    
    # Calculate Gini coefficient
    cumsum = np.cumsum(sorted_contributions)
    return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0.0

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
    metrics = {
        "avg_contribution": [],
        "social_welfare": [],
        "contribution_variance": [],
        "gini_coefficient": [],
        "action_diversity": []
    }
    
    # Add individual contributions for each agent
    for i in range(len(agents)):
        metrics[f"agent_{i}_contrib"] = []
    
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
        
        # Calculate average contributions per agent (over the episode)
        avg_episode_contributions = [c / steps_per_episode for c in episode_contributions]
        
        # Calculate metrics
        avg_contribution = np.mean(avg_episode_contributions)
        social_welfare = np.sum(episode_rewards)
        contribution_variance = np.var(avg_episode_contributions)
        gini = gini_coefficient(avg_episode_contributions)
        action_diversity = len(episode_actions)
        
        # Store metrics
        metrics["avg_contribution"].append(avg_contribution)
        metrics["social_welfare"].append(social_welfare)
        metrics["contribution_variance"].append(contribution_variance)
        metrics["gini_coefficient"].append(gini)
        metrics["action_diversity"].append(action_diversity)
        
        # Store individual contributions
        for i in range(len(agents)):
            metrics[f"agent_{i}_contrib"].append(avg_episode_contributions[i])
        
        # Print progress every 100 episodes
        if episode % 100 == 0:
            print(f"Episode {episode}/{num_episodes} - Avg Contribution: {avg_contribution:.4f}, Social Welfare: {social_welfare:.4f}")
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'episode': range(num_episodes),
        'avg_contribution': metrics['avg_contribution'],
        'social_welfare': metrics['social_welfare'],
        'contribution_variance': metrics['contribution_variance'],
        'gini_coefficient': metrics['gini_coefficient'],
        'action_diversity': metrics['action_diversity']
    })
    
    # Add individual agent contributions
    for i in range(len(agents)):
        metrics_df[f'agent_{i}_contrib'] = metrics[f'agent_{i}_contrib']
    
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
    
    return metrics

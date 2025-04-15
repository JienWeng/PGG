import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Union, Sequence
from pathlib import Path
from ast import literal_eval

def gini(x: Sequence[float]) -> float:
    """
    Calculate Gini coefficient of array x.
    
    Args:
        x: Array of values
        
    Returns:
        Gini coefficient (0 = perfect equality, 1 = perfect inequality)
    """
    # Convert to float and sort
    x = np.array(x, dtype=float)
    sorted_x = np.sort(x)
    n = len(x)
    
    # Edge case
    if n == 0 or np.sum(x) == 0:
        return 0.0
        
    # Calculate cumulative sum and Gini coefficient
    cumsum = np.cumsum(sorted_x)
    return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n

def load_q_values(filename: str, is_double_q: bool) -> Dict[int, Dict[Tuple[Tuple[float, float], float], float]]:
    """Load Q-table from CSV."""
    df = pd.read_csv(filename)
    df['state'] = df['state'].apply(literal_eval)  # Convert string tuples to actual tuples
    
    q_values = {}
    for agent in df['agent'].unique():
        agent_data = df[df['agent'] == agent]
        q_values[agent] = {}
        
        if is_double_q:
            for _, row in agent_data.iterrows():
                q_values[agent][(row['state'], row['action'])] = (row['q_a_value'] + row['q_b_value']) / 2
        else:
            for _, row in agent_data.iterrows():
                q_values[agent][(row['state'], row['action'])] = row['q_value']
    
    return q_values

def compute_metrics(
    q_values: Dict[int, Dict[Tuple[Tuple[float, float], float], float]],
    is_double_q: bool
) -> Dict[int, Dict[str, Any]]:
    """Compute Q-table metrics."""
    metrics = {}
    
    for agent, agent_q_values in q_values.items():
        # Group Q-values by state
        state_values = {}
        for (state, action), q_value in agent_q_values.items():
            if state not in state_values:
                state_values[state] = {}
            state_values[state][action] = q_value
        
        # Calculate metrics for each state
        preferred_actions = {}
        avg_q_values = {}
        q_variances = {}
        
        # Process each state
        for state, actions in state_values.items():
            preferred_actions[state] = max(actions.items(), key=lambda x: x[1])[0]
            avg_q_values[state] = np.mean(list(actions.values()))
            q_variances[state] = np.var(list(actions.values()))
        
        # Calculate cooperation score
        cooperation_score = sum(1 for a in preferred_actions.values() if a >= 0.5) / len(preferred_actions)
        
        # Calculate action balance using Gini coefficient
        action_counts = np.zeros(25)  # Assuming 25 actions
        for action in preferred_actions.values():
            action_idx = int(action * 24)  # Convert action to index
            action_counts[action_idx] += 1
        
        if sum(action_counts) > 0:
            sorted_counts = np.sort(action_counts)
            cumsum = np.cumsum(sorted_counts)
            action_balance = 1 - 2 * np.sum(cumsum[:-1]) / (len(action_counts) * cumsum[-1])
        else:
            action_balance = 0.0
        
        metrics[agent] = {
            "preferred_action": preferred_actions,
            "avg_q_value": avg_q_values,
            "q_variance": q_variances,
            "cooperation_score": cooperation_score,
            "action_balance": action_balance
        }
    
    return metrics

def evaluate_q_tables(
    q_qvalues: Dict[int, Dict[Tuple[Tuple[float, float], float], float]],
    double_q_qvalues: Dict[int, Dict[Tuple[Tuple[float, float], float], float]],
    output_dir: str,
    r: float
) -> None:
    """Evaluate and visualize Q-tables."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    results = []
    
    # Process each agent's data
    for algo, data, name in [("q", q_qvalues, "Q-Learning"), 
                           ("double_q", double_q_qvalues, "Double Q-Learning")]:
        for agent in data.keys():
            # Get states and sort them by endowment then contribution
            states = sorted(set(state for (state, _) in data[agent].keys()))
            actions = []
            
            # Collect preferred actions for each state
            for state in states:
                state_actions = {action: data[agent][(state, action)] 
                               for (s, action) in data[agent].keys() if s == state}
                actions.append(max(state_actions.items(), key=lambda x: x[1])[0])
            
            # Convert to numpy array and reshape based on new dimensions
            actions = np.array(actions)
            action_matrix = actions.reshape(7, 11)  # 7 endowment levels, 11 contribution levels
            
            # Create heatmap
            plt.figure(figsize=(12, 8))
            sns.heatmap(action_matrix, cmap='RdYlBu', center=0.5)
            plt.title(f'{name} Preferred Actions - Agent {agent} (r={r})')
            plt.xlabel('Contribution Level')
            plt.ylabel('Endowment Level')
            
            # Update x and y tick labels to match new bins
            plt.xticks(np.arange(11), ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', 
                                     '0.6', '0.7', '0.8', '0.9', '1.0'])
            plt.yticks(np.arange(7), ['0.0', '0.25', '0.5', '0.75', '1.0', '1.25', '1.5'])
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/{algo}_preferred_actions_agent_{agent}_r{r}.png')
            plt.close()
            
            # Calculate metrics
            cooperation_score = np.mean(actions >= 0.5)
            action_balance = 1 - gini(np.bincount(
                (actions * 24).astype(int), minlength=25
            ) / len(actions))
            
            # Add metrics to results
            results.append({
                'algorithm': name,
                'agent': agent,
                'cooperation_score': cooperation_score,
                'action_balance': action_balance
            })
    
    # Save results
    pd.DataFrame(results).to_csv(
        f'{output_dir}/q_table_summary_r{r}.csv',
        index=False
    )

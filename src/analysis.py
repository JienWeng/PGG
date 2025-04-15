import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Union
from pathlib import Path

def load_metrics(filename: str) -> Dict[str, List[float]]:
    """Load metrics from CSV."""
    df = pd.read_csv(filename)
    return {col: df[col].tolist() for col in df.columns}

def smooth_metrics(metrics: Dict[str, List[float]], window: int = 50) -> Dict[str, List[float]]:
    """
    Apply moving average smoothing to metrics.
    
    Args:
        metrics: Dictionary of metric arrays
        window: Moving average window size
        
    Returns:
        Dictionary of smoothed metric arrays
    """
    smoothed = {}
    for key, values in metrics.items():
        if key != 'episode':  # Don't smooth episode numbers
            series = pd.Series(values)
            smoothed[key] = series.rolling(window=window, min_periods=1, center=True).mean().tolist()
        else:
            smoothed[key] = values
    return smoothed

def plot_metrics(
    q_metrics: Dict[str, List[float]],
    double_q_metrics: Dict[str, List[float]],
    output_dir: str,
    r: float
) -> None:
    """Plot episode-wise metrics."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Apply smoothing with larger window for smoother curves
    window = 100  # Increased window size for smoother plots
    q_smoothed = smooth_metrics(q_metrics, window)
    double_q_smoothed = smooth_metrics(double_q_metrics, window)
    episodes = range(len(q_smoothed['avg_contribution']))
    
    # Set style for better visualization
    plt.style.use('seaborn-paper')
    plt.rcParams['lines.linewidth'] = 2.0
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    
    # Common figure settings
    figure_settings = {
        'figsize': (10, 6),
    }
    
    # Common plot settings
    plot_settings = {
        'alpha': 0.8,
    }
    
    # Colors for individual agent plots
    colors = plt.cm.viridis(np.linspace(0, 1, 4))

    # 1. Average Contribution Comparison
    plt.figure(**figure_settings)
    plt.plot(episodes, q_smoothed['avg_contribution'], label='Q-Learning', **plot_settings)
    plt.plot(episodes, double_q_smoothed['avg_contribution'], label='Double Q-Learning', **plot_settings)
    plt.xlabel('Episode')
    plt.ylabel('Average Contribution')
    plt.title(f'Contribution Comparison (r={r})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_dir}/contribution_comparison_r{r}.png', bbox_inches='tight', dpi=300)
    plt.close()

    # 2. Social Welfare Comparison
    plt.figure(**figure_settings)
    plt.plot(episodes, q_smoothed['social_welfare'], label='Q-Learning', **plot_settings)
    plt.plot(episodes, double_q_smoothed['social_welfare'], label='Double Q-Learning', **plot_settings)
    plt.xlabel('Episode')
    plt.ylabel('Social Welfare')
    plt.title(f'Social Welfare Comparison (r={r})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_dir}/welfare_comparison_r{r}.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # 3. Gini Coefficient Comparison
    plt.figure(**figure_settings)
    plt.plot(episodes, q_smoothed['gini_coefficient'], label='Q-Learning', **plot_settings)
    plt.plot(episodes, double_q_smoothed['gini_coefficient'], label='Double Q-Learning', **plot_settings)
    plt.xlabel('Episode')
    plt.ylabel('Gini Coefficient')
    plt.title(f'Inequality Comparison (r={r})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_dir}/gini_comparison_r{r}.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # 4. Action Diversity Comparison
    plt.figure(**figure_settings)
    plt.plot(episodes, q_smoothed['action_diversity'], label='Q-Learning', **plot_settings)
    plt.plot(episodes, double_q_smoothed['action_diversity'], label='Double Q-Learning', **plot_settings)
    plt.xlabel('Episode')
    plt.ylabel('Action Diversity')
    plt.title(f'Action Diversity Comparison (r={r})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_dir}/action_diversity_r{r}.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # 5. Individual Contributions (Q-Learning)
    plt.figure(**figure_settings)
    for i in range(4):
        plt.plot(episodes, q_smoothed[f'agent_{i}_contrib'], 
                label=f'Agent {i} (e={0.5*(i+1)})',
                color=colors[i],
                **plot_settings)
    plt.xlabel('Episode')
    plt.ylabel('Contribution')
    plt.title(f'Individual Contributions - Q-Learning (r={r})')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/q_individual_contribs_r{r}.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # 6. Individual Contributions (Double Q-Learning)
    plt.figure(**figure_settings)
    for i in range(4):
        plt.plot(episodes, double_q_smoothed[f'agent_{i}_contrib'], 
                label=f'Agent {i} (e={0.5*(i+1)})',
                color=colors[i],
                **plot_settings)
    plt.xlabel('Episode')
    plt.ylabel('Contribution')
    plt.title(f'Individual Contributions - Double Q-Learning (r={r})')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/double_q_individual_contribs_r{r}.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # 7. Final Contributions Comparison
    plt.figure(**figure_settings)
    final_q = q_smoothed['avg_contribution'][-1]
    final_double_q = double_q_smoothed['avg_contribution'][-1]
    plt.bar(['Q-Learning', 'Double Q-Learning'], [final_q, final_double_q], alpha=0.8)
    plt.ylabel('Final Average Contribution')
    plt.title(f'Final Contribution Comparison (r={r})')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_dir}/final_contribution_r{r}.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_heatmap(
    q_qvalues: pd.DataFrame,
    double_q_qvalues: pd.DataFrame,
    output_dir: str,
    r: float
) -> None:
    """Plot Q-value heatmaps."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Process Q-Learning heatmaps
    for agent in q_qvalues['agent'].unique():
        agent_data = q_qvalues[q_qvalues['agent'] == agent].copy()  # Create explicit copy
        pivot_table = pd.pivot_table(
            agent_data,
            values='q_value',
            index='state',
            columns='action'
        )
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_table, center=0)
        plt.title(f'Q-Values Heatmap - Agent {agent} (r={r})')
        plt.xlabel('Action')
        plt.ylabel('State')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/q_heatmap_agent_{agent}_r{r}.png')
        plt.close()
    
    # Process Double Q-Learning heatmaps
    for agent in double_q_qvalues['agent'].unique():
        agent_data = double_q_qvalues[double_q_qvalues['agent'] == agent].copy()  # Create explicit copy
        agent_data.loc[:, 'avg_q'] = (agent_data['q_a_value'] + agent_data['q_b_value']) / 2
        pivot_table = pd.pivot_table(
            agent_data,
            values='avg_q',
            index='state',
            columns='action'
        )
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_table, cmap='viridis', center=0)
        plt.title(f'Double Q-Values Heatmap - Agent {agent} (r={r})')
        plt.xlabel('Action')
        plt.ylabel('State')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/double_q_heatmap_agent_{agent}_r{r}.png')
        plt.close()

def save_summary(
    q_metrics: Dict[str, List[float]],
    double_q_metrics: Dict[str, List[float]],
    output_dir: str,
    r: float
) -> None:
    """Save smoothed metric summary."""
    # Apply smoothing
    q_smoothed = smooth_metrics(q_metrics)
    double_q_smoothed = smooth_metrics(double_q_metrics)
    
    # Metrics to analyze (excluding 'episode')
    metrics_to_analyze = [
        'avg_contribution', 'social_welfare', 'gini_coefficient', 
        'action_diversity', 'agent_0_contrib', 'agent_1_contrib',
        'agent_2_contrib', 'agent_3_contrib'
    ]
    
    summary_data = []
    
    # Process Q-Learning metrics
    for metric in metrics_to_analyze:
        values = q_smoothed[metric]
        summary_data.append({
            'algorithm': 'Q-Learning',
            'metric': metric,
            'mean': np.mean(values),
            'std': np.std(values)
        })
    
    # Process Double Q-Learning metrics
    for metric in metrics_to_analyze:
        values = double_q_smoothed[metric]
        summary_data.append({
            'algorithm': 'Double Q-Learning',
            'metric': metric,
            'mean': np.mean(values),
            'std': np.std(values)
        })
    
    # Save summary
    pd.DataFrame(summary_data).to_csv(
        f'{output_dir}/summary_r{r}.csv',
        index=False
    )

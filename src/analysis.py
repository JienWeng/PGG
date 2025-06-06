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
    """Apply moving average smoothing to metrics."""
    smoothed = {}
    for key, values in metrics.items():
        if key != 'episode':
            series = pd.Series(values)
            smoothed[key] = series.rolling(window=window, min_periods=1, center=True).mean().tolist()
        else:
            smoothed[key] = values
    return smoothed

def plot_metrics(q_metrics: Dict, double_q_metrics: Dict, output_dir: str, r: float) -> None:
    """Plot all metrics with consistent settings."""
    # Plotting settings
    plt.rcParams.update({
        'figure.figsize': (10, 6),
        'lines.linewidth': 2.0,
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 300
    })
    
    # Data collection settings
    window = 50  # Smoothing window
    last_n = 1000  # Analysis of last N episodes
    
    # Create output directory structure
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    analysis_dir = os.path.join(output_dir, f'analysis_r{r}')
    Path(analysis_dir).mkdir(parents=True, exist_ok=True)

    # Objective 1: Normalized contribution rates and social welfare
    plot_normalized_contributions(q_metrics, double_q_metrics, output_dir, r)
    analyze_social_welfare(q_metrics, double_q_metrics, output_dir, r)
    
    # Objective 2: Individual contributions
    plot_individual_contributions(q_metrics, double_q_metrics, output_dir, r)
    
    # Objective 3: Shapley value analysis
    plot_shapley_analysis(q_metrics, double_q_metrics, output_dir, r)

def plot_normalized_contributions(q_metrics: Dict, double_q_metrics: Dict, output_dir: str, r: float):
    """Plot normalized contribution rates against r_t for each endowment."""
    fig, ax = plt.subplots()
    markers = ['o', 's', '^', 'D']
    colors = plt.cm.Set2(np.linspace(0, 1, 4))
    
    for i, e in enumerate([0.5 * (i+1) for i in range(4)]):
        # Q-Learning
        norm_contrib = [c/e * 100 for c in q_metrics[f'agent_{i}_contrib']]
        ax.scatter(r * np.ones_like(norm_contrib[-1000:]), norm_contrib[-1000:],
                  marker=markers[i], c=[colors[i]], alpha=0.3,
                  label=f'Q-Learn e={e}')
        
        # Double Q-Learning
        norm_contrib = [c/e * 100 for c in double_q_metrics[f'agent_{i}_contrib']]
        ax.scatter(r + 0.1 + np.zeros_like(norm_contrib[-1000:]), norm_contrib[-1000:],
                  marker=markers[i], c=[colors[i]], alpha=0.3,
                  label=f'Double-Q e={e}')
    
    ax.set_xlabel('Multiplication Factor (r)')
    ax.set_ylabel('Normalized Contribution Rate (%)')
    ax.set_title(f'Last 1000 Episodes Contribution Rates (r={r})')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/normalized_contributions_r{r}.png', bbox_inches='tight')
    plt.close()

def analyze_social_welfare(q_metrics: Dict, double_q_metrics: Dict, output_dir: str, r: float):
    """Analyze social welfare and create summary statistics."""
    # Plot social welfare over episodes with 50-episode smoothing
    q_welfare = pd.Series(q_metrics['social_welfare']).rolling(50, min_periods=1).mean()
    dq_welfare = pd.Series(double_q_metrics['social_welfare']).rolling(50, min_periods=1).mean()
    
    fig, ax = plt.subplots()
    ax.plot(range(len(q_welfare)), q_welfare, label='Q-Learning')
    ax.plot(range(len(dq_welfare)), dq_welfare, label='Double Q-Learning')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Social Welfare')
    ax.set_title(f'Social Welfare Over Episodes (r={r})')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/social_welfare_r{r}.png')
    plt.close()
    
    # Summary statistics for last 1000 episodes
    stats = pd.DataFrame({
        'Algorithm': ['Q-Learning', 'Double Q-Learning'],
        'Mean Contribution (%)': [
            np.mean(q_metrics['avg_contribution'][-1000:]) * 100,
            np.mean(double_q_metrics['avg_contribution'][-1000:]) * 100
        ],
        'Std Dev (%)': [
            np.std(q_metrics['avg_contribution'][-1000:]) * 100,
            np.std(double_q_metrics['avg_contribution'][-1000:]) * 100
        ]
    })
    stats.to_csv(f'{output_dir}/contribution_stats_r{r}.csv', index=False)

def plot_individual_contributions(q_metrics: Dict, double_q_metrics: Dict, output_dir: str, r: float):
    """Plot individual contributions with 50-episode smoothing window."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    colors = plt.cm.Set2(np.linspace(0, 1, 4))
    window = 50
    
    # Q-Learning contributions
    for i in range(4):
        contrib = pd.Series(q_metrics[f'agent_{i}_contrib']).rolling(window, min_periods=1).mean()
        ax1.plot(range(len(contrib)), contrib, 
                label=f'Agent {i} (e={0.5*(i+1)})',
                color=colors[i])
    ax1.set_title(f'Q-Learning Individual Contributions (r={r})')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Contribution')
    ax1.legend()
    
    # Double Q-Learning contributions
    for i in range(4):
        contrib = pd.Series(double_q_metrics[f'agent_{i}_contrib']).rolling(window, min_periods=1).mean()
        ax2.plot(range(len(contrib)), contrib, 
                label=f'Agent {i} (e={0.5*(i+1)})',
                color=colors[i])
    ax2.set_title(f'Double Q-Learning Individual Contributions (r={r})')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Contribution')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/individual_contributions_r{r}.png')
    plt.close()

def plot_shapley_analysis(q_metrics: Dict, double_q_metrics: Dict, output_dir: str, r: float):
    """Plot Shapley values analysis with 50-episode smoothing."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    colors = plt.cm.Set2(np.linspace(0, 1, 4))
    window = 50
    
    # Q-Learning Shapley values
    for i in range(4):
        shapley = pd.Series(q_metrics[f'agent_{i}_shapley']).rolling(window, min_periods=1).mean()
        ax1.plot(range(len(shapley)), shapley, 
                label=f'Agent {i} (e={0.5*(i+1)})',
                color=colors[i])
    ax1.set_title(f'Q-Learning Shapley Values (r={r})')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Shapley Value')
    ax1.legend()
    
    # Double Q-Learning Shapley values
    for i in range(4):
        shapley = pd.Series(double_q_metrics[f'agent_{i}_shapley']).rolling(window, min_periods=1).mean()
        ax2.plot(range(len(shapley)), shapley, 
                label=f'Agent {i} (e={0.5*(i+1)})',
                color=colors[i])
    ax2.set_title(f'Double Q-Learning Shapley Values (r={r})')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Shapley Value')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/shapley_values_r{r}.png')
    plt.close()

import os
from typing import Dict, Tuple, Any, List
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def aggregate_seed_data(base_dir: str, r: float, seeds: List[int], is_double_q: bool) -> pd.DataFrame:
    """Aggregate metrics across seeds."""
    metrics_dfs = []
    prefix = "double_q" if is_double_q else "q"
    
    for seed in seeds:
        seed_dir = os.path.join(base_dir, f"r{r}", f"seed{seed}")
        if os.path.exists(seed_dir):
            metrics_file = os.path.join(seed_dir, f"{prefix}_r{r}_seed{seed}_metrics.csv")
            if os.path.exists(metrics_file):
                metrics_dfs.append(pd.read_csv(metrics_file))
                
    if not metrics_dfs:
        print(f"Warning: Missing data for r={r}, {'Double Q' if is_double_q else 'Q'}-Learning")
        return pd.DataFrame()
        
    return pd.concat(metrics_dfs).groupby(level=0).mean()

def conduct_statistical_tests(q_metrics: pd.DataFrame, double_q_metrics: pd.DataFrame, output_dir: str, r: float) -> Dict:
    """Conduct paired t-tests for metrics."""
    metrics = {
        'avg_contribution': 'Average Contribution',
        'social_welfare': 'Social Welfare',
        'fairness': 'Fairness (Shapley-based)',
    }
    
    results = {}
    for metric, label in metrics.items():
        t_stat, p_value = stats.ttest_rel(
            q_metrics[metric].iloc[-1000:],
            double_q_metrics[metric].iloc[-1000:],
            alternative='two-sided'
        )
        
        results[metric] = {
            'metric': label,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    pd.DataFrame([{
        'Metric': v['metric'],
        'T-statistic': v['t_statistic'],
        'P-value': v['p_value'],
        'Significant': v['significant']
    } for v in results.values()]).to_csv(
        os.path.join(output_dir, f'statistical_tests_r{r}.csv'), 
        index=False
    )
    
    return results

def plot_required_metrics(q_metrics: pd.DataFrame, double_q_metrics: pd.DataFrame, output_dir: str, r: float):
    """Plot the five required visualizations."""
    # 1. Normalized contribution rates
    plt.figure(figsize=(10, 6))
    for i in range(4):
        endowment = 0.5 * (i + 1)
        q_norm = [c/endowment * 100 for c in q_metrics[f'agent_{i}_contrib']]
        dq_norm = [c/endowment * 100 for c in double_q_metrics[f'agent_{i}_contrib']]
        
        plt.scatter([r] * len(q_norm[-1000:]), q_norm[-1000:], alpha=0.3, label=f'Q-Learn e={endowment}')
        plt.scatter([r+0.1] * len(dq_norm[-1000:]), dq_norm[-1000:], alpha=0.3, label=f'Double-Q e={endowment}')
    
    plt.xlabel('Multiplication Factor (r)')
    plt.ylabel('Normalized Contribution Rate (%)')
    plt.title(f'Last 1000 Episodes Contribution Rates (r={r})')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'normalized_contributions_r{r}.png'))
    plt.close()
    
    # 2. Social welfare over episodes
    plt.figure(figsize=(10, 6))
    episodes = range(len(q_metrics))
    plt.plot(episodes, q_metrics['social_welfare'].rolling(50).mean(), label='Q-Learning')
    plt.plot(episodes, double_q_metrics['social_welfare'].rolling(50).mean(), label='Double Q-Learning')
    plt.xlabel('Episode')
    plt.ylabel('Social Welfare')
    plt.title(f'Social Welfare Evolution (r={r})')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'social_welfare_r{r}.png'))
    plt.close()
    
    # 3. Individual contributions
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    for i in range(4):
        ax1.plot(episodes, pd.Series(q_metrics[f'agent_{i}_contrib']).rolling(50).mean(),
                label=f'Agent {i} (e={0.5*(i+1)})')
        ax2.plot(episodes, pd.Series(double_q_metrics[f'agent_{i}_contrib']).rolling(50).mean(),
                label=f'Agent {i} (e={0.5*(i+1)})')
    
    ax1.set_title(f'Q-Learning Individual Contributions (r={r})')
    ax2.set_title(f'Double Q-Learning Individual Contributions (r={r})')
    ax1.legend()
    ax2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'individual_contributions_r{r}.png'))
    plt.close()

    # 4. Shapley values
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    for i in range(4):
        ax1.plot(episodes, pd.Series(q_metrics[f'agent_{i}_shapley']).rolling(50).mean(),
                label=f'Agent {i} (e={0.5*(i+1)})')
        ax2.plot(episodes, pd.Series(double_q_metrics[f'agent_{i}_shapley']).rolling(50).mean(),
                label=f'Agent {i} (e={0.5*(i+1)})')
    
    ax1.set_title(f'Q-Learning Shapley Values (r={r})')
    ax2.set_title(f'Double Q-Learning Shapley Values (r={r})')
    ax1.legend()
    ax2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'shapley_values_r{r}.png'))
    plt.close()

    # Save mean and std dev of contribution rates
    stats_df = pd.DataFrame({
        'Algorithm': ['Q-Learning', 'Double Q-Learning'],
        'Mean Contribution (%)': [
            q_metrics['avg_contribution'].iloc[-1000:].mean() / 200 * 100 ,
            double_q_metrics['avg_contribution'].iloc[-1000:].mean() / 200 * 100
        ],
        'Std Dev (%)': [
            q_metrics['avg_contribution'].iloc[-1000:].std() / 200 * 100,
            double_q_metrics['avg_contribution'].iloc[-1000:].std() / 200 * 100
        ]
    })
    stats_df.to_csv(os.path.join(output_dir, f'contribution_stats_r{r}.csv'), index=False)

def plot_shapley_analysis(q_metrics: pd.DataFrame, double_q_metrics: pd.DataFrame, output_dir: str, r: float):
    """Plot Shapley values distribution for each agent."""
    # Get last 1000 episodes data
    window = -1000
    
    # Prepare data
    agents = range(4)
    q_shapley_values = [q_metrics[f'agent_{i}_shapley'].iloc[window:].mean() for i in agents]
    dq_shapley_values = [double_q_metrics[f'agent_{i}_shapley'].iloc[window:].mean() for i in agents]
    q_shapley_std = [q_metrics[f'agent_{i}_shapley'].iloc[window:].std() for i in agents]
    dq_shapley_std = [double_q_metrics[f'agent_{i}_shapley'].iloc[window:].std() for i in agents]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot Q-Learning Shapley values
    x = np.arange(len(agents))
    width = 0.8
    bars1 = ax1.bar(x, q_shapley_values, width, yerr=q_shapley_std, capsize=5,
                    label=[f'e={0.5*(i+1)}' for i in agents])
    
    ax1.set_ylabel('Shapley Value')
    ax1.set_title(f'Q-Learning Shapley Values (r={r})')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Agent {i}' for i in agents])
    
    # Add value labels
    for bar, std in zip(bars1, q_shapley_std):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}\n±{std:.2f}',
                ha='center', va='bottom')
    
    # Plot Double Q-Learning Shapley values
    bars2 = ax2.bar(x, dq_shapley_values, width, yerr=dq_shapley_std, capsize=5,
                    label=[f'e={0.5*(i+1)}' for i in agents])
    
    ax2.set_ylabel('Shapley Value')
    ax2.set_title(f'Double Q-Learning Shapley Values (r={r})')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'Agent {i}' for i in agents])
    
    # Add value labels
    for bar, std in zip(bars2, dq_shapley_std):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}\n±{std:.2f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'shapley_distribution_r{r}.png'), dpi=300)
    plt.close()

def plot_shap_summary(q_metrics: pd.DataFrame, double_q_metrics: pd.DataFrame, output_dir: str, r: float):
    """Create SHAP summary plot for agent contributions."""
    import shap
    import numpy as np
    
    # Prepare feature matrix
    features = []
    feature_names = []
    
    # Get last 1000 episodes data for both Q and Double Q
    for i in range(4):
        # Q-Learning data
        q_contrib = q_metrics[f'agent_{i}_contrib'].iloc[-1000:].values
        features.append(q_contrib)
        feature_names.append(f'Agent {i} (Q)')
        
        # Double Q-Learning data
        dq_contrib = double_q_metrics[f'agent_{i}_contrib'].iloc[-1000:].values
        features.append(dq_contrib)
        feature_names.append(f'Agent {i} (DQ)')
    
    # Convert to numpy array and transpose
    X = np.array(features).T
    
    # Calculate SHAP values (using KernelExplainer as an example)
    background = shap.kmeans(X, 100)  # Background distribution
    explainer = shap.KernelExplainer(lambda x: x.sum(1), background)
    shap_values = explainer.shap_values(X)
    
    # Create SHAP summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values,
        X,
        feature_names=feature_names,
        show=False,
        max_display=8  # Show all agents
    )
    
    # Customize plot
    plt.title(f'SHAP Summary Plot (r={r})')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'shap_summary_r{r}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_summary(
    q_metrics: Dict[float, pd.DataFrame],
    double_q_metrics: Dict[float, pd.DataFrame],
    output_dir: str
) -> None:
    """
    Plot smoothed summary of contribution vs. r with error bars.
    Proportion calculated as: (actual contribution / maximum possible contribution) * 100
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    r_values = sorted(q_metrics.keys())
    q_means = []
    q_stds = []
    double_q_means = []
    double_q_stds = []
    
    window = 50  # Smoothing window
    for r in r_values:
        # Calculate proportions for Q-Learning
        q_proportions = []
        for i in range(4):
            max_contribution = 0.5 * (i + 1) * 200 # Maximum possible contribution for agent i
            agent_contrib = q_metrics[r][f'agent_{i}_contrib']
            q_proportions.extend(agent_contrib / max_contribution)  # Convert to proportion
        
        q_smooth = pd.Series(q_proportions).rolling(window=window, min_periods=1, center=True).mean()
        q_means.append(q_smooth.mean() * 100)  # Convert to percentage
        q_stds.append(q_smooth.std() * 100)
        
        # Calculate proportions for Double Q-Learning
        dq_proportions = []
        for i in range(4):
            max_contribution = 0.5 * (i + 1) * 200
            agent_contrib = double_q_metrics[r][f'agent_{i}_contrib']
            dq_proportions.extend(agent_contrib / max_contribution)
            
        dq_smooth = pd.Series(dq_proportions).rolling(window=window, min_periods=1, center=True).mean()
        double_q_means.append(dq_smooth.mean() * 100)
        double_q_stds.append(dq_smooth.std() * 100)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    # Plot Q-Learning with error bars
    plt.errorbar(r_values, q_means, yerr=q_stds, 
                fmt='o-', color='#1f77b4', capsize=5, 
                label='Q-Learning', linewidth=2, markersize=8)
    
    # Plot Double Q-Learning with error bars
    plt.errorbar(r_values, double_q_means, yerr=double_q_stds,
                fmt='s-', color='#ff7f0e', capsize=5,
                label='Double Q-Learning', linewidth=2, markersize=8)
    
    # Add benchmark lines
    plt.axhline(y=25, color='gray', linestyle='--', label='25% Benchmark', alpha=0.5)
    plt.axhline(y=50, color='gray', linestyle='-.', label='50% Benchmark', alpha=0.5)
    
    plt.xlabel('Multiplication Factor (r)', fontsize=12)
    plt.ylabel('Average Contribution (% of endowment)', fontsize=12)
    plt.title('Average Contribution vs. Multiplication Factor', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc='upper left')
    
    # Set axis limits
    plt.xlim(min(r_values) - 0.1, max(r_values) + 0.1)
    plt.ylim(0, 100)  # Percentage scale
    plt.xticks(r_values)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'contribution_vs_r.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def save_consolidated_results(
    metrics_by_r: Dict[float, Tuple[pd.DataFrame, pd.DataFrame]], 
    test_results_by_r: Dict[float, Dict], 
    output_dir: str
) -> None:
    """Save consolidated contribution statistics and test results across all r values."""
    
    # Prepare contribution statistics data
    contribution_stats = []
    for r, (q_metrics, dq_metrics) in metrics_by_r.items():
        # Calculate stats for last 1000 episodes
        q_stats = {
            'r': r,
            'Algorithm': 'Q-Learning',
            'Mean Contribution (%)': q_metrics['avg_contribution'].iloc[-1000:].mean() / 200 * 100,
            'Std Dev (%)': q_metrics['avg_contribution'].iloc[-1000:].std() / 200 * 100
        }
        
        dq_stats = {
            'r': r,
            'Algorithm': 'Double Q-Learning',
            'Mean Contribution (%)': dq_metrics['avg_contribution'].iloc[-1000:].mean() / 200 * 100,
            'Std Dev (%)': dq_metrics['avg_contribution'].iloc[-1000:].std() / 200 * 100
        }
        
        contribution_stats.extend([q_stats, dq_stats])
    
    # Save contribution statistics
    pd.DataFrame(contribution_stats).to_csv(
        os.path.join(output_dir, 'contribution_statistics_all_r.csv'),
        index=False
    )
    
    # Prepare statistical test results
    test_stats = []
    for r, results in test_results_by_r.items():
        for metric, stats in results.items():
            test_stats.append({
                'r': r,
                'Metric': stats['metric'],
                'T-statistic': stats['t_statistic'],
                'P-value': stats['p_value'],
                'Significant': stats['significant']
            })
    
    # Save statistical test results
    pd.DataFrame(test_stats).to_csv(
        os.path.join(output_dir, 'statistical_tests_all_r.csv'),
        index=False
    )

def main() -> None:
    """Modified main function to save consolidated results."""
    base_dir = "results"
    output_dir = "results_analysis"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    metrics_by_r = {}
    test_results_by_r = {}
    r_dirs = [d for d in os.listdir(base_dir) if d.startswith("r")]
    multiplication_factors = sorted([float(d.replace("r", "")) for d in r_dirs])
    seeds = [42, 123, 456]
    
    print(f"Found data for r values: {multiplication_factors}")
    
    for r in multiplication_factors:
        print(f"\nProcessing r = {r}")
        r_output_dir = os.path.join(output_dir, f"r{r}")
        Path(r_output_dir).mkdir(parents=True, exist_ok=True)
        
        # Aggregate data
        q_metrics = aggregate_seed_data(base_dir, r, seeds, False)
        double_q_metrics = aggregate_seed_data(base_dir, r, seeds, True)
        
        if not q_metrics.empty and not double_q_metrics.empty:
            metrics_by_r[r] = (q_metrics, double_q_metrics)
            
            # Generate plots and get test results
            plot_required_metrics(q_metrics, double_q_metrics, r_output_dir, r)
            test_results = conduct_statistical_tests(q_metrics, double_q_metrics, r_output_dir, r)
            test_results_by_r[r] = test_results
            
            print(f"Statistical Test Results (r={r}):")
            for metric, results in test_results.items():
                sig = "significant" if results['significant'] else "not significant"
                print(f"{results['metric']}: p={results['p_value']:.4f} ({sig})")
    
    # Generate summary plot
    print("\nGenerating contribution summary plot...")
    q_metrics_by_r = {r: metrics[0] for r, metrics in metrics_by_r.items()}
    double_q_metrics_by_r = {r: metrics[1] for r, metrics in metrics_by_r.items()}
    plot_summary(q_metrics_by_r, double_q_metrics_by_r, output_dir)
    
    # Save consolidated results
    print("\nSaving consolidated results...")
    save_consolidated_results(metrics_by_r, test_results_by_r, output_dir)

if __name__ == "__main__":
    main()
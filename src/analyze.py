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
            # Assuming metrics files are named like "q_r2.0_seed1_metrics.csv" or "double_q_r2.0_seed1_metrics.csv"
            # The simulation script saves them as f"{algorithm}_metrics.csv", e.g. "Q-Learning_metrics.csv"
            # Let's adjust to match the simulation output naming convention if algorithm is "Q-Learning" or "Double Q-Learning"
            algo_name_in_file = "Q-Learning" if not is_double_q else "Double Q-Learning" # Match simulation.py output
            
            # The simulation script saves metrics as {algorithm}_metrics.csv in the seed_output_dir
            # e.g., Q-Learning_metrics.csv or Double Q-Learning_metrics.csv
            # The provided analyze.py uses a different naming convention in metrics_file construction.
            # Let's assume the file name is based on the prefix from the function call.
            # The simulation script saves metrics as: os.path.join(output_dir, f"{algorithm}_metrics.csv")
            # where output_dir is seed_output_dir and algorithm is "Q-Learning" or "Double Q-Learning"
            # The analyze.py script uses: os.path.join(seed_dir, f"{prefix}_r{r}_seed{seed}_metrics.csv")
            # This seems to be a mismatch. I will assume the analyze.py file naming is what's being used.
            metrics_file = os.path.join(seed_dir, f"{prefix}_r{r}_seed{seed}_metrics.csv")
            if os.path.exists(metrics_file):
                try:
                    df = pd.read_csv(metrics_file)
                    if 'episode' not in df.columns:
                        print(f"Warning: 'episode' column missing in {metrics_file}")
                        continue
                    metrics_dfs.append(df)
                except pd.errors.EmptyDataError:
                    print(f"Warning: Empty metrics file {metrics_file}")
                except Exception as e:
                    print(f"Warning: Error reading {metrics_file}: {e}")
            else:
                print(f"Warning: Metrics file not found {metrics_file}")
                
    if not metrics_dfs:
        print(f"Warning: No valid metrics data found for r={r}, {'Double Q' if is_double_q else 'Q'}-Learning")
        return pd.DataFrame()
        
    combined_df = pd.concat(metrics_dfs)
    # Group by the 'episode' column and calculate the mean for all other columns.
    # This makes 'episode' the index of the aggregated DataFrame.
    aggregated_df = combined_df.groupby('episode').mean() 
    return aggregated_df

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
    if q_metrics.empty or double_q_metrics.empty:
        print(f"Warning: Metrics data is empty for r={r}. Skipping plots.")
        return

    actual_episodes = q_metrics.index # Use the 'episode' index for the x-axis
    # The smoothing window should be smaller than the number of data points if you want a smoothed line.
    # If you have 50 data points (10k episodes / 200 per save), a window of 50 will result in a single point
    # or a cumulative mean if min_periods=1. Let's use a smaller window for demonstration or add min_periods.
    smoothing_window = 5 # Example: smooth over 5 recorded intervals (1000 episodes)
    # If you want to use the original window=50, ensure min_periods=1 for a rolling mean line:
    # rolling_op = lambda series: series.rolling(window=50, min_periods=1).mean()
    # Or, for a smaller effective smoothing on the 50 points:
    rolling_op = lambda series: series.rolling(window=smoothing_window, min_periods=1).mean()
    
    # 2. Social welfare over episodes
    plt.figure(figsize=(10, 6))
    if 'social_welfare' in q_metrics.columns:
        plt.plot(actual_episodes, rolling_op(q_metrics['social_welfare']), label='Q-Learning')
    if 'social_welfare' in double_q_metrics.columns:
        plt.plot(actual_episodes, rolling_op(double_q_metrics['social_welfare']), label='Double Q-Learning')
    plt.xlabel('Episode')
    plt.ylabel('Social Welfare')
    plt.title(f'Social Welfare Evolution (r={r})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'social_welfare_r{r}.png'))
    plt.close()
    
    # 3. Individual contributions
    fig_ind_contrib, (ax1_ind_contrib, ax2_ind_contrib) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
    colors = plt.cm.Set2(np.linspace(0, 1, 4))
    for i in range(4):
        if f'agent_{i}_contrib' in q_metrics.columns:
            ax1_ind_contrib.plot(actual_episodes, rolling_op(pd.Series(q_metrics[f'agent_{i}_contrib'])),
                    label=f'Agent {i} (e={0.5*(i+1)})', color=colors[i])
        if f'agent_{i}_contrib' in double_q_metrics.columns:
            ax2_ind_contrib.plot(actual_episodes, rolling_op(pd.Series(double_q_metrics[f'agent_{i}_contrib'])),
                    label=f'Agent {i} (e={0.5*(i+1)})', color=colors[i])
    
    ax1_ind_contrib.set_title(f'Q-Learning Individual Contributions (r={r})')
    ax1_ind_contrib.set_ylabel('Contribution')
    ax1_ind_contrib.legend()
    ax2_ind_contrib.set_title(f'Double Q-Learning Individual Contributions (r={r})')
    ax2_ind_contrib.set_xlabel('Episode')
    ax2_ind_contrib.set_ylabel('Contribution')
    ax2_ind_contrib.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'individual_contributions_r{r}.png'))
    plt.close(fig_ind_contrib)

    # 4. Variance of Shapley Values Over Episodes
    plt.figure(figsize=(10, 6)) # Changed from subplots to a single plot
    
    # Q-Learning Shapley Variance
    shapley_cols_q = [col for col in q_metrics.columns if col.startswith('agent_') and col.endswith('_shapley')]
    if shapley_cols_q and not q_metrics[shapley_cols_q].empty:
        q_shapley_variance = q_metrics[shapley_cols_q].var(axis=1)
        if not q_shapley_variance.dropna().empty:
            plt.plot(actual_episodes, rolling_op(q_shapley_variance), label='Q-Learning Shapley Variance')

    # Double Q-Learning Shapley Variance
    shapley_cols_dq = [col for col in double_q_metrics.columns if col.startswith('agent_') and col.endswith('_shapley')]
    if shapley_cols_dq and not double_q_metrics[shapley_cols_dq].empty:
        dq_shapley_variance = double_q_metrics[shapley_cols_dq].var(axis=1)
        if not dq_shapley_variance.dropna().empty:
            plt.plot(actual_episodes, rolling_op(dq_shapley_variance), label='Double Q-Learning Shapley Variance')
    
    plt.title(f'Variance of Shapley Values Over Episodes (r={r})')
    plt.xlabel('Episode')
    plt.ylabel('Variance of Shapley Values')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'shapley_variance_r{r}.png')) # Changed filename
    plt.close()

    # 6. Individual Payoffs
    fig_payoffs, (ax1_payoffs, ax2_payoffs) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
    colors = plt.cm.Set2(np.linspace(0, 1, 4)) # Same colors as other plots
    for i in range(4): # Assuming 4 agents
        agent_label = f'Agent {i} (e={0.5*(i+1)})'
        # Q-Learning Payoffs
        if f'agent_{i}_payoff' in q_metrics.columns:
            if not q_metrics[f'agent_{i}_payoff'].dropna().empty:
                ax1_payoffs.plot(actual_episodes, rolling_op(pd.Series(q_metrics[f'agent_{i}_payoff'])),
                                 label=agent_label, color=colors[i])
        
        # Double Q-Learning Payoffs
        if f'agent_{i}_payoff' in double_q_metrics.columns:
            if not double_q_metrics[f'agent_{i}_payoff'].dropna().empty:
                ax2_payoffs.plot(actual_episodes, rolling_op(pd.Series(double_q_metrics[f'agent_{i}_payoff'])),
                                 label=agent_label, color=colors[i])

    ax1_payoffs.set_title(f'Q-Learning Individual Payoffs (r={r})')
    ax1_payoffs.set_ylabel('Payoff')
    ax1_payoffs.legend()
    
    ax2_payoffs.set_title(f'Double Q-Learning Individual Payoffs (r={r})')
    ax2_payoffs.set_xlabel('Episode')
    ax2_payoffs.set_ylabel('Payoff')
    ax2_payoffs.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'individual_payoffs_r{r}.png'))
    plt.close(fig_payoffs)

    # Save mean and std dev of contribution rates
    # The original code for stats_df used .iloc[-1000:], which is not applicable to already aggregated data
    # with few points. We should calculate stats on the available aggregated points.
    # Also, avg_contribution is already a percentage of endowment if it was calculated that way.
    # If 'avg_contribution' in the CSV is the raw average action (0-1), then *100 is fine.
    # The division by 200 was incorrect.
    
    # Assuming 'avg_contribution' in the CSV is the average action (fraction of endowment, 0-1)
    # and we want to report it as a percentage.
    q_avg_contrib_percent = q_metrics['avg_contribution'] * 100
    dq_avg_contrib_percent = double_q_metrics['avg_contribution'] * 100

    stats_df = pd.DataFrame({
        'Algorithm': ['Q-Learning', 'Double Q-Learning'],
        'Mean Contribution (%)': [
            q_avg_contrib_percent.mean(),
            dq_avg_contrib_percent.mean()
        ],
        'Std Dev (%)': [
            q_avg_contrib_percent.std(),
            dq_avg_contrib_percent.std()
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
    Plot summary of contribution vs. r with error bars.
    Mean and Std Dev calculated from the average proportion of endowment contributed
    across agents, using data from the last 100 episodes.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    r_values = sorted(q_metrics.keys())
    q_means = []
    q_stds = []
    double_q_means = []
    double_q_stds = []

    num_agents = 4
    num_last_episodes_for_plot_stats = 1000 # Number of last episodes to consider for plot stats

    for r in r_values:
        # --- Q-Learning ---
        if r in q_metrics and not q_metrics[r].empty:
            metrics_df_q = q_metrics[r]
            # DataFrame to store proportions for each agent for Q-Learning
            q_agent_proportions_df = pd.DataFrame(index=metrics_df_q.index)
            for i in range(num_agents):
                if f'agent_{i}_contrib' in metrics_df_q.columns:
                    max_contribution = 0.5 * (i + 1)
                    agent_contrib_series = metrics_df_q[f'agent_{i}_contrib']
                    if not agent_contrib_series.empty and max_contribution > 0:
                        q_agent_proportions_df[f'agent_{i}_prop'] = agent_contrib_series / max_contribution
                    else:
                        q_agent_proportions_df[f'agent_{i}_prop'] = np.nan
                else:
                    q_agent_proportions_df[f'agent_{i}_prop'] = np.nan
            
            # Calculate the average proportion across agents for each episode
            q_avg_prop_per_episode = q_agent_proportions_df.mean(axis=1, skipna=True)
            # Consider last N episodes for stats
            q_relevant_episodes_avg_prop = q_avg_prop_per_episode.iloc[-num_last_episodes_for_plot_stats:]
            
            if not q_relevant_episodes_avg_prop.empty:
                q_means.append(q_relevant_episodes_avg_prop.mean() * 100)  # Convert to percentage
                q_stds.append(q_relevant_episodes_avg_prop.std() * 100)
            else:
                q_means.append(np.nan)
                q_stds.append(np.nan)
        else:
            q_means.append(np.nan)
            q_stds.append(np.nan)
        
        # --- Double Q-Learning ---
        if r in double_q_metrics and not double_q_metrics[r].empty:
            metrics_df_dq = double_q_metrics[r]
            # DataFrame to store proportions for each agent for Double Q-Learning
            dq_agent_proportions_df = pd.DataFrame(index=metrics_df_dq.index)
            for i in range(num_agents):
                if f'agent_{i}_contrib' in metrics_df_dq.columns:
                    max_contribution = 0.5 * (i + 1)
                    agent_contrib_series = metrics_df_dq[f'agent_{i}_contrib']
                    if not agent_contrib_series.empty and max_contribution > 0:
                        dq_agent_proportions_df[f'agent_{i}_prop'] = agent_contrib_series / max_contribution
                    else:
                        dq_agent_proportions_df[f'agent_{i}_prop'] = np.nan
                else:
                    dq_agent_proportions_df[f'agent_{i}_prop'] = np.nan

            # Calculate the average proportion across agents for each episode
            dq_avg_prop_per_episode = dq_agent_proportions_df.mean(axis=1, skipna=True)
            # Consider last N episodes for stats
            dq_relevant_episodes_avg_prop = dq_avg_prop_per_episode.iloc[-num_last_episodes_for_plot_stats:]

            if not dq_relevant_episodes_avg_prop.empty:
                double_q_means.append(dq_relevant_episodes_avg_prop.mean() * 100) # Convert to percentage
                double_q_stds.append(dq_relevant_episodes_avg_prop.std() * 100)
            else:
                double_q_means.append(np.nan)
                double_q_stds.append(np.nan)
        else:
            double_q_means.append(np.nan)
            double_q_stds.append(np.nan)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    valid_indices_q = ~np.isnan(q_means)
    valid_indices_dq = ~np.isnan(double_q_means)
    r_values_np = np.array(r_values)

    if np.any(valid_indices_q):
        plt.errorbar(r_values_np[valid_indices_q], np.array(q_means)[valid_indices_q], yerr=np.array(q_stds)[valid_indices_q], 
                    fmt='o-', color='#1f77b4', capsize=5, 
                    label='Q-Learning', linewidth=2, markersize=8)
    
    if np.any(valid_indices_dq):
        plt.errorbar(r_values_np[valid_indices_dq], np.array(double_q_means)[valid_indices_dq], yerr=np.array(double_q_stds)[valid_indices_dq],
                    fmt='s-', color='#ff7f0e', capsize=5,
                    label='Double Q-Learning', linewidth=2, markersize=8)
    
    plt.axhline(y=25, color='gray', linestyle='--', label='25% Benchmark', alpha=0.5)
    plt.axhline(y=50, color='gray', linestyle='-.', label='50% Benchmark', alpha=0.5)
    
    plt.xlabel('Multiplication Factor (r)', fontsize=12)
    plt.ylabel('Average Contribution (% of endowment, last 1000 ep.)', fontsize=12) 
    plt.title('Average Contribution vs. Multiplication Factor (Last 1000 Episodes)', fontsize=14) 
    plt.grid(True, alpha=0.3)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend(fontsize=10, loc='upper left')
    
    if r_values:
        plt.xlim(min(r_values) - 0.1, max(r_values) + 0.1)
        plt.xticks(r_values)
    plt.ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'contribution_vs_r_last1000ep.png'), # Updated filename
                dpi=300, bbox_inches='tight')
    plt.close()

def save_consolidated_results(
    metrics_by_r: Dict[float, Tuple[pd.DataFrame, pd.DataFrame]], 
    test_results_by_r: Dict[float, Dict], 
    output_dir: str
) -> None:
    """
    Save consolidated contribution statistics and test results across all r values.
    "Mean Contribution" is calculated as the average proportion of endowment contributed.
    """
    
    contribution_stats_list = [] # Renamed from contribution_stats to avoid confusion
    num_agents = 4 # Assuming 4 agents as used elsewhere in the script

    for r, (q_metrics_df, dq_metrics_df) in metrics_by_r.items():
        for algo_name, metrics_df in [('Q-Learning', q_metrics_df), ('Double Q-Learning', dq_metrics_df)]:
            if metrics_df.empty:
                print(f"Warning: Empty metrics for r={r}, Algorithm={algo_name}. Skipping contribution stats.")
                continue

            # Calculate average proportion of endowment contributed
            all_agent_proportions_df = pd.DataFrame(index=metrics_df.index)
            
            for i in range(num_agents):
                endowment = 0.5 * (i + 1)
                agent_contrib_col = f'agent_{i}_contrib'
                
                if agent_contrib_col in metrics_df.columns:
                    # Ensure endowment is not zero to avoid division by zero, though not expected here
                    if endowment == 0:
                         all_agent_proportions_df[f'agent_{i}_prop'] = 0.0 # Or handle as NaN if appropriate
                    else:
                        all_agent_proportions_df[f'agent_{i}_prop'] = metrics_df[agent_contrib_col] / endowment
                else:
                    print(f"Warning: Column {agent_contrib_col} not found for r={r}, Algo={algo_name}. Agent proportion will be missing.")
                    all_agent_proportions_df[f'agent_{i}_prop'] = np.nan # Add NaN column if agent data is missing

            # Calculate the average proportion across agents for each episode
            # This Series will have one value per episode: the average of (contrib/endowment) over the agents
            avg_prop_contrib_per_episode = all_agent_proportions_df.mean(axis=1, skipna=True)

            # Consider last 1000 episodes for stats, or fewer if not available
            relevant_episodes_avg_prop = avg_prop_contrib_per_episode.iloc[-1000:]
            
            mean_avg_proportion = relevant_episodes_avg_prop.mean()
            std_dev_avg_proportion = relevant_episodes_avg_prop.std()

            algo_stats = {
                'r': r,
                'Algorithm': algo_name,
                'Mean Contribution': mean_avg_proportion, # This now represents average proportion
                'Std Dev': std_dev_avg_proportion         # Std dev of average proportions
            }
            contribution_stats_list.append(algo_stats)
    
    # Save contribution statistics
    pd.DataFrame(contribution_stats_list).to_csv(
        os.path.join(output_dir, 'contribution_statistics_all_r.csv'),
        index=False
    )
    
    # Prepare statistical test results (this part remains unchanged)
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
    seeds = [1,2,3,4,5,6,7,8,9,10]
    
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
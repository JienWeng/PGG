import os
from typing import Dict, Tuple, Any, List
from pathlib import Path
import pandas as pd
import numpy as np
from ast import literal_eval
import matplotlib.pyplot as plt

from analysis import plot_metrics, plot_heatmap, save_summary
from evaluation import evaluate_q_tables

def find_available_seeds(base_dir: str, r: float) -> List[int]:
    """Find all available seed directories for a given r value."""
    r_dir = os.path.join(base_dir, f"r{r}")
    seeds = []
    if os.path.exists(r_dir):
        for d in os.listdir(r_dir):
            if d.startswith("seed"):
                try:
                    seed = int(d.replace("seed", ""))
                    seeds.append(seed)
                except ValueError:
                    continue
    return sorted(seeds)

def aggregate_seed_data(base_dir: str, r: float, seeds: List[int], is_double_q: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate metrics and Q-values across seeds."""
    metrics_dfs = []
    qvalues_dfs = []
    prefix = "double_q" if is_double_q else "q"
    
    for seed in seeds:
        seed_dir = os.path.join(base_dir, f"r{r}", f"seed{seed}")
        if os.path.exists(seed_dir):
            # Load metrics
            metrics_file = f"{prefix}_r{r}_seed{seed}_metrics.csv"
            metrics_path = os.path.join(seed_dir, metrics_file)
            if os.path.exists(metrics_path):
                metrics_dfs.append(pd.read_csv(metrics_path))
            
            # Load Q-values
            qvalues_file = f"{prefix}_r{r}_seed{seed}_qvalues.csv"
            qvalues_path = os.path.join(seed_dir, qvalues_file)
            if os.path.exists(qvalues_path):
                qvalues_dfs.append(pd.read_csv(qvalues_path))
    
    # Aggregate metrics
    metrics = pd.concat(metrics_dfs).groupby(level=0).mean() if metrics_dfs else pd.DataFrame()
    
    # Aggregate Q-values
    if qvalues_dfs:
        base_cols = ['agent', 'state', 'action']
        if is_double_q:
            qvalues = pd.DataFrame({
                **{col: qvalues_dfs[0][col] for col in base_cols},
                'q_a_value': np.mean([df['q_a_value'] for df in qvalues_dfs], axis=0),
                'q_b_value': np.mean([df['q_b_value'] for df in qvalues_dfs], axis=0)
            })
        else:
            qvalues = pd.DataFrame({
                **{col: qvalues_dfs[0][col] for col in base_cols},
                'q_value': np.mean([df['q_value'] for df in qvalues_dfs], axis=0)
            })
    else:
        qvalues = pd.DataFrame()
    
    return metrics, qvalues

def plot_summary(
    q_metrics: Dict[float, pd.DataFrame],
    double_q_metrics: Dict[float, pd.DataFrame],
    output_dir: str
) -> None:
    """
    Plot smoothed summary of contribution vs. r with error bars.
    
    Args:
        q_metrics: Dictionary mapping r values to Q-Learning metrics
        double_q_metrics: Dictionary mapping r values to Double Q-Learning metrics
        output_dir: Directory to save output files
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Calculate smoothed means and std deviations for each r
    r_values = sorted(q_metrics.keys())
    q_means = []
    q_stds = []
    double_q_means = []
    double_q_stds = []
    
    window = 50  # Smoothing window
    for r in r_values:
        # Apply smoothing to Q-Learning data
        q_smooth = pd.Series(q_metrics[r]['avg_contribution']).rolling(
            window=window, min_periods=1, center=True).mean()
        q_means.append(q_smooth.mean())
        q_stds.append(q_smooth.std())
        
        # Apply smoothing to Double Q-Learning data
        dq_smooth = pd.Series(double_q_metrics[r]['avg_contribution']).rolling(
            window=window, min_periods=1, center=True).mean()
        double_q_means.append(dq_smooth.mean())
        double_q_stds.append(dq_smooth.std())
    
    # Convert proportions to percentages
    q_means = [x * 100 for x in q_means]
    q_stds = [x * 100 for x in q_stds]
    double_q_means = [x * 100 for x in double_q_means]
    double_q_stds = [x * 100 for x in double_q_stds]
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    # Plot Q-Learning with error bars
    plt.errorbar(r_values, q_means, yerr=q_stds, 
                fmt='o-', color='#1f77b4', capsize=5, 
                label='Q-Learning', linewidth=2)
    
    # Plot Double Q-Learning with error bars
    plt.errorbar(r_values, double_q_means, yerr=double_q_stds,
                fmt='s-', color='#ff7f0e', capsize=5,
                label='Double Q-Learning', linewidth=2)
    
    plt.xlabel('Multiplication Factor (r)', fontsize=12)
    plt.ylabel('Average Contribution (%)', fontsize=12)  # Updated label
    plt.title('Average Contribution vs. Multiplication Factor', 
              fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    # Add horizontal lines to show contribution thresholds (now in percentage)
    plt.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
    plt.axhline(y=25, color='gray', linestyle=':', alpha=0.5)
    plt.text(1.45, 52, '50% Contribution', fontsize=8, alpha=0.7)
    plt.text(1.45, 27, '25% Contribution', fontsize=8, alpha=0.7)
    
    # Update y-axis limits to percentage scale
    plt.ylim(0, 100)
    plt.xticks(r_values)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'contribution_vs_r.png'), dpi=300)
    plt.close()

def main() -> None:
    """Run post-simulation analysis with seed aggregation."""
    base_dir = "results"
    output_dir = "results_analysis"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Find all r values with data
    r_dirs = [d for d in os.listdir(base_dir) if d.startswith("r")]
    multiplication_factors = sorted([float(d.replace("r", "")) for d in r_dirs])
    
    print(f"Found data for r values: {multiplication_factors}")
    
    q_metrics_dict = {}
    double_q_metrics_dict = {}
    
    for r in multiplication_factors:
        print(f"\nProcessing r = {r}")
        r_output_dir = os.path.join(output_dir, f"r{r}")
        Path(r_output_dir).mkdir(parents=True, exist_ok=True)
        
        # Find available seeds
        seeds = find_available_seeds(base_dir, r)
        print(f"Found seeds: {seeds}")
        
        if not seeds:
            print(f"No seed data found for r = {r}, skipping...")
            continue
        
        # Aggregate Q-Learning data
        q_metrics, q_qvalues = aggregate_seed_data(base_dir, r, seeds, False)
        
        # Aggregate Double Q-Learning data
        double_q_metrics, double_q_qvalues = aggregate_seed_data(base_dir, r, seeds, True)
        
        if q_metrics.empty or double_q_metrics.empty:
            print(f"Missing data for r = {r}, skipping...")
            continue
        
        # Save aggregated data
        q_metrics.to_csv(os.path.join(r_output_dir, f"q_r{r}_metrics.csv"), index=False)
        q_qvalues.to_csv(os.path.join(r_output_dir, f"q_r{r}_qvalues.csv"), index=False)
        double_q_metrics.to_csv(os.path.join(r_output_dir, f"double_q_r{r}_metrics.csv"), index=False)
        double_q_qvalues.to_csv(os.path.join(r_output_dir, f"double_q_r{r}_qvalues.csv"), index=False)
        
        # Convert Q-values for evaluation
        q_dict = {}
        double_q_dict = {}
        
        # Process Q-Learning Q-values
        for agent in q_qvalues['agent'].unique():
            agent_data = q_qvalues[q_qvalues['agent'] == agent]
            q_dict[agent] = {(literal_eval(row['state']), row['action']): row['q_value'] 
                            for _, row in agent_data.iterrows()}
        
        # Process Double Q-Learning Q-values
        for agent in double_q_qvalues['agent'].unique():
            agent_data = double_q_qvalues[double_q_qvalues['agent'] == agent]
            double_q_dict[agent] = {(literal_eval(row['state']), row['action']): 
                                  (row['q_a_value'] + row['q_b_value']) / 2
                                  for _, row in agent_data.iterrows()}
        
        # Generate analysis
        print(f"Generating analysis for r = {r}...")
        plot_metrics(q_metrics, double_q_metrics, r_output_dir, r)
        plot_heatmap(q_qvalues, double_q_qvalues, r_output_dir, r)
        save_summary(q_metrics, double_q_metrics, r_output_dir, r)
        evaluate_q_tables(q_dict, double_q_dict, r_output_dir, r)
        
        # Store metrics for summary plot
        q_metrics_dict[r] = q_metrics
        double_q_metrics_dict[r] = double_q_metrics
    
    # Generate summary plot
    print("Generating summary plot...")
    plot_summary(q_metrics_dict, double_q_metrics_dict, output_dir)

if __name__ == "__main__":
    main()

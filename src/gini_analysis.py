import os
from typing import Dict, List
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_gini_data(base_dir: str, r_values: List[float]) -> Dict[str, Dict[float, pd.Series]]:
    """Load and aggregate Gini coefficient data across seeds."""
    data = {'q': {}, 'double_q': {}}
    
    for r in r_values:
        r_dir = os.path.join(base_dir, f"r{r}")
        if not os.path.exists(r_dir):
            continue
            
        # Find all seed directories
        seed_dirs = [d for d in os.listdir(r_dir) if d.startswith('seed')]
        if not seed_dirs:
            continue
            
        q_dfs = []
        dq_dfs = []
        
        # Load data from each seed
        for seed_dir in seed_dirs:
            seed_path = os.path.join(r_dir, seed_dir)
            
            # Load Q-Learning data
            q_file = f"q_r{r}_seed{seed_dir.replace('seed', '')}_metrics.csv"
            q_path = os.path.join(seed_path, q_file)
            if os.path.exists(q_path):
                q_dfs.append(pd.read_csv(q_path)['gini_coefficient'])
                
            # Load Double Q-Learning data
            dq_file = f"double_q_r{r}_seed{seed_dir.replace('seed', '')}_metrics.csv"
            dq_path = os.path.join(seed_path, dq_file)
            if os.path.exists(dq_path):
                dq_dfs.append(pd.read_csv(dq_path)['gini_coefficient'])
        
        # Average across seeds
        if q_dfs:
            data['q'][r] = pd.concat(q_dfs, axis=1).mean(axis=1)
        if dq_dfs:
            data['double_q'][r] = pd.concat(dq_dfs, axis=1).mean(axis=1)
    
    return data

def plot_gini_episodes(data: Dict[str, Dict[float, pd.Series]], output_dir: str):
    """Create separate plots of Gini coefficients over episodes for each algorithm."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Specific r values to plot
    target_r_values = [2.0, 3.0, 4.0]
    available_r = [r for r in sorted(data['q'].keys()) if r in target_r_values]
    colors = plt.cm.viridis(np.linspace(0, 1, len(target_r_values)))
    window = 100  # Smoothing window
    
    # Plot settings dictionary
    plot_settings = {
        'q': {
            'title': 'Q-Learning Gini Coefficient Evolution',
            'filename': 'q_gini_episodes.png'
        },
        'double_q': {
            'title': 'Double Q-Learning Gini Coefficient Evolution',
            'filename': 'double_q_gini_episodes.png'
        }
    }
    
    # Create separate plots for each algorithm
    for algo in ['q', 'double_q']:
        plt.figure(figsize=(10, 6))
        
        # Plot each r value
        for r, color in zip(available_r, colors):
            if r in data[algo]:
                # Apply smoothing
                smoothed = data[algo][r].rolling(
                    window=window, min_periods=1, center=True).mean()
                plt.plot(smoothed, color=color, label=f'r={r}', linewidth=2)
        
        plt.title(plot_settings[algo]['title'], fontsize=12, pad=20)
        plt.xlabel('Episode', fontsize=10)
        plt.ylabel('Gini Coefficient', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        
        # Add reference lines with improved positioning
        plt.axhline(y=0.4, color='gray', linestyle=':', alpha=0.5)
        plt.axhline(y=0.2, color='gray', linestyle=':', alpha=0.5)
        plt.text(500, 0.41, 'High Inequality', fontsize=8, alpha=0.7)
        plt.text(500, 0.21, 'Low Inequality', fontsize=8, alpha=0.7)
        plt.ylim(0, 0.6)
        
        # Save plot with adjusted layout
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, plot_settings[algo]['filename']),
            dpi=300,
            bbox_inches='tight',
            pad_inches=0.2
        )
        plt.close()

def main():
    """Run Gini coefficient analysis."""
    base_dir = "results"
    output_dir = "results_analysis"
    
    # Get available r values
    r_dirs = [d for d in os.listdir(base_dir) if d.startswith("r")]
    r_values = sorted([float(d.replace("r", "")) for d in r_dirs])
    
    print(f"Analyzing Gini coefficients for r values: {r_values}")
    
    # Load and process data
    data = load_gini_data(base_dir, r_values)
    
    # Generate plots
    plot_gini_episodes(data, output_dir)
    print("Gini coefficient analysis complete.")

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from typing import List, Dict, Optional, IO

def get_mean_metric_per_seed(
    base_dir: str,
    r_value: float,
    seeds: List[int],
    is_double_q: bool,
    metric_name: str, # e.g., "avg_contribution", "social_welfare", or "std_dev_contrib"
    num_last_points_for_stat: int = 50
) -> Dict[int, float]:
    """
    Loads metrics for each seed, extracts/calculates the specified metric,
    and returns the mean of this metric over the last N points for each seed.
    """
    seed_mean_metrics: Dict[int, float] = {}
    algo_prefix = "double_q" if is_double_q else "q"

    for seed in seeds:
        seed_dir_path = Path(base_dir) / f"r{r_value}" / f"seed{seed}"
        metrics_file = seed_dir_path / f"{algo_prefix}_r{r_value}_seed{seed}_metrics.csv"

        if metrics_file.exists():
            try:
                df = pd.read_csv(metrics_file)
                if 'episode' not in df.columns:
                    print(f"Warning: 'episode' column missing in {metrics_file}")
                    continue
                if len(df) == 0:
                    print(f"Warning: Empty DataFrame after loading {metrics_file}")
                    continue

                target_metric_series = None
                current_metric_col_name = metric_name

                if metric_name == "std_dev_contrib":
                    contrib_cols = [col for col in df.columns if col.startswith('agent_') and col.endswith('_contrib')]
                    if not contrib_cols or len(contrib_cols) < 2:
                        print(f"Warning: Not enough agent contribution columns to calculate std_dev_contrib in {metrics_file}")
                        continue
                    df['calculated_std_dev_contrib'] = df[contrib_cols].std(axis=1, skipna=True)
                    current_metric_col_name = 'calculated_std_dev_contrib'
                
                if current_metric_col_name not in df.columns:
                    print(f"Warning: Metric column '{current_metric_col_name}' not found in {metrics_file}")
                    continue
                
                target_metric_series = df[current_metric_col_name].copy() # Use .copy() to avoid SettingWithCopyWarning
                target_metric_series.dropna(inplace=True)

                if target_metric_series.empty:
                    print(f"Warning: Target metric series for '{current_metric_col_name}' is empty or all NaN in {metrics_file}")
                    continue
                
                if len(target_metric_series) >= num_last_points_for_stat:
                    mean_val_for_seed = target_metric_series.iloc[-num_last_points_for_stat:].mean()
                elif len(target_metric_series) > 0:
                    mean_val_for_seed = target_metric_series.mean()
                else: # Should be caught by target_metric_series.empty already
                    print(f"Warning: No data points available for metric '{current_metric_col_name}' in {metrics_file} after processing.")
                    continue
                
                if pd.notna(mean_val_for_seed):
                    seed_mean_metrics[seed] = mean_val_for_seed
                else:
                    print(f"Warning: NaN result for mean of '{current_metric_col_name}' for {metrics_file}")

            except pd.errors.EmptyDataError:
                print(f"Warning: Empty metrics file {metrics_file}")
            except Exception as e:
                print(f"Warning: Error reading or processing {metrics_file}: {e}")
        else:
            print(f"Warning: Metrics file not found {metrics_file}")
            
    return seed_mean_metrics

def perform_paired_t_test(
    data1_dict: Dict[int, float],
    data2_dict: Dict[int, float],
    label1: str,
    label2: str,
    hypothesis_description: str,
    file_handler: Optional[IO] = None
):
    """
    Performs a paired t-test on data from common seeds and prints/writes results.
    (Copied and adapted from statistical_testing_exp3.py)
    """
    output_lines = []
    output_lines.append(f"\n--- {hypothesis_description} ---")
    
    common_seeds = sorted(list(set(data1_dict.keys()) & set(data2_dict.keys())))
    
    valid_common_seeds = [
        s for s in common_seeds 
        if s in data1_dict and pd.notna(data1_dict[s]) and s in data2_dict and pd.notna(data2_dict[s])
    ]

    if len(valid_common_seeds) < 2:
        output_lines.append(f"Not enough valid (non-NaN) paired samples for t-test (found {len(valid_common_seeds)}). Required at least 2.")
        output_lines.append(f"Data for {label1}: {len(data1_dict)} seeds (raw). Data for {label2}: {len(data2_dict)} seeds (raw).")
        if file_handler:
            for line in output_lines: file_handler.write(line + "\n")
        else:
            for line in output_lines: print(line)
        return

    paired_values1_final = [data1_dict[s] for s in valid_common_seeds]
    paired_values2_final = [data2_dict[s] for s in valid_common_seeds]

    mean1 = np.mean(paired_values1_final)
    mean2 = np.mean(paired_values2_final)
    
    t_stat, p_value = stats.ttest_rel(paired_values1_final, paired_values2_final, alternative='two-sided')
    
    output_lines.append(f"Comparing: {label1} vs. {label2}")
    output_lines.append(f"Number of paired samples (seeds) used in test: {len(valid_common_seeds)}")
    output_lines.append(f"Mean for {label1}: {mean1:.6f}")
    output_lines.append(f"Mean for {label2}: {mean2:.6f}")
    output_lines.append(f"Paired t-statistic: {t_stat:.4f}")
    output_lines.append(f"P-value (two-tailed): {p_value:.4f}")
    
    alpha = 0.05
    if p_value < alpha:
        output_lines.append("Result: Significant difference detected.")
        if mean1 > mean2:
            output_lines.append(f"Direction: {label1} has a significantly HIGHER mean than {label2}.")
        elif mean2 > mean1:
            output_lines.append(f"Direction: {label2} has a significantly HIGHER mean than {label1}.")
        else:
            output_lines.append("Direction: Means are effectively equal.")
    else:
        output_lines.append("Result: No significant difference detected.")
    output_lines.append("--------------------------------------")

    if file_handler:
        for line in output_lines: file_handler.write(line + "\n")
    else:
        for line in output_lines: print(line)

def main():
    report_filename = "statistical_report_exp1_exp2.txt"
    r_values_to_run = [2.0, 3.5]  # Run tests for both r=2.0 and r=3.5
    seeds_to_test = list(range(1, 11)) # Seeds 1 to 10
    num_last_points = 50 
    base_dir = "results/" # Only use the main results directory

    print(f"Starting statistical tests for Experiments 1 & 2 (r values: {r_values_to_run})")
    print(f"Using data from: {base_dir}")
    print(f"Using last {num_last_points} recorded metric points for averaging per seed.")
    print(f"Report will be saved to: {report_filename}")

    with open(report_filename, 'w') as report_file:
        report_file.write(f"Statistical Test Report for Experiments 1 & 2\n")
        report_file.write(f"Data sourced from: {base_dir}\n")
        report_file.write(f"Using last {num_last_points} recorded metric points for averaging per seed.\n")

        for r_value_to_test in r_values_to_run:
            report_file.write(f"\n\n======================================\n")
            report_file.write(f"   RESULTS FOR r = {r_value_to_test}\n")
            report_file.write(f"======================================\n")
            print(f"\n--- Running tests for r = {r_value_to_test} ---")

            # --- Experiment 1 ---
            report_file.write("\n\n=== Experiment 1: Q-Learning vs. Double Q-Learning ===\n")
            
            # Exp 1.1: Average Contribution Rate
            print("Exp 1.1: Testing Average Contribution Rate...")
            q_avg_contrib = get_mean_metric_per_seed(base_dir, r_value_to_test, seeds_to_test, False, "avg_contribution", num_last_points)
            dq_avg_contrib = get_mean_metric_per_seed(base_dir, r_value_to_test, seeds_to_test, True, "avg_contribution", num_last_points)
            perform_paired_t_test(q_avg_contrib, dq_avg_contrib, "Q-Learning", "Double Q-Learning", 
                                  "Exp 1.1: Average Contribution Rate", report_file)

            # Exp 1.2: Total Social Welfare
            print("Exp 1.2: Testing Total Social Welfare...")
            q_social_welfare = get_mean_metric_per_seed(base_dir, r_value_to_test, seeds_to_test, False, "social_welfare", num_last_points)
            dq_social_welfare = get_mean_metric_per_seed(base_dir, r_value_to_test, seeds_to_test, True, "social_welfare", num_last_points)
            perform_paired_t_test(q_social_welfare, dq_social_welfare, "Q-Learning", "Double Q-Learning", 
                                  "Exp 1.2: Total Social Welfare", report_file)

            # Exp 1.3: Standard Deviation of Contributions
            print("Exp 1.3: Testing Standard Deviation of Contributions...")
            q_std_dev_contrib = get_mean_metric_per_seed(base_dir, r_value_to_test, seeds_to_test, False, "std_dev_contrib", num_last_points)
            dq_std_dev_contrib = get_mean_metric_per_seed(base_dir, r_value_to_test, seeds_to_test, True, "std_dev_contrib", num_last_points)
            perform_paired_t_test(q_std_dev_contrib, dq_std_dev_contrib, "Q-Learning", "Double Q-Learning", 
                                  "Exp 1.3: Standard Deviation of Contributions", report_file)

            # --- Experiment 2 ---
            report_file.write("\n\n=== Experiment 2: Q-Learning vs. Double Q-Learning ===\n")
            report_file.write("Note: 'Mean Individual Contribution' is interpreted as 'Average Contribution Rate'.\n")
            
            # Exp 2.1: Mean Individual Contribution (interpreted as avg_contribution)
            print("Exp 2.1: Testing Mean Individual Contribution (as Avg. Contribution Rate)...")
            perform_paired_t_test(q_avg_contrib, dq_avg_contrib, "Q-Learning", "Double Q-Learning", 
                                  "Exp 2.1: Mean Individual Contribution (Avg. Contribution Rate)", report_file)
                                  
    print(f"\nStatistical report saved to {report_filename}")

if __name__ == "__main__":
    main()
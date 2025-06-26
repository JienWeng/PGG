import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from typing import List, Dict, Tuple, Optional, IO # Added Optional and IO

def get_mean_shapley_variance_per_seed(
    base_dir: str,
    r_value: float,
    seeds: List[int],
    is_double_q: bool,
    num_last_points_for_stat: int = 50 # Number of last recorded metric points to average over
) -> Dict[int, float]:
    """
    Loads metrics for each seed using a constructed filename, 
    calculates per-episode Shapley variance, and returns the mean 
    of this variance over the last N points for each seed.
    """
    seed_mean_shapley_variances: Dict[int, float] = {}
    algo_prefix = "double_q" if is_double_q else "q"

    for seed in seeds:
        seed_dir = Path(base_dir) / f"r{r_value}" / f"seed{seed}"
        metrics_file = seed_dir / f"{algo_prefix}_r{r_value}_seed{seed}_metrics.csv"

        if metrics_file.exists():
            try:
                df = pd.read_csv(metrics_file)
                if 'episode' not in df.columns:
                    print(f"Warning: 'episode' column missing in {metrics_file}") # Console warning
                    continue

                shapley_cols = [col for col in df.columns if col.startswith('agent_') and col.endswith('_shapley')]
                if not shapley_cols:
                    print(f"Warning: No Shapley columns found in {metrics_file}") # Console warning
                    continue
                
                if len(df) == 0:
                    print(f"Warning: Empty DataFrame after loading {metrics_file}") # Console warning
                    continue

                df['shapley_variance'] = df[shapley_cols].var(axis=1, skipna=True)
                df.dropna(subset=['shapley_variance'], inplace=True)

                if len(df) == 0:
                    print(f"Warning: DataFrame became empty after dropping NaNs in shapley_variance for {metrics_file}") # Console warning
                    continue

                if len(df['shapley_variance']) >= num_last_points_for_stat:
                    mean_var_for_seed = df['shapley_variance'].iloc[-num_last_points_for_stat:].mean()
                else:
                    mean_var_for_seed = df['shapley_variance'].mean() 
                
                if pd.notna(mean_var_for_seed):
                    seed_mean_shapley_variances[seed] = mean_var_for_seed
                else:
                    print(f"Warning: NaN result for mean Shapley variance for {metrics_file} (possibly all NaNs in last points).") # Console warning

            except pd.errors.EmptyDataError:
                print(f"Warning: Empty metrics file {metrics_file}") # Console warning
            except Exception as e:
                print(f"Warning: Error reading or processing {metrics_file}: {e}") # Console warning
        else:
            print(f"Warning: Metrics file not found {metrics_file}") # Console warning
            
    return seed_mean_shapley_variances

def perform_paired_t_test(
    data1_dict: Dict[int, float],
    data2_dict: Dict[int, float],
    label1: str,
    label2: str,
    hypothesis_description: str,
    file_handler: Optional[IO] = None # Added file_handler argument
):
    """
    Performs a paired t-test on data from common seeds and prints/writes results.
    """
    output_lines = []
    output_lines.append(f"\n--- {hypothesis_description} ---")
    
    common_seeds = sorted(list(set(data1_dict.keys()) & set(data2_dict.keys())))
    
    if len(common_seeds) < 2:
        output_lines.append(f"Not enough common seeds for a paired t-test (found {len(common_seeds)}). Required at least 2.")
        output_lines.append(f"Data for {label1}: {len(data1_dict)} seeds. Data for {label2}: {len(data2_dict)} seeds.")
        if file_handler:
            for line in output_lines:
                file_handler.write(line + "\n")
        else:
            for line in output_lines:
                print(line)
        return

    paired_values1 = [data1_dict[s] for s in common_seeds if s in data1_dict and pd.notna(data1_dict[s])]
    paired_values2 = [data2_dict[s] for s in common_seeds if s in data2_dict and pd.notna(data2_dict[s])]
    
    # Re-filter common_seeds to only those that have valid entries in both dicts for pairing
    valid_common_seeds = [
        s for s in common_seeds 
        if s in data1_dict and pd.notna(data1_dict[s]) and s in data2_dict and pd.notna(data2_dict[s])
    ]

    if len(valid_common_seeds) < 2:
        output_lines.append(f"Not enough valid (non-NaN) paired samples for t-test (found {len(valid_common_seeds)}). Required at least 2.")
        if file_handler:
            for line in output_lines:
                file_handler.write(line + "\n")
        else:
            for line in output_lines:
                print(line)
        return

    paired_values1_final = [data1_dict[s] for s in valid_common_seeds]
    paired_values2_final = [data2_dict[s] for s in valid_common_seeds]

    mean1 = np.mean(paired_values1_final)
    mean2 = np.mean(paired_values2_final)
    
    t_stat, p_value = stats.ttest_rel(paired_values1_final, paired_values2_final, alternative='two-sided') # nan_policy='omit' is default for ttest_rel
    
    output_lines.append(f"Comparing: {label1} vs. {label2}")
    output_lines.append(f"Number of paired samples (seeds) used in test: {len(valid_common_seeds)}")
    output_lines.append(f"Mean Shapley Variance for {label1}: {mean1:.6f}")
    output_lines.append(f"Mean Shapley Variance for {label2}: {mean2:.6f}")
    output_lines.append(f"Paired t-statistic: {t_stat:.4f}")
    output_lines.append(f"P-value (two-tailed): {p_value:.4f}")
    
    alpha = 0.05
    if p_value < alpha:
        output_lines.append("Result: Significant difference detected.")
        if mean1 > mean2:
            output_lines.append(f"Direction: {label1} has a significantly HIGHER mean Shapley variance than {label2}.")
        elif mean2 > mean1:
            output_lines.append(f"Direction: {label2} has a significantly HIGHER mean Shapley variance than {label1}.")
        else:
            output_lines.append("Direction: Means are effectively equal (though p-value suggests significance, this is unlikely if means are truly identical).")
    else:
        output_lines.append("Result: No significant difference detected.")
    output_lines.append("--------------------------------------")

    if file_handler:
        for line in output_lines:
            file_handler.write(line + "\n")
    else:
        for line in output_lines:
            print(line)


def main():
    # Configuration
    r_values_to_run = [2.0, 3.5]
    seeds_to_test = list(range(1, 11))  # Seeds 1 to 10
    num_last_points = 50
    base_dir_26_level = "results/"
    base_dir_3_level = "results_control/"

    for r_value_to_test in r_values_to_run:
        # --- Setup for the current r-value ---
        report_filename = f"statistical_report_exp3_r{r_value_to_test}.txt"

        # Console print for script start
        print(f"\n===== STARTING TESTS FOR r = {r_value_to_test} =====")
        print(f"Using last {num_last_points} recorded metric points for averaging Shapley variance per seed.")
        print(f"Report will be saved to: {report_filename}")

        with open(report_filename, 'w') as report_file:
            report_file.write(f"Statistical Test Report for Experiment 3 (r = {r_value_to_test})\n")
            report_file.write(f"Using last {num_last_points} recorded metric points for averaging Shapley variance per seed.\n")
            report_file.write(f"Comparing data from '{base_dir_26_level}' (26-level AS) and '{base_dir_3_level}' (3-level AS).\n")

            # --- Load data for all conditions for the current r-value ---
            print(f"Loading data for r={r_value_to_test}...")
            print("  Loading data for 26-level action space...")
            q_shapley_var_26_level = get_mean_shapley_variance_per_seed(
                base_dir_26_level, r_value_to_test, seeds_to_test, is_double_q=False, num_last_points_for_stat=num_last_points
            )
            dq_shapley_var_26_level = get_mean_shapley_variance_per_seed(
                base_dir_26_level, r_value_to_test, seeds_to_test, is_double_q=True, num_last_points_for_stat=num_last_points
            )
            print("  Loading data for 3-level action space...")
            q_shapley_var_3_level = get_mean_shapley_variance_per_seed(
                base_dir_3_level, r_value_to_test, seeds_to_test, is_double_q=False, num_last_points_for_stat=num_last_points
            )
            dq_shapley_var_3_level = get_mean_shapley_variance_per_seed(
                base_dir_3_level, r_value_to_test, seeds_to_test, is_double_q=True, num_last_points_for_stat=num_last_points
            )
            print("Data loading complete. Performing tests...")

            # --- Primary Hypotheses: Comparing algorithms within each action space ---
            perform_paired_t_test(
                q_shapley_var_26_level, dq_shapley_var_26_level,
                "Q-Learning (26-level AS)", "Double Q-Learning (26-level AS)",
                f"Primary Hypothesis 1: Algorithm comparison in 26-level Action Space (r={r_value_to_test})",
                file_handler=report_file
            )

            perform_paired_t_test(
                q_shapley_var_3_level, dq_shapley_var_3_level,
                "Q-Learning (3-level AS)", "Double Q-Learning (3-level AS)",
                f"Primary Hypothesis 2: Algorithm comparison in 3-level Action Space (r={r_value_to_test})",
                file_handler=report_file
            )

            # --- Secondary Hypotheses: Comparing action space impact for each algorithm ---
            perform_paired_t_test(
                q_shapley_var_3_level, q_shapley_var_26_level,
                "Q-Learning (3-level AS)", "Q-Learning (26-level AS)",
                f"Secondary Hypothesis 1: Action space impact for Q-Learning (r={r_value_to_test})",
                file_handler=report_file
            )

            perform_paired_t_test(
                dq_shapley_var_3_level, dq_shapley_var_26_level,
                "Double Q-Learning (3-level AS)", "Double Q-Learning (26-level AS)",
                f"Secondary Hypothesis 2: Action space impact for Double Q-Learning (r={r_value_to_test})",
                file_handler=report_file
            )

        print(f"Statistical report for r={r_value_to_test} saved to {report_filename}")
        print(f"===== FINISHED TESTS FOR r = {r_value_to_test} =====")


if __name__ == "__main__":
    main()
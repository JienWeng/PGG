# Public Goods Game MARL Analysis

This repository contains analysis of multi-agent reinforcement learning experiments in a Public Goods Game environment.

## Structure
- `src/`: Code for environment, agents, simulation, analysis, evaluation.
- `results/`: Outputs (CSVs, plots) per r.

## Setup
```bash
python src/main.py
python src/analyze.py
```

## Plot Descriptions

### Overall Analysis

1. **Contribution vs. Multiplication Factor (contribution_vs_r.png)**
   - X-axis: Multiplication factor r (1.5, 2.0, 2.5)
   - Y-axis: Average contribution proportion (0-1)
   - Type: Line plot with error bars
   - Error bars: Standard deviation across episodes (after smoothing)
   - Reference lines: 25% and 50% contribution thresholds
   - Interpretation: 
     - Shows how cooperation level changes with increasing returns on public good
     - Steeper slope indicates stronger influence of multiplication factor
     - Error bars show stability of strategies
     - Comparison between Q and Double Q reveals algorithmic differences
     - Crossing threshold lines indicates meaningful cooperation levels

### Per-r Analysis

For each r value, the following plots are generated:

1. **Contribution Comparison (contribution_comparison_r{r}.png)**
   - X-axis: Episode number (0-10000)
   - Y-axis: Average contribution proportion (0-1)
   - Type: Smoothed line plot
   - Interpretation: Learning progression of contribution behavior. Convergence indicates stable strategies.

2. **Social Welfare Comparison (welfare_comparison_r{r}.png)**
   - X-axis: Episode number (0-10000)
   - Y-axis: Total group payoff
   - Type: Smoothed line plot
   - Interpretation: Overall group performance. Higher values indicate more efficient outcomes.

3. **Inequality Comparison (gini_comparison_r{r}.png)**
   - X-axis: Episode number (0-10000)
   - Y-axis: Gini coefficient (0-1)
   - Type: Smoothed line plot
   - Interpretation: Fairness of outcomes. Lower values indicate more equal payoff distribution.

4. **Action Diversity (action_diversity_r{r}.png)**
   - X-axis: Episode number (0-10000)
   - Y-axis: Entropy of action distributions
   - Type: Smoothed line plot
   - Interpretation: Strategy exploration/exploitation balance. Higher values indicate more diverse strategies.

5. **Individual Contributions (q_individual_contribs_r{r}.png, double_q_individual_contribs_r{r}.png)**
   - X-axis: Episode number (0-10000)
   - Y-axis: Contribution proportion (0-1)
   - Type: Multiple smoothed lines (one per agent)
   - Interpretation: Individual agent behaviors. Shows endowment effects and agent specialization.

6. **Final Contribution Comparison (final_contribution_r{r}.png)**
   - X-axis: Algorithm type
   - Y-axis: Final average contribution
   - Type: Bar plot
   - Interpretation: Final learned strategies. Direct comparison of algorithm effectiveness.

7. **Q-Value Heatmaps (q_heatmap_agent_{i}_r{r}.png, double_q_heatmap_agent_{i}_r{r}.png)**
   - X-axis: Action space (0-1)
   - Y-axis: State space (endowment × group contribution)
   - Type: Heatmap
   - Interpretation: Learned value functions. Shows how agents value different state-action pairs.

8. **Preferred Actions (q_preferred_actions_agent_{i}_r{r}.png, double_q_preferred_actions_agent_{i}_r{r}.png)**
   - X-axis: Group contribution (0-1)
   - Y-axis: Agent endowment (0-2)
   - Type: Heatmap
   - Interpretation: Policy visualization. Shows what actions agents prefer in different states.

## Summary Files

1. **Metric Summary (summary_r{r}.csv)**
   - Contains: Mean and standard deviation of all metrics
   - Use: Quantitative comparison of algorithms

2. **Q-Table Summary (q_table_summary_r{r}.csv)**
   - Contains: Cooperation scores and action balance metrics
   - Use: Analysis of learned policies

## Key Findings

- **Endowment Effects**: Compare individual contribution plots across agents with different endowments
- **Learning Stability**: Check convergence in contribution comparison plots
- **Cooperation Level**: Analyze contribution_vs_r.png for r's influence
- **Fairness**: Monitor Gini coefficient for distributional effects
- **Exploration**: Track action diversity for learning dynamics
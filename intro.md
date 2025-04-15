### Introduction

#### 1. Background Study
Public Goods Games (PGGs) serve as a fundamental framework for studying cooperative behavior in social dilemmas, where individuals must balance personal gain against collective welfare. In a PGG, agents decide how much of their resources to contribute to a shared pool, which is then multiplied by a factor \( r \) and redistributed equally among all participants, regardless of their contributions. This setup often leads to the "free-rider problem," where rational agents may contribute minimally to maximize personal payoffs, resulting in suboptimal social outcomes. Traditional economic studies, such as those by Isaac and Walker (1988), have shown that cooperation in PGGs increases with higher multiplication factors (\( r \)), but real-world applications, such as charity systems or environmental resource management, often involve heterogeneous agents with varying endowments, complicating the dynamics of cooperation.

Recent advancements in multi-agent reinforcement learning (MARL) provide a powerful tool to model and analyze such social dilemmas in computational settings. MARL enables agents to learn optimal strategies through trial-and-error interactions with their environment and other agents, offering insights into emergent cooperative behaviors. Algorithms like Q-Learning and Double Q-Learning have been widely applied to MARL problems due to their ability to handle complex, dynamic environments. For instance, Leibo et al. (2017) demonstrated that MARL agents in social dilemmas can learn cooperative strategies under specific conditions, such as high rewards for collective action. However, most studies assume homogeneous agents or limited action spaces, which do not fully capture real-world scenarios where agents have diverse resources (e.g., endowments in charity systems) and a wide range of contribution options (e.g., donation amounts). Furthermore, the impact of these factors on fairness—measured through metrics like the Gini coefficient—and social welfare remains underexplored in MARL-based PGGs.

This study leverages MARL to simulate a PGG with heterogeneous endowments (\( E_i \in [0.5, 1.0, 1.5, 2.0] \)) and a large action space (25 discrete actions from 0 to 1.0), comparing Q-Learning and Double Q-Learning under varying multiplication factors (\( r \in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0] \)). The experiment incorporates reward shaping (\( \beta = 0.5 \)) to incentivize cooperation and examines fairness and welfare over 10,000 episodes. The findings aim to inform real-world applications, such as optimizing charity systems like ZakatFlow, where equitable contributions are crucial for maximizing community impact.

#### 2. Objectives
The primary goal of this research is to investigate the dynamics of cooperation, fairness, and social welfare in a MARL-based PGG under varying conditions. The specific objectives are:

1. **To Analyze the Influence of the Multiplication Factor (\( r \)) on Cooperation**: Test how \( r \in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0] \) affects average contributions and cooperation scores, comparing Q-Learning and Double Q-Learning.
2. **To Examine the Effect of Heterogeneous Endowments on Contribution Strategies**: Investigate how agents with endowments \( E_i \in [0.5, 1.0, 1.5, 2.0] \) adapt their contribution strategies, particularly at \( r=4.0 \), where cooperation is expected to be highest.
3. **To Evaluate the Impact of a Large Action Space on Fairness**: Assess how a large action space (25 discrete actions) influences fairness (via Gini coefficient) and action balance, especially under high cooperation incentives (\( r=4.0 \)).

#### 3. Significance
This research bridges the gap between theoretical PGG studies and practical applications by using MARL to model complex social dilemmas with heterogeneous agents and diverse action choices. The findings offer several key contributions:

- **Advancement in MARL for Social Dilemmas**: By comparing Q-Learning and Double Q-Learning, this study provides insights into which algorithms better promote cooperation and fairness in PGGs, contributing to the growing field of MARL in social simulations.
- **Real-World Applications**: The results have direct implications for designing equitable charity systems, such as ZakatFlow, where donors have varying resources and contribution options. Higher \( r \) (e.g., matching donations) can incentivize cooperation, while understanding endowment effects ensures fair resource distribution.
- **Policy Design**: The study highlights the importance of incentives and fairness in collective action problems, informing policies for resource management, climate action, and other public goods scenarios where cooperation is critical.

#### 4. Problem Statements
Despite the potential of MARL to model PGGs, several challenges remain unaddressed:

1. **Limited Understanding of \( r \)'s Impact in MARL**: While economic studies suggest that higher \( r \) promotes cooperation, its effect in MARL settings with learning agents is unclear, especially when comparing algorithms like Q-Learning and Double Q-Learning.
2. **Effect of Heterogeneous Endowments**: Most MARL studies assume homogeneous agents, but real-world scenarios involve agents with diverse resources. It is uncertain how endowments affect contribution strategies and whether high \( r \) can mitigate free-riding among low-endowment agents.
3. **Fairness with Large Action Spaces**: A large action space offers flexibility in contributions but may lead to unequal outcomes (high Gini coefficient) if agents exploit it to free-ride. The impact on fairness and action balance in MARL-based PGGs is underexplored.
4. **Welfare Trends Over Time**: Preliminary results at \( r=2.0 \) showed decreasing contributions and welfare over episodes, indicating a failure to sustain cooperation. It is unclear whether parameter adjustments (e.g., higher \( r \), reward shaping) can reverse this trend.

#### 5. Scope
This study focuses on a MARL-based PGG with the following scope:

- **Environment**: A PGG with 4 agents, endowments \( E_i \in [0.5, 1.0, 1.5, 2.0] \), and multiplication factors \( r \in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0] \).
- **Algorithms**: Comparison of Q-Learning and Double Q-Learning, with parameters including a large action space (25 actions), reward shaping (\( \beta = 0.5 \)), learning rate (0.05), discount factor (0.95), and 10,000 episodes (200 steps each).
- **Metrics**: Cooperation (average contribution, cooperation score), fairness (Gini coefficient, action balance), and social welfare.
- **Limitations**: The study does not explore continuous action spaces, larger group sizes, or alternative MARL algorithms (e.g., policy gradients). Real-world data integration (e.g., actual charity donation patterns) is beyond the current scope.

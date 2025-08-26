# Temporal-Difference-Algorithms-RL

## Overview
This assignment implements and compares two fundamental Temporal Difference (TD) learning algorithms in Reinforcement Learning: **Q-learning** and **SARSA**. TD learning is a combination of Monte Carlo ideas and Dynamic Programming ideas, allowing agents to learn directly from raw experience without a model of the environment's dynamics.

## Problem Environment: Cliff Walking

The algorithms are tested on the **Cliff Walking** environment, a classic grid-world problem:

- **Grid Size**: 4×12 grid world
- **Start Position**: Bottom-left corner (3, 0)
- **Goal Position**: Bottom-right corner (3, 11)
- **Cliff**: The bottom row (row 3) from position (3, 1) to (3, 10) contains dangerous cliff states
- **Actions**: Four possible actions: Up, Down, Left, Right
- **Rewards**:
  - Safe states: -1 per step
  - Cliff states: -100 (agent falls off and returns to start)
  - Goal state: Episode terminates

## Implemented Algorithms

### 1. Q-Learning (`Q-learning.py`)
**Off-policy TD control** algorithm that learns the optimal action-value function.

**Key Features:**
- Uses epsilon-greedy policy for exploration
- Updates Q-values using the maximum Q-value of the next state (off-policy)
- Implements epsilon decay for better convergence
- Hyperparameters:
  - Learning rate (α): 0.05
  - Discount factor (γ): 0.90
  - Initial epsilon: 0.2
  - Decay rate: 1.25

**Algorithm:**
```
Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
```

### 2. SARSA (`sarsa_method.py`)
**On-policy TD control** algorithm that learns the action-value function.

**Key Features:**
- Uses epsilon-greedy policy for both action selection and evaluation
- Updates Q-values using the actual action taken in the next state (on-policy)
- Same hyperparameters as Q-learning for fair comparison
- Implements epsilon decay for exploration-exploitation balance

**Algorithm:**
```
Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]
```

## Key Differences

| Aspect | Q-Learning | SARSA |
|--------|------------|-------|
| **Policy Type** | Off-policy | On-policy |
| **Update Rule** | Uses max Q(s',a') | Uses Q(s',a') for actual action taken |
| **Exploration** | More aggressive (can learn optimal policy while following suboptimal policy) | More conservative (learns policy it's actually following) |
| **Performance** | Generally finds better paths but may be less safe during learning | Generally safer during learning but may converge to suboptimal policy |

## Code Structure

### Common Components
- **Environment Setup**: Grid world representation with rewards and state definitions
- **Action Policy**: Epsilon-greedy exploration strategy with boundary handling
- **State Transitions**: Movement logic with cliff detection and episode termination
- **Visualization**: Grid display with optimal path arrows and learning curves

### Key Functions
- `action_policy(state, Q, epsilon)`: Epsilon-greedy action selection
- `new_state(state, action)`: State transition function
- `Q_policy(state, Q)` / `max_policy(state, Q)`: Greedy action selection for evaluation

## Running the Code

### Prerequisites
```bash
pip install numpy matplotlib
```

### Execution
```bash
# Run Q-learning algorithm
python Q-learning.py

# Run SARSA algorithm
python sarsa_method.py
```

### Output
Both algorithms provide:
1. **Q-value Matrix**: Printed Q-values for all state-action pairs
2. **Optimal Path Visualization**: Grid world with arrows showing the learned optimal path
3. **Learning Curve**: Plot of average rewards per 100 episodes over training

## Results Analysis

### Expected Behavior
- **Q-Learning**: Should find the optimal path along the edge of the cliff (more risky but optimal)
- **SARSA**: Should find a safer path slightly away from the cliff edge
- **Convergence**: Both should show decreasing negative rewards over episodes

### Performance Metrics
- **Episode Length**: Number of steps to reach goal
- **Total Reward**: Cumulative reward per episode
- **Policy Quality**: Optimality of the learned path

## Learning Parameters

### Hyperparameter Tuning
- **Learning Rate (α)**: Controls update step size (0.05 provides stable learning)
- **Discount Factor (γ)**: Balances immediate vs. future rewards (0.90 for long-term planning)
- **Epsilon (ε)**: Controls exploration vs. exploitation (decays from 0.2)
- **Decay Rate**: Controls how quickly exploration decreases (1.25 for gradual decay)

### Training Configuration
- **Episodes**: 5000 training episodes
- **Evaluation**: Performance measured every 100 episodes
- **Convergence**: Algorithms typically converge within 1000-2000 episodes

## Theoretical Background

### Temporal Difference Learning
TD methods combine the advantages of:
- **Monte Carlo**: Learning from actual experience
- **Dynamic Programming**: Bootstrapping from estimated values

### Q-Learning vs SARSA
- **Q-Learning**: Learns optimal policy regardless of behavior policy
- **SARSA**: Learns policy that balances exploration and exploitation

## Extensions and Improvements

### Potential Enhancements
1. **Experience Replay**: Store and reuse past experiences
2. **Double Q-Learning**: Reduce overestimation bias
3. **Prioritized Experience Replay**: Focus on important transitions
4. **Adaptive Learning Rates**: Dynamic adjustment based on state visit frequency

### Alternative Environments
- **Grid World Variations**: Different sizes and obstacle patterns
- **Continuous State Spaces**: Function approximation with neural networks
- **Multi-Agent Scenarios**: Competitive or cooperative environments

## References

- Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction
- Watkins, C. J. C. H. (1989). Learning from delayed rewards
- Rummery, G. A., & Niranjan, M. (1994). On-line Q-learning using connectionist systems

## License

This project is licensed under the terms specified in the LICENSE file.

---

**Note**: This implementation demonstrates fundamental concepts in reinforcement learning and serves as a foundation for understanding more advanced algorithms and techniques.

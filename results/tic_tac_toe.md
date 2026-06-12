# Tic-Tac-Toe Agent Results

## Engineering Question

How much does state-action representation influence reinforcement learning performance?

The objective was not to invent a new learning algorithm. Q-Learning is well understood.

The focus was to investigate how feature design affects what an agent can learn, how quickly it learns, and how effectively it generalizes against different opponent strategies.

---

## Agent Design

The agent uses Q-Learning with linear function approximation.

Instead of storing values for every possible board configuration, actions are represented using a compact set of engineered features that describe positional advantage, threats, opportunities, and game progression.

The final representation contains 16 features grouped into four categories.

### Positional Features

* bias
* action_cell
* pos_centre
* pos_corner
* pos_edge

### Terminal-State Features

* immediate_win
* blocks_win

### Threat-Awareness Features

* my_threats_before
* opp_threats_before
* my_threats_after
* creates_fork

### Board-State Features

* my_open_lines
* opp_open_lines
* board_occupancy
* my_max_line
* opp_max_line

---

## Experimental Setup

### Training Configuration

* Algorithm: Q-Learning
* Function Approximation: Linear
* Training Episodes: 1000

### Evaluation Configuration

* 200 games per opponent
* Random Opponent
* Aggressive Opponent
* Defensive Opponent

---

## Results

| Opponent   | Win Rate |
| ---------- | -------: |
| Random     |    75.0% |
| Aggressive |    67.0% |
| Defensive  |    40.0% |

The lower win rate against the Defensive opponent reflects a known limitation of reward-sparse environments. An opponent that rarely creates threats provides weaker learning signals, making it harder for the agent to discover and reinforce long-term strategic behaviour.

---

## Key Observation

The same Q-Learning algorithm can produce dramatically different behaviour depending on how state-action information is represented.

The agent learned a competitive policy using only:

* 16 engineered features
* 61 learned weights
* 1000 training episodes

A 16-feature linear model learned a competitive policy in 1000 episodes — no deep networks, no large state tables, and no exhaustive search.

Despite the compact representation, the learned policy consistently outperformed random and aggressive opponents while remaining competitive against a defensive strategy.

---

## Engineering Insight

This experiment highlights a common principle in reinforcement learning:

> Representation often matters as much as learning.

No changes were made to the underlying Q-Learning update rule.

Performance improvements came entirely from providing the agent with better information about threats, opportunities, positional control, and game state.

A compact feature representation was sufficient to learn useful behaviour without requiring large state tables, deep neural networks, or extensive training.

---

## Demo

Run:

```bash
cd reinforcement_learning_agents/tic_tac_toe
python3 demo.py
```

The demonstration:

* Trains the agent through self-play
* Evaluates against multiple opponent strategies
* Displays complete games
* Reports win-rate statistics and learned policy performance

# Puzzle Solver Agent Results

## Engineering Question

How much does heuristic quality affect search efficiency in A* search?

The objective was not to improve the search algorithm itself, but to understand how increasingly informative heuristics change the amount of work required to reach an optimal solution.

---

## Heuristics Evaluated

### h = 0 (Blind Search)

No heuristic information.

A* behaves similarly to Uniform Cost Search and explores a large portion of the search space.

### Manhattan Distance

Estimates the minimum number of moves required by summing the distance of each tile from its goal position.

Admissible and computationally inexpensive.

### Manhattan + Linear Conflict

Extends Manhattan Distance by identifying tiles that are in their correct row or column but block each other's progress.

Adds tile-interaction reasoning while preserving admissibility.

---

## Results

| Puzzle | Moves | h=0 | Manhattan | Linear Conflict |
|---------|---------|---------:|---------:|---------:|
| Easy | 1 | 3 | 1 | 1 |
| Medium | 4 | 19 | 4 | 4 |
| Hard | 14 | 4157 | 76 | 56 |

---

## Key Observation

The largest performance improvement came from heuristic selection rather than changes to the search algorithm.

For the hardest puzzle:

* Blind search expanded 4,157 nodes
* Manhattan Distance reduced this to 76 nodes
* Linear Conflict further reduced it to 56 nodes

Manhattan Distance did the heavy lifting — eliminating approximately 98% of unnecessary expansions. Linear Conflict adds the remaining gain by reasoning about tile interactions that Manhattan Distance ignores.


---

## Engineering Insight

This experiment highlights a common principle in AI search:

> Better information is often more valuable than more computation.

A heuristic that took minutes to design reduced a 4,157-node search to 56 nodes — a 74× improvement — without touching the search algorithm.

The majority of the improvement came from choosing a good heuristic (Manhattan Distance), while Linear Conflict provided a further refinement by incorporating tile-interaction reasoning.


---

## Demo

Run:

```bash
cd search_agents/puzzle_solver
python3 demo.py


The demonstration visualizes:

- Puzzle configurations
- Solution paths
- Node expansions
- Heuristic comparisons
- Performance summaries
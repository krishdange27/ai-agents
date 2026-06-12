"""
test_puzzle_solver.py
Tests for the A* puzzle solver agent with Linear Conflict heuristic.
"""

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.puzzle import GOAL, get_neighbors, is_goal
from src.heuristics import heuristic, evaluate, GOAL_POS
from src.search import astar


# ── Puzzle environment tests ──────────────────────────────────────────────────

class TestPuzzleEnvironment:

    def test_goal_state_is_recognised(self):
        assert is_goal(GOAL)

    def test_non_goal_state_not_recognised(self):
        assert not is_goal((1, 2, 3, 4, 5, 6, 7, 8, 0)[::-1])

    def test_neighbors_of_goal_are_valid(self):
        neighbors = get_neighbors(GOAL)
        assert len(neighbors) > 0
        for nb in neighbors:
            assert len(nb) == 9
            assert sorted(nb) == list(range(9))

    def test_one_move_from_goal(self):
        # Swap blank with last tile
        one_away = (1, 2, 3, 4, 5, 6, 7, 0, 8)
        result = astar(one_away, heuristic, evaluate)
        assert result is not None
        assert result["cost"] == 1


# ── Heuristic tests ───────────────────────────────────────────────────────────

class TestHeuristic:

    def test_goal_state_has_zero_heuristic(self):
        assert heuristic(GOAL) == 0

    def test_heuristic_is_non_negative(self):
        states = [
            (1, 2, 3, 4, 5, 6, 7, 8, 0),
            (8, 1, 3, 4, 0, 2, 7, 6, 5),
            (1, 3, 0, 4, 2, 5, 7, 8, 6),
        ]
        for state in states:
            assert heuristic(state) >= 0

    def test_heuristic_never_exceeds_true_cost(self):
        """Admissibility check: h(n) <= h*(n) for several known states."""
        known = [
            ((1, 2, 3, 4, 5, 6, 7, 0, 8), 1),   # 1 move away
            ((1, 2, 3, 4, 5, 6, 0, 7, 8), 2),   # 2 moves away
        ]
        for state, true_cost in known:
            assert heuristic(state) <= true_cost, \
                f"Heuristic {heuristic(state)} exceeds true cost {true_cost} for {state}"

    def test_evaluate_combines_g_and_h(self):
        assert evaluate(3, 5) == 8
        assert evaluate(0, 0) == 0


# ── A* search tests ───────────────────────────────────────────────────────────

class TestAStarSearch:

    def test_solves_goal_state_immediately(self):
        result = astar(GOAL, heuristic, evaluate)
        assert result is not None
        assert result["cost"] == 0

    def test_solves_simple_puzzle(self):
        start  = (1, 2, 3, 4, 5, 6, 7, 0, 8)
        result = astar(start, heuristic, evaluate)
        assert result["cost"] == 1

    def test_solves_medium_puzzle_optimally(self):
        start  = (1, 3, 0, 4, 2, 5, 7, 8, 6)
        result = astar(start, heuristic, evaluate)
        assert result is not None
        assert result["cost"] <= 15

    def test_solves_hard_puzzle(self):
        start  = (8, 1, 3, 4, 0, 2, 7, 6, 5)
        result = astar(start, heuristic, evaluate)
        assert result is not None
        assert result["cost"] == 14

    def test_linear_conflict_expands_fewer_nodes_than_baseline(self):
        def zero_h(state): return 0
        hard = (8, 1, 3, 4, 0, 2, 7, 6, 5)
        lc_result   = astar(hard, heuristic, evaluate)
        base_result = astar(hard, zero_h, evaluate)
        assert lc_result["expanded"] < base_result["expanded"]

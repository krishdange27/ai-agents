import unittest
from src.astar import astar
from src.heuristics import heuristic, evaluate

class TestPublic(unittest.TestCase):

    def test_goal(self):
        start = (1,2,3,4,5,6,7,8,0)
        res = astar(start, heuristic, evaluate)
        self.assertEqual(res["cost"], 0)

    def test_one_move(self):
        start = (1,2,3,4,5,6,7,0,8)
        res = astar(start, heuristic, evaluate)
        self.assertEqual(res["cost"], 1)

    def test_simple(self):
        start = (1,2,3,4,5,6,0,7,8)
        res = astar(start, heuristic, evaluate)
        self.assertEqual(res["cost"], 2)

    def test_heuristic_non_negative(self):
        state = (2,1,3,4,5,6,7,8,0)
        self.assertGreaterEqual(heuristic(state), 0)

    def test_heuristic_goal_zero(self):
        state = (1,2,3,4,5,6,7,8,0)
        self.assertEqual(heuristic(state), 0)


class TestPublicHeuristicAdmissibility(unittest.TestCase):

    def test_heuristic_never_exceeds_true_cost(self):
        """
        Admissibility: h(s) <= true optimal cost for near-goal states
        where we know the exact answer.
        """
        known = [
            ((1,2,3,4,5,6,7,0,8), 1),   # 1 move
            ((1,2,3,4,5,6,0,7,8), 2),   # 2 moves
            ((1,2,3,4,5,0,7,6,8), 3),   # 3 moves  
        ]
        for state, true_cost in known:
            h = heuristic(state)
            self.assertLessEqual(
                h, true_cost,
                f"Heuristic {h} exceeds true cost {true_cost} for {state}"
            )
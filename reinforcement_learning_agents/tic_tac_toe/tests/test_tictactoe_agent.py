"""
test_tictactoe_agent.py
Tests for the Q-learning TicTacToe agent — environment, features, reward, and training.
"""

import pytest
import random
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.environment import TicTacToe
from src.agents import QLearningAgent, RandomAgent
from src.training import train, evaluate
from src.agent import extract_features, reward_function, epsilon_schedule
from src.utils import OpponentAdapter, state_3x3_to_flat, flat_to_state_3x3, board_to_string
from src.opponents import RandomOpponent, DefensiveOpponent, AggressiveOpponent

random.seed(42)


# ── Environment tests ─────────────────────────────────────────────────────────

class TestEnvironment:

    def test_reset_returns_empty_board(self):
        env = TicTacToe()
        state = env.reset()
        assert all(state[r][c] == 0 for r in range(3) for c in range(3))

    def test_valid_actions_on_empty_board(self):
        env = TicTacToe()
        env.reset()
        actions = env.get_valid_actions()
        assert len(actions) == 9

    def test_step_fills_cell(self):
        env = TicTacToe()
        state = env.reset()
        new_state, _, _ = env.step((0, 0))
        assert new_state[0][0] != 0

    def test_win_detected(self):
        env = TicTacToe()
        env.reset()
        env.step((0, 0)); env.step((1, 0))
        env.step((0, 1)); env.step((1, 1))
        _, reward, done = env.step((0, 2))
        assert done
        assert reward == 1


# ── Feature extraction tests ──────────────────────────────────────────────────

class TestFeatureExtraction:

    def test_features_returns_list_of_tuples(self):
        env = TicTacToe()
        state = env.reset()
        features = extract_features(state, (0, 0))
        assert isinstance(features, list)
        assert all(isinstance(f, tuple) and len(f) == 2 for f in features)

    def test_features_are_hashable(self):
        env = TicTacToe()
        state = env.reset()
        features = extract_features(state, (1, 1))
        for f in features:
            hash(f)  # should not raise

    def test_centre_action_flagged(self):
        env = TicTacToe()
        state = env.reset()
        features = dict(extract_features(state, (1, 1)))
        assert features["pos_centre"] == 1
        assert features["pos_corner"] == 0

    def test_corner_action_flagged(self):
        env = TicTacToe()
        state = env.reset()
        features = dict(extract_features(state, (0, 0)))
        assert features["pos_corner"] == 1
        assert features["pos_centre"] == 0

    def test_immediate_win_detected(self):
        # Board: player has (0,0) and (0,1) — playing (0,2) wins
        state = (
            (1, 1, 0),
            (0, 0, 0),
            (0, 0, 0),
        )
        features = dict(extract_features(state, (0, 2)))
        assert features["immediate_win"] == 1

    def test_blocking_move_detected(self):
        # Opponent has (0,0) and (0,1) — playing (0,2) blocks them
        state = (
            (-1, -1, 0),
            (0,  0,  0),
            (0,  0,  0),
        )
        features = dict(extract_features(state, (0, 2)))
        assert features["blocks_win"] == 1


# ── Reward function tests ─────────────────────────────────────────────────────

class TestRewardFunction:

    def test_winning_move_gives_positive_reward(self):
        state = (
            (1, 1, 0),
            (0, 0, 0),
            (0, 0, 0),
        )
        next_state = (
            (1, 1, 1),
            (0, 0, 0),
            (0, 0, 0),
        )
        assert reward_function(state, (0, 2), next_state) > 0

    def test_losing_state_gives_negative_reward(self):
        state = (
            (0, 0, 0),
            (0, 0, 0),
            (0, 0, 0),
        )
        next_state = (
            (-1, -1, -1),
            (0,  0,  0),
            (0,  0,  0),
        )
        assert reward_function(state, (0, 0), next_state) < 0

    def test_non_terminal_reward_is_small(self):
        state = (
            (0, 0, 0),
            (0, 0, 0),
            (0, 0, 0),
        )
        next_state = (
            (1, 0, 0),
            (0, 0, 0),
            (0, 0, 0),
        )
        r = reward_function(state, (0, 0), next_state)
        assert abs(r) < 0.5


# ── Epsilon schedule tests ────────────────────────────────────────────────────

class TestEpsilonSchedule:

    def test_epsilon_starts_high(self):
        assert epsilon_schedule(0) >= 0.5

    def test_epsilon_decays_over_time(self):
        assert epsilon_schedule(500) < epsilon_schedule(0)

    def test_epsilon_never_below_floor(self):
        for ep in [0, 100, 500, 1000, 5000]:
            assert epsilon_schedule(ep) >= 0.05

    def test_epsilon_never_above_one(self):
        for ep in [0, 1, 10]:
            assert epsilon_schedule(ep) <= 1.0


# ── Training and evaluation tests ────────────────────────────────────────────

class TestTrainingAndEvaluation:

    def test_agent_trains_without_error(self):
        env   = TicTacToe()
        agent = QLearningAgent(extract_features, reward_function)
        train(env, agent, RandomAgent(), episodes=20)

    def test_agent_improves_over_random_baseline(self):
        env   = TicTacToe()
        agent = QLearningAgent(extract_features, reward_function, alpha=0.5, gamma=0.95)
        train(env, agent, RandomAgent(), episodes=500)
        agent.epsilon = 0.0
        win_rate = evaluate(env, agent, RandomAgent(), games=100)
        assert win_rate > 0.5, f"Expected > 50% win rate, got {win_rate:.2f}"

    def test_opponent_adapter_works(self):
        env   = TicTacToe()
        agent = QLearningAgent(extract_features, reward_function)
        adapted = OpponentAdapter(RandomOpponent())
        train(env, agent, adapted, episodes=10)

    def test_evaluate_returns_value_between_0_and_1(self):
        env   = TicTacToe()
        agent = QLearningAgent(extract_features, reward_function)
        rate  = evaluate(env, agent, RandomAgent(), games=20)
        assert 0.0 <= rate <= 1.0


# ── Utility tests ─────────────────────────────────────────────────────────────

class TestUtils:

    def test_state_conversion_round_trip(self):
        env   = TicTacToe()
        state = env.reset()
        flat  = state_3x3_to_flat(state)
        back  = flat_to_state_3x3(flat)
        assert state == back

    def test_flat_has_correct_symbols(self):
        state = ((1, -1, 0), (0, 1, 0), (-1, 0, 1))
        flat  = state_3x3_to_flat(state)
        assert set(flat).issubset({"X", "O", " "})

    def test_board_to_string_renders(self):
        env    = TicTacToe()
        state  = env.reset()
        output = board_to_string(state)
        assert isinstance(output, str)
        assert len(output) > 0

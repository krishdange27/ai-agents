import unittest
from src.tictactoe import TicTacToe
from src.agents import QLearningAgent, RandomAgent
from src.training import train, evaluate
import src.design as design

import random
random.seed(42)
# note, same random seed will be used for hidden tests as well 

class TestPublic(unittest.TestCase):

    def test_training_runs(self):
        env = TicTacToe()
        agent = QLearningAgent(design.extract_features, design.reward_function)
        opponent = RandomAgent()

        train(env, agent, opponent, episodes=10)

        self.assertTrue(len(agent.q) > 0)

    def test_evaluation_range(self):
        env = TicTacToe()
        agent = QLearningAgent(design.extract_features, design.reward_function)
        opponent = RandomAgent()

        train(env, agent, opponent, episodes=20)
        score = evaluate(env, agent, opponent, games=10)

        self.assertTrue(0 <= score <= 1)

    def test_valid_actions(self):
        env = TicTacToe()
        actions = env.get_valid_actions()
        self.assertEqual(len(actions), 9)

class TestPublicAdditional(unittest.TestCase):

    def test_reset_clears_board(self):
        env = TicTacToe()
        env.board[0][0] = 1
        env.reset()
        self.assertTrue(all(cell == 0 for row in env.board for cell in row))

    def test_step_changes_state(self):
        env = TicTacToe()
        state = env.reset()
        action = (0, 0)
        next_state, _, _ = env.step(action)
        self.assertNotEqual(state, next_state)

    def test_invalid_reuse_of_cell(self):
        env = TicTacToe()
        env.reset()
        env.step((0, 0))
        actions = env.get_valid_actions()
        self.assertNotIn((0, 0), actions)

    def test_draw_condition(self):
        env = TicTacToe()
        env.board = [
            [1, -1, 1],
            [1, -1, -1],
            [-1, 1, 1]
        ]
        winner = env.check_winner()
        self.assertEqual(winner, 0)

    def test_feature_function_returns_iterable(self):
        env = TicTacToe()
        state = env.reset()
        features = design.extract_features(state, (0, 0))
        self.assertTrue(hasattr(features, "__iter__"))

    def test_feature_function_not_empty(self):
        env = TicTacToe()
        state = env.reset()
        features = design.extract_features(state, (0, 0))
        self.assertTrue(len(features) > 0)

    def test_reward_function_callable(self):
        env = TicTacToe()
        state = env.reset()
        next_state, _, _ = env.step((0, 0))
        r = design.reward_function(state, (0, 0), next_state)
        self.assertIsInstance(r, (int, float))

    def test_q_values_update(self):
        env = TicTacToe()
        agent = QLearningAgent(design.extract_features, design.reward_function)
        opponent = RandomAgent()

        train(env, agent, opponent, episodes=5)
        self.assertTrue(any(abs(v) > 0 for v in agent.q.values()))

    def test_agent_action_valid(self):
        env = TicTacToe()
        agent = QLearningAgent(design.extract_features, design.reward_function)

        state = env.reset()
        action = agent.act(state, env.get_valid_actions())
        self.assertIn(action, env.get_valid_actions())

    def test_multiple_training_runs_stable(self):
        env = TicTacToe()
        agent = QLearningAgent(design.extract_features, design.reward_function)
        opponent = RandomAgent()

        train(env, agent, opponent, episodes=30)
        score1 = evaluate(env, agent, opponent, games=10)
        score2 = evaluate(env, agent, opponent, games=10)

        self.assertTrue(abs(score1 - score2) < 0.5)

    def test_epsilon_effect(self):
        env = TicTacToe()
        agent = QLearningAgent(design.extract_features, design.reward_function, epsilon=1.0)

        state = env.reset()
        actions = env.get_valid_actions()

        # with epsilon=1, actions should vary
        samples = set(agent.act(state, actions) for _ in range(10))
        self.assertTrue(len(samples) > 1)

    def test_small_training_does_not_crash(self):
        env = TicTacToe()
        agent = QLearningAgent(design.extract_features, design.reward_function)
        opponent = RandomAgent()

        try:
            train(env, agent, opponent, episodes=1)
        except Exception as e:
            self.fail(f"Training crashed with exception: {e}")

    def test_state_representation_hashable(self):
        env = TicTacToe()
        state = env.reset()
        try:
            hash(state)
        except TypeError:
            self.fail("State should be hashable")

    def test_symmetry_basic(self):
        env = TicTacToe()
        env.board = [
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, 0]
        ]
        state = env.get_state()
        features1 = design.extract_features(state, (0, 1))
        features2 = design.extract_features(state, (1, 0))

        # not enforcing equality, but should not crash
        self.assertTrue(len(features1) > 0 and len(features2) > 0)


class TestPublicRewardProperties(unittest.TestCase):

    def test_reward_win_is_positive(self):
        """Winning move must yield a positive reward."""
        env = TicTacToe()
        env.board = [
            [1, 1, 0],
            [-1, -1, 0],
            [0, 0, 0]
        ]
        env.current_player = 1
        state = env.get_state()
        _, _, _ = env.step((0, 2))   # winning move for player 1
        next_state = env.get_state()
        r = design.reward_function(state, (0, 2), next_state)
        self.assertGreater(r, 0)

    def test_reward_loss_is_negative(self):
        """A state transition leading to a loss must yield negative reward."""
        env = TicTacToe()
        env.board = [
            [-1, -1, 0],
            [1, 1, 0],
            [0, 0, 0]
        ]
        env.current_player = -1
        state = env.get_state()
        _, _, _ = env.step((0, 2))   # opponent wins
        next_state = env.get_state()
        r = design.reward_function(state, (0, 2), next_state)
        self.assertLess(r, 0)
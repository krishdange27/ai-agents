"""
demo.py — TicTacToe Reinforcement Learning Agent
--------------------------------------------------
Trains a Q-learning agent from scratch using self-play against a random
opponent, then evaluates it against three opponent types and plays a
live game so you can watch the agent in action.

Run from the tic_tac_toe/ folder:
    python demo.py
"""

import sys
import os
import random
sys.path.insert(0, os.path.dirname(__file__))

from src.environment import TicTacToe
from src.agents import QLearningAgent, RandomAgent
from src.training import train, evaluate
from src.opponents import RandomOpponent, DefensiveOpponent, AggressiveOpponent
from src.agent import extract_features, reward_function, epsilon_schedule
from src.utils import OpponentAdapter, board_to_string

TRAIN_EPISODES = 1000
EVAL_GAMES     = 200
RANDOM_SEED    = 42

random.seed(RANDOM_SEED)


def train_with_schedule(env, agent, opponent, episodes):
    """Train the agent, updating epsilon each episode."""
    for ep in range(episodes):
        agent.epsilon = epsilon_schedule(ep)
        state = env.reset()
        done = False
        while not done:
            actions = env.get_valid_actions()
            action = agent.act(state, actions)
            next_state, env_reward, done = env.step(action)
            reward = reward_function(state, action, next_state)
            if not done:
                opp_action = opponent.act(next_state, env.get_valid_actions())
                next_state, _, done = env.step(opp_action)
            next_actions = env.get_valid_actions() if not done else []
            agent.update(state, action, reward, next_state, next_actions)
            state = next_state


def play_live_game(env, agent, opponent_label, opponent):
    """Play one game and print the board after every move."""
    print(f"\n  Live game vs {opponent_label}")
    print(f"  Agent = X   Opponent = O")
    print("  " + "─" * 20)

    state = env.reset()
    done = False
    move = 0

    print(f"\n  Start:")
    print("  " + board_to_string(state).replace("\n", "\n  "))

    while not done:
        # Agent's turn
        action = agent.act(state, env.get_valid_actions())
        state, reward, done = env.step(action)
        move += 1
        print(f"\n  Move {move} — Agent plays {action}:")
        print("  " + board_to_string(state).replace("\n", "\n  "))
        if done:
            print(f"\n  Result: {'Agent wins! 🎉' if reward == 1 else 'Draw'}")
            break

        # Opponent's turn
        opp_action = opponent.act(state, env.get_valid_actions())
        state, reward, done = env.step(opp_action)
        move += 1
        print(f"\n  Move {move} — Opponent plays {opp_action}:")
        print("  " + board_to_string(state).replace("\n", "\n  "))
        if done:
            print(f"\n  Result: {'Opponent wins' if reward == -1 else 'Draw'}")
            break


def main():
    print("=" * 60)
    print("  TicTacToe Agent — Q-Learning with Function Approximation")
    print("=" * 60)

    env   = TicTacToe()
    agent = QLearningAgent(extract_features, reward_function, alpha=0.5, gamma=0.95)

    # ── Training ──────────────────────────────────────────────────
    print(f"\n[1] Training for {TRAIN_EPISODES} episodes vs random opponent...")
    train_with_schedule(env, agent, RandomAgent(), TRAIN_EPISODES)
    agent.epsilon = 0.0   # pure exploitation for evaluation
    print(f"    Training complete. Q-table entries: {len(agent.q)}")

    # ── Evaluation ────────────────────────────────────────────────
    opponents = [
        ("Random",     RandomAgent()),
        ("Defensive",  OpponentAdapter(DefensiveOpponent())),
        ("Aggressive", OpponentAdapter(AggressiveOpponent())),
    ]

    print(f"\n[2] Evaluating over {EVAL_GAMES} games per opponent type:")
    print(f"\n    {'Opponent':<12} {'Win Rate':>9} {'Result'}")
    print(f"    {'-'*12} {'-'*9} {'-'*20}")

    eval_results = []
    for label, opp in opponents:
        win_rate = evaluate(env, agent, opp, games=EVAL_GAMES)
        bar = "█" * int(win_rate * 20)
        eval_results.append((label, win_rate))
        print(f"    {label:<12} {win_rate*100:>8.1f}%  {bar}")

    # ── Live game ─────────────────────────────────────────────────
    print(f"\n[3] Watching the agent play...")
    agent.epsilon = 0.0
    play_live_game(env, agent, "Random", RandomAgent())

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Results Summary")
    print(f"{'='*60}")
    print(f"  Training episodes : {TRAIN_EPISODES}")
    print(f"  Q-table size      : {len(agent.q)} feature weights")
    for label, wr in eval_results:
        print(f"  Win rate vs {label:<12}: {wr*100:.1f}%")
    print()


if __name__ == "__main__":
    main()

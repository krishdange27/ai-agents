import random

class RandomOpponent:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def choose_action(self, state):
        actions = [i for i, v in enumerate(state) if v == " "]
        return self.rng.choice(actions)


class DefensiveOpponent:
    """
    Blocks opponent wins if possible, else random
    """

    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def choose_action(self, state):
        # Try to block X
        for a in self.available_actions(state):
            new_state = state.copy()
            new_state[a] = "X"
            if self.check_winner(new_state, "X"):
                return a

        return self.rng.choice(self.available_actions(state))

    def available_actions(self, state):
        return [i for i, v in enumerate(state) if v == " "]

    def check_winner(self, board, player):
        wins = [
            (0,1,2), (3,4,5), (6,7,8),
            (0,3,6), (1,4,7), (2,5,8),
            (0,4,8), (2,4,6)
        ]
        return any(board[i] == board[j] == board[k] == player for i, j, k in wins)


class AggressiveOpponent:
    """
    Tries to win if possible, else random
    """

    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def choose_action(self, state):
        # Try to win as O
        for a in self.available_actions(state):
            new_state = state.copy()
            new_state[a] = "O"
            if self.check_winner(new_state, "O"):
                return a

        return self.rng.choice(self.available_actions(state))

    def available_actions(self, state):
        return [i for i, v in enumerate(state) if v == " "]

    def check_winner(self, board, player):
        wins = [
            (0,1,2), (3,4,5), (6,7,8),
            (0,3,6), (1,4,7), (2,5,8),
            (0,4,8), (2,4,6)
        ]
        return any(board[i] == board[j] == board[k] == player for i, j, k in wins)
    


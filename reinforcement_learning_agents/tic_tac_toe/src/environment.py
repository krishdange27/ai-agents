import random
import copy

class TicTacToe:
    def __init__(self, size=3, win_k=3):
        self.size = size
        self.win_k = win_k
        self.reset()

    def reset(self):
        self.board = [[0]*self.size for _ in range(self.size)]
        self.current_player = 1
        return self.get_state()

    def get_state(self):
        return tuple(tuple(row) for row in self.board)

    def get_valid_actions(self):
        return [(i, j) for i in range(self.size)
                for j in range(self.size) if self.board[i][j] == 0]

    def step(self, action):
        i, j = action
        self.board[i][j] = self.current_player

        winner = self.check_winner()
        done = winner is not None

        reward = 0
        if done:
            if winner == 1:
                reward = 1
            elif winner == -1:
                reward = -1
            else:
                reward = 0

        self.current_player *= -1
        return self.get_state(), reward, done

    def check_winner(self):
        lines = []

        # rows & cols
        for i in range(self.size):
            lines.append(self.board[i])
            lines.append([self.board[j][i] for j in range(self.size)])

        # diagonals
        lines.append([self.board[i][i] for i in range(self.size)])
        lines.append([self.board[i][self.size-1-i] for i in range(self.size)])

        for line in lines:
            for i in range(len(line) - self.win_k + 1):
                segment = line[i:i+self.win_k]
                if abs(sum(segment)) == self.win_k:
                    return segment[0]

        if all(cell != 0 for row in self.board for cell in row):
            return 0  # draw

        return None
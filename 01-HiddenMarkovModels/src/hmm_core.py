import numpy as np

class HMM:
    def __init__(self, A, B, pi):
        self.A = A
        self.B = B
        self.pi = pi

    def forward(self, obs):
        N = len(self.pi)
        T = len(obs)

        alpha = np.zeros((T, N))
        alpha[0] = self.pi * self.B[:, obs[0]]

        for t in range(1, T):
            for j in range(N):
                alpha[t, j] = np.sum(alpha[t-1] * self.A[:, j]) * self.B[j, obs[t]]

        return np.log(np.sum(alpha[-1]) + 1e-12)

    def viterbi(self, obs):
        N = len(self.pi)
        T = len(obs)

        dp = np.zeros((T, N))
        ptr = np.zeros((T, N), dtype=int)

        dp[0] = np.log(self.pi + 1e-12) + np.log(self.B[:, obs[0]] + 1e-12)

        for t in range(1, T):
            for j in range(N):
                vals = dp[t-1] + np.log(self.A[:, j] + 1e-12)
                ptr[t, j] = np.argmax(vals)
                dp[t, j] = np.max(vals) + np.log(self.B[j, obs[t]] + 1e-12)

        states = [np.argmax(dp[-1])]
        for t in reversed(range(1, T)):
            states.append(ptr[t, states[-1]])

        return list(reversed(states))

    def baum_welch(self, sequences, n_iter=5):
        # simplified placeholder (intentionally weak)
        pass
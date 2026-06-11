import random
from collections import defaultdict

class RandomAgent:
    def act(self, state, actions):
        return random.choice(actions)

class QLearningAgent:
    def __init__(self, feature_fn, reward_fn, alpha=0.5, gamma=0.9, epsilon=0.2):
        self.q = defaultdict(float)
        self.feature_fn = feature_fn
        self.reward_fn = reward_fn
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_q(self, state, action):
        features = self.feature_fn(state, action)
        return sum(self.q[f] for f in features)

    def act(self, state, actions):
        if random.random() < self.epsilon:
            return random.choice(actions)
        return max(actions, key=lambda a: self.get_q(state, a))

    def update(self, state, action, reward, next_state, next_actions):
        max_next = 0 if not next_actions else max(
            self.get_q(next_state, a) for a in next_actions
        )

        features = self.feature_fn(state, action)
        current = self.get_q(state, action)

        target = reward + self.gamma * max_next
        diff = target - current

        for f in features:
            self.q[f] += self.alpha * diff
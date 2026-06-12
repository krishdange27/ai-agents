def train(env, agent, opponent, episodes=200):
    for _ in range(episodes):
        state = env.reset()
        done = False

        while not done:
            actions = env.get_valid_actions()

            action = agent.act(state, actions)
            next_state, reward, done = env.step(action)

            if not done:
                opp_action = opponent.act(next_state, env.get_valid_actions())
                next_state, opp_reward, done = env.step(opp_action)
                reward -= opp_reward

            next_actions = env.get_valid_actions() if not done else []

            agent.update(state, action, reward, next_state, next_actions)
            state = next_state


def evaluate(env, agent, opponent, games=50):
    wins = 0

    for _ in range(games):
        state = env.reset()
        done = False

        while not done:
            action = agent.act(state, env.get_valid_actions())
            state, reward, done = env.step(action)

            if done:
                if reward == 1:
                    wins += 1
                break

            opp_action = opponent.act(state, env.get_valid_actions())
            state, reward, done = env.step(opp_action)

    return wins / games
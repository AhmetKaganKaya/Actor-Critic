import gym
from Agent import Agent
import numpy as np

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    agent = Agent(gamma = 0.99, n_actions = env.action_space.n, input_dims = env.observation_space.sample().shape[0])
    scores, eps_history = [], []
    n_games = 0

    while True:
        score = 0
        done = False
        observation = env.reset()
        timer = 0
        while not done:
            action, distribution = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.learn(state = observation, action = action, next_state = observation_, reward= reward
                        , done = done, distribution = distribution)
            observation = observation_
            timer += 1

            # if i > n_games - 5:
            #     env.render()
        scores.append(score)
        n_games += 1
        avg_score = np.mean(scores[-100:])
        if avg_score == 500:
            break
        print("Episode: {}, Score: {}, Avg Score: {}, Timer: {}".format(n_games + 1, score, avg_score, timer))

    env.close()
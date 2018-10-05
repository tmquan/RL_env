from rllab.envs.base import Env
from rllab.spaces import Box
from rllab.envs.base import Step
import numpy as np
import gym


if __name__ == '__main__':
    env = gym.make("Pong-v0")
    MAX_NUM_EPISODES = 5000
    for episode in range(MAX_NUM_EPISODES):
        done = False
        obs = env.reset()
        total_reward = 0.0 # To keep track of the total reward obtained in each
        episode
        step = 0
        while not done:
            env.render()

            # Sample random action. This will be replaced by our agent's action when we start developing the agent algorithms

            action = env.action_space.sample() 
            # Send the action to the environment and receive the next_state, reward and whether done or not
            next_state, reward, done, info = env.step(action) 
            total_reward += reward
            step += 1
            obs = next_state
        print("Episode #{} ended in {} steps.total_reward={}".format(episode, step+1, total_reward))
    env.close()
from rllab.envs.base import Env
from rllab.spaces import Box
from rllab.envs.base import Step
import numpy as np
import gym

class PointEnv(Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        self.__version__ = "0.0.1"
        print("Init PointEnv")
        # Modify the observation space, low, high and shape values according to your custom environment's needs
        # self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(3,))
        # Modify the action space, and dimension according to your custom environment's needs
        # self.action_space = gym.spaces.Discrete(4)

    def step(self, action):
        """
        Runs one time-step of the environment's dynamics. The reset() method is called at the end of every episode
        :param action: The action to be executed in the environment
        :return: (observation, reward, done, info)
            observation (object):
                Observation from the environment at the current time-step
            reward (float):
                Reward from the environment due to the previous action performed
            done (bool):
                a boolean, indicating whether the episode has ended
            info (dict):
                a dictionary containing additional information about the previous action
        """
        # Implement your step method here
        # return (observation, reward, done, info)
        self._state = self._state + action
        # print('Step state:', self._state)
        x, y = self._state
        reward = - (x ** 2 + y ** 2) ** 0.5
        done = abs(x) < 0.01 and abs(y) < 0.01
        next_observation = np.copy(self._state)
        return Step(observation=next_observation, reward=reward, done=done)

    def reset(self):
        """
        Reset the environment state and returns an initial observation

        Returns
        -------
        observation (object): The initial observation for the new episode after reset
        :return:
        """

        # Implement your reset method here
        # return observation
        self._state = np.random.uniform(-1, 1, size=(2,))
        observation = np.copy(self._state)
        return observation


    def render(self, mode='human', close=False):
        """

        :param mode:
        :return:
        """
        # return
        print('Current state:', self._state)
        # return

    @property
    def observation_space(self):
        return Box(low=-np.inf, high=np.inf, shape=(2,))

    @property
    def action_space(self):
        return Box(low=-0.1, high=0.1, shape=(2,))

if __name__ == '__main__':
    env = PointEnv()

    MAX_NUM_EPISODES = 10
    MAX_STEPS_PER_EPISODE = 50000


    for episode in range(MAX_NUM_EPISODES):
        obs = env.reset()
        for step in range(MAX_STEPS_PER_EPISODE):
            env.render()
            action = env.action_space.sample()# Sample random action. This will be replaced by our agent's action when we start developing the agentalgorithms
            next_state, reward, done, info = env.step(action) # Send the action to the environment and receive the next_state, reward and whether done or not
            obs = next_state
            if done is True:
                print("\n Episode #{} ended in {} steps.".format(episode, step+1))
            break



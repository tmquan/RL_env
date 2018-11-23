from rllab.envs.base import Env
from rllab.spaces import Box
from rllab.envs.base import Step
import numpy as np
import gym

from QuadLocEnvironment import *

import ray
from ray.tune.registry import register_env
from ray.tune import run_experiments
from ray.rllib.agents import ppo

def env_creator(env_config={}):
    env = QuadLocEnv(dataDir='/Users/tmquan/RL_env/data/', num=20)
    num = env.action_space.n
    # print("Action:", num)
    env.reset()
    return env

# env = QuadLocEnv(dataDir='data/', num=20)
register_env("QuadLocEnv-v0", env_creator)

ray.init()
run_experiments({
        "demo": {
            "run": "DQN",
            "env": "QuadLocEnv-v0",
       		"config": {
                "env_config": {
                    "corridor_length": 5,
                },
            },
        },
    })
# trainer = ppo.PPOAgent(
#     env="QuadLocEnv-v0", 
#     config={
#     "env_config": {},  # config to pass to env creator
#     }
# )

# while True:
#     print(trainer.train())
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
    env = QuadLocEnv(dataDir='/home/Pearl/quantm/RL_env/data/', num=500)
    num = env.action_space.n
    # print("Action:", num)
    env.reset()
    return env

# env = QuadLocEnv(dataDir='data/', num=20)
register_env("QuadLocEnv-v0", env_creator)

# ray.init()
ray.init(num_cpus=4, num_gpus=1)
run_experiments({
        "demo": {
            "run": "A3C",
            "env": "QuadLocEnv-v0",
       		# "config": {
         #        "env_config": {
         #            "version": 0.1,
         #        },
         #    },
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
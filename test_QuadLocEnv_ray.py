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
    env = QuadLocEnv(dataDir='/home/Pearl/quantm/CVPR18/segmentation/data/train/', num=20)
    return env

env = QuadLocEnv(dataDir='/home/Pearl/quantm/CVPR18/segmentation/data/train/', num=20)
register_env("QuadLocEnv-v0", lambda: env)

ray.init()
run_experiments({
        "demo": {
            "run": "PPO",
            "env": "QuadLocEnv-v0",
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
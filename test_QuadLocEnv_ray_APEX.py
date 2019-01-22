import ray
import argparse
import os
import numpy as np
import gym
import tensorflow as tf

from ray import tune
from ray.tune.registry import register_env
from tensorflow import layers
from ray.rllib.models import ModelCatalog, Model
from ray.rllib.models.misc import flatten, normc_initializer



from tensorpack import *
NUM_ACTIONS = 4

class ConvNet2D(Model):
    def _build_layers(self, image, num_outputs=NUM_ACTIONS, options=None):
        image = tf.cast(image, tf.float32) / 255.0
        with argscope(Conv2D, activation=tf.nn.relu):
            l = Conv2D('conv0', image, 32, 5)
            l = MaxPooling('pool0', l, 2)
            l = Conv2D('conv1', l, 32, 5)
            l = MaxPooling('pool1', l, 2)
            l = Conv2D('conv2', l, 64, 4)
            l = MaxPooling('pool2', l, 2)
            l = Conv2D('conv3', l, 64, 3)

        l = FullyConnected('fc0', l, 512)
        l = PReLU('prelu', l)
        logits = FullyConnected('fc-pi', l, NUM_ACTIONS)    # unnormalized policy
        value = FullyConnected('fc-v', l, 1)
        return value #logits, l


ModelCatalog.register_custom_model("ConvNet2D", ConvNet2D)




#############
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

register_env("QuadLocEnv-v0", env_creator)



############
ray.init(use_raylet=True, redis_password=os.urandom(128).hex())
register_env("QuadLocEnv-v0", env_creator)

experiment_spec = {
    "custom_env": {
        "run": "APEX",
        "env": "QuadLocEnv-v0",
#             "restore": checkpoint,
        "config": {
            "model": {
                "custom_model": "ConvNet2D",
            },
        },
           "trial_resources":{
               "cpu": 24,
               "gpu": 1,
           },
        "checkpoint_freq": 10,
    },
}
tune.run_experiments(experiment_spec)
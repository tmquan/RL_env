from gym.envs.registration import register


register(
    id='ImageEnv-v0', 
    entry_point='gym_image.envs:ImageEnv', 
)

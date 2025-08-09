from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import gym
import random
import numpy as np

seed = 42
random.seed(seed)
np.random.seed(seed)

env_id = "airgym:airsim-academic-v3"
env = gym.make(env_id)


# 导入模型
model = PPO.load("logs/PPO_Academic_V1_0508_3D_take_angle_150000_steps.zip")


evaluate_policy(model, env, n_eval_episodes=20)
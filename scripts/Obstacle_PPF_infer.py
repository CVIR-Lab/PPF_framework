import os
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)

from stable_baselines3 import PPO
from stable_baselines3 import TD3
from stable_baselines3.common.env_util import make_vec_env
import gym
import os
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize
import gym
import random
import numpy as np
import torch           # PyTorch

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# 若你在训练时只是锁 cuDNN：
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
torch.use_deterministic_algorithms(True)


from stable_baselines3.common.monitor import Monitor


from stable_baselines3.common.vec_env import DummyVecEnv
from airgym.envs.AirSimAction import AirSimAction


def make_env(seed=SEED):
    def _init():
        env = AirSimAction(seed=seed)
        env.action_space.seed(seed)              # 锁 action sampling                     # 记录 episode 长度/回报
        return env
    return _init

eval_env = DummyVecEnv([make_env(SEED)])

# 导入模型
model = PPO.load(r"G:\PPF_Framework\scripts\logs\PPO_Pretrained_aca_v2_0603_obstacle_avoidance.zip")



from stable_baselines3.common.evaluation import evaluate_policy

mean_r, std_r = evaluate_policy(
    model,
    eval_env,
    n_eval_episodes=20,
    deterministic=True,       # ← 关键：π(a|s) 取均值动作
    render=False
)



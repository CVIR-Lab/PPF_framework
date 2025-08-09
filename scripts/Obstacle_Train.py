import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
import gym
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3 import PPO
from stable_baselines3 import TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
import gym
import os
import torch.nn as nn
import torch
from airgym.envs.AirSimPursuit import AirSimPursuit
from airgym.envs.AirSimAction import AirSimAction
from typing import Callable
from gym import Env


from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

import random


SEED = 42                                  # 你想用的固定种子
os.environ["PYTHONHASHSEED"] = str(SEED)   # 锁定 Python 哈希随机化（3.3+）

random.seed(SEED)          # 纯 Python 随机
np.random.seed(SEED)       # NumPy 随机
torch.manual_seed(SEED)    # CPU 上的 Torch
torch.cuda.manual_seed_all(SEED)   # GPU 上的 Torch（若用 GPU）

# 为了彻底可复现，关闭 cuDNN 探测和非确定性算法
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)   # PyTorch≥1.13


# 自定义回调，记录成功率
# class SuccessRateCallback(BaseCallback):
#     """
#     回调函数用于记录和计算训练过程中的成功率。
#     """
#     def __init__(self, verbose=0):
#         super(SuccessRateCallback, self).__init__(verbose)
#         self.success_counts = 0
#         self.episode_counts = 0
#         self.prev_success_epochs = 0
#         self.prev_total_epochs = 0
#
#     def _on_step(self) -> bool:
#         """
#         在每一步训练时调用。
#         """
#         if 'infos' in self.locals:
#             infos = self.locals['infos']
#             # 对于单环境环境，infos 是一个长度为1的列表
#             for info in infos:
#                 # 获取当前的累计成功回合数和总回合数
#                 current_success_epochs = info.get('success_epochs', 0)
#                 current_total_epochs = info.get('total_epochs', 0)
#
#                 # 计算自上次记录以来的新成功回合数和新总回合数
#                 new_success = current_success_epochs - self.prev_success_epochs
#                 new_total = current_total_epochs - self.prev_total_epochs
#
#                 # 更新累积计数
#                 self.success_counts += new_success
#                 self.episode_counts += new_total
#
#                 # 更新上次记录的计数
#                 self.prev_success_epochs = current_success_epochs
#                 self.prev_total_epochs = current_total_epochs
#
#             # 计算并记录成功率
#             if self.episode_counts > 0:
#                 success_rate = self.success_counts / self.episode_counts
#                 self.logger.record('successful_rate', success_rate)
#                 if self.verbose > 0:
#                     print(f"Step: {self.num_timesteps}, Success Rate: {success_rate:.2f}")
#
#         return True


class SuccessRateCallback(BaseCallback):
    def __init__(self, save_path, init_success=0, init_total=0,
                 save_every=10_000, verbose=0):
        super().__init__(verbose)
        self.success_counts = init_success
        self.episode_counts = init_total
        self.prev_success_epochs = 0
        self.prev_total_epochs = 0
        self.save_path = save_path
        self.save_every = save_every

    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [])
        for info in infos:
            cur_succ = info.get('success_epochs', 0)
            cur_total = info.get('total_epochs', 0)
            self.success_counts += cur_succ - self.prev_success_epochs
            self.episode_counts += cur_total - self.prev_total_epochs
            self.prev_success_epochs, self.prev_total_epochs = cur_succ, cur_total

        # TensorBoard 记录
        if self.episode_counts:
            self.logger.record(
                'successful_rate',
                self.success_counts / self.episode_counts
            )

        # 每 save_every 步把计数写入磁盘
        if self.n_calls % self.save_every == 0:
            np.savez(self.save_path,
                     success=self.success_counts,
                     total=self.episode_counts)
        return True






########################################################
#  新增: 可学习潜势网络训练回调 (DPBA)
#  每隔一定步数, 我们取出 info["raw_obs"] / info["distance"]
#  然后调用 env.update_potential_network(...) 来逼近 (-distance)
########################################################
# class PotentialTrainingCallback(BaseCallback):
#     def __init__(self, env, train_freq=2048, verbose=0):
#         """
#         :param env: 传入已经创建好的 AirSimPursuit 实例(或VecEnv单环境)
#         :param train_freq: 每收集多少条 (obs,distance) 后执行一次训练
#         """
#         super(PotentialTrainingCallback, self).__init__(verbose)
#         self.env = env
#         self.train_freq = train_freq
#         self.memory_obs = []
#         self.memory_targets = []
#
#     def _on_step(self) -> bool:
#         # 收集当前 step 产生的信息
#         infos = self.locals.get('infos', [])
#         for info in infos:
#             obs = info.get("raw_obs", None)
#             dist = info.get("distance", None)
#             if obs is not None and dist is not None:
#                 self.memory_obs.append(obs)
#                 self.memory_targets.append(-dist)
#
#         # 如果数据达到 train_freq，则执行一次潜势网络的训练
#         if len(self.memory_obs) >= self.train_freq:
#             # 1. 找到真实的底层环境 AirSimPursuit
#             raw_env = self.training_env
#             # 如果是 VecEnv，就取第一个子环境
#             if hasattr(raw_env, "envs"):
#                 raw_env = raw_env.envs[0]
#             # 如果有 Monitor 包装，再拆一次
#             if hasattr(raw_env, "env"):
#                 raw_env = raw_env.env
#
#             # 2. 调用底层环境的 update_potential_network
#             raw_env.update_potential_network(
#                 batch_obs=self.memory_obs,
#                 batch_targets=self.memory_targets
#             )
#             if self.verbose > 0:
#                 print(f"DPBA: potential net updated on {len(self.memory_obs)} samples.")
#
#             # 3. 清空缓存
#             self.memory_obs.clear()
#             self.memory_targets.clear()
#
#         return True

# 创建环境
# env_id = "airgym:airsim-academic-v1"  # 替换为您的环境

def make_env(seed: int = 0) -> Callable[[], Env]:
    def _init() -> Env:          # ← 明确告诉分析器返回 Env
        env = AirSimAction(seed=seed)
        env.action_space.seed(seed)
        return env
    return _init

env = DummyVecEnv([make_env(SEED)])
env = VecMonitor(env, filename=None)

# env = gym.make(env_id)
# 然后获取真正的 AirSimPursuit 实例
# airsim_env = env.unwrapped  # 若未包 DummyVecEnv/Monitor 等，可直接这样

# 配置 TensorBoard 日志
log_dir = "./logs/"
os.makedirs(log_dir, exist_ok=True)

MODEL_PATH = os.path.join(
    log_dir, "PPO_Pretrained_aca_v2_0603_obstacle_avoidance_975000_steps.zip"
)

TB_RUN_NAME = "PPO_58"                   # 你 TensorBoard 里已有的子目录
EXTRA_STEPS = 1000000                   # 想追加的 timesteps


counter_file = os.path.join(log_dir, "success_counter.npz")
if os.path.exists(counter_file):
    data = np.load(counter_file)
    init_succ = int(data['success'])
    init_total = int(data['total'])
else:
    init_succ = init_total = 0

success_rate_callback = SuccessRateCallback(
    save_path=counter_file,
    init_success=init_succ,
    init_total=init_total,
    save_every=10_000
)




# 定义策略参数
policy_kwargs = dict(
    net_arch=[dict(pi=[128, 128], vf=[128, 128])],
    activation_fn=nn.ReLU
)

# 创建 PPO 模型
model = PPO("MlpPolicy", env, seed=SEED, n_steps=2048, policy_kwargs=policy_kwargs,verbose=1,tensorboard_log=log_dir, batch_size=128,device='cuda')


#接续训练PPO

# model: PPO = PPO.load(
#     MODEL_PATH,
#     env=env,                      # 必须传入 env，才能继续 .learn()
#     device="cuda",                # 跟之前一致
#     tensorboard_log=log_dir       # 仍然写到同一个 logs 根目录
# )

# model = TD3('CnnPolicy', env, verbose=1,1
#                 learning_starts=2000,
#                 batch_size=128,
#                 train_freq=(200, 'step'),
#                 gradient_steps=200,
#                 tensorboard_log=log_dir,
#                 buffer_size=50000, seed=0,device='cuda')

# 设置训练回调
checkpoint_callback = CheckpointCallback(save_freq=5000, save_path=log_dir, name_prefix="PPO_Pretrained_aca_v2_obstacle_avoidance")


# 这里新增潜势网络训练回调 (DPBA)
# potential_training_callback = PotentialTrainingCallback(env, train_freq=2048, verbose=1)

# 开始训练
total_timesteps =1_000_000
# model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, success_rate_callback, potential_training_callback], progress_bar=True)
model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, success_rate_callback], progress_bar=True)


#接续训练

# model.learn(
#     total_timesteps=EXTRA_STEPS,
#     callback=[checkpoint_callback, success_rate_callback],
#     progress_bar=True,
#     reset_num_timesteps=False,   # 关键！从 455 000 继续累计
#     tb_log_name=TB_RUN_NAME      # 关键！写进同一目录，曲线连在一起
# )



# 保存模型
model.save(os.path.join(log_dir, "PPO_Pretrained_aca_v2_obstacle_avoidance"))

# 保存潜势网络权重
# if isinstance(airsim_env, AirSimPursuit):
    # torch.save(airsim_env.potential_network.state_dict(), "05_21_potential_net.pth")




# 关闭环境
env.close()
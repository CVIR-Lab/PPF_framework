import airsim
import numpy as np
import gym
from gym import spaces
from PIL import Image
import math
import time
import cv2
from dynamics.QuadcopterDrone import QuadcopterDrone
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# #####################################################
# 1) 新增: 潜势网络定义 (DPBA所需)
# #####################################################
class PotentialNetwork(nn.Module):
    def __init__(self, state_dim=7, hidden_dim=64):
        super(PotentialNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)  # 输出标量(潜势)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)  # shape: (batch_size, 1)
        return out



class AirSimUAVPursuit(gym.Env):
    def __init__(self):
        super(AirSimUAVPursuit, self).__init__()

        # 连接到 AirSim
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True,"Drone1")
        self.client.armDisarm(True)

        # 使用MultirotorDynamicsAirsim定义的动作和状态空间
        self.dynamics = QuadcopterDrone()
        self.action_space = self.dynamics.action_space

        #start position
        start_position = [0, 0, 1]

        self.dynamics.set_start(start_position, random_angle=math.pi * 2)

        # self.work_space_x = [-4, 34]
        # self.work_space_y = [-20, 20]
        # self.work_space_z = [0.5, 8]

        self.max_episode_steps = 1000
        self.max_depth_meters = 20
        self.screen_height = 144
        self.screen_width = 256


        #这里以字典为观测空间，目的在于把图像与数据进行融合
        # self.observation_space = spaces.Dict({"image": spaces.Box(low=0, high=255, shape=(self.screen_height, self.screen_width, 1), dtype=np.uint8),
        #                                     "state": spaces.Box(low=np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf]), high=np.array([np.inf, np.inf, np.inf, np.inf,np.inf, np.inf]), dtype=np.float32)})

        '''[self.target_angel,self.target_z_angel, self.yaw, self.distance]'''
        self.observation_space = spaces.Box(low=np.array([-np.inf, -np.inf, -np.inf, -np.inf]),
                                            high=np.array([np.inf, np.inf, np.inf, np.inf]),
                                            dtype=np.float32)


        self.x_position = self.dynamics.get_position()[0]
        self.y_position = self.dynamics.get_position()[1]
        self.z_position = self.dynamics.get_position()[2]

        self.target_x_position = self.dynamics.get_targetposition()[0]
        self.target_y_position = self.dynamics.get_targetposition()[1]
        self.target_z_position = self.dynamics.get_targetposition()[2]

        self.distance = self.dynamics.get_distance()
        self.x_distance = self.dynamics.get_x_distance()
        self.y_distance = self.dynamics.get_y_distance()
        self.z_distance = self.dynamics.get_z_distance()

        self.target_angel = self.dynamics.get_target_angel()
        # self.yaw = self.dynamics.getPitchRollYaw()[2]

        self.yaw = self.dynamics.getyaw()
        self.target_z_angel = self.dynamics.get_z_angel()


        self.trajectory_list = []
        self.episodes = 0

        self.current_step = 0

        # 初始化前一时刻的距离为None
        self.previous_distance = None

        # 初始化前一时刻的朝向目标无人机的角度为None
        self.previous_target_angel = None

        # 初始化前一时刻的朝向目标无人机的势能
        self.previous_energy = None

        self.total_rewards = 0

        self.altitude = self.dynamics.get_altitude()

        self.total_epochs = 0
        self.success_epochs = 0

        # 定义一个折扣因子 gamma（若只想简单测试可设为 1.0）(PBRS)
        self.gamma = 0.99
        # 用来保存上一时刻的势函数值 \Phi(s)，初始化为 None(PBRS)

        ######################################################
        # 2) 新增: DPBA 相关变量
        #    - 创建潜势网络 + 优化器
        #    - 设置折扣因子 gamma
        #    - previous_potential 用于存储上一时刻潜势值
        ######################################################
        self.potential_network = PotentialNetwork(state_dim=4, hidden_dim=64)
        self.potential_optimizer = optim.Adam(self.potential_network.parameters(), lr=1e-3)



    def reset(self):

        # reset state
        self.dynamics.reset()

        # state_info = np.array([self.target_angel, self.yaw, self.distance])  # 获取额外状态信息

        state_info = [self.target_angel, self.target_z_angel, self.yaw, self.distance]

        # state_info = np.array([self.x_position, self.y_position, self.z_position, self.target_x_position, self.target_y_position, self.target_z_position, self.yaw])  # 获取额外状态信息

        obs = state_info

        self.current_step = 0
        self.previous_distance = None  # 重置前一时刻的距离

        self.previous_target_angel = None# 重置前一时刻的角度
        self.previous_energy = None  # 重置前一时刻的势能

        # 重置时势函数也要清空(PBRS)
        self.previous_potential = None


        return obs


    def render(self, mode='human'):
        pass



    def step(self, action):

        self.dynamics.set_action(action, execute=True)
        # 更新位置信息
        position_ue4 = self.dynamics.get_position()
        self.trajectory_list.append(position_ue4)

        # 更新距离
        self.distance = self.dynamics.get_distance()
        self.target_z_angel = self.dynamics.get_z_angel()
        # 更新角度
        self.target_angel = self.dynamics.get_target_angel()

        self.yaw = self.dynamics.getyaw()

        self.altitude = self.dynamics.get_altitude()

        self.x_position = self.dynamics.get_position()[0]
        self.y_position = self.dynamics.get_position()[1]
        self.z_position = self.dynamics.get_position()[2]

        self.target_x_position = self.dynamics.get_targetposition()[0]
        self.target_y_position = self.dynamics.get_targetposition()[1]
        self.target_z_position = self.dynamics.get_targetposition()[2]


        # state_info = np.array([self.target_angel, self.yaw, self.distance])  # 更新额外状态信息

        state_info = [self.target_angel, self.target_z_angel, self.yaw, self.distance]
        # state_info = np.array([self.x_position, self.y_position, self.z_position, self.target_x_position, self.target_y_position, self.target_z_position, self.yaw])  # 更新额外状态信息
        obs = state_info

        # print("obs:", obs)
        # print('target_angel:', self.target_angel)
        # print('yaw:', self.yaw)
        # print('distance:', self.distance)

        # 初始化 reward
        reward = 0

        done = False
# PBRS
#         current_potential = -self.distance  # 这里的势函数就是负距离
#
#         # 如果这是本回合第一次执行 step，则 previous_potential 还没有值，
#         # 可以直接赋值，避免产生 NoneType 的差值
#         if self.previous_potential is None:
#             self.previous_potential = current_potential
#
#         # 计算 shaping 奖励
#         shaping_reward = self.gamma * (current_potential - self.previous_potential)
#
#         # 加到原有环境奖励
#         reward += shaping_reward
#
#         # 更新 previous_potential
#         self.previous_potential = current_potential


# 计算势能奖励
#         k = 160
#
#         pi = math.pi
#
#         theta = self.target_z_angel
#
#         fai = self.target_angel
#
#         d = self.distance
#
#         if -90 <= fai <= 0 or 0 < fai <= 90:
#             positive_real_roots = self.calculate_PP_positive((270 - theta) / 180 * pi, (fai - 180) / 180 * pi)
#             r = positive_real_roots
#         else:
#             positive_real_roots = self.calculate_PP_negative((270 - theta) / 180 * pi, (fai - 180) / 180 * pi)
#             r = -positive_real_roots
#
#         Lr = d / r
#
#         energy = k / Lr
#
#         # Lr_z = Lr * math.cos((270 - theta) / 180 * pi)
#
#         if self.previous_energy is not None:
#             Energy_pp_reward = energy - self.previous_energy
#             reward += Energy_pp_reward
#
#         self.previous_energy = energy


#实际上是水滴场的终止条件
        # if 0 < d/r <= 0.5:
        #     reward += 50
        #     print('d/r=', d / r)
        #     done = True
        #     self.total_epochs += 1
        #     success = True
        #     self.success_epochs += 1
#势能奖励结束

#计算距离奖励
        # '''根据距离的变化来计算奖励'''
        # reward_factor = 3
        # epsilon = 1e-6
        # distance_reward = 0
        # if self.previous_distance is not None and self.previous_distance > 0:
        #     distance_change = self.previous_distance - self.distance
        #     relative_change = distance_change / (self.previous_distance + epsilon)
        #
        #     if distance_change > 0:
        #         # 使用tanh限制奖励范围
        #         distance_reward += math.tanh(relative_change) * reward_factor
        #     else:
        #         # 使用tanh限制惩罚范围
        #         distance_reward -= math.tanh(-relative_change) * reward_factor
        #
        #     # 可选：归一化奖励
        #     distance_reward = max(min(distance_reward, reward_factor), -reward_factor)
        # reward += distance_reward
        # # 更新前一时刻的距离为当前距离
        # self.previous_distance = self.distance
#距离奖励结束
        if self.distance <= 3:
            reward += 50
            done = True
            self.total_epochs += 1
            self.success_epochs += 1

        if self.current_step > 1000:
            reward -= 30
            done = True
            self.total_epochs += 1

        # 2) =============== DPBA 整形奖励 ===============
        current_potential = self.compute_potential_value(obs)
        if self.previous_potential is None:
            # 若是本回合第一步, 没有前一时刻潜势, 则先初始化
            self.previous_potential = current_potential

        shaping_reward = self.gamma * (current_potential - self.previous_potential)
        reward += shaping_reward
        self.previous_potential = current_potential
        # ===============================================


        # 3) 在 info 中保留成功率数据 + 追加 raw_obs, distance 等
        info = {
            "success_epochs": self.success_epochs,
            "total_epochs": self.total_epochs,
            "raw_obs": obs,  # 用于在回调中采样 (负距离监督)
            "distance": self.distance  # ditto
        }



        self.total_rewards += reward
        self.current_step += 1

        # 设置 info 字典，传递当前的成功回合数和总回合数（DPBA等）
        # info = {
        #     "success_epochs": self.success_epochs,
        #     "total_epochs": self.total_epochs
        # }


        return obs, reward, done, info

        ######################################################
        # 3) 提供一个更新潜势网络的方法 (供外部回调调用)
        #    用 "负距离" 作为训练目标 (示例: DPBA)
        ######################################################
    def update_potential_network(self, batch_obs, batch_targets):
        """
        batch_obs: list/ndarray of shape [N, 3]  (obs = [target_angel, yaw, distance])
        batch_targets: list/ndarray of shape [N]  (期望的潜势值, 这里 -distance)
        """
        if len(batch_obs) == 0:
            return

        obs_tensor = torch.tensor(batch_obs, dtype=torch.float32)
        tgt_tensor = torch.tensor(batch_targets, dtype=torch.float32).unsqueeze(-1)

        pred = self.potential_network(obs_tensor)
        loss = F.mse_loss(pred, tgt_tensor)

        self.potential_optimizer.zero_grad()
        loss.backward()
        self.potential_optimizer.step()
    #
    #     ######################################################
    #     # 4) 计算当前观测 obs 对应的潜势数值
    #     ######################################################
    def compute_potential_value(self, obs):
        """
        给定当前状态 obs=[target_angel, yaw, distance],
        返回潜势网络输出的标量
        """
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            val = self.potential_network(obs_tensor)
            return val.item()



    def calculate_PP_positive(self, theta, fai):
        a = 1
        b = -2 * math.sin(theta) * math.cos(fai)
        c = 5 - (17 * (math.sin(theta) ** 2) * (math.cos(fai) ** 2))
        d = -14 * math.sin(theta) * math.cos(fai)
        e = -5

        roots = np.roots([a, b, c, d, e])

        real_roots = [root for root in roots if np.isreal(root)]

        # Convert to real floats if the imaginary part is 0
        positive_real_roots = [np.real(root).item() for root in real_roots if root.imag == 0 and root.real > 0]

        return positive_real_roots[0]

    def calculate_PP_negative(self, theta, fai):
        a = 1
        b = 2 * math.sin(theta) * math.cos(fai)
        c = 5 - (17 * (math.sin(theta) ** 2) * (math.cos(fai) ** 2))
        d = 14 * math.sin(theta) * math.cos(fai)
        e = -5

        roots = np.roots([a, b, c, d, e])

        real_roots = [root for root in roots if np.isreal(root)]

        # Convert to real floats if the imaginary part is 0
        positive_real_roots = [np.real(root).item() for root in real_roots if root.imag == 0 and root.real > 0]

        return positive_real_roots[0]



    def is_done(self):
        episode_done = False

        is_not_inside_workspace_now = self.is_not_inside_workspace()

        too_close_to_obstable = self.is_crashed()

        distance_get_goal = self.disbool()

        angel_not_in = self.calculate_angle_inside()

        z_angel_not_in = self.calculate_z_angle_inside()



        # We see if we are outside the Learning Space
        episode_done = is_not_inside_workspace_now or \
            too_close_to_obstable or \
            distance_get_goal or\
            angel_not_in or\
            z_angel_not_in


        return episode_done

    def calculate_angle_inside(self):
        not_in_angel = False
        angel_info = self.target_angel
        if -180< angel_info < -90 or 90< angel_info <180:
            not_in_angel = True

        return not_in_angel

    def calculate_z_angle_inside(self):
        z_not_in_angel = False
        angel_z_info = self.target_z_angel
        if -180 < angel_z_info <= -90 or 90 <= angel_z_info <180:
            z_not_in_angel = True

        return z_not_in_angel




    def disbool(self):
        disbool = False
        distance = self.distance
        if distance < 2.5:
            disbool = True


        return disbool




    def close(self):
        # 关闭环境
        self.client.enableApiControl(False)


    def is_crashed(self):
        is_crashed = False
        collision_info = self.client.simGetCollisionInfo()
        if collision_info.has_collided:
            is_crashed = True

        return is_crashed

    def is_not_inside_workspace(self):
        """
        Check if the Drone is inside the Workspace defined
        """
        is_not_inside = False
        current_position = self.dynamics.get_position()

        if current_position[0] < self.work_space_x[0] or current_position[0] > self.work_space_x[1] or \
            current_position[1] < self.work_space_y[0] or current_position[1] > self.work_space_y[1] or \
                current_position[2] < self.work_space_z[0] or current_position[2] > self.work_space_z[1]:
            is_not_inside = True

        return is_not_inside



    def calculate_angle_reward(self, current_angle):
        angle_limit = 45  # 角度限制
        angle_reward = 0  # 初始化奖励
        done = False

        # 如果角度在±45度范围内
        if -angle_limit <= current_angle <= angle_limit:
            # 接近0度时给予正奖励，越接近0度奖励越大
            angle_reward = (angle_limit - abs(current_angle)) / angle_limit
        else:
            # 角度超出限制，给予负奖励并结束回合
            angle_reward = -1
            done = True

        # 你可以调整奖励的比例以满足你的需要
        return angle_reward, done

if __name__ == '__main__':
    AirSimUAVPursuit= AirSimUAVPursuit()
    # print(AirSimPursuit.target_angel)
    #
    # for i in range(10):
    #     print(AirSimPursuit.target_z_angel)
    # print(AirSimPursuit.yaw)

    # a = AirSimUAVPursuit.reset()
    # # 确保观测值是正确的类型
    # print("Initial observation type:", type(a))
    # print("Initial observation shape:",
    #       a.shape if isinstance(a, np.ndarray) else "Not an ndarray")





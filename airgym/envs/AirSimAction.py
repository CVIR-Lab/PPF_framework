import airsim
import numpy as np
import gym
from gym import spaces
from PIL import Image
import math
import copy
import time
import cv2
from dynamics.QuadcopterDrone import QuadcopterDrone
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from typing import Optional, List, Dict
import random
from gym.utils.seeding import np_random

import matplotlib.pyplot as plt



PRETRAIN_PATH = r"G:\PPF_Framework\scripts\logs\PPO_Pretrained_aca_v1_3dim.zip"
PRETRAIN_MODEL = PPO.load(PRETRAIN_PATH, device="cpu", print_system_info=False)
PRETRAIN_MODEL.policy.eval()
for p in PRETRAIN_MODEL.policy.parameters():
    p.requires_grad_(False)


class AirSimAction(gym.Env):
    def __init__(self, seed: Optional[int] = None):
        super(AirSimAction, self).__init__()

        self.device = torch.device('cuda')

        # 连接到 AirSim
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True,"Drone1")
        self.client.armDisarm(True)


        #RGB图像
        # self.image_request = airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
        self.image_request1 = airsim.ImageRequest("0", airsim.ImageType.DepthVis, True)
        self.image_request2 = airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)

        # #深度图像
        # self.image_request = airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True)

        # 使用MultirotorDynamicsAirsim定义的动作和状态空间
        self.dynamics = QuadcopterDrone()
        self.action_space = self.dynamics.action_space

        #start position
        start_position = [0, 0, 1]

        self.dynamics.set_start(start_position, random_angle=math.pi * 2)

        self.work_space_x = [-4, 34]
        self.work_space_y = [-20, 20]
        self.work_space_z = [0.5, 8]

        # self.crash_distance = 1
        # self.accept_radius = 2

        self.max_depth_meters = 20
        self.screen_height = 128
        self.screen_width = 128

        self.seed(seed)


        self.theta_tadd1 = 0

        self.observation_space = spaces.Box(low=np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf]),
                                            high=np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]),
                                            dtype=np.float32)

        # self.env_id = "airgym:airsim-academic-v1"  # 替换为您的环境
        # self.env = gym.make(self.env_id)
        # self.model = PPO.load("G:\\PPField_A\\scripts\\logs\\PPO_Academic_V1_10000_steps.zip")

        # ---------- 第一阶段模型 ----------

        self.pre_model = PRETRAIN_MODEL

        #----------------------------------


        self.distance = self.dynamics.get_distance()

        #获取激光雷达的最小最大角度
        self.detect_obstacles_min_angle, self.detect_obstacles_max_angle = self.dynamics.detect_obstacles()

        self.target_angel = self.dynamics.get_target_angel()
        self.yaw = self.dynamics.getPitchRollYaw()[2]
        self.target_z_angel = self.dynamics.get_z_angel()


        self.trajectory_list = []
        self.episodes = 0

        self.current_step = 0

        # 初始化前一时刻的朝向目标无人机的角度为None
        self.previous_target_angel = None

        # 初始化前一时刻的朝向目标无人机的势能
        self.previous_energy = None

        self.total_rewards = 0

        self.altitude = self.dynamics.get_altitude()



        #模型评估
        # model_weights_path = 'G:\\Interactive_Learning_change\\scripts\\cnnlogs\\model_weights_0924_125000.pth'  # 替换为实际的模型路径
        # self.CameraConvNet.load_state_dict(torch.load(model_weights_path))

        # self.model_copy = copy.deepcopy(self.CameraConvNet)

        self.Cnnoutput = 0

        self.last_obs = 0

        self.pre_position = [0, 0, 0]

        self.pre_kesai = 0

        self.fai = 0
        self.pre_yaw = 0
        self.position = 0


        self.num_count = 0

        self.yaw_predict = 0

        self.energy_predicted = 0
        self.pre_energy_predicted = 0
        self.energy_now = 0

        self.test = np.random.uniform(1, 5)

        self.step_count = 0

        self.total_epochs = 0
        self.success_epochs = 0


        #tesT


    def seed(self, seed: Optional[int] = None) -> List[int]:
        """
        Gym 旧接口：返回 [seed]；Gymnasium ≥0.28 亦兼容。
        """
        self.np_random, seed = np_random(seed)  # Gym 自带 RNG
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        return [seed]




    def reset(self):
        # reset state
        self.step_count = 0
        self.total_rewards = 0
        self.dynamics.reset()

        '''强化学习1的部分开始'''
        self.distance = self.dynamics.get_distance()
        self.target_angel = self.dynamics.get_target_angel()
        self.yaw = self.dynamics.getPitchRollYaw()[2]
        self.detect_obstacles_min_angle, self.detect_obstacles_max_angle = self.dynamics.detect_obstacles()

        reset_obs = np.array([self.target_angel, self.yaw, self.distance])
        # print(reset_obs)

        # === 2) 用一阶段模型推理下一步动作 ===
        with torch.no_grad():
            pred_action, _ = self.pre_model.predict(reset_obs, deterministic=True)

        next_pose = self.dynamics.simulate_action(pred_action)
        yaw_pred = next_pose["yaw"]
        angel_predicted = self.dynamics.get_predicted_target_angle(
            {"x": next_pose["x"], "y": next_pose["y"]}
        )

        # 使用模型预测动作
        # action_predict, _states = self.model.predict(reset_obs, deterministic=True)
        #
        # position_yaw_predict = self.dynamics.simulate_action(action_predict) #获取预测的新位置
        #
        # new_position = {
        #     "x": position_yaw_predict["x"],
        #     "y": position_yaw_predict["y"]
        # }

        # angel_predicted = self.dynamics.get_predicted_target_angle(new_position)

        # 预估的相差角
        self.theta_tadd1 = angel_predicted
        '''强化学习1的部分结束'''
        '''预测出了动作，下一时刻预估的夹角'''

        # 获取并保存预测的yaw值
        # self.yaw_predict = position_yaw_predict["yaw"]  #当前时刻预估出来的t+1yaw

        self.yaw_predict = yaw_pred

        # 获取状态信息
        state_info = np.array([self.yaw, self.yaw_predict, self.detect_obstacles_min_angle,
                               self.detect_obstacles_max_angle, self.target_angel,
                               angel_predicted, self.distance])

        obs = state_info


        self.current_step = 0
        self.previous_distance = None  # 重置前一时刻的距离

        self.previous_target_angel = None # 重置前一时刻的角度
        self.previous_energy = None  # 重置前一时刻的势能

        self.trajectory_list = []

        self.pre_yaw = self.yaw #TODO
        self.pre_position = self.position

        return obs


    def merge_rgb_and_depth(self,rgb_scene, depth_image):
        # 检查深度图像，确保它是归一化的
        depth_image_normalized = depth_image / np.max(depth_image)

        # 验证图像数据有效性
        if rgb_scene is None or depth_image is None:
            print("Invalid image data received. Resetting environment.")
            return None  # 或者选择一个合适的默认值

        # 确保图像数据形状正确
        if rgb_scene.shape[:2] != depth_image.shape[:2]:
            print("Image shape mismatch. Resetting environment.")
            return None  # 或者选择一个合适的默认值

        # 将深度图像调整为[144,256,1]形状
        depth_image_reshaped = depth_image_normalized.reshape(depth_image.shape[0], depth_image.shape[1], 1)

        # 将深度图像作为第四个通道添加到RGB图像上
        combined_image = np.concatenate((rgb_scene, depth_image_reshaped), axis=2)

        return combined_image


    def transform_RGB(self, responses):
        response = responses[0]
        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)  # 将字节数据转换为1D数组
        img_rgb = img1d.reshape(response.height, response.width, 3)  # 重塑数组为3通道图像
        img_normalized = img_rgb.astype(np.float32) / 255.0  # 归一化到0-1

        # 裁剪图像为128x128x3
        target_height, target_width = 128, 128
        img_resized = cv2.resize(img_normalized, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        img_normalized = img_resized


        return img_normalized

    def preprocess_image(self, img):
        # 转换颜色通道顺序 HWC 到 CHW
        img_transposed = np.transpose(img, (2, 0, 1))  # (3, 128, 128)

        # 转换为torch tensor
        img_tensor = torch.from_numpy(img_transposed).float()

        # 添加批次维度
        img_tensor = img_tensor.unsqueeze(0)  # (1, 3, 128, 128)

        return img_tensor

    def get_depth_image(self,responses):


        depth_img = airsim.list_to_2d_float_array(
            responses[0].image_data_float, responses[0].width,
            responses[0].height)

        depth_meter = depth_img * 100

        return depth_meter

    def get_obs_image(self,responses):
        # Normal mode: get depth image then transfer to matrix with state
        # 1. get current depth image and transfer to 0-255  0-20m 255-0m
        image = self.get_depth_image(responses)  # 0-6550400.0 float 32
        image_resize = cv2.resize(image, (self.screen_width,self.screen_height))
        # switch 0 and 255
        image_scaled = np.clip(image_resize, 0, self.max_depth_meters) / self.max_depth_meters * 255


        image_normalized = image_scaled / 255.0

        return image_normalized


    def get_obs_depthimage(self,depth_image):
        # 检查深度图像，确保它是归一化的
        depth_image_normalized = depth_image / np.max(depth_image)

        # 将深度图像调整为[144,256,1]形状
        depth_image_reshaped = depth_image_normalized.reshape(depth_image.shape[0], depth_image.shape[1], 1)

        return depth_image_reshaped


    def render(self, mode='human'):
        if mode == 'rgb_array':

            depth_image = self.client.simGetImages([self.image_request1])
            depth_image = self.get_obs_image(depth_image)
            depth_image = self.get_obs_depthimage(depth_image)
            # print(depth_image.shape)

            rgb_scene = self.client.simGetImages([self.image_request2])
            rgb_scene = self.transform_RGB(rgb_scene)
            # print(rgb_scene.shape)
            # observation = self.merge_rgb_and_depth(rgb_scene, depth_image)

            observation = depth_image

            # return self.transform_image(responses)
            return observation

        else:
            # 其他渲染模式的处理
            pass

    def frozen_copy(self):
        self.model_copy.load_state_dict(self.CameraConvNet.state_dict())
        for param in self.model_copy.parameters():
            param.requires_grad = False



    def step(self, action):


        '''动作分割'''

        self.dynamics.set_action(action)
        '''动作分割'''
        # print("-----step start-----")
        self.num_count += 1

        # 更新位置信息
        position_ue4 = self.dynamics.get_position()
        self.trajectory_list.append(position_ue4)

        # 更新距离
        self.distance = self.dynamics.get_distance()
        self.target_z_angel = self.target_z_angel

        # 更新角度
        self.target_angel = self.dynamics.get_target_angel()

        self.yaw = self.dynamics.getPitchRollYaw()[2] #弧度制  #当前时刻

        self.detect_obstacles_min_angle, self.detect_obstacles_max_angle = self.dynamics.detect_obstacles()
        # print(self.detect_obstacles_min_angle, self.detect_obstacles_max_angle)
        #更新位置
        self.position = self.dynamics.get_position()

        self.current_obs = np.array([self.target_angel, self.yaw, self.distance])
        # print("current obs:", self.current_obs)


        with torch.no_grad():
            pred_action, _ = self.pre_model.predict(self.current_obs, deterministic=True)

        next_pose = self.dynamics.simulate_action(pred_action)
        yaw_pred = next_pose["yaw"]
        angel_predicted = self.dynamics.get_predicted_target_angle(
            {"x": next_pose["x"], "y": next_pose["y"]}
        )

        # 使用模型预测动作
        # action_predict, _states = self.model.predict(self.current_obs, deterministic=True)
        #
        # position_yaw_predict = self.dynamics.simulate_action(action_predict)  # 获取预测的新位置
        #
        # new_position = {
        #     "x": position_yaw_predict["x"],
        #     "y": position_yaw_predict["y"]
        # }
        # print("predict_position:", new_position)

        # angel_predicted = self.dynamics.get_predicted_target_angle(new_position)

        # print("angel_predicted:", angel_predicted)

        # 预估的相差角
        self.theta_tadd2 = angel_predicted


        # 初始化 reward
        reward = 0
        done = False
        #
        # if self.yaw < self.detect_obstacles_min_angle or self.yaw > self.detect_obstacles_max_angle:
        #     if self.is_crashed():
        #         angel_change = 3 * math.exp(-abs(self.yaw - self.yaw_predict))
        #         reward -= self.total_rewards
        #         reward -= 100 * angel_change
        #         done = True
        #
        #     else:
        #         angel_change = 3 * math.exp(-abs(self.yaw - self.yaw_predict)) - 2.1
        #         reward += 0.1 * angel_change
        #
        # elif self.yaw >= self.detect_obstacles_min_angle and self.yaw <= self.detect_obstacles_max_angle:

        if self.is_crashed():
            angel_change = math.exp(-abs(self.yaw - self.yaw_predict))
            reward -= self.total_rewards
            reward -= 100 * angel_change
            done = True
            self.total_epochs += 1
        else:
            angel_change = 3 * math.exp(-abs(self.yaw - self.yaw_predict)) - 2.1
            reward += 0.1 * angel_change


        if self.current_step > 1000:
            reward -= self.total_rewards
            reward -= 100
            done = True
            self.total_epochs += 1

        if self.distance < 2.5:
            # 当与目标距离小于2.5时
            reward += 100 - (0.1 * self.step_count)
            done = True
            self.total_epochs += 1
            self.success_epochs += 1

        self.total_rewards += reward
        self.current_step += 1


        # if self.num_count % 5000 == 0:
        #     # 保存模型权重
        #     torch.save(self.CameraConvNet.state_dict(), f'G:\\Interactive_Learning_change\\scripts\\cnnlogs\\model_weights_0924_{self.num_count}.pth')

        # 获取并保存预测的yaw值
        self.yaw_predict = yaw_pred
        # self.yaw_predict = position_yaw_predict["yaw"]
        # print("predict_yaw:", self.yaw_predict)

        state_info = np.array([
            self.yaw,
            self.yaw_predict,
            self.detect_obstacles_min_angle,
            self.detect_obstacles_max_angle,
            self.target_angel,
            angel_predicted,
            self.distance
        ])

        obs = state_info

        # self.dynamics.slove_lidar_data()

        # 设置 info 字典，传递当前的成功回合数和总回合数（DPBA等）
        info = {
            "success_epochs": self.success_epochs,
            "total_epochs": self.total_epochs
        }

        return obs, reward, done, info


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
    AirSimAction = AirSimAction()

    # print(AirSimPursuit.target_angel)
    #
    # for i in range(10):
    #     print(AirSimPursuit.target_z_angel)
    # print(AirSimPursuit.yaw)

    # AirSimAction.reset()



import airsim
import numpy as np
import math
from gym import spaces
from sklearn.cluster import DBSCAN
import csv


class QuadcopterDrone():

    def __init__(self) -> None:

        # config
        '''这部分是是否使用三维，如果改为False，就是连续二维动作'''
        self.navigation_3d = False

        self.using_velocity_state = False
        self.dt = 0.1

        # AirSim Client
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        # start and goal position
        self.start_position = [0, 0, 0]
        self.start_random_angle = None

        # states
        self.x = 0
        self.y = 0
        self.z = 0
        self.v_xy = 0
        self.v_z = 0
        self.yaw = 0
        self.yaw_rate = 0

        # cmd
        self.v_xy_sp = 0
        self.v_z_sp = 0
        self.yaw_rate_sp = 0
        '''
           self.v_xy_sp：这个变量可能代表水平方向（X-Y平面）的速度设定点（setpoint）。它被初始化为0，意味着初始时刻没有水平方向的速度要求。

           self.v_z_sp：这个变量可能代表垂直方向（Z轴）的速度设定点。它同样被初始化为0，表示初始时刻没有垂直方向的速度要求。

           self.yaw_rate_sp：这个变量可能代表偏航角速度的设定点。偏航角是指物体围绕垂直轴的旋转角度，因此这个变量控制着旋转速度。它被初始化为0，意味着初始时刻没有旋转速度要求。
        '''



        # action space
        self.acc_xy_max = 2.0
        # self.v_xy_max = 5
        # self.v_xy_min = 2

        # self.v_xy_max = 7
        # self.v_xy_min = 3

        # self.v_xy_max = 7.5
        # self.v_xy_min = 3
        # self.v_z_max = 2.0

        self.v_xy_max = 8
        self.v_xy_min = 4
        self.v_z_max = 3.0
        # self.v_z_max = 1.0
        self.yaw_rate_max_deg = 30.0
        self.yaw_rate_max_rad = math.radians(self.yaw_rate_max_deg)
        self.max_vertical_difference = 5

        if self.navigation_3d:
            if self.using_velocity_state:
                self.state_feature_length = 6
            else:
                self.state_feature_length = 3
            self.action_space = spaces.Box(low=np.array([self.v_xy_min, -self.v_z_max, -self.yaw_rate_max_rad]),
                                           high=np.array([self.v_xy_max, self.v_z_max, self.yaw_rate_max_rad]),
                                           dtype=np.float32)
        else:
            if self.using_velocity_state:
                self.state_feature_length = 4
            else:
                self.state_feature_length = 2
            self.action_space = spaces.Box(low=np.array([self.v_xy_min, -self.yaw_rate_max_rad]),
                                           high=np.array([self.v_xy_max, self.yaw_rate_max_rad]),
                                           dtype=np.float32)
    def reset(self):
        self.client.reset()

        # reset start
        # yaw_noise = self.start_random_angle * np.random.random()

        '''这一部分就是开局根据yaw noise来设定朝着的方向的，'''


        # set airsim pose
        pose = self.client.simGetVehiclePose()
        pose.position.x_val = self.start_position[0]
        pose.position.y_val = self.start_position[1]
        pose.position.z_val = - self.start_position[2]
        # pose.orientation = airsim.to_quaternion(0, 0, yaw_noise)

        self.client.simSetVehiclePose(pose, True)

        self.client.simPause(False)
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        # take off
        self.client.moveToZAsync(-self.start_position[2], 2).join()

        self.client.simPause(True)

    def set_action(self, action, execute=True):

        self.v_xy_sp = action[0] * 0.7
        self.yaw_rate_sp = action[-1] * 2
        if self.navigation_3d:
            self.v_z_sp = float(action[1])
        else:
            self.v_z_sp = 0

        self.yaw = self.get_attitude()[2]
        self.yaw_sp = self.yaw + self.yaw_rate_sp * self.dt

        if self.yaw_sp > math.radians(180):
            self.yaw_sp -= math.pi * 2
        elif self.yaw_sp < math.radians(-180):
            self.yaw_sp += math.pi * 2

        vx_local_sp = self.v_xy_sp * math.cos(self.yaw_sp)
        vy_local_sp = self.v_xy_sp * math.sin(self.yaw_sp)

        if execute:
            self.client.simPause(False)
            if len(action) == 2:
                self.client.moveByVelocityZAsync(vx_local_sp, vy_local_sp, -self.start_position[2], self.dt,
                                                 drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                                                 yaw_mode=airsim.YawMode(is_rate=True,
                                                                         yaw_or_rate=math.degrees(self.yaw_rate_sp))).join()
                # self.client.moveByVelocityZAsync(vx_local_sp, vy_local_sp, -self.start_position[2], self.dt,
                #                                 drivetrain=airsim.DrivetrainType.ForwardOnly,
                #                                 yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=math.degrees(0))).join()
            elif len(action) == 3:
                self.client.moveByVelocityAsync(vx_local_sp, vy_local_sp, -self.v_z_sp, self.dt,
                                                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                                                yaw_mode=airsim.YawMode(is_rate=True,
                                                                        yaw_or_rate=math.degrees(self.yaw_rate_sp))).join()

            self.client.simPause(True)

    def simulate_action(self, action):

        current_position = self.client.simGetVehiclePose().position
        self.v_xy_sp = action[0] * 0.7
        self.yaw_rate_sp = action[-1] * 2
        if self.navigation_3d:
            self.v_z_sp = float(action[1])
        else:
            self.v_z_sp = 0

        self.yaw = self.get_attitude()[2]
        self.yaw_sp = self.yaw + self.yaw_rate_sp * self.dt

        if self.yaw_sp > math.radians(180):
            self.yaw_sp -= math.pi * 2
        elif self.yaw_sp < math.radians(-180):
            self.yaw_sp += math.pi * 2

        # Calculate the simulated local velocities
        vx_local_sp = self.v_xy_sp * math.cos(self.yaw_sp)
        vy_local_sp = self.v_xy_sp * math.sin(self.yaw_sp)

        # Assume the drone moves in this direction for the duration of `self.dt`
        # 使用正确的属性访问方法
        simulated_new_position = {
            "x": current_position.x_val + vx_local_sp * self.dt,
            "y": current_position.y_val + vy_local_sp * self.dt,
            "z": current_position.z_val + self.v_z_sp * self.dt, # 注意：这里假设 z_val 是向上为正，向下为负
            "yaw": self.yaw_sp  # Add the new yaw to the output
        }

        return simulated_new_position

    def detect_obstacles(self, lidar_name='LidarSensor1', vehicle_name='Drone1', distance_threshold=5.0):
        # 获取激光雷达数据
        lidar_data = self.client.getLidarData(lidar_name=lidar_name, vehicle_name=vehicle_name)

        # 检查数据有效性
        if len(lidar_data.point_cloud) < 3:
            # print("未收到激光雷达数据")
            return 180, 180

        # 将点云数据转换为 numpy 数组
        points = np.array(lidar_data.point_cloud, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 3), 3))

        # 计算每个点的距离和水平角度
        distances = np.linalg.norm(points, axis=1)
        angles = np.degrees(np.arctan2(points[:, 1], points[:, 0]))

        # 筛选出距离小于阈值的点（障碍物点）
        obstacle_indices = np.where(distances <= distance_threshold)
        obstacle_angles = angles[obstacle_indices]

        if len(obstacle_angles) == 0:
            # print("未检测到障碍物")
            return 180, 180

        # 将角度转换为弧度，并用于聚类
        obstacle_angles_rad = np.radians(obstacle_angles)
        X = obstacle_angles_rad.reshape(-1, 1)
        clustering = DBSCAN(eps=np.radians(5), min_samples=5).fit(X)
        labels = clustering.labels_

        # 分析聚类结果
        unique_labels = set(labels)
        max_span = 0
        cluster_min_angle = float('inf')
        cluster_max_angle = float('-inf')
        for label in unique_labels:
            if label == -1:
                continue
            class_member_mask = (labels == label)
            cluster_angles = obstacle_angles[class_member_mask]

            cluster_min_angle = np.min(cluster_angles)
            cluster_max_angle = np.max(cluster_angles)

            # print(f"检测到障碍物，角度范围为 {cluster_min_angle:.2f} 度 到 {cluster_max_angle:.2f} 度")
            max_span = max(max_span, cluster_max_angle - cluster_min_angle)
        return cluster_min_angle, cluster_max_angle

    def get_predicted_target_angle(self, simulated_position):
        # Get the target position from the simulation environment or configuration
        goal = self.client.simGetObjectPose("BP_DJI_2").position

        pitch, roll, yaw = self.getPitchRollYaw()
        yaw = math.degrees(yaw)

        # Calculate the angle to the target from the simulated position
        pos_angle = math.atan2(goal.y_val - simulated_position["y"], goal.x_val - simulated_position["x"])
        pos_angle = math.degrees(pos_angle) % 360

        track = math.radians(pos_angle - yaw)
        angel = ((math.degrees(track) - 180) % 360) - 180

        return angel


    def set_start(self, position, random_angle):
        self.start_position = position
        self.start_random_angle = random_angle

    def get_position(self):
        position = self.client.simGetVehiclePose().position
        return [position.x_val, position.y_val, -position.z_val]


    def get_targetposition(self):
        target = self.client.simGetObjectPose("BP_DJI_2").position
        return [target.x_val, target.y_val, -target.z_val]


    def get_distance(self):

        position = self.client.simGetVehiclePose().position
        target = self.client.simGetObjectPose("BP_DJI_2").position

        position_x = position.x_val
        position_y = position.y_val
        position_z = position.z_val

        Tar_x = target.x_val
        Tar_y = target.y_val
        Tar_z = target.z_val

        distance = math.sqrt((Tar_x-position_x)**2 + (Tar_y-position_y) **2 + (Tar_z-position_z)**2 )

        return distance


    def get_altitude(self):

        position = self.client.simGetVehiclePose().position

        altitude = -position.z_val


        return altitude


    def get_velocity(self):
        states = self.client.getMultirotorState()
        linear_velocity = states.kinematics_estimated.linear_velocity
        angular_velocity = states.kinematics_estimated.angular_velocity

        velocity_xy = math.sqrt(pow(linear_velocity.x_val, 2) + pow(linear_velocity.y_val, 2))
        velocity_z = linear_velocity.z_val
        yaw_rate = angular_velocity.z_val

        return [velocity_xy, -velocity_z, yaw_rate]

    def get_attitude(self):
        self.state_current_attitude = self.client.simGetVehiclePose().orientation
        return airsim.to_eularian_angles(self.state_current_attitude)

    def get_attitude_cmd(self):
        return [0.0, 0.0, self.yaw_sp]

    def get_x_distance(self):
        position = self.client.simGetVehiclePose().position
        target = self.client.simGetObjectPose("BP_DJI_2").position

        position_x = position.x_val


        Tar_x = target.x_val


        x_distance = math.sqrt((Tar_x - position_x) ** 2)

        return x_distance


    def get_y_distance(self):
        position = self.client.simGetVehiclePose().position
        target = self.client.simGetObjectPose("BP_DJI_2").position

        position_y = position.y_val


        Tar_y = target.y_val


        y_distance = math.sqrt((Tar_y - position_y) ** 2)

        return y_distance

    def get_z_distance(self):
        position = self.client.simGetVehiclePose().position
        target = self.client.simGetObjectPose("BP_DJI_2").position

        position_z = -position.z_val


        Tar_z = -target.z_val


        z_distance = math.sqrt((Tar_z - position_z) ** 2)

        return z_distance

    def get_target_angel(self):

        pos = self.client.simGetObjectPose('Drone1').position
        goal = self.client.simGetObjectPose("BP_DJI_2").position
        # Test = self.client.simGetObjectPose("DJIDrone_2").position

        '''   "BP_DJI_2"    "DJIDrone_2"   '''

        pitch, roll, yaw = self.getPitchRollYaw()
        yaw = math.degrees(yaw)

        # 由于Vector3r对象不可通过索引访问，我们使用.x_val, .y_val, .z_val属性访问其坐标
        pos_angle = math.atan2(goal.y_val - pos.y_val, goal.x_val - pos.x_val)
        pos_angle = math.degrees(pos_angle) % 360

        # 计算从当前位置到目标位置的相对方向角
        track = math.radians(pos_angle - yaw)
        angel = ((math.degrees(track) - 180) % 360) - 180

        return angel




    def getPitchRollYaw(self):

        my_ori = self.client.simGetObjectPose('Drone1').orientation

        pitch, roll, yaw = self.eularian_angles(my_ori)

        return pitch, roll, yaw

    def getyaw(self):

        my_ori = self.client.simGetObjectPose('Drone1').orientation
        yaw = self.eularian_angles(my_ori)[2]
        yaw = math.degrees(yaw)

        return yaw



    def eularian_angles(self,q):
        z = q.z_val
        y = q.y_val
        x = q.x_val
        w = q.w_val
        ysqr = y * y

        # roll (x-axis rotation)
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + ysqr)
        roll = math.atan2(t0, t1)

        # pitch (y-axis rotation)
        t2 = +2.0 * (w * y - z * x)
        if (t2 > 1.0):
            t2 = 1
        if (t2 < -1.0):
            t2 = -1.0
        pitch = math.asin(t2)

        # yaw (z-axis rotation)
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (ysqr + z * z)
        yaw = math.atan2(t3, t4)

        return (pitch, roll, yaw)



    def get_z_angel(self):
        # 获取目标无人机和当前无人机的位置
        target_position = self.client.simGetObjectPose("BP_DJI_2").position
        my_position = self.client.simGetObjectPose('Drone1').position

        '''   "DJIDrone_2"      "BP_DJI_2"  '''

        # 计算两个无人机之间的位置差
        delta_x = target_position.x_val - my_position.x_val
        delta_y = target_position.y_val - my_position.y_val
        delta_z = target_position.z_val - my_position.z_val

        # 计算水平距离（在地面平面上的距离）
        horizontal_distance = math.sqrt(delta_x ** 2 + delta_y ** 2)

        # 使用atan2计算垂直角度，这里atan2的好处是可以处理delta_z和horizontal_distance中的任何一个为零的情况
        vertical_angle = math.atan2(delta_z, horizontal_distance)

        # 将角度从弧度转换为度
        vertical_angle_degrees = math.degrees(vertical_angle)

        return vertical_angle_degrees



    def save_obstacle_positions(self, csv_path="obstacles.csv"):
        """
        将场景里所有指定障碍物的 (x, y, z) 中心坐标一次性写入 CSV。
        不返回值；只做文件写入。
        """
        # ① 手动列出所有障碍物名称（一次写全）
        obstacle_names = [
            'SM_Palm_A_01_23', 'SM_Palm_A_47', 'SM_Palm_A_02_32', 'SM_Palm_A_44', 'SM_Palm_A_03_17',
            'SM_Palm_A_29', 'SM_Palm_A_04_5', 'SM_Palm_A_41', 'SM_Palm_A_50', 'SM_Palm_A_53',
            'SM_Palm_A_56', 'SM_Palm_A_65', 'SM_Palm_A_77', 'SM_Palm_A_83', 'SM_Palm_A_86',
            'SM_Palm_A_101', 'SM_Palm_A_155', 'SM_Palm_A_158', 'SM_Palm_A_163',
            'SM_Palm_B_01_8', 'SM_Palm_B_14', 'SM_Palm_B_02_11', 'SM_Palm_B_20', 'SM_Palm_B_03_35',
            'SM_Palm_B_38', 'SM_Palm_B_26', 'SM_Palm_B_59', 'SM_Palm_B_134', 'SM_Palm_B_68',
            'SM_Palm_B_74', 'SM_Palm_B_80', 'SM_Palm_B_89', 'SM_Palm_B_98',
            'SM_SwampTree_71', 'SM_SwampTree_02_33', 'SM_SwampTree_107',
            'SM_Tree_Large_10', 'SM_Tree_Large_110', 'SM_Tree_Large_113',
            'SM_Tree_Large_149', 'SM_Tree_Large_152',
            'SM_Tree_Medium_01_95', 'SM_Tree_Medium_146', 'SM_Tree_Medium_02_92',
            'SM_Tree_Medium_137', 'SM_Tree_Medium_140', 'SM_Tree_Medium_143'
        ]

        # ② 若你想给半径，统一写成 1.0；后期手动改也行
        default_r = 1.0

        # ③ 查询并写 CSV
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["name", "x", "y", "z", "r"])  # 表头
            for name in obstacle_names:
                pose = self.client.simGetObjectPose(name)
                if pose.position:  # 若名字找不到会给默认 Pose
                    p = pose.position
                    writer.writerow([name,
                                     float(p.x_val),
                                     float(p.y_val),
                                     float(p.z_val),
                                     default_r])
        print(f"✓ {csv_path} written with {len(obstacle_names)} rows")



if __name__ == '__main__':
    QuadcopterDrone = QuadcopterDrone()
    # print(MultirotorDynamicsAirsim.yaw_rate_max_rad)


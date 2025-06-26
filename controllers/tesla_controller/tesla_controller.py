from controller import Supervisor
import numpy as np
import socket
import json
import os
from statistics import variance as var 

class CustomCarEnv:
    
    def __init__(self):

        self.robot = Supervisor()
        self.timestep = int(self.robot.getBasicTimeStep())

        #==== Timestep management ====
        self.global_steps = 0
        self.udr_start_steps = 200_000 # Threshold to start UDR (e.g. 1M steps)
        # ============================

        #===== Motor setup =====
        self.left_motor = self.robot.getDevice('left_rear_wheel')
        self.right_motor = self.robot.getDevice('right_rear_wheel')

        self.front_left_steer = self.robot.getDevice('left_steer')
        self.front_right_steer = self.robot.getDevice('right_steer')

        self.max_speed = 52 #rad/s
        self.max_back_speed = 0 #rad/s

        self.max_steering = 0.6 #rad

        self.norm_max = 1
        self.norm_min = 0

        if self.front_left_steer is None or self.front_right_steer is None:
            print("ERROR: Steering actuators not found.")
            exit()

        #===== Lidar management =====
        self.lidar_front = self.robot.getDevice('lidar_front')

        if self.lidar_front is not None:
            self.lidar_front.enable(self.timestep)
            self.lidar_front.enablePointCloud()
        else:
            print("WARNING: Front lidar not found.")
        
        # self.lidar_front.getMaxRange() # DEBUG

        self.collision_th = 1.0
        #==========================

        self.max_timesteps = 2000 # Adjusted for a shorter 100m track
        self.curr_timestep = 0
        self.curr_episode = 0

        #===== Road parameters =====
        self.road_length = 120
        self.road_width = 12

        #===== Supervisor: get reference to Tesla node =====
        self.car_node = self.robot.getFromDef("tesla3")
        self.translation_field = self.car_node.getField('translation')
        self.rotation_field = self.car_node.getField('rotation')
        self.default_car_pos = self.translation_field.getSFVec3f()

        if self.left_motor is None or self.right_motor is None:
            print("ERROR: Motors not found.")
            exit()

        # Set motors to velocity control mode
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        self.front_left_steer.setPosition(0.0)
        self.front_right_steer.setPosition(0.0)

        self.target_node = self.robot.getFromDef('target')
        self.target_translation = self.target_node.getField('translation')
        self.target_pos = self.target_translation.getSFVec3f()

        self.distance_target_threshold = 3.0

        print(f'Initial target coordinates: {self.target_pos}') #DEBUG

        self.target = {
            'node':self.target_node,
            'translation': self.target_translation,
            'rotation':self.target_node.getField('rotation')
        }
        #==================

        #===== Sensors =====
        self.gps = self.robot.getDevice('gps')
        if self.gps is not None:
            self.gps.enable(self.timestep)
        else:
            print("WARNING: GPS not found.")

        self.imu = self.robot.getDevice('inertial unit')
        if self.imu is not None:
            self.imu.enable(self.timestep)
        else:
            print("WARNING: Inertial Unit not found.")
        self.max_angle_roll_pitch_yaw = np.pi
        #========================

        #===== Reward handling =====
        self.total_reward = 0.0
        self.reward_print_interval = 50   # Print every 50 steps
        #=========================

        #====== Anti-Blockage mechanism =====
        self.block_counter = 0
        self.max_block_steps = 50
        self.last_pos = None
        self.block_movement_threshold = 0.01   # Minimum distance to consider movement
        self.min_speed_threshold = 0.05    # Avg speed under which car is considered almost stopped

        # === Domain Randomization setup ===
        self.enable_domain_randomization = True 
        self.num_obstacles = 6
        self.obstacle_nodes = [] # List to store obstacle nodes

        # Get obstacle node references from Webots world
        for i in range(0, self.num_obstacles): 
            obstacle_node = self.robot.getFromDef(f'ostacolo_{i}')
            if obstacle_node:
                self.obstacle_nodes.append(obstacle_node)
            else:
                print(f"WARNING: Obstacle OBSTACLE_{i} not found. Create the DEFs in your Webots world.")
        # -----------------------------------------------

        # WANDB-logging
        self.episode_trajectory = []
        self.last_50_results = []
        self.episode_counters = {
            'target_reached': 0,
            'collision': 0,
            'timeout': 0,
            'falling': 0,
            'blocked': 0
        }

        #------------------------------------------------

        self.reset() # Initial reset to set up the environment
    
    def step(self, action):
        # wandb custom logging
        info = {}

        avg_speed = action[0] * self.max_speed
        self.left_motor.setVelocity(avg_speed)
        self.right_motor.setVelocity(avg_speed)

        # Set steering angle for front wheels
        self.front_left_steer.setPosition(action[1] * self.max_steering) 
        self.front_right_steer.setPosition(action[1] * self.max_steering)
        
        if self.robot.step(self.timestep) == -1:
            print("Simulation interrupted during step.")
            return None, 0.0, True, {}

        obs = self._get_obs()
        reward = self._compute_reward(obs)

        self.total_reward += reward
        self.global_steps += 1

        if self.curr_timestep % self.reward_print_interval == 0:
            print(f"[{self.curr_timestep}] Cumulative reward episode {self.curr_episode}: {self.total_reward:.2f}")

        done, cause = self._check_done(obs)
        self.curr_timestep += 1

        # ------- WANDB Step-level logging -------
        pos = self.gps.getValues()
        target_coords = self.target['translation'].getSFVec3f()
        raw_dist = np.linalg.norm([target_coords[0] - pos[0], target_coords[1] - pos[1]])
        proximity = np.exp(-raw_dist)

        self.episode_trajectory.append((pos[0], pos[1]))

        car_pos = obs[1:3] * [self.road_length, self.road_width]
        target_pos = obs[7:9] * [self.road_length, self.road_width]

        front_lidar = obs[10:20]  # normalized lidar
        norm_thresh = self.collision_th / self.lidar_front.getMaxRange()
        collision_flag = int(np.any(front_lidar < norm_thresh))
        proximity_to_obstacle = np.min(front_lidar) * self.lidar_front.getMaxRange()

        info.update({
            "episode": int(self.curr_episode),
            "final_distance_to_target": float(raw_dist),
            "mean_proximity_reward": float(proximity),
            "proximity_to_obstacle": float(proximity_to_obstacle),
            "avg_speed": float(avg_speed),
            "reward_step": float(reward),
            "collision_flag": int(collision_flag)
        })
        #-----------------------------------------------------
        info["episode"] = {
                "r": float(self.total_reward),
                "l": int(self.curr_timestep)
        }

        if done:
            print(f"[END][{cause}] Episode {self.curr_episode} ended with cumulative reward: {self.total_reward:.2f}\n")
            print("==========================")
            
            self.curr_episode += 1

            if len(self.episode_trajectory) > 1:
                distance_covered = np.linalg.norm(np.array(self.episode_trajectory[-1]) - np.array(self.episode_trajectory[0]))
                traj_x, traj_y = zip(*self.episode_trajectory)
                traj_var = var(traj_x) + var(traj_y)
            else:
                distance_covered = 0.0
                traj_var = 0.0

            efficiency = self.total_reward / max(1, self.curr_timestep)

            self.last_50_results.append(cause)
            if self.curr_episode > 0 and self.curr_episode % 50 == 0:
                self.successes_last_50 = self.last_50_results.count("target_reached")
                info.update({
                    "successes_every_50": int(self.successes_last_50)
                },step=self.curr_episode)
                self.last_50_results = []

            info.update({
                "distance_covered": float(distance_covered),
                "trajectory_variance": float(traj_var),
                "efficiency_score": float(efficiency),
                "episode_result": cause
            })

            self.episode_trajectory = []

            if cause in self.episode_counters:
                self.episode_counters[cause] += 1

                info.update({
                    "episode_type/target_reached": self.episode_counters['target_reached'],
                    "episode_type/collision": self.episode_counters['collision'],
                    "episode_type/timeout": self.episode_counters['timeout'],
                    "episode_type/falling": self.episode_counters['falling'],
                    "episode_type/blocked": self.episode_counters['blocked'],
                })
            # =======================================================================

            obs = self.reset()
            return obs, reward, True, info

        return obs, reward, False, info


    def _get_obs(self):
        #===== Speed management =====
        left_velocity = self.left_motor.getVelocity()
        right_velocity = self.right_motor.getVelocity()
        avg_velocity = 0.5 * (left_velocity + right_velocity)

        avg_velocity_norm = avg_velocity / self.max_speed
        avg_velocity_norm = np.array([avg_velocity_norm], dtype=np.float32)

        #===== GPS position management =====
        pos = np.array(self.gps.getValues() if self.gps else [0.0, 0.0, 0.0], dtype=np.float32)
        pos_norm = np.copy(pos)

        # Normalize GPS values for observation space
        pos_norm[0] /= self.road_length
        pos_norm[1] /= self.road_width/2

        #===== IMU (Roll, Pitch, Yaw) management =====
        imu = np.array(self.imu.getRollPitchYaw() if self.imu else [0.0, 0.0, 0.0], dtype=np.float32)

        # Normalize angular values (in radians) for observation space
        orientation_obs_norm = np.copy(imu)
        orientation_obs_norm[0] /= self.max_angle_roll_pitch_yaw  # Roll
        orientation_obs_norm[1] /= self.max_angle_roll_pitch_yaw/2  # Pitch
        orientation_obs_norm[2] /= self.max_angle_roll_pitch_yaw  # Yaw

        #===== Lidar management =====
        lidar_front_values = np.array(self.lidar_front.getRangeImage(), dtype=np.float32)

        lidar_front_values[np.isinf(lidar_front_values)] = self.lidar_front.getMaxRange()

        num_samples = 10
        step = max(1, len(lidar_front_values) // num_samples)
        lidar_front_samples_obs = lidar_front_values[::step][:num_samples]
        lidar_front_samples_norm = lidar_front_samples_obs / self.lidar_front.getMaxRange()

        #===== Target position management =====
        target_coords = np.array(self.target['translation'].getSFVec3f(), dtype=np.float32)
        target_coords_norm = np.copy(target_coords)
        target_coords_norm[0] /= self.road_length
        target_coords_norm[1] /= (self.road_width / 2)
        target_coords_norm[2] /= 1.0

        # Observation vector concatenation
        obs_space = np.concatenate(
            [
            avg_velocity_norm,             # 1 value
            pos_norm,                      # 3 values
            orientation_obs_norm,         # 3 values
            target_coords_norm,           # 3 values
            lidar_front_samples_norm,     # 10 values
            ], dtype=np.float32)
        return obs_space

    def _compute_reward(self, obs):
        reward = 0.0

        # Initialize reward components for debugging
        reward_components = {
            'progress_reward': 0.0,
            'target_reached_reward': 0.0,
            'collision_penalty': 0.0,
            'falling_penalty': 0.0,
            'blocked_penalty': 0.0,
            'time_penalty': 0.0,
            'proximity_penalty': 0.0
        }

        avg_speed = obs[0] * self.max_speed

        # Denormalize Tesla position
        tesla_pos_norm = obs[1:4]
        tesla_pos = np.copy(tesla_pos_norm)
        tesla_pos[0] *= self.road_length
        tesla_pos[1] *= self.road_width

        orientation_norm = obs[4:7]
        orientation = np.copy(orientation_norm) * self.max_angle_roll_pitch_yaw

        target_norm = obs[7:10]
        target = np.copy(target_norm)
        target[0] *= self.road_length
        target[1] *= self.road_width

        lidars_norm = obs[10:20]
        lidars = np.copy(lidars_norm) * self.lidar_front.getMaxRange()

        current_distance_from_target = np.linalg.norm(target[:2] - tesla_pos[:2])  # Only X and Y

        # 1. Progress Reward
        if self.prev_distance is not None:
            distance_reduction = self.prev_distance - current_distance_from_target
            reward_components['progress_reward'] = distance_reduction * 15.0
        self.prev_distance = current_distance_from_target

        # 2. Target Reached Reward
        if current_distance_from_target < self.distance_target_threshold:
            reward_components['target_reached_reward'] = 100.0

        # 3. Collision Penalty
        min_distance_lidar = np.min(lidars)
        collision = (min_distance_lidar < self.collision_th and avg_speed > 0.1)
        if collision:
            reward_components['collision_penalty'] = -50.0

        # 4. Falling Penalty
        max_angle_for_fall = 0.2
        falling = abs(orientation[0]) > max_angle_for_fall or abs(orientation[1]) > max_angle_for_fall
        if falling:
            reward_components['falling_penalty'] = -75.0

        # 5. Blocked Penalty
        if self.block_counter > 0:
            reward_components['blocked_penalty'] = -0.75

        # 6. Speed Reward
        target_speed = 26
        speed_deviation = abs(avg_speed - target_speed)
        reward_components['speed_reward'] = -0.01 * speed_deviation

        # 8. Time Penalty
        reward_components['time_penalty'] = -0.01

        # 9. Proximity Penalty
        proximity_warning_dist_front = 1.5
        if not collision:
            if min_distance_lidar < proximity_warning_dist_front:
                proximity = proximity_warning_dist_front - min_distance_lidar
                reward_components['proximity_penalty'] += -10.0 * proximity

        # Final reward (before clipping)
        raw_reward = sum(reward_components.values())
        reward = np.clip(raw_reward, -50.0, 50.0)

        return reward

    def _check_done(self, obs):

        avg_speed = obs[0] * self.max_speed

        tesla_pos_norm = obs[1:4]
        tesla_pos = np.copy(tesla_pos_norm)
        tesla_pos[0] *= self.road_length
        tesla_pos[1] *= self.road_width

        orientation_norm = obs[4:7]
        orientation = np.copy(orientation_norm) * self.max_angle_roll_pitch_yaw

        target_norm = obs[7:10]
        target = np.copy(target_norm)
        target[0] *= self.road_length
        target[1] *= self.road_width

        lidars_norm = obs[10:20]
        lidars = np.copy(lidars_norm) * self.lidar_front.getMaxRange()

        current_distance_from_target = np.linalg.norm(target[:2] - tesla_pos[:2])

        cause = None

        #===== Collision check =====
        collision = np.min(lidars) < self.collision_th
        if collision:
            cause = 'collision'

        #===== Timeout check =====
        timeout = self.curr_timestep >= self.max_timesteps
        if timeout and cause is None:
            cause = 'timeout'

        #===== Target reached check =====
        target_reached = current_distance_from_target < self.distance_target_threshold
        if target_reached and cause is None:
            cause = 'target_reached'

        #===== Falling check =====
        max_angle_for_fall = 0.2
        falling = abs(orientation[0]) > max_angle_for_fall or abs(orientation[1]) > max_angle_for_fall
        if falling and cause is None:
            cause = 'falling'

        #===== Anti-block check =====
        if self.last_pos is not None:
            delta_movement = np.linalg.norm(tesla_pos - self.last_pos)
            if delta_movement < self.block_movement_threshold or abs(avg_speed) < self.min_speed_threshold:
                self.block_counter += 1
            else:
                self.block_counter = 0
        self.last_pos = tesla_pos

        is_blocked = self.block_counter > self.max_block_steps
        if is_blocked and cause is None:
            cause = 'blocked'

        done = collision or timeout or target_reached or falling or is_blocked

        return done, cause

    def reset(self):
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        self.front_left_steer.setPosition(0.0)
        self.front_right_steer.setPosition(0.0)

        self.translation_field.setSFVec3f(self.default_car_pos)
        self.rotation_field.setSFRotation([0, 1, 0, 0])

        self.prev_distance = None
        self.curr_timestep = 0
        self.total_reward = 0.0
        self.block_counter = 0
        self.last_pos = None

        if self.enable_domain_randomization and self.global_steps > self.udr_start_steps:
            self.udr()

        self.car_node.setVelocity([0, 0, 0, 0, 0, 0])  # Full physical reset of the car
        self.car_node.resetPhysics()

        for _ in range(50):  # Small wait to stabilize the simulator
            self.robot.step(self.timestep)

        print(f"Starting Episode {self.curr_episode}") 

        return self._get_obs()

    def udr(self):
        if not self.enable_domain_randomization:
            return

        # PHASE 1: Move all existing obstacles off the track
        for obstacle_node in self.obstacle_nodes:
            if obstacle_node:
                translation_field = obstacle_node.getField('translation')
                current_z = translation_field.getSFVec3f()[2] 
                idx = self.obstacle_nodes.index(obstacle_node)
                translation_field.setSFVec3f([1000.0 + idx * 10.0, 1000.0, current_z])
                rotation_field = obstacle_node.getField('rotation')
                if rotation_field:
                    rotation_field.setSFRotation([0, 1, 0, 0])  # Reset to default rotation

        # === Randomize car Y position ===
        random_y_offset = np.random.uniform(-self.road_width / 4, self.road_width / 4)
        new_car_pos = list(self.default_car_pos) 
        new_car_pos[1] += random_y_offset 
        self.translation_field.setSFVec3f(new_car_pos)

        # === Slightly rotate the car ===
        random_yaw_angle = np.random.uniform(np.deg2rad(-15), np.deg2rad(15))
        self.rotation_field.setSFRotation([0, 0, 1, random_yaw_angle])

        # === Randomize 3 obstacles on road ===
        if len(self.obstacle_nodes) >= 3:
            self.three_random_obstacles = np.random.choice(
                self.obstacle_nodes,
                3,
                replace=False
            )
        else:
            self.three_random_obstacles = np.array(self.obstacle_nodes)

        available_x_sections = [
            [20.0, 50.0],
            [55.0, 75.0],
            [80.0, 100.0]
        ]
        
        if len(available_x_sections) >= len(self.three_random_obstacles):
            shuffled_section_indices = np.random.permutation(len(available_x_sections))[:len(self.three_random_obstacles)]
        else:
            shuffled_section_indices = np.random.permutation(len(available_x_sections))

        for i, obstacle_node in enumerate(self.three_random_obstacles):
            if obstacle_node:
                obstacle_translation_field = obstacle_node.getField('translation')
                x_min, x_max = available_x_sections[shuffled_section_indices[i]]
                random_x = np.random.uniform(x_min, x_max)

                y_buffer = 2.0  # Keep obstacle away from road edges
                random_y = np.random.uniform(-self.road_width / 2 + y_buffer, self.road_width / 2 - y_buffer)
                
                z = obstacle_translation_field.getSFVec3f()[2]
                obstacle_translation_field.setSFVec3f([random_x, random_y, z])




def _recv_all(sock):
    buffer = b""
    while True:
        part = sock.recv(8192)  # Default was: 1024
        buffer += part
        if len(part) < 8192:
            break
    return buffer

if __name__ == '__main__':
    # --- Socket server for external RL communication ---
    HOST = '127.0.0.1'

    try:
        PORT = int(os.environ.get("PORT", 10000))
        print(f"[Tesla Controller] PORT = {PORT}")

    except (TypeError, ValueError):
        print("ERROR: Environment variable PORT is not set. Exiting.")
        exit()

    env = CustomCarEnv()

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((HOST, PORT))
            s.listen(1)
            print("Webots controller listening on port", PORT, "...")

            conn, addr = s.accept()
            with conn:
                print(f"Connected to: {addr}")
                while True:
                    data = _recv_all(conn)
                    
                    if not data:
                        print("Client disconnected.")
                        break

                    try:
                        msg = json.loads(data.decode())
                    except json.JSONDecodeError:
                        print(f"JSON decode error: {data.decode()}")
                        continue

                    if msg['cmd'] == 'reset':
                        obs = env.reset()
                        conn.send(json.dumps({'obs': obs.tolist()}).encode())

                    elif msg['cmd'] == 'step':
                        obs, reward, done, info = env.step(msg['action'])
                        conn.send(json.dumps({
                            'obs': obs.tolist(),
                            'reward': float(reward),
                            'done': bool(done),
                            'info': info
                        }).encode())

                    elif msg['cmd'] == 'exit':
                        print("Received 'exit' command.")
                        env.robot.simulationSetMode(0)
                        env.robot.simulationQuit(0)
                        
                        break

    except Exception as e:
        print(f"Socket server error: {e}")

    finally:
        print("Closing Webots controller.")

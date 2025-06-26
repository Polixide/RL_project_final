import numpy as np
import socket
import json
import gymnasium as gym
from gymnasium import spaces

class WebotsRemoteEnv(gym.Env):
    def __init__(self,port):
        super().__init__()

        self.host = '127.0.0.1'
        self.port = port
        self.conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.conn.connect((self.host, self.port))

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)

        '''
        OBSERVATION SPACE:
        0   : combined_velocity_norm
        1-3 : pos_x, pos_y, pos_z _norm
        4-6 : roll, pitch, yaw _norm
        7-9   : target_x_norm, target_y_norm, target_z_norm
        10-19: front_lidar_samples (10 values) _norm
        '''

        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0]), 
            high=np.array([1.0, 1.0]),   #[acceleration, rotation]
            dtype=np.float32
        )

    
    def step(self, action):
        msg = json.dumps({'cmd': 'step', 'action': action.tolist()}).encode()
        self.conn.send(msg)
        response = _recv_all(self.conn)

        data = json.loads(response.decode())
        obs = np.array(data['obs'], dtype=np.float32)
        reward = data['reward']
        done = data['done']
        info = data.get('info',{})

        return obs, reward, done , False , info

    def reset(self, seed=None, options=None):
        self.conn.send(json.dumps({'cmd': 'reset'}).encode())
        response = _recv_all(self.conn)
        
        data = json.loads(response.decode())
        obs = np.array(data['obs'], dtype=np.float32)
        return obs , {}

    def close(self):
        self.conn.send(json.dumps({'cmd': 'exit'}).encode())
        self.conn.close()

def _recv_all(sock):
    buffer = b""
    while True:
        part = sock.recv(8192) #initially it was: 1024
        if not part:
            break
        buffer += part
        if len(part) < 8192:
            break
    return buffer


import os
import sys
import copy
import time
import math
import torch
import pickle
import logging
import itertools
import numpy as np
from my_env import *
from network_init import *
from normalization import *
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple
from torch.utils.data.sampler import SubsetRandomSampler, BatchSampler

def global_target_pose(action, pos_and_heading):
    # 获取当前状态
    cur_x = pos_and_heading[0]
    cur_y = pos_and_heading[1]
    cur_heading = pos_and_heading[2]
    cur_speed = pos_and_heading[3]
    # m/s转换为km/h
    cur_speed = cur_speed * 3.6

    # 进行clip切割
    action[0] = np.clip(action[0],-1,1)
    action[1] = np.clip(action[1],-1,1)

    # 获取当前角度动作
    angle = action[1] * 0.1

    # 获取当前位移动作
    speed = cur_speed + action[0] * 2
    speed = np.clip(speed,0,1e10)
    magnitude = speed * 1000 / 3600 * 0.1

    # 新的全局朝向
    new_heading = cur_heading + angle
    new_heading = (new_heading + np.pi) % (2 * np.pi) - np.pi
    # 当前位置
    cur_coord = (cur_x + 1j * cur_y)
    # 新位置
    new_pos = cur_coord + magnitude * np.exp(1j * (new_heading + np.pi / 2))
    x_coord = np.real(new_pos)
    y_coord = np.imag(new_pos)
    # targetpose目标
    target_pose = np.array(
        [
            x_coord, 
            y_coord, 
            new_heading,
            0.1,
        ],
        dtype=object,
    )
    return target_pose

class Params():
    def __init__(self, actor_input_dim, critic_input_dim, action_dim, hidden_dim, heads_num, key_dim, value_dim, \
                 layer_num, mlp_hidden_dim, seq_len, test_env, save_path, eval_num, device, load_counter, \
                 actor_net_type, critic_net_type):
        self.actor_input_dim = actor_input_dim
        self.critic_input_dim = critic_input_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.heads_num = heads_num
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.layer_num = layer_num
        self.mlp_hidden_dim = mlp_hidden_dim
        self.seq_len = seq_len
        self.save_path = save_path
        self.device = device
        self.load_counter = load_counter
        self.actor_net_type = actor_net_type
        self.critic_net_type = critic_net_type
        self.test_env = test_env
        self.eval_num = eval_num


class test():
    def __init__(self, params):
        # 环境设定
        self.test_env = params.test_env
        # 网络维度
        self.actor_input_dim = params.actor_input_dim
        self.critic_input_dim = params.critic_input_dim
        self.action_dim = params.action_dim
        self.hidden_dim = params.hidden_dim
        self.heads_num = params.heads_num
        self.key_dim = params.key_dim
        self.value_dim = params.value_dim
        self.layer_num = params.layer_num
        self.mlp_hidden_dim = params.mlp_hidden_dim
        self.seq_len = params.seq_len
        # 保存路径
        self.save_path = params.save_path
        # 是否加载模型
        self.load_counter = params.load_counter
        # 添加Actor网络
        self.actor_net_type = params.actor_net_type
        self.critic_net_type = params.critic_net_type
        if params.actor_net_type == "mlp":
            self.actor_net = Mlp_Actor_model_action2(self.actor_input_dim, self.action_dim, params.device).to(device=params.device)
        elif params.actor_net_type == "cnn1":
            self.actor_net = CNN_Actor_model_action2(self.action_dim, params.device).to(device=params.device)
        # 预测次数
        self.eval_num = params.eval_num
        # 设备
        self.device = device
        
    # 选择动作
    def select_action(self, state, action=None, determine=False):
        with torch.no_grad():
            mu, sigma, action, log_prob, entropy = self.actor_net(state, action, determine=determine)
        return mu, sigma, action, log_prob, entropy

    # 加载参数
    def load_param(self):
        actor_path = os.path.join(self.save_path, "actor_net_" + str(self.counter) + ".pth")
        critic_path = os.path.join(self.save_path, "critic_net_" + str(self.counter) + ".pth")
        norm_path = os.path.join(self.save_path, "norm_" + str(self.counter) + ".pkl")
        self.actor_net.load_state_dict(torch.load(actor_path, map_location=self.device))
        self.critic_net.load_state_dict(torch.load(critic_path, map_location=self.device))

    def reshape_obs(self, obs):
        return_obs = []
        for i in range(self.env_num):
            if params.actor_net_type == "mlp":
                return_obs.append(list(np.squeeze(obs[i]["final_obs"],axis=0)))
            elif params.actor_net_type == "cnn1" or params.actor_net_type == "cnn2":
                return_obs.append(list(np.squeeze(obs[i]["rgb"],axis=0)))
        return_obs = np.array(return_obs, dtype=np.float32)
        return return_obs

    def get_action_obs(self, obs):
        return_obs = []
        for i in range(self.env_num):
            return_obs.append(list(np.squeeze(obs[i]["pos_and_heading"],axis=0)))
        return_obs = np.array(return_obs, dtype=np.float32)
        return return_obs

    def get_pose_action(self, action, action_state):
        target_pose_action = []
        for i in range(self.env_num):
            temp_target_pose_action = global_target_pose(action[i], action_state[i])
            target_pose_action.append(temp_target_pose_action)
        target_pose_action = np.array(target_pose_action, dtype=np.float32)
        return target_pose_action

    def clamp_action(self, output_action):
        # 尝试使用clamp切分 clamp0
        action_low = [0, 0, -1]
        action_high = [1, 1, 1]
        output_action = output_action.transpose(0,1)
        for i in range(self.action_dim):
            output_action[i] = torch.clamp(output_action[i], action_low[i], action_high[i])
        output_action = output_action.transpose(0,1)
        return output_action

    def eval_step(self, obs):
        if self.actor_net_type == "mlp":
            state = np.squeeze(obs["final_obs"],axis=0)
        elif self.actor_net_type == "cnn1" or self.actor_net_type == "cnn2":
            state = np.squeeze(obs["rgb"],axis=0)
        # actor_state = self.actor_state_norm(state, update=False)
        actor_state = torch.from_numpy(state).float().to(self.device)
        actor_state = torch.unsqueeze(actor_state,0)
        action_state = np.squeeze(obs["pos_and_heading"],axis=0)
        # 生成动作
        with torch.no_grad():
            mu, sigma, action, log_prob, entropy = self.select_action(actor_state, determine=False)
        action = global_target_pose(action[0].cpu().numpy(), action_state)
        return action

    def eval_episode(self):
        state = self.test_env.reset()
        all_reward = 0
        while True:
            action = self.eval_step(state)
            state, reward, done, info = self.test_env.step(action)
            all_reward += reward
            if done:
                break
        print("reward", all_reward)
        return all_reward

    def eval_episodes(self):
        return_rewards = []
        for i in range(self.eval_num):
            all_reward = self.eval_episode()
            return_rewards.append(all_reward)
        print('steps={}\t test_return_reward={:.5f}'.format(self.counter, np.mean(return_rewards)))
        return np.mean(return_rewards)




if __name__ == "__main__":
    ###################################
    # 环境参数设置
    ###################################
    # 模型参数
    actor_input_dim = 23
    action_dim = 3
    hidden_dim = 256
    heads_num = 2
    key_dim = 64
    value_dim = 64
    layer_num = 2
    mlp_hidden_dim = 512
    seq_len = 1083
    critic_input_dim = 23
    critic_hidden_dim = 1024
    # 场景列表
    scenarios = ["1_to_2lane_left_turn_c", "1_to_2lane_left_turn_t", "3lane_merge_single_agent", \
                 "3lane_cruise_single_agent", "3lane_overtake"]
    # 是否使用sumo的可视化
    sumo_gui =  True  # If False, enables sumo-gui display.
    # 观测尺寸
    img_meters = 50 
    # 图像尺寸
    img_pixels = 112 
    # 叠加观测数量
    num_stack = 3
    # 日志地址
    logdir = ""
    # 是否使用visdom
    visdom = False
    # 是否使用headless 与visdom相反
    headless = True
    # 使用场景号
    env_class = 0
    #############################################
    # 环境定义
    #############################################
    env_params = ENV_Params(scenarios, sumo_gui, img_meters, num_stack, img_pixels, logdir, visdom, headless)
    test_env = MY_ENV(env_params, env_class)
    # 保存文件路径
    path_name = 'tt_new'
    save_path = os.path.join(os.getcwd(), path_name)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    # 训练参数
    eval_num = 100
    # gpu设置
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    # 载入checkpoint
    load_counter = 250
    # actor_net_type: mlp/cnn1
    actor_net_type = "cnn1"
    critic_net_type = "cnn1"
    params = Params(actor_input_dim, critic_input_dim, action_dim, hidden_dim, heads_num, key_dim, value_dim, layer_num, \
                    mlp_hidden_dim, seq_len, test_env, save_path, eval_num, device, load_counter, actor_net_type, critic_net_type)
    my_test = test(params)
    my_test.eval_episodes()
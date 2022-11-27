import os
import sys
import copy
import time
import math
import torch
import pickle
import logging
import itertools
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
import numpy as np
from my_env import *
from network_init_new import *
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
    
    # 进行clip切割
    action[0] = np.clip(action[0],-1,1)
    action[1] = np.clip(action[1],-1,1)
    
    # 获取当前动作 最大72km/h
    angle = action[1] * 0.1
    magnitude = (action[0] + 1)
    # magnitude = speed * 1000 / 3600 * time_delta
    
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
    def __init__(self, env, save_path, generate_num, model_path, load_counter, actor_net_type, actor_input_dim, \
                 action_dim, heads_num, key_dim, value_dim, layer_num, mlp_hidden_dim, device, scenario_name):
        self.env = env
        self.save_path = save_path
        self.generate_num = generate_num
        self.model_path = model_path
        self.load_counter = load_counter
        self.actor_net_type = actor_net_type
        self.actor_input_dim = actor_input_dim
        self.action_dim = action_dim
        self.heads_num = heads_num
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.layer_num = layer_num
        self.mlp_hidden_dim = mlp_hidden_dim
        self.device = device
        self.scenario_name = scenario_name

class Image_generator():
    def __init__(self, params):
        self.env = params.env
        self.scenario_name = params.scenario_name
        self.save_path = os.path.join(params.save_path, self.scenario_name)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        self.generate_num = params.generate_num
        self.model_path = params.model_path
        self.load_counter = params.load_counter
        self.device = params.device
        # 添加Actor网络
        self.actor_net_type = actor_net_type
        if params.actor_net_type == "mlp":
            self.actor_net = Mlp_Actor_model_action_pose(params.actor_input_dim, params.action_dim, params.device).to(device=params.device)
        elif params.actor_net_type == "cnn1":
            self.actor_net = CNN_Actor_model_action_pose(params.action_dim, params.device).to(device=params.device)
        elif params.actor_net_type == "mix1":
            self.actor_net = Mix_Actor_model_action_pose(params.actor_input_dim, params.action_dim, params.device).to(device=params.device)
        elif params.actor_net_type == "mix2":
            self.actor_net = All_Mix_Actor_model_action_pose(params.actor_input_dim, params.action_dim, params.heads_num, params.key_dim, params.value_dim, params.layer_num, params.mlp_hidden_dim, params.device).to(device=params.device)
        # 加载
        self.load_param()
        # 保存计数
        self.pic_count = 0
        # 全部观测
        self.all_data = []

    # 选择动作
    def select_action(self, all_vector_state, rgb_state, together_vector_state, trans_vector_state, judge_state, action=None, determine=False):
        with torch.no_grad():
            mu, sigma, action, log_prob, entropy = self.actor_net(all_vector_state, rgb_state, together_vector_state, trans_vector_state, judge_state, action, determine=determine)
        return mu, sigma, action, log_prob, entropy

    # 加载参数
    def load_param(self):
        actor_path = os.path.join(self.model_path, "actor_net_" + str(self.load_counter) + ".pth")
        norm_path = os.path.join(self.model_path, "norm_" + str(self.load_counter) + ".pkl")
        self.actor_net.load_state_dict(torch.load(actor_path, map_location=self.device))

    def reshape_obs(self, obs):
        return_obs = []
        for i in range(self.env_num):
            if params.actor_net_type == "mlp":
                return_obs.append(list(np.squeeze(obs[i]["final_obs"],axis=0)))
            elif params.actor_net_type == "cnn1" or params.actor_net_type == "cnn2":
                # print("rgb shape",obs[i]["rgb"].shape)
                return_obs.append(list(np.squeeze(obs[i]["rgb"],axis=0)))
        return_obs = np.array(return_obs, dtype=np.float32)
        return return_obs
    
    def reshape_obs_new(self, obs):
        all_vector_obs = []
        rgb_obs = []
        togrther_vector_obs = []
        trans_vector_obs = []
        judge_obs = []
        for i in range(self.env_num):
            all_vector_obs.append(list(np.squeeze(obs[i]["brief_final_obs"],axis=0)))
            rgb_obs.append(list(np.squeeze(obs[i]["rgb"],axis=0)))
            togrther_vector_obs.append(list(np.squeeze(obs[i]["brief_together_obs"],axis=0)))
            trans_vector_obs.append(list(np.squeeze(obs[i]["brief_transformer_pos_and_heading_2d"],axis=0)))
            judge_obs.append(list(np.squeeze(obs[i]["brief_judge"],axis=0)))
        all_vector_obs = np.array(all_vector_obs, dtype=np.float32)
        rgb_obs = np.array(rgb_obs, dtype=np.float32)
        togrther_vector_obs = np.array(togrther_vector_obs, dtype=np.float32)
        trans_vector_obs = np.array(trans_vector_obs, dtype=np.float32)
        judge_obs = np.array(judge_obs, dtype=np.float32)
        return all_vector_obs, rgb_obs, togrther_vector_obs, trans_vector_obs, judge_obs

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

    def numpy_2_torch(self, all_vector_state, rgb_state, together_vector_state, trans_vector_state, judge_state):
        all_vector_state = torch.from_numpy(all_vector_state).float().to(self.device)
        rgb_state = torch.from_numpy(rgb_state).float().to(self.device)
        together_vector_state = torch.from_numpy(together_vector_state).float().to(self.device)
        trans_vector_state = torch.from_numpy(trans_vector_state).float().to(self.device)
        judge_state = torch.from_numpy(judge_state).float().to(self.device)
        return all_vector_state, rgb_state, together_vector_state, trans_vector_state, judge_state

    def state_unsqueeze(self, all_vector_state, rgb_state, together_vector_state, trans_vector_state, judge_state):
        all_vector_state = torch.unsqueeze(all_vector_state,0)
        rgb_state = torch.unsqueeze(rgb_state,0)
        together_vector_state = torch.unsqueeze(together_vector_state,0)
        trans_vector_state = torch.unsqueeze(trans_vector_state,0)
        judge_state = torch.unsqueeze(judge_state,0)
        return all_vector_state, rgb_state, together_vector_state, trans_vector_state, judge_state

    def eval_step(self, obs):
        all_vector_state = np.squeeze(obs["brief_final_obs"],axis=0)
        rgb_state = np.squeeze(obs["rgb"],axis=0)
        together_vector_state = np.squeeze(obs["brief_together_obs"],axis=0)
        trans_vector_state = np.squeeze(obs["brief_transformer_pos_and_heading_2d"],axis=0)
        judge_state = np.squeeze(obs["brief_judge"],axis=0)
        self.all_data.append(rgb_state)
        """
        temp_rgb_state = rgb_state.astype().transpose(1,2,0)
        temp_rgb_state = np.ascontiguousarray(temp_rgb_state)
        print("aaaaaaaaa")
        plt.imshow(temp_rgb_state)
        plt.show()
        print("bbbbbbbbb")
        while True:
            pass
        """

        self.pic_count += 1

        # 转为torch
        all_vector_state, rgb_state, together_vector_state, trans_vector_state, judge_state = self.numpy_2_torch(all_vector_state, rgb_state, together_vector_state, trans_vector_state, judge_state)
        # 增大维度
        all_vector_state, rgb_state, together_vector_state, trans_vector_state, judge_state = self.state_unsqueeze(all_vector_state, rgb_state, together_vector_state, trans_vector_state, judge_state)
        # 获取 action state
        action_state = np.squeeze(obs["pos_and_heading"],axis=0)
        # 生成动作
        with torch.no_grad():
            mu, sigma, action, log_prob, entropy = self.select_action(all_vector_state, rgb_state, together_vector_state, trans_vector_state, judge_state, determine=True)
        action = global_target_pose(action[0].cpu().numpy(), action_state)
        return action

    def eval_episode(self):
        state = self.env.reset()
        all_reward = 0
        while True:
            action = self.eval_step(state)
            state, reward, done, info = self.env.step(action)
            all_reward += reward
            if done:
                break
        return all_reward

    def generate(self):
        while self.pic_count < self.generate_num:
            self.eval_episode()
        np.save(self.save_path + "/data.npy", np.array(self.all_data))


if __name__ =="__main__":
    # 模型参数 100/45
    actor_input_dim = 45
    action_dim = 2
    hidden_dim = 256
    heads_num = 2
    key_dim = 64
    value_dim = 64
    layer_num = 2
    mlp_hidden_dim = 512
    ###################################
    # 环境参数设置
    ###################################
    # 场景列表
    scenarios = ["1_to_2lane_left_turn_c", "1_to_2lane_left_turn_t", "3lane_merge_single_agent", \
                 "3lane_cruise_single_agent", "3lane_overtake", "3lane_cut_in"]
    # 是否使用sumo的可视化
    sumo_gui =  False  # If False, enables sumo-gui display.
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
    env_class = 5
    #############################################
    # 环境定义
    #############################################
    
    env_params = ENV_Params(scenarios, sumo_gui, img_meters, num_stack, img_pixels, logdir, visdom, headless)
    """
    train_envs = []
    for i in range(6):
        env = MY_ENV(env_params, i)
        envs.append(env)
    """
    env = MY_ENV(env_params, env_class)
    scenario_name = scenarios[env_class]
    # 模型路径
    """
    model_pathes = ['/root/deep_learning/SMARTS/SMARTS/competition/track1/train_sun/a_new_model', \
                    '/root/deep_learning/SMARTS/SMARTS/competition/track1/train_sun/a_new_model', \
                    '/root/deep_learning/SMARTS/SMARTS/competition/track1/train_sun/a_new_model', \ 
                    '/root/deep_learning/SMARTS/SMARTS/competition/track1/train_sun/a_new_model', \
                    '/root/deep_learning/SMARTS/SMARTS/competition/track1/train_sun/a_new_model', \
                    '/root/deep_learning/SMARTS/SMARTS/competition/track1/train_sun/a_new_model' ]
    """
    model_path = '/root/deep_learning/SMARTS/SMARTS/competition/track1/train_sun/a_newest_model_pose2_scen5_0003'
    # 加载的checkpoint
    load_counter = 100
    # 保存路径
    save_path = "/root/deep_learning/SMARTS/SMARTS/DATA"
    # 生成数量
    generate_num = 10000
    # 网络类型
    actor_net_type = "mix2"
    # 设备选择
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    params = Params(env, save_path, generate_num, model_path, load_counter, actor_net_type, actor_input_dim, \
                    action_dim, heads_num, key_dim, value_dim, layer_num, mlp_hidden_dim, device, scenario_name)
    my_img_generator = Image_generator(params)
    my_img_generator.generate()
    
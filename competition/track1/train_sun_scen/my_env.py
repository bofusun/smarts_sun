from typing import Any, Dict, List

import os
import gym
import shutil
import numpy as np
from train.info import Info
from train.reward import Reward
from multiprocessing import Process, Pipe
from train.action import Action as DiscreteAction
from train.my_obs import Concatenate, FilterObs, SaveObs
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, SubprocVecEnv

from smarts.core.controllers import ActionSpaceType
from smarts.env.wrappers.format_action import FormatAction
from smarts.env.wrappers.format_obs import FormatObs
from smarts.env.wrappers.frame_stack import FrameStack
from smarts.env.wrappers.single_agent import SingleAgent


class ENV_Params():
    def __init__(self, scenarios, sumo_gui, img_meters, num_stack, img_pixels, logdir, visdom, headless):
        # 场景列表
        self.scenarios = scenarios
        # 是否使用sumo的可视化
        self.sumo_gui = sumo_gui  
        # 观测尺寸
        self.img_meters = img_meters 
        # 图像尺寸
        self.img_pixels = img_pixels 
        # 叠加观测数量
        self.num_stack = num_stack 
        # 日志地址
        self.logdir = logdir
        # 可视化
        self.visdom = visdom
        # 是否使用headless
        self.headless = headless
        
    
class MY_ENV():
    def __init__(self, env_params, env_class):
        self.scenarios = env_params.scenarios
        self.sumo_gui = env_params.sumo_gui 
        self.img_meters = env_params.img_meters 
        self.img_pixels = env_params.img_pixels
        self.num_stack = env_params.num_stack
        self.logdir = env_params.logdir
        self.visdom = env_params.visdom
        self.headless = env_params.headless
        self.env_class = env_class
        self.env = gym.make("smarts.env:multi-scenario-v0",
                            scenario = self.scenarios[self.env_class],
                            img_meters = self.img_meters,
                            img_pixels = self.img_pixels,
                            action_space="TargetPose",
                            sumo_headless = not self.sumo_gui,  # If False, enables sumo-gui display.
                            visdom = self.visdom,
                            headless = self.headless
                            )
        self.wrappers = self.Wrappers()
        for wrapper in self.wrappers:
            self.env = wrapper(self.env)

    def Wrappers(self):
        # fmt: off
        wrappers = [
            # 设置观测空间格式，满足gym格式
            FormatObs,
            # 设置动作空间格式，满足gym格式 TargetPose Continuous
            lambda env: FormatAction(env=env, space=ActionSpaceType["TargetPose"]),
            Info,
            # 奖励函数设置
            Reward,
            # 保存观测设置
            SaveObs,
            # 将连续动作变为离散动作
            # DiscreteAction,
            # 过滤出需要的观测
            FilterObs,
            # 用于把观测叠加在一起，做多个时间步的观测
            # lambda env: FrameStack(env=env, num_stack=self.num_stack),
            # 把叠加的字典放入numpy数组里
            # lambda env: Concatenate(env=env, channels_order="first"),
            # 将接口修改为单代理接口，该接口与gym等库兼容。
            SingleAgent,
            lambda env: DummyVecEnv([lambda: env]),
            lambda env: VecMonitor(venv=env, filename=str(self.logdir), info_keywords=("is_success",))
        ]
        # fmt: on
        return wrappers
        
    def reset(self):
        """
        if self.scenarios[self.env_class] == "1_to_2lane_left_turn_c":
            glb_path = "/root/deep_learning/SMARTS/SMARTS/smarts/scenarios/intersection/1_to_2lane_left_turn_c/map.glb"
            glb_back_path = "/root/deep_learning/SMARTS/SMARTS/smarts/scenarios/intersection/1_to_2lane_left_turn_c/map_backup.glb"
        elif self.scenarios[self.env_class] == "1_to_2lane_left_turn_t":
            glb_path = "/root/deep_learning/SMARTS/SMARTS/smarts/scenarios/intersection/1_to_2lane_left_turn_t/map.glb"
            glb_back_path = "/root/deep_learning/SMARTS/SMARTS/smarts/scenarios/intersection/1_to_2lane_left_turn_t/map_backup.glb"
        elif self.scenarios[self.env_class] == "3lane_merge_single_agent":
            glb_path = "/root/deep_learning/SMARTS/SMARTS/smarts/scenarios/merge/3lane_single_agent/map.glb"
            glb_back_path = "/root/deep_learning/SMARTS/SMARTS/smarts/scenarios/merge/3lane_single_agent/map_backup.glb"
        elif self.scenarios[self.env_class] == "3lane_cruise_single_agent":
            glb_path = "/root/deep_learning/SMARTS/SMARTS/smarts/scenarios/straight/3lane_cruise_single_agent/map.glb"
            glb_back_path = "/root/deep_learning/SMARTS/SMARTS/smarts/scenarios/straight/3lane_cruise_single_agent/map_backup.glb"
        elif self.scenarios[self.env_class] == "3lane_overtake":
            glb_path = "/root/deep_learning/SMARTS/SMARTS/smarts/scenarios/straight/3lane_overtake/map.glb"
            glb_back_path = "/root/deep_learning/SMARTS/SMARTS/smarts/scenarios/straight/3lane_overtake/map_backup.glb"
        if not os.path.exists(glb_path):
            print("not exits:", glb_path)
            shutil.copyfile(glb_back_path, glb_path)
        """
        obs = self.env.reset()
        return obs
        
    def step(self, action):
        """
        # {0: np.array([100,0.5,0.5,0])}
        if self.scenarios[self.env_class] == "1_to_2lane_left_turn_c":
            glb_path = "/root/deep_learning/SMARTS/SMARTS/smarts/scenarios/intersection/1_to_2lane_left_turn_c/map.glb"
            glb_back_path = "/root/deep_learning/SMARTS/SMARTS/smarts/scenarios/intersection/1_to_2lane_left_turn_c/map_backup.glb"
        elif self.scenarios[self.env_class] == "1_to_2lane_left_turn_t":
            glb_path = "/root/deep_learning/SMARTS/SMARTS/smarts/scenarios/intersection/1_to_2lane_left_turn_t/map.glb"
            glb_back_path = "/root/deep_learning/SMARTS/SMARTS/smarts/scenarios/intersection/1_to_2lane_left_turn_t/map_backup.glb"
        elif self.scenarios[self.env_class] == "3lane_merge_single_agent":
            glb_path = "/root/deep_learning/SMARTS/SMARTS/smarts/scenarios/merge/3lane_single_agent/map.glb"
            glb_back_path = "/root/deep_learning/SMARTS/SMARTS/smarts/scenarios/merge/3lane_single_agent/map_backup.glb"
        elif self.scenarios[self.env_class] == "3lane_cruise_single_agent":
            glb_path = "/root/deep_learning/SMARTS/SMARTS/smarts/scenarios/straight/3lane_cruise_single_agent/map.glb"
            glb_back_path = "/root/deep_learning/SMARTS/SMARTS/smarts/scenarios/straight/3lane_cruise_single_agent/map_backup.glb"
        elif self.scenarios[self.env_class] == "3lane_overtake":
            glb_path = "/root/deep_learning/SMARTS/SMARTS/smarts/scenarios/straight/3lane_overtake/map.glb"
            glb_back_path = "/root/deep_learning/SMARTS/SMARTS/smarts/scenarios/straight/3lane_overtake/map_backup.glb"
        if not os.path.exists(glb_path):
            print("not exits:", glb_path)
            shutil.copyfile(glb_back_path, glb_path)
        """
        obs, reward, done, info = self.env.step({0: np.array(action)})
        if done:
            print("-------------------------------done------------------------------------")
            # obs = self.env.reset()
        # obs, reward, done, info = self.env.step(np.array(action))
        return obs, reward, done, info
        
def run_once(env):
    obs = env.reset()
    done = False
    while not done:
        # action = [np.random.uniform(-1e10, 1e10), np.random.uniform(-1e10, 1e10), \
        #           np.random.uniform(-np.pi, np.pi), np.random.uniform(0, 1e10)]
        action = [1,0,0]
        obs, reward, done, info = env.step(action)
        # print("obs", obs)
        print("action", action)
        print("reward", reward)
        print("done", done)
        # print("info", info)
        print('-'*20)
    # env.close()


def run_envs_once(envs):
    obss = envs.reset()
    # print("obss", obss)
    # actions = [[np.random.uniform(-1e10, 1e10), np.random.uniform(-1e10, 1e10), \
    #               np.random.uniform(-np.pi, np.pi), np.random.uniform(0, 1e10)] for i in range(len(envs.envs))]
    done = False
    while not done:
        actions = [[0,0,-1] for i in range(len(envs.envs))]
        obss, rewards, dones, infos = envs.step(actions)
        print("actions", actions)
        print("rewards", rewards)
        print("dones", dones)
        print('-'*20)
        done = dones[0]


def worker(conn, env):
    while True:
        cmd, data = conn.recv()
        if cmd == "step":
            observation, reward, done, info = env.step(data)
            if done:
                observation = env.reset()
            conn.send((observation, reward, done, info))
        elif cmd == "reset":
            observation = env.reset()
            conn.send(observation)
        else:
            raise NotImplementedError


class ParallelEnv():
    def __init__(self, envs):
        assert len(envs) >= 1 
        self.envs = envs

        self.locals = []
        for env in self.envs[1:]:
            local, remote = Pipe()
            self.locals.append(local)
            p = Process(target=worker, args=(remote, env))
            # p.daemon = True
            p.start()
            remote.close()

    def reset(self):
        for local in self.locals:
            local.send(("reset", None))
        results = [self.envs[0].reset()] + [local.recv() for local in self.locals]    
        # results = zip(*[self.envs[0].reset()] + [local.recv() for local in self.locals])
        return results

    def step(self, actions):
        for local, action in zip(self.locals, actions[1:]):
            local.send(("step", action))
        observation, reward, done, info = self.envs[0].step(actions[0])
        # if done:
        #     observation= self.envs[0].reset()
        results = zip(*[(observation, reward, done, info)] + [local.recv() for local in self.locals])
        return results


if __name__ == "__main__":
    ###################################
    # 参数设置
    ###################################
    # 场景列表
    scenarios = ["1_to_2lane_left_turn_c", "1_to_2lane_left_turn_t", "3lane_merge_single_agent", \
                 "3lane_cruise_single_agent", "3lane_overtake"]
    # 是否使用sumo的可视化
    sumo_gui = False  # If False, enables sumo-gui display.
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
    env_class = 1
    #############################################
    # 环境定义
    #############################################
    env_params = ENV_Params(scenarios, sumo_gui, img_meters, num_stack, img_pixels, logdir, visdom, headless)
    # env = MY_ENV(env_params, env_class)
    envs = []
    for i in range(1):
        env = MY_ENV(env_params, env_class)
        envs.append(env)
    envs = ParallelEnv(envs)
    for i in range(5):
        print("-"*20)
        print("iter:", i)
        run_envs_once(envs)
    # run_once(env)











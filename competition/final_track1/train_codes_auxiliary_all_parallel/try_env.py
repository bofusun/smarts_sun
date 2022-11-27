from typing import Any, Dict, List

import gym
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from train.action import Action as DiscreteAction
from train.info import Info
from train.observation import Concatenate, FilterObs, SaveObs
from train.reward import Reward

from smarts.core.controllers import ActionSpaceType
from smarts.env.wrappers.format_action import FormatAction
from smarts.env.wrappers.format_obs import FormatObs
from smarts.env.wrappers.frame_stack import FrameStack
from smarts.env.wrappers.single_agent import SingleAgent


class Params():
    def __init__(self):
        self.scenarios = ["1_to_2lane_left_turn_c", "1_to_2lane_left_turn_t", "3lane_merge_single_agent", \
                          "3lane_cruise_single_agent", "3lane_overtake"]
        self.sumo_gui =  True  # If False, enables sumo-gui display.
        self.img_meters = 50 # Observation image area size in meters.
        self.img_pixels = 112 # Observation image size in pixels.
        # 叠加观测数量
        self.num_stack = 1 
        # 日志地址
        self.logdir = ""
        


class MY_ENV():
    def __init__(self, params):
        self.scenarios = params.scenarios
        self.sumo_gui = params.sumo_gui 
        self.img_meters = params.img_meters 
        self.img_pixels = params.img_pixels
        self.num_stack = params.num_stack
        self.logdir = params.logdir
        self.env = gym.make("smarts.env:multi-scenario-v0",
                            scenario = self.scenarios[4],
                            img_meters = self.img_meters,
                            img_pixels = self.img_pixels,
                            sumo_headless = self.sumo_gui,  # If False, enables sumo-gui display.
                            visdom = True,
                            headless = False
                            )
        self.wrappers = self.Wrappers()
        for wrapper in self.wrappers:
            self.env = wrapper(self.env)
        print(dir(self.env))
    
    def Wrappers(self):
        # fmt: off
        wrappers = [
            # 设置观测空间格式，满足gym格式
            FormatObs,
            # 设置动作空间格式，满足gym格式
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
            # lambda env: DummyVecEnv([lambda: env]),
            # lambda env: VecMonitor(venv=env, filename=str(self.logdir), info_keywords=("is_success",))
        ]
        # fmt: on
        return wrappers
    
    def reset(self):
        obs = self.env.reset()
        return obs
        

    def step(self, action):
        # [100,0.5,0.5,0]
        # obs, reward, done, info = self.env.step({0: np.array(action)})
        obs, reward, done, info = self.env.step(np.array(action))
        return obs, reward, done, info

    def run_once(self):
        obs = self.reset()
        # print("obs", obs)
        done = False
        while not done:
            action = [np.random.uniform(-1e10, 1e10), np.random.uniform(-1e10, 1e10), \
                      np.random.uniform(-np.pi, np.pi), np.random.uniform(0, 1e10)]
            obs, reward, done, info = self.step(action)
            # print("obs", obs)
            print("action", action)
            print("reward", reward)
            print('-'*20)
            # print("done", done)
            # print("info", info)
            if done:
                obs = self.reset()
        self.env.close()

if __name__ =="__main__":
    params = Params()
    my_env = MY_ENV(params)
    my_env.run_once()



from typing import Any, Dict
import copy
import gym
import math
import matplotlib
import scipy.misc as misc
import numpy as np
from train.obs_function import ttc_by_path, ego_ttc_calc, heading_to_degree, get_goal_layer, signed_dist_to_line, radians_to_vec

class SaveObs(gym.ObservationWrapper):
    """Saves several selected observation parameters."""

    def __init__(self, env: gym.Env):
        """
        Args:
            env (gym.Env): Environment to be wrapped.
        """
        super().__init__(env)
        self.saved_obs: Dict[str, Dict[str, Any]]

    def observation(self, obs: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Saves the wrapped environment's observation.

        Note: Users should not directly call this method.
        """

        obs_data = {}
        for agent_id, agent_obs in obs.items():
            obs_data.update(
                {
                    agent_id: {
                        "pos": copy.deepcopy(agent_obs["ego"]["pos"]),
                        "heading": copy.deepcopy(agent_obs["ego"]["heading"]),
                    }
                }
            )
        self.saved_obs = obs_data

        return obs




class FilterObs(gym.ObservationWrapper):
    """Filter only the selected observation parameters."""

    def __init__(self, env: gym.Env):
        """
        Args:
            env (gym.Env): Environment to be wrapped.
        """
        super().__init__(env)

        self.observation_space = gym.spaces.Dict(
            {agent_id: gym.spaces.Dict(
                    {
                        # 自己车的位置与方向
                        "pos_and_heading": gym.spaces.Box(low=-1e10, high=1e10, shape=(5,), dtype=np.float32, ),
                        # rgb图像
                        "rgb": gym.spaces.Box(low=0, high=255, shape=(agent_obs_space["rgb"].shape[-1]+1,) + agent_obs_space["rgb"].shape[:-1], dtype=np.uint8,),
                        # 所有观测放在一起的向量观测
                        "brief_final_obs": gym.spaces.Box(low=-1e10, high=+1e10, shape=(91,), dtype=np.float32, ),
                        # transformer的二维观测
                        "brief_transformer_pos_and_heading_2d": gym.spaces.Box(low=-1e10, high=+1e10, shape=(11,5), dtype=np.float32, ),
                        # 单独的向量观测
                        "brief_together_obs": gym.spaces.Box(low=-1e10, high=+1e10, shape=(36,), dtype=np.float32, ),
                        # 车辆辨别观测
                        "brief_judge": gym.spaces.Box(low=-1e10, high=+1e10, shape=(11,), dtype=np.float32, ),
                        # 场景观测
                        "brief_scenario_obs": gym.spaces.Box(low=-1e10, high=+1e10, shape=(1,), dtype=np.float32, ),
                    }
                )
                for agent_id, agent_obs_space in env.observation_space.spaces.items()
            }
        )
    

    def observation(self, obs: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Adapts the wrapped environment's observation.

        Note: Users should not directly call this method.
        """
        wrapped_obs = {}
        for agent_id, agent_obs in obs.items():
            #########################################################
            # 1. pos_and_heading观测 4
            #########################################################
            # 自己车的信息
            ego_heading = (agent_obs["ego"]["heading"] + np.pi) % (2 * np.pi) - np.pi
            ego_pos = agent_obs["ego"]["pos"]
            ego_speed = agent_obs["ego"]["speed"]
            # 生成 pos_and_heading
            pos_and_heading = [ego_pos[0], ego_pos[1], ego_heading, ego_speed, 0]

            #########################################################
            # 2. rgb图像观测
            #########################################################
            rgb = agent_obs["rgb"]
            rgb = rgb.transpose(2, 0, 1)
            goal_rgb = get_goal_layer(agent_obs["mission"]["goal_pos"][0], agent_obs["mission"]["goal_pos"][1], \
                                      agent_obs["ego"]["pos"][0], agent_obs["ego"]["pos"][1], agent_obs["ego"]["heading"])
            rgb = np.concatenate((rgb, goal_rgb), axis=0)

            #########################################################
            # 3. transformer的二维输入 11 * 4
            #########################################################
            # 邻居车辆的绝对和相对位置、方向和速度
            nv_poses = []
            for i in range(10): 
                nv_pos = get_trans_coor(agent_obs['neighbors']['pos'][i][0], agent_obs['neighbors']['pos'][i][1], ego_heading)
                nv_poses.append(nv_pos)

            neighbor_pos_and_heading_2d = [[nv_poses[i][0], nv_poses[i][1], \
                                            agent_obs['neighbors']['heading'][i] - ego_heading, agent_obs['neighbors']['speed'][i]]
                                            for i in range(10)]
            # 判别是否有邻居车
            neighbor_judge = [0 if agent_obs['neighbors']['pos'][i][0] == 0 and agent_obs['neighbors']['pos'][i][1] ==0 else 1 for i in range(10)]
            # 邻居车辆的距离
            neighbor_distance = [((agent_obs['neighbors']['pos'][i][0] - ego_pos[0]) ** 2 + \
                                  (agent_obs['neighbors']['pos'][i][1] - ego_pos[1]) ** 2) ** 0.5 \
                                  for i in range(10)]
            neighbor_distance = [[neighbor_distance[i] * neighbor_judge[i]] for i in range(10)]
            # 加入车辆距离
            new_neighbor_pos_and_heading_2d = []
            for i in range(10):
                new_neighbor_pos_and_heading_2d.append(neighbor_pos_and_heading_2d[i]+neighbor_distance[i])
            neighbor_pos_and_heading_2d = new_neighbor_pos_and_heading_2d
            # 如果没有车将所有元素置为-1
            for i in range(10):
                if neighbor_judge[i]==0:
                    neighbor_pos_and_heading_2d[i] = [-1 for i in range(9)]
            # 一维的邻居信息
            neighbor_pos_and_heading_1d = [item for sublist in neighbor_pos_and_heading_2d for item in sublist]
            # transformer加入自己车的信息
            temp_ego_pos = get_trans_coor(ego_pos[0], ego_pos[1], ego_heading)
            temp_pos_and_heading = [temp_ego_pos[0], temp_ego_pos[1], 0, ego_speed]
            temp_pos_and_heading.append(0)
            temp = [temp_pos_and_heading]
            transformer_pos_and_heading_2d = temp + neighbor_pos_and_heading_2d

            #####################################################
            # 4. 邻居车判断观测 11
            #####################################################
            brief_judge = [1]
            brief_judge.extend([neighbor_judge[i] for i in range(10)])

            #####################################################
            # 5. 向量观测 10*3
            #####################################################
            # waypoint未来选择的10个点
            indices = np.array([0, 1, 2, 3, 5, 7, 9, 12, 15, 19])
            # 如果waypoint点不够,则剩下的取最大号
            wps_len = [len(path) for path in agent_obs["waypoints"]['heading']]
            max_len_lane_index = np.argmax(wps_len)
            max_len = np.max(wps_len)
            sample_wp_heading = [agent_obs["waypoints"]["heading"][max_len_lane_index][i] for i in indices]
            # waypoint观测
            sample_wp_info = []
            for i in indices:
                wapoint_pos = get_trans_coor(agent_obs["waypoints"]["pos"][max_len_lane_index][i][0], \
                                             agent_obs["waypoints"]["pos"][max_len_lane_index][i][1], ego_heading)
                temp_sample_wp_info = [wapoint_pos[0], 
                                       wapoint_pos[1],
                                       agent_obs["waypoints"]["heading"][max_len_lane_index][i] - ego_heading]
                sample_wp_info.extend(temp_sample_wp_info)
            # 获取waypoint的位置和方向
            waypoint_pos_and_heading = sample_wp_info

            # 目标的角度 4
            goal_pos = agent_obs["mission"]["goal_pos"]
            rel_pos = goal_pos - ego_pos
            goal_angle = np.angle(rel_pos[0] + 1j * rel_pos[1]) - np.pi / 2
            goal_angle = (goal_angle + np.pi) % (2 * np.pi) - np.pi
            goal_final_pos = get_trans_coor(agent_obs["mission"]["goal_pos"][0], agent_obs["mission"]["goal_pos"][1], ego_heading)
            # 目标的方向
            goal_heading = goal_angle - ego_heading
            goal_heading = (goal_heading + np.pi) % (2 * np.pi) - np.pi
            goal_heading = [goal_heading]
            # 目标的距离
            goal_distance = [np.linalg.norm(agent_obs["mission"]["goal_pos"] - agent_obs["ego"]["pos"])]
            # 获取goal的位置和方向
            goal_pos_and_heading = [goal_final_pos[0], goal_final_pos[1], goal_heading[0] - ego_heading, goal_distance[0]]  
            
            #########################################################################################################
            # 根据 主车状态 邻居车辆的状态 waypionts点的路径 距离最近的waypoints来得到车道的ttc dist rel_speed 等信息
            # 处理waypoints最终得到wp_paths --list 包含4条待选path
            # wp_path -- list 包含20个waypoints 其中每个waypoints是一个字典
            #########################################################################################################
            smarts_wps = agent_obs['waypoints'] 
            wp = {}
            wp_paths = []
            for path_index in range(4):
                wp_path = []
                for wps_index in range(20):
                    wp = {wp_feature:smarts_wps[wp_feature][path_index][wps_index] for wp_feature, _ in smarts_wps.items()}
                    wp_path.append(wp)
                wp_paths.append(wp_path)
            
            closest_wps = [path[0] for path in wp_paths]
            closest_wp = min(closest_wps, key=lambda wp: np.linalg.norm(wp['pos']-ego_pos))
            # 距离道路中心线的距离
            temp_ego_pos = ego_pos[:2]
            temp_closest_wp_pos = closest_wp["pos"][:2]
            temp_closest_wp_heading = closest_wp["heading"]
            temp_closest_wp_vec = radians_to_vec(temp_closest_wp_heading)
            # 获取距离
            signed_dist_from_center = signed_dist_to_line(temp_ego_pos, temp_closest_wp_pos, temp_closest_wp_vec)
            lane_width = closest_wp["lane_width"] * 0.5
            norm_dist_from_center = signed_dist_from_center / lane_width
            if np.isnan(norm_dist_from_center) or np.isinf(norm_dist_from_center):
                norm_dist_from_center = 0
            norm_dist_from_center = np.array([norm_dist_from_center], dtype=np.float32)
            # 添加一个限制速度
            speed_limit = [agent_obs["waypoints"]["speed_limit"].max()]

            ########################################################################
            # 6. 场景观测
            ########################################################################
            ["1_to_3lane_left_turn_c", "1_to_3lane_left_turn_c_1", "1_to_3lane_left_turn_t", \
             "lanes_merge_single_agent_1", "lanes_merge_single_agent_2", "lanes_merge_single_agent_3", \
             "lanes_cruise_single_agent_1", "lanes_cruise_single_agent_2", "lanes_cruise_single_agent_3", \
             "lanes_overtake_1", "lanes_overtake_2", "lanes_overtake_3", "lanes_cut_in_1", "lanes_cut_in_2"]        
            ["1_to_3lane_left_turn_c", "1_to_3lane_left_turn_t", "3lane_merge_single_agent", \
             "3lane_cruise_single_agent", "3lane_overtake", "3lane_cut_in"]
            if self.scenario == "1_to_3lane_left_turn_c":
                scenario_obs = np.array([0])
            elif self.scenario == "1_to_3lane_left_turn_t":
                scenario_obs = np.array([1])
            elif self.scenario == "3lane_merge_single_agent":
                scenario_obs = np.array([2])
            elif self.scenario == "3lane_cruise_single_agent":
                scenario_obs = np.array([3])
            elif self.scenario == "3lane_overtake":
                scenario_obs = np.array([4])
            elif self.scenario == "3lane_cut_in":
                scenario_obs = np.array([5])
            
            # 自己车位置方向与速度 5
            brief_ego_pos_and_heading = np.array(pos_and_heading, dtype=np.float32)
            # transformer二维观测 11*5
            brief_transformer_pos_and_heading_2d = np.array(transformer_pos_and_heading_2d, dtype=np.float32)
            # 邻居车辆一维观测 10*5
            brief_neighbor_pos_and_heading_1d = np.array(neighbor_pos_and_heading_1d, dtype=np.float32)
            # waypoint位置方向与headingerror 10*3
            brief_waypoint_pos_and_heading = np.array(waypoint_pos_and_heading, dtype=np.float32)
            # 目标位置方向与距离 4
            brief_goal_pos_and_heading = np.array(goal_pos_and_heading, dtype=np.float32)
            # 全部向量观测 91
            brief_final_obs = np.concatenate((brief_ego_pos_and_heading, brief_neighbor_pos_and_heading_1d,\
                                              brief_waypoint_pos_and_heading, brief_goal_pos_and_heading, norm_dist_from_center, speed_limit), axis=0)
            # 共同向量观测 36
            brief_together_obs = np.concatenate((brief_waypoint_pos_and_heading, brief_goal_pos_and_heading, norm_dist_from_center, speed_limit), axis=0)

            # 返回观测
            wrapped_obs.update(
                {
                    agent_id: {
                        "pos_and_heading": np.array(pos_and_heading, dtype=np.float32),
                        "rgb": np.uint8(rgb),
                        "brief_final_obs": brief_final_obs,
                        "brief_transformer_pos_and_heading_2d": brief_transformer_pos_and_heading_2d,
                        "brief_together_obs": brief_together_obs,
                        "brief_judge": brief_judge,
                        "brief_scenario_obs": scenario_obs,
                    }
                }
            )
        return wrapped_obs

class Concatenate(gym.ObservationWrapper):
    """Concatenates data from stacked dictionaries. Only works with nested gym.spaces.Box .
    Dimension to stack over is determined by `channels_order`.
    """

    def __init__(self, env: gym.Env, channels_order: str = "first"):
        """
        Args:
            env (gym.Env): Environment to be wrapped.
            channels_order (str): A string, either "first" or "last", specifying
                the dimension over which to stack each observation.
        """
        super().__init__(env)

        self._repeat_axis = {
            "first": 0,
            "last": -1,
        }.get(channels_order)

        for agent_name, agent_space in env.observation_space.spaces.items():
            for subspaces in agent_space:
                for key, space in subspaces.spaces.items():
                    assert isinstance(space, gym.spaces.Box), (
                        f"Concatenate only works with nested gym.spaces.Box. "
                        f"Got agent {agent_name} with key {key} and space {space}."
                    )

        _, agent_space = next(iter(env.observation_space.spaces.items()))
        self._num_stack = len(agent_space)
        self._keys = agent_space[0].spaces.keys()

        obs_space = {}
        for agent_name, agent_space in env.observation_space.spaces.items():
            subspaces = {}
            for key, space in agent_space[0].spaces.items():
                low = np.repeat(space.low, self._num_stack, axis=self._repeat_axis)
                high = np.repeat(space.high, self._num_stack, axis=self._repeat_axis)
                subspaces[key] = gym.spaces.Box(low=low, high=high, dtype=space.dtype)
            obs_space.update({agent_name: gym.spaces.Dict(subspaces)})
        self.observation_space = gym.spaces.Dict(obs_space)

    def observation(self, obs):
        """Adapts the wrapped environment's observation.

        Note: Users should not directly call this method.
        """

        wrapped_obs = {}
        for agent_id, agent_obs in obs.items():
            stacked_obs = {}
            for key in self._keys:
                val = [obs[key] for obs in agent_obs]
                print("key", key)
                print("val", val)
                stacked_obs[key] = np.concatenate(val, axis=self._repeat_axis)
                print("stacked_obs[key]", stacked_obs[key])
            wrapped_obs.update({agent_id: stacked_obs})

        return wrapped_obs

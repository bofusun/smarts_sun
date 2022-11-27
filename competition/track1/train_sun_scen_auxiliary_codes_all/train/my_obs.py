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
        """
        self.observation_space = gym.spaces.Dict(
            {agent_id: gym.spaces.Dict(
                    {
                        # To make car follow the waypoints
                        # distance from lane center
                        "distance_from_center": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,), dtype=np.float32, ),
                        # relative heading angle from 10 waypoints in 50 forehead waypoints
                        "heading_errors": gym.spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32, ),
                        # Car attributes
                        # ego speed
                        "speed": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,), dtype=np.float32, ),
                        # ego steering
                        "steering": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,), dtype=np.float32, ),
                        # To make car learn to slow down, overtake or dodge
                        # distance to the closest car in each lane
                        "lane_dist": gym.spaces.Box(low=-1e10, high=1e10, shape=(5,), dtype=np.float32, ),
                        # time to collide to the closest car in each lane
                        "lane_ttc": gym.spaces.Box(low=-1e10, high=1e10, shape=(5,), dtype=np.float32, ),
                        # ego lane closest social vehicle relative speed
                        "closest_lane_nv_rel_speed": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,), dtype=np.float32, ),
                        # distance to the closest car in possible intersection direction
                        "intersection_ttc": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,), dtype=np.float32, ),
                        # time to collide to the closest car in possible intersection direction
                        "intersection_distance": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,), dtype=np.float32, ),
                        # intersection closest social vehicle relative speed
                        "closest_its_nv_rel_speed": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,), dtype=np.float32, ),
                        # intersection closest social vehicle relative position in vehicle heading coordinate
                        "closest_its_nv_rel_pos": gym.spaces.Box(low=-1e10, high=1e10, shape=(2,), dtype=np.float32, ),
                        "goal_distance": gym.spaces.Box(low=-1e10, high=+1e10, shape=(1,), dtype=np.float32, ),
                        "goal_heading": gym.spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32, ),
                    }
                )
                for agent_id, agent_obs_space in env.observation_space.spaces.items()
            }
        )
        """
        # for agent_id, agent_obs_space in env.observation_space.spaces.items():
        #     print("agent_obs_space[rgb].shape[-1]", agent_obs_space["rgb"].shape[-1])
        #     print("agent_obs_space[rgb].shape[:-1]", agent_obs_space["rgb"].shape[:-1])
        #     print("(agent_obs_space[rgb].shape[-1],) + agent_obs_space[rgb].shape[:-1]", (agent_obs_space["rgb"].shape[-1],) + agent_obs_space["rgb"].shape[:-1])

        self.observation_space = gym.spaces.Dict(
            {agent_id: gym.spaces.Dict(
                    {
                        # To make car follow the waypoints
                        "pos_and_heading": gym.spaces.Box(low=-1e10, high=1e10, shape=(8,), dtype=np.float32, ),
                        # rgb图像
                        "rgb": gym.spaces.Box(low=0, high=255, shape=(agent_obs_space["rgb"].shape[-1]+1,) + agent_obs_space["rgb"].shape[:-1], dtype=np.uint8,),
                        # distance from lane center
                        # "distance_from_center": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,), dtype=np.float32, ),
                        # relative heading angle from 10 waypoints in 50 forehead waypoints
                        "heading_errors": gym.spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32, ),
                        # Car attributes
                        # ego speed
                        "speed": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,), dtype=np.float32, ),
                        # ego steering
                        "heading": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,), dtype=np.float32, ),
                        # To make car learn to slow down, overtake or dodge
                        # distance to the closest car in each lane
                        "lane_dist": gym.spaces.Box(low=-1e10, high=1e10, shape=(4,), dtype=np.float32, ),
                        # time to collide to the closest car in each lane
                        "lane_ttc": gym.spaces.Box(low=-1e10, high=1e10, shape=(4,), dtype=np.float32, ),
                        # ego lane closest social vehicle relative speed
                        "closest_lane_nv_rel_speed": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,), dtype=np.float32, ),
                        # 目标距离
                        "goal_distance": gym.spaces.Box(low=-1e10, high=+1e10, shape=(1,), dtype=np.float32, ),
                        "goal_heading": gym.spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32, ),
                        "final_obs": gym.spaces.Box(low=-1e10, high=+1e10, shape=(23,), dtype=np.float32, ),
                        # "brief_final_obs": gym.spaces.Box(low=-1e10, high=+1e10, shape=(100,), dtype=np.float32, ),
                        "brief_final_obs": gym.spaces.Box(low=-1e10, high=+1e10, shape=(167,), dtype=np.float32, ),
                        # "brief_transformer_pos_and_heading_2d": gym.spaces.Box(low=-1e10, high=+1e10, shape=(11,5), dtype=np.float32, ),
                        "brief_transformer_pos_and_heading_2d": gym.spaces.Box(low=-1e10, high=+1e10, shape=(11,9), dtype=np.float32, ),
                        # "brief_together_obs": gym.spaces.Box(low=-1e10, high=+1e10, shape=(45,), dtype=np.float32, ),
                        "brief_together_obs": gym.spaces.Box(low=-1e10, high=+1e10, shape=(68,), dtype=np.float32, ),
                        "brief_judge": gym.spaces.Box(low=-1e10, high=+1e10, shape=(11,), dtype=np.float32, ),
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
            # rgb图像观测
            rgb = agent_obs["rgb"]
            rgb = rgb.transpose(2, 0, 1)
            goal_rgb = get_goal_layer(agent_obs["mission"]["goal_pos"][0], agent_obs["mission"]["goal_pos"][1], \
                                      agent_obs["ego"]["pos"][0], agent_obs["ego"]["pos"][1], agent_obs["ego"]["heading"])
            
            # matplotlib.image.imsave('/root/deep_learning/SMARTS/SMARTS/competition/track1/train_sun/train/out.jpg', rgb)
            rgb = np.concatenate((rgb, goal_rgb), axis=0)
            # 车辆与目标的距离
            goal_distance = [np.linalg.norm(agent_obs["mission"]["goal_pos"] - agent_obs["ego"]["pos"])]
            # 车辆的方向
            ego_heading = (agent_obs["ego"]["heading"] + np.pi) % (2 * np.pi) - np.pi
            # 车辆的位置
            ego_pos = agent_obs["ego"]["pos"]
            # 目标的位置
            goal_pos = agent_obs["mission"]["goal_pos"]
            # 与目标的位置差
            rel_pos = goal_pos - ego_pos
            # 目标的角度
            goal_angle = np.angle(rel_pos[0] + 1j * rel_pos[1]) - np.pi / 2
            goal_angle = (goal_angle + np.pi) % (2 * np.pi) - np.pi
            # 目标的方向
            goal_heading = goal_angle - ego_heading
            goal_heading = (goal_heading + np.pi) % (2 * np.pi) - np.pi
            goal_heading = [goal_heading]
            # 邻居车辆相关信息
            nv_states = agent_obs['neighbors']
            
            # 设定自己车辆的状态
            ego_state = agent_obs["ego"]
            # 车辆与道路中央的距离
            wps_pos = agent_obs["waypoints"]["pos"]
            closest_wp_pos = np.array([path[0] for path in wps_pos])
            closest_wp_count = np.argmin((closest_wp_pos - ego_state["pos"])**2)
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
                temp_sample_wp_info = [agent_obs["waypoints"]["heading"][max_len_lane_index][i], 
                                       agent_obs["waypoints"]["pos"][max_len_lane_index][i][0],
                                       agent_obs["waypoints"]["pos"][max_len_lane_index][i][1],
                                       agent_obs["waypoints"]["heading"][max_len_lane_index][i] - agent_obs["ego"]["heading"],
                                       agent_obs["waypoints"]["pos"][max_len_lane_index][i][0] - ego_pos[0],
                                       agent_obs["waypoints"]["pos"][max_len_lane_index][i][1] - ego_pos[1]]
                sample_wp_info.extend(temp_sample_wp_info)
            
            last_wp_index = 0
            for i, wp_index in enumerate(indices):
                if wp_index > max_len - 1:
                    indices[i:] = last_wp_index
                    break
                last_wp_index = wp_index
            
            #################################################
            # 方向错误的sin值
            #################################################
            heading_errors = [wp_heading - agent_obs["ego"]["heading"] for wp_heading in sample_wp_heading]
            
            #######################################################
            # ego heading
            #######################################################
            if np.isnan(ego_state["heading"]):
                ego_heading = 0
            else: 
                ego_heading = ego_state["heading"]
            
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

            # 转化成nv里面包含相应的信息
            nv = {}
            neighborhood_vehicle_states = []
            for nv_index in range(10):
                nv = {nv_feature:nv_states[nv_feature][nv_index] for nv_feature, _ in nv_states.items()}
                neighborhood_vehicle_states.append(nv)


            (
                lane_ttc,
                lane_dist,
                closest_lane_nv_rel_speed,
                intersection_ttc,
                intersection_distance,
                closest_its_nv_rel_speed,
                closest_its_nv_rel_pos,
                overtake_behind_index,
                not_eq_len_behind_index,
            ) = ttc_by_path(
                ego_state, wp_paths, neighborhood_vehicle_states, closest_wp
            )
            
            ego_speed = ego_state["speed"]
            # 距离道路中心线的距离
            temp_ego_pos = ego_pos[:2]
            temp_closest_wp_pos = closest_wp["pos"][:2]
            temp_closest_wp_heading = closest_wp["heading"]
            temp_closest_wp_vec = radians_to_vec(temp_closest_wp_heading)

            signed_dist_from_center = signed_dist_to_line(temp_ego_pos, temp_closest_wp_pos, temp_closest_wp_vec)
            lane_width = closest_wp["lane_width"] * 0.5
            norm_dist_from_center = signed_dist_from_center / lane_width
            norm_dist_from_center = np.array([norm_dist_from_center], dtype=np.float32)
            # 新向量观测
            pos_and_heading = [ego_pos[0], ego_pos[1], ego_heading, ego_speed, 0, 0, 0, 0]
            neighbor_pos_and_heading_2d = [[agent_obs['neighbors']['pos'][i][0], agent_obs['neighbors']['pos'][i][1], \
                                            agent_obs['neighbors']['heading'][i], agent_obs['neighbors']['speed'][i], \
                                            agent_obs['neighbors']['pos'][i][0] - pos_and_heading[0], agent_obs['neighbors']['pos'][i][1] - pos_and_heading[1], \
                                            agent_obs['neighbors']['heading'][i] - pos_and_heading[2], agent_obs['neighbors']['speed'][i] - pos_and_heading[3]] \
                                            for i in range(10)]
            brief_judge = [1]
            neighbor_judge = [0 if neighbor_pos_and_heading_2d[i]==[0,0,0,0] else 1 for i in range(10)]
            brief_judge.extend([0 if neighbor_pos_and_heading_2d[i]==[0,0,0,0] else 1 for i in range(10)])
            # 邻居车辆的距离
            neighbor_distance = [((agent_obs['neighbors']['pos'][i][0] - ego_pos[0]) ** 2 + \
                                  (agent_obs['neighbors']['pos'][i][1] - ego_pos[1]) ** 2) ** 0.5 \
                                  for i in range(10)]
            neighbor_distance = [[neighbor_distance[i] * neighbor_judge[i]] for i in range(10)]
            # print("neighbor_distance", neighbor_distance)
            new_neighbor_pos_and_heading_2d = []
            # 加入车辆距离
            for i in range(10):
                new_neighbor_pos_and_heading_2d.append(neighbor_pos_and_heading_2d[i]+neighbor_distance[i])
            neighbor_pos_and_heading_2d = new_neighbor_pos_and_heading_2d
            # print("neighbor_pos_and_heading_2d", neighbor_pos_and_heading_2d)
            # 展开
            neighbor_pos_and_heading_1d = [item for sublist in neighbor_pos_and_heading_2d for item in sublist]
            waypoint_pos_and_heading = sample_wp_info
            goal_pos_and_heading = [agent_obs["mission"]["goal_pos"][0], agent_obs["mission"]["goal_pos"][1], goal_heading[0] , agent_obs["mission"]["goal_pos"][0] - pos_and_heading[0], agent_obs["mission"]["goal_pos"][1] - pos_and_heading[1], goal_heading[0] - pos_and_heading[2], goal_distance[0]]
            temp_pos_and_heading = pos_and_heading.copy()
            temp_pos_and_heading.append(0)
            temp = [temp_pos_and_heading]
            transformer_pos_and_heading_2d = temp + neighbor_pos_and_heading_2d
            # print("#######################################################")
            # print("agent_obs", agent_obs)
            # print("closest_wp", closest_wp_count)

            # print("heading_errors",np.array(heading_errors, dtype=np.float32,))
            # print("speed",np.array([ego_state["speed"] / 120], dtype=np.float32,))
            # print("heading",np.array([ego_heading / (0.5 * math.pi)], dtype=np.float32,))
            # print("goal_distance",np.array(goal_distance, dtype=np.float32,))
            # print("goal_heading",np.array(goal_heading, dtype=np.float32,))
            # print("lane_ttc", np.array(lane_ttc, dtype=np.float32,))
            # print("lane_dist", np.array(lane_dist, dtype=np.float32,))
            # print("closest_lane_nv_rel_speed", np.array([closest_lane_nv_rel_speed], dtype=np.float32,))
            # print("pos_and_heading", pos_and_heading)
            # print("angular_velocity", ego_state["angular_velocity"])
            # print("angular_acceleration", ego_state["angular_acceleration"])
            # print("angular_jerk", ego_state["angular_jerk"])
            # print("linear_velocity", ego_state["linear_velocity"])
            # print("linear_acceleration", ego_state["linear_acceleration"])
            # print("linear_jerk", ego_state["linear_jerk"])
            # print("box", ego_state["box"])
            # print("speed km/h", pos_and_heading[3]*3.6)

            final_pos_and_heading = np.array(pos_and_heading, dtype=np.float32)
            final_heading_errors = np.array(heading_errors, dtype=np.float32,)
            final_speed = np.array([ego_state["speed"] / 120], dtype=np.float32,)
            final_heading = np.array([ego_heading / (0.5 * math.pi)], dtype=np.float32,)
            final_goal_distance = np.array(goal_distance, dtype=np.float32,)
            final_goal_heading = np.array(goal_heading, dtype=np.float32,)
            final_lane_ttc = np.array(lane_ttc, dtype=np.float32,)
            final_lane_dist = np.array(lane_dist, dtype=np.float32,)
            final_closest_lane_nv_rel_speed = np.array([closest_lane_nv_rel_speed], dtype=np.float32,)

            final_obs = np.concatenate((final_heading_errors, final_speed, final_heading, final_goal_distance, final_goal_heading, \
                                        final_lane_ttc, final_lane_dist, final_closest_lane_nv_rel_speed), axis=0)
            # print("final_obs", final_obs)
            # 场景信息
            ["1_to_2lane_left_turn_c", "1_to_2lane_left_turn_t", "3lane_merge_single_agent", \
            "3lane_cruise_single_agent", "3lane_overtake", "3lane_cut_in"]
            if self.scenario == "1_to_2lane_left_turn_c":
                scenario_obs = np.array([0])
            elif self.scenario == "1_to_2lane_left_turn_t":
                scenario_obs = np.array([1])
            elif self.scenario == "3lane_merge_single_agent":
                scenario_obs = np.array([2])
            elif self.scenario == "3lane_cruise_single_agent":
                scenario_obs = np.array([3])
            elif self.scenario == "3lane_overtake":
                scenario_obs = np.array([4])
            elif self.scenario == "3lane_cut_in":
                scenario_obs = np.array([5])
            # 简洁版向量观测
            # 自己车位置方向与速度 9
            brief_ego_pos_and_heading = np.array(temp_pos_and_heading, dtype=np.float32)
            # 邻居车位置方向与速度 10*9
            brief_neighbor_pos_and_heading_1d = np.array(neighbor_pos_and_heading_1d, dtype=np.float32)
            # waypoint位置方向与headingerror 10*6
            brief_waypoint_pos_and_heading = np.array(waypoint_pos_and_heading, dtype=np.float32)
            # 目标位置方向与距离 7
            # print("goal_pos_and_heading", goal_pos_and_heading)
            brief_goal_pos_and_heading = np.array(goal_pos_and_heading, dtype=np.float32)
            # 简洁版最终向量观测 100
            # print("brief_ego_pos_and_heading", brief_ego_pos_and_heading)
            # print("norm_dist_from_center", norm_dist_from_center)
            brief_final_obs = np.concatenate((brief_ego_pos_and_heading, brief_neighbor_pos_and_heading_1d,\
                                              brief_waypoint_pos_and_heading, brief_goal_pos_and_heading, norm_dist_from_center), axis=0)
            # transformer观测 11*9
            brief_transformer_pos_and_heading_2d = np.array(transformer_pos_and_heading_2d, dtype=np.float32)
            # 共同观测 45
            brief_together_obs = np.concatenate((brief_waypoint_pos_and_heading, brief_goal_pos_and_heading, norm_dist_from_center), axis=0)
            # 返回观测
            wrapped_obs.update(
                {
                    agent_id: {
                        "pos_and_heading": np.array(pos_and_heading, dtype=np.float32),
                        "rgb": np.uint8(rgb),
                        "heading_errors": np.array(heading_errors, dtype=np.float32,),
                        "speed": np.array([ego_state["speed"] / 120], dtype=np.float32,),
                        "heading": np.array([ego_heading / (0.5 * math.pi)], dtype=np.float32,),
                        "lane_ttc": np.array(lane_ttc, dtype=np.float32,),
                        "lane_dist": np.array(lane_dist, dtype=np.float32,),
                        "closest_lane_nv_rel_speed":  np.array([closest_lane_nv_rel_speed], dtype=np.float32,),
                        "goal_distance": np.array(goal_distance, dtype=np.float32,),
                        "goal_heading": np.array(goal_heading, dtype=np.float32,),
                        "final_obs": final_obs,
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

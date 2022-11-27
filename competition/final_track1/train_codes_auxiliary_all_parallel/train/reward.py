from typing import Any, Dict

import gym
import numpy as np
from train.obs_function import ttc_by_path, ego_ttc_calc, heading_to_degree, get_goal_layer
from train.obs_function import signed_dist_to_line, radians_to_vec

class Reward(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.my_count = 0

    def reset(self, **kwargs):
        self.my_count = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        """Adapts the wrapped environment's step.

        Note: Users should not directly call this method.
        """
        self.my_count += 1
        obs, reward, done, info = self.env.step(action)
        wrapped_reward = self._reward(obs, reward)

        for agent_id, agent_done in done.items():
            if agent_id != "__all__" and agent_done == True:
                if obs[agent_id]["events"]["reached_goal"]:
                    print(f"{agent_id}: Hooray! Reached goal.")
                elif obs[agent_id]["events"]["reached_max_episode_steps"]:
                    print(f"{agent_id}: Reached max episode steps.")
                elif (
                    obs[agent_id]["events"]["collisions"]
                    | obs[agent_id]["events"]["off_road"]
                    | obs[agent_id]["events"]["off_route"]
                    | obs[agent_id]["events"]["on_shoulder"]
                    | obs[agent_id]["events"]["wrong_way"]
                ):
                    pass
                else:
                    print("Events: ", obs[agent_id]["events"])
                    raise Exception("Episode ended for unknown reason.")

        return obs, wrapped_reward, done, info
    

    def designed_reward(self, obs, agent_id, agent_reward):
        agent_obs = obs[agent_id]
        # 主车相关的信息
        ego_state = obs[agent_id]["ego"]
        ego_pos = ego_state["pos"]
        ego_speed = ego_state["speed"]
        # waypoint相关信息
        smarts_wps = obs[agent_id]['waypoints'] 
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
        ###################################################################
        # 1. 距离道路中心的距离惩罚
        ###################################################################
        temp_ego_pos = ego_pos[:2]
        closest_wp_pos = closest_wp["pos"][:2]
        closest_wp_heading = closest_wp["heading"]
        closest_wp_vec = radians_to_vec(closest_wp_heading)
        # 获取道路中心距离
        signed_dist_from_center = signed_dist_to_line(temp_ego_pos, closest_wp_pos, closest_wp_vec)
        lane_hwidth = closest_wp["lane_width"] * 0.5
        norm_dist_from_center = signed_dist_from_center / lane_hwidth
        if np.isnan(norm_dist_from_center) or np.isinf(norm_dist_from_center):
            norm_dist_from_center = 0
        # 道路中心惩罚
        dist_center_reward = - np.abs(norm_dist_from_center)
        # print("dist_center_reward:{}".format(dist_center_reward))

        ###################################################################
        # 2. 空车道惩罚
        ###################################################################
        # 空车道的奖励 以及行驶到空车道后的速度奖励
        empty_lane_reward = 0 
        lane_max_speed = closest_wp["speed_limit"]
        # print("lane_max_speed:{}".format(lane_max_speed))
        ego_lane_index = ego_state["lane_index"]
        # 当所有道路一样长度时奖励行驶在最空直道，当存在汇合时 道路不一样长时 对最长道路给予奖励 其他道路给予惩罚
        wps_len = [len(path) for path in agent_obs["waypoints"]['heading']]
        # 邻居车辆信息
        nv_states = agent_obs['neighbors']
        # print("nv_speed:{}".format(nv_states["speed"]))
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


        if len(set(wps_len))<=1:
            # return 0
            if lane_dist[ego_lane_index] < 1:
                # 找到最空的一条有效道路
                # 当存在多条最优时在任意一条上既可以给奖励
                max_dist_lane_indexs = [i for i,x in enumerate(lane_dist) if x==max(lane_dist)]
                if ego_lane_index not in max_dist_lane_indexs and lane_dist[max_dist_lane_indexs[0]] - lane_dist[ego_lane_index] > 0.1:
                    empty_lane_reward = -0.05 * agent_reward
                    # print("*"*30)
                    # print(" current lane is busy and another is empty, we should change lane !!!")
                    # print("*"*30)
                """
                else:
                    empty_lane_reward = 0.05 * agent_reward
                """
        ########################################################
        # 3. 车辆距离惩罚
        ########################################################
        lane_distance_penalty = 0
        safe_speed_limit = 18.5
        distance_ratio = ego_speed / safe_speed_limit
        safe_distance = 0.1 * distance_ratio + 0.03
        if lane_dist[ego_lane_index] < safe_distance:
            lane_distance_penalty = -1 * 1 * (safe_distance - lane_dist[ego_lane_index])
        # print("lane_distance_penalty", lane_distance_penalty)

        # 限制速度
        speed_limit = closest_wp["speed_limit"]
        speed_penalty = 0
        if ego_speed > speed_limit:
            speed_penalty = -(ego_speed - speed_limit) / speed_limit
        # print("speed", ego_speed)

        return 0.1*dist_center_reward + empty_lane_reward + lane_distance_penalty + speed_penalty



    def _reward(
        self, obs: Dict[str, Dict[str, Any]], env_reward: Dict[str, np.float64]
    ) -> Dict[str, np.float64]:
        reward = {agent_id: np.float64(0) for agent_id in env_reward.keys()}

        for agent_id, agent_reward in env_reward.items():
            # Reward for distance travelled
            reward[agent_id] += np.float64(agent_reward * 0.1)
            print("agent_fault_reward:{}".format(agent_reward * 0.1))

            # Designed Reward 
            designed_reward = self.designed_reward(obs, agent_id, agent_reward)
            # reward[agent_id] += np.float64(designed_reward)
           
            # print("Designed_reward:{}".format(designed_reward))
            print("*"*30)

            # Penalty for colliding
            if obs[agent_id]["events"]["collisions"]:
                reward[agent_id] -= np.float64(10)
                print(f"{agent_id}: Collided.")
                break

            # Penalty for driving off road
            if obs[agent_id]["events"]["off_road"]:
                reward[agent_id] -= np.float64(10)
                print(f"{agent_id}: Went off road.")
                break

            # Penalty for driving off route
            if obs[agent_id]["events"]["off_route"]:
                reward[agent_id] -= np.float64(10)
                print(f"{agent_id}: Went off route.")
                break

            # Penalty for driving on road shoulder
            if obs[agent_id]["events"]["on_shoulder"]:
                reward[agent_id] -= np.float64(2)
                break

            # Penalty for driving on wrong way
            if obs[agent_id]["events"]["wrong_way"]:
                reward[agent_id] -= np.float64(10)
                print(f"{agent_id}: Went wrong way.")
                break

            # Reward for reaching goal 30
            if obs[agent_id]["events"]["reached_goal"]:
                reward[agent_id] += np.float64(20)
                step_reward = (1 - self.my_count / 200) * 30
                step_reward = np.clip(step_reward,0,31)
                reward[agent_id] += np.float64(step_reward)
                print("reached_reward:", step_reward)
            # print("my_count:", self.my_count)

           

        return reward

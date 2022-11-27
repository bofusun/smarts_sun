"""
this file contains tuned obs function and reward function
fix ttc calculate
"""
import math

import gym
import numpy as np

from smarts.core.agent import AgentSpec
from smarts.core.agent_interface import AgentInterface
from smarts.core.agent_interface import OGM, NeighborhoodVehicles
from smarts.core.controllers import ActionSpaceType

MAX_LANES = 5  # The maximum number of lanes we expect to see in any scenario.
lane_crash_flag = False  # used for training to signal a flipped car
intersection_crash_flag = False  # used for training to signal intersect crash

# ==================================================
# Continous Action Space
# throttle, brake, steering
# ==================================================

ACTION_SPACE = gym.spaces.Box(
    low=np.array([0.0, 0.0, -1.0]), high=np.array([1.0, 1.0, 1.0]), dtype=np.float32
)

# ==================================================
# Observation Space
# This observation space should match the output of observation(..) below
# ==================================================
OBSERVATION_SPACE_29 = gym.spaces.Dict(
    {
        # To make car follow the waypoints
        # distance from lane center
        "distance_from_center": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
        # relative heading angle from 10 waypoints in 50 forehead waypoints
        "heading_errors": gym.spaces.Box(low=-1.0, high=1.0, shape=(10,)),
        # Car attributes
        # ego speed
        "speed": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
        # ego steering
        "steering": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
        # To make car learn to slow down, overtake or dodge
        # distance to the closest car in each lane
        "lane_dist": gym.spaces.Box(low=-1e10, high=1e10, shape=(5,)),
        # time to collide to the closest car in each lane
        "lane_ttc": gym.spaces.Box(low=-1e10, high=1e10, shape=(5,)),
        # ego lane closest social vehicle relative speed
        "closest_lane_nv_rel_speed": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
        # distance to the closest car in possible intersection direction
        "intersection_ttc": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
        # time to collide to the closest car in possible intersection direction
        "intersection_distance": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
        # intersection closest social vehicle relative speed
        "closest_its_nv_rel_speed": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
        # intersection closest social vehicle relative position in vehicle heading coordinate
        "closest_its_nv_rel_pos": gym.spaces.Box(low=-1e10, high=1e10, shape=(2,)),
    }
)

OBSERVATION_SPACE_24 = gym.spaces.Dict(
    {
        # To make car follow the waypoints
        # distance from lane center
        "distance_from_center": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
        # relative heading angle from 10 waypoints in 50 forehead waypoints
        "heading_errors": gym.spaces.Box(low=-1.0, high=1.0, shape=(10,)),
        # Car attributes
        # ego speed
        "speed": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
        # ego steering
        "steering": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
        # To make car learn to slow down, overtake or dodge
        # distance to the closest car in each lane
        "lane_dist": gym.spaces.Box(low=-1e10, high=1e10, shape=(5,)),
        # time to collide to the closest car in each lane
        "lane_ttc": gym.spaces.Box(low=-1e10, high=1e10, shape=(5,)),
        # ego lane closest social vehicle relative speed
        "closest_lane_nv_rel_speed": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
    }
)


def heading_to_degree(heading):
    # +y = 0 rad. Note the 0 means up direction
    return np.degrees((heading + math.pi) % (2 * math.pi))


def heading_to_vec(heading):
    # axis x: right, y:up
    angle = (heading + math.pi * 0.5) % (2 * math.pi)
    return np.array([math.cos(angle), math.sin(angle)])

def same_edge_behind_nv(ego,neighborhood_vehicle_states,close_nv_index):
    ego_edge_id = ego.lane_id[:-2]
    # 同edge 同向车道
    same_edge_nv = [nv for nv in neighborhood_vehicle_states if nv.lane_id[:-2] == ego_edge_id]
    nv_poses = np.array([nv.position for nv in same_edge_nv])
    # print('nv_pos:',nv_poses)
    ego_pos = np.array([ego.position[:2]])
    # print('ego_pos:',ego_pos)
    if nv_poses.size:
        nv_ego_distance = np.linalg.norm(nv_poses[:, :2][:, np.newaxis] - ego_pos, axis=2)
        # print('nv_ego_distance:',nv_ego_distance)
        overtake_all_close_nv_index = np.where(nv_ego_distance < 30)[0]
        no_eq_len_all_close_nv_index = np.where(nv_ego_distance < 10)[0]
        if not close_nv_index.size:
            return [(neighborhood_vehicle_states[i],nv_ego_distance[i]) for i in overtake_all_close_nv_index],[(neighborhood_vehicle_states[i],nv_ego_distance[i]) for i in no_eq_len_all_close_nv_index]
        else:
            overtake_close_behind_nv_index = [i for i in overtake_all_close_nv_index if i not in close_nv_index]
            no_eq_len_close_behind_nv_index = [i for i in no_eq_len_all_close_nv_index if i not in close_nv_index]
            return [(neighborhood_vehicle_states[i],nv_ego_distance[i]) for i in overtake_close_behind_nv_index],[(neighborhood_vehicle_states[i],nv_ego_distance[i]) for i in no_eq_len_close_behind_nv_index]
    else:
        return [],[]

def ttc_by_path(ego, wp_paths, neighborhood_vehicle_states, ego_closest_wp):
    global lane_crash_flag
    global intersection_crash_flag

    # init flag, dist, ttc, headings
    lane_crash_flag = False
    intersection_crash_flag = False

    # default 10s
    lane_ttc = np.array([1] * 5, dtype=float)
    # default 100m
    lane_dist = np.array([1] * 5, dtype=float)
    # default 120km/h
    closest_lane_nv_rel_speed = 1

    intersection_ttc = 1
    intersection_distance = 1
    closest_its_nv_rel_speed = 1
    # default 100m
    closest_its_nv_rel_pos = np.array([1, 1])

    # here to set invalid value to 0
    wp_paths_num = len(wp_paths)
    lane_ttc[wp_paths_num:] = 0
    lane_dist[wp_paths_num:] = 0
    overtake_behind_index = []
    not_eq_len_behind_index = []
    lane_crash = 0
    intersection_crash = 0

    # return if no neighbour vehicle or off the routes(no waypoint paths)
    if not neighborhood_vehicle_states or not wp_paths_num:
        return (
            lane_ttc,
            lane_dist,
            closest_lane_nv_rel_speed,
            intersection_ttc,
            intersection_distance,
            closest_its_nv_rel_speed,
            closest_its_nv_rel_pos,
            overtake_behind_index,
            not_eq_len_behind_index,
            lane_crash,
            intersection_crash,
        )

    # merge waypoint paths (consider might not the same length)
    merge_waypoint_paths = []
    for wp_path in wp_paths:
        merge_waypoint_paths += wp_path

    wp_poses = np.array([wp.pos for wp in merge_waypoint_paths])

    # compute neighbour vehicle closest wp
    nv_poses = np.array([nv.position for nv in neighborhood_vehicle_states])
    nv_wp_distance = np.linalg.norm(nv_poses[:, :2][:, np.newaxis] - wp_poses, axis=2)
    nv_closest_wp_index = np.argmin(nv_wp_distance, axis=1)
    nv_closest_distance = np.min(nv_wp_distance, axis=1)

    # get not in same lane id social vehicles(intersect vehicles and behind vehicles)
    wp_lane_ids = np.array([wp.lane_id for wp in merge_waypoint_paths])
    nv_lane_ids = np.array([nv.lane_id for nv in neighborhood_vehicle_states])
    not_in_same_lane_id = nv_lane_ids[:, np.newaxis] != wp_lane_ids
    not_in_same_lane_id = np.all(not_in_same_lane_id, axis=1)

    ego_edge_id = ego.lane_id[1:-2] if ego.lane_id[0] == "-" else ego.lane_id[:-2]
    nv_edge_ids = np.array(
        [
            nv.lane_id[1:-2] if nv.lane_id[0] == "-" else nv.lane_id[:-2]
            for nv in neighborhood_vehicle_states
        ]
    )
    not_in_ego_edge_id = nv_edge_ids[:, np.newaxis] != ego_edge_id
    not_in_ego_edge_id = np.squeeze(not_in_ego_edge_id, axis=1)

    is_not_closed_nv = not_in_same_lane_id & not_in_ego_edge_id
    not_closed_nv_index = np.where(is_not_closed_nv)[0]

    # filter sv not close to the waypoints including behind the ego or ahead past the end of the waypoints
    close_nv_index = np.where(nv_closest_distance < 2)[0]

    # 后车
    overtake_behind_nv,not_eq_len_behind_nv = same_edge_behind_nv(ego,neighborhood_vehicle_states,close_nv_index)
    # print('overtake_behind_nv:',overtake_behind_nv)
    # print('not_eq_len_behind_nv:',not_eq_len_behind_nv)
    if len(overtake_behind_nv):
        for v in overtake_behind_nv:
            overtake_behind_index.append(v[0].lane_index)

    if len(not_eq_len_behind_nv):
        for v in not_eq_len_behind_nv:
            not_eq_len_behind_index.append(v[0].lane_index)



    if not close_nv_index.size:
        pass
    else:
        close_nv = [neighborhood_vehicle_states[i] for i in close_nv_index]

        # calculate waypoints distance to ego car along the routes
        wps_with_lane_dist_list = []
        for wp_path in wp_paths:
            path_wp_poses = np.array([wp.pos for wp in wp_path])
            wp_poses_shift = np.roll(path_wp_poses, 1, axis=0)
            wps_with_lane_dist = np.linalg.norm(path_wp_poses - wp_poses_shift, axis=1)
            wps_with_lane_dist[0] = 0
            wps_with_lane_dist = np.cumsum(wps_with_lane_dist)
            wps_with_lane_dist_list += wps_with_lane_dist.tolist()
        wps_with_lane_dist_list = np.array(wps_with_lane_dist_list)

        # get neighbour vehicle closest waypoints index
        nv_closest_wp_index = nv_closest_wp_index[close_nv_index]
        # ego car and neighbour car distance, not very accurate since use the closest wp
        ego_nv_distance = wps_with_lane_dist_list[nv_closest_wp_index]

        # get neighbour vehicle lane index
        nv_lane_index = np.array(
            [merge_waypoint_paths[i].lane_index for i in nv_closest_wp_index]
        )

        # get wp path lane index
        lane_index_list = [wp_path[0].lane_index for wp_path in wp_paths]

        for i, lane_index in enumerate(lane_index_list):
            # get same lane vehicle
            same_lane_nv_index = np.where(nv_lane_index == lane_index)[0]
            if not same_lane_nv_index.size:
                continue
            same_lane_nv_distance = ego_nv_distance[same_lane_nv_index]
            closest_nv_index = same_lane_nv_index[np.argmin(same_lane_nv_distance)]
            closest_nv = close_nv[closest_nv_index]
            closest_nv_speed = closest_nv.speed
            closest_nv_heading = closest_nv.heading
            # radius to degree
            closest_nv_heading = heading_to_degree(closest_nv_heading)

            closest_nv_pos = closest_nv.position[:2]
            bounding_box = closest_nv.bounding_box

            # map the heading to make it consistent with the position coordination
            map_heading = (closest_nv_heading + 90) % 360
            map_heading_radius = np.radians(map_heading)
            nv_heading_vec = np.array(
                [np.cos(map_heading_radius), np.sin(map_heading_radius)]
            )
            nv_heading_vertical_vec = np.array([-nv_heading_vec[1], nv_heading_vec[0]])

            # get four edge center position (consider one vehicle take over two lanes when change lane)
            # maybe not necessary
            closest_nv_front = closest_nv_pos + bounding_box.length * nv_heading_vec
            closest_nv_behind = closest_nv_pos - bounding_box.length * nv_heading_vec
            closest_nv_left = (
                closest_nv_pos + bounding_box.width * nv_heading_vertical_vec
            )
            closest_nv_right = (
                closest_nv_pos - bounding_box.width * nv_heading_vertical_vec
            )
            edge_points = np.array(
                [closest_nv_front, closest_nv_behind, closest_nv_left, closest_nv_right]
            )

            ep_wp_distance = np.linalg.norm(
                edge_points[:, np.newaxis] - wp_poses, axis=2
            )
            ep_closed_wp_index = np.argmin(ep_wp_distance, axis=1)
            ep_closed_wp_lane_index = set(
                [merge_waypoint_paths[i].lane_index for i in ep_closed_wp_index]
                + [lane_index]
            )

            min_distance = np.min(same_lane_nv_distance)

            if ego_closest_wp.lane_index in ep_closed_wp_lane_index:
                if min_distance < 6:
                    lane_crash_flag = True
                    lane_crash = 1
                    print('!!! lane crash !!!')

                nv_wp_heading = (
                    closest_nv_heading
                    - heading_to_degree(
                        merge_waypoint_paths[
                            nv_closest_wp_index[closest_nv_index]
                        ].heading
                    )
                ) % 360

                # find those car just get from intersection lane into ego lane
                if nv_wp_heading > 30 and nv_wp_heading < 330:
                    relative_close_nv_heading = closest_nv_heading - heading_to_degree(
                        ego.heading
                    )
                    # map nv speed to ego car heading
                    map_close_nv_speed = closest_nv_speed * np.cos(
                        np.radians(relative_close_nv_heading)
                    )
                    closest_lane_nv_rel_speed = min(
                        closest_lane_nv_rel_speed,
                        (map_close_nv_speed - ego.speed) * 3.6 / 120,
                    )
                else:
                    closest_lane_nv_rel_speed = min(
                        closest_lane_nv_rel_speed,
                        (closest_nv_speed - ego.speed) * 3.6 / 120,
                    )

            relative_speed_m_per_s = ego.speed - closest_nv_speed

            if abs(relative_speed_m_per_s) < 1e-5:
                relative_speed_m_per_s = 1e-5

            ttc = min_distance / relative_speed_m_per_s
            # normalized into 10s
            ttc /= 10

            for j in ep_closed_wp_lane_index:
                if min_distance / 100 < lane_dist[j]:
                    # normalize into 100m
                    lane_dist[j] = min_distance / 100

                if ttc <= 0:
                    continue

                if j == ego_closest_wp.lane_index:
                    if ttc < 0.1:
                        lane_crash_flag = True
                        lane_crash = 1
                        print('!!! lane crash !!!')

                if ttc < lane_ttc[j]:
                    lane_ttc[j] = ttc

    # get vehicles not in the waypoints lane
    if not not_closed_nv_index.size:
        pass
    else:
        filter_nv = [neighborhood_vehicle_states[i] for i in not_closed_nv_index]

        nv_pos = np.array([nv.position for nv in filter_nv])[:, :2]
        nv_heading = heading_to_degree(np.array([nv.heading for nv in filter_nv]))
        nv_speed = np.array([nv.speed for nv in filter_nv])

        ego_pos = ego.position[:2]
        ego_heading = heading_to_degree(ego.heading)
        ego_speed = ego.speed
        nv_to_ego_vec = nv_pos - ego_pos

        line_heading = (
            (np.arctan2(nv_to_ego_vec[:, 1], nv_to_ego_vec[:, 0]) * 180 / np.pi) - 90
        ) % 360
        nv_to_line_heading = (nv_heading - line_heading) % 360
        ego_to_line_heading = (ego_heading - line_heading) % 360

        # judge two heading whether will intersect
        same_region = (nv_to_line_heading - 180) * (
            ego_to_line_heading - 180
        ) > 0  # both right of line or left of line
        ego_to_nv_heading = ego_to_line_heading - nv_to_line_heading
        valid_relative_angle = (
            (nv_to_line_heading - 180 > 0) & (ego_to_nv_heading > 0)
        ) | ((nv_to_line_heading - 180 < 0) & (ego_to_nv_heading < 0))

        # emit behind vehicles
        valid_intersect_angle = np.abs(line_heading - ego_heading) < 90

        # emit patient vehicles which stay in the intersection
        not_patient_nv = nv_speed > 0.01

        # get valid intersection sv
        intersect_sv_index = np.where(
            same_region & valid_relative_angle & valid_intersect_angle & not_patient_nv
        )[0]

        if not intersect_sv_index.size:
            pass
        else:
            its_nv_pos = nv_pos[intersect_sv_index][:, :2]
            its_nv_speed = nv_speed[intersect_sv_index]
            its_nv_to_line_heading = nv_to_line_heading[intersect_sv_index]
            line_heading = line_heading[intersect_sv_index]
            # ego_to_line_heading = ego_to_line_heading[intersect_sv_index]

            # get intersection closest vehicle
            ego_nv_distance = np.linalg.norm(its_nv_pos - ego_pos, axis=1)
            ego_closest_its_nv_index = np.argmin(ego_nv_distance)
            ego_closest_its_nv_distance = ego_nv_distance[ego_closest_its_nv_index]

            line_heading = line_heading[ego_closest_its_nv_index]
            ego_to_line_heading = (
                heading_to_degree(ego_closest_wp.heading) - line_heading
            ) % 360

            ego_closest_its_nv_speed = its_nv_speed[ego_closest_its_nv_index]
            its_closest_nv_to_line_heading = its_nv_to_line_heading[
                ego_closest_its_nv_index
            ]
            # rel speed along ego-nv line
            closest_nv_rel_speed = ego_speed * np.cos(
                np.radians(ego_to_line_heading)
            ) - ego_closest_its_nv_speed * np.cos(
                np.radians(its_closest_nv_to_line_heading)
            )
            closest_nv_rel_speed_m_s = closest_nv_rel_speed
            if abs(closest_nv_rel_speed_m_s) < 1e-5:
                closest_nv_rel_speed_m_s = 1e-5
            ttc = ego_closest_its_nv_distance / closest_nv_rel_speed_m_s

            intersection_ttc = min(intersection_ttc, ttc / 10)
            intersection_distance = min(
                intersection_distance, ego_closest_its_nv_distance / 100
            )

            # transform relative pos to ego car heading coordinate
            rotate_axis_angle = np.radians(90 - ego_to_line_heading)
            closest_its_nv_rel_pos = (
                np.array(
                    [
                        ego_closest_its_nv_distance * np.cos(rotate_axis_angle),
                        ego_closest_its_nv_distance * np.sin(rotate_axis_angle),
                    ]
                )
                / 100
            )

            closest_its_nv_rel_speed = min(
                closest_its_nv_rel_speed, -closest_nv_rel_speed * 3.6 / 120
            )

            if ttc < 0:
                pass
            else:
                intersection_ttc = min(intersection_ttc, ttc / 10)
                intersection_distance = min(
                    intersection_distance, ego_closest_its_nv_distance / 100
                )

                # if to collide in 3s, make it slow down
                if ttc < 2 or ego_closest_its_nv_distance < 6:
                    intersection_crash_flag = True
                    intersection_crash = 1
                    print('!!! intersection crash !!!')

    return (
        lane_ttc,
        lane_dist,
        closest_lane_nv_rel_speed,
        intersection_ttc,
        intersection_distance,
        closest_its_nv_rel_speed,
        closest_its_nv_rel_pos,
        overtake_behind_index,
        not_eq_len_behind_index,
        lane_crash,
        intersection_crash,
        
    )


def ego_ttc_calc(ego_lane_index, ttc_by_path, lane_dist):
    # transform lane ttc and dist to make ego lane in the array center

    # index need to be set to zero
    # 4: [0,1], 3:[0], 2:[], 1:[4], 0:[3,4]
    zero_index = [[3, 4], [4], [], [0], [0, 1]]
    zero_index = zero_index[ego_lane_index]

    ttc_by_path[zero_index] = 0
    lane_ttc = np.roll(ttc_by_path, 2 - ego_lane_index)
    lane_dist[zero_index] = 0
    ego_lane_dist = np.roll(lane_dist, 2 - ego_lane_index)

    return lane_ttc, ego_lane_dist


def get_distance_from_center(env_obs):
    ego_state = env_obs.ego_vehicle_state
    wp_paths = env_obs.waypoint_paths
    closest_wps = [path[0] for path in wp_paths]

    # distance of vehicle from center of lane
    closest_wp = min(closest_wps, key=lambda wp: wp.dist_to(ego_state.position))
    signed_dist_from_center = closest_wp.signed_lateral_error(ego_state.position)
    lane_hwidth = closest_wp.lane_width * 0.5
    norm_dist_from_center = signed_dist_from_center / lane_hwidth

    return norm_dist_from_center

def get_longestpath_or_emptypath_distance_frome_center(env_obs,ego_lane_index,lane_dist,overtake_behind_index,not_eq_len_behind_index):
    not_eq_len_road = 0
    ego_state = env_obs.ego_vehicle_state
    wp_paths = env_obs.waypoint_paths
    closest_wps = [path[0] for path in wp_paths]
    # print(closest_wps)
    # print('')
    indices = np.array([0, 1, 2, 3, 5, 8, 13, 21, 34, 50])
    # solve case that wps are not enough, then assume the left heading to be same with the last valid.
    wps_len = [len(path) for path in wp_paths]
    max_len_lane_index = np.argmax(wps_len)
    max_len = np.max(wps_len)
    # print('max_lane_index:',max_len_lane_index)

    sample_wp_path = [wp_paths[max_len_lane_index][i] for i in indices]
    lane_heading_errors = np.array([
        math.sin(wp.relative_heading(sample_wp_path[0].heading)) for wp in sample_wp_path
    ])

    # is_all_same_length = True

    # 换道可能会出现 前方出现两条道但目前只有一条道的情况(需要判断)
    # 道路一样长 且为直道时起作用 
    if len(set(wps_len))<=1:
        print('等长路')
        if lane_dist[ego_lane_index] < 1:
            # 找到最空的一条有效道路
            max_dist_lane_indexs = [i for i,x in enumerate(lane_dist) if x==max(lane_dist)]
            # 当本车道不是最空车道 且 空道足够空时
            if ego_lane_index not in max_dist_lane_indexs and lane_dist[max_dist_lane_indexs[0]] - lane_dist[ego_lane_index] > 0.3 and lane_dist[ego_lane_index] <= 0.2:   
                # # 当不是弯道时
                sin_limit = 0.17  # sin阈值(正负0.17约等于正负10度之间)
                acceptable_heading_errors = [abs(error) < sin_limit for error in lane_heading_errors]
                if False not in acceptable_heading_errors:
                    # 如果为直道则找最空
                    # distance of vehicle from center of lane
                    empty_index = empty_lane_index(env_obs,lane_dist,overtake_behind_index)
                    print('empty index:',empty_index)

                    closest_wp = closest_wps[empty_index]
                    # print('closet_wp:',closest_wp)
                    signed_dist_from_center = closest_wp.signed_lateral_error(ego_state.position)
                    lane_hwidth = closest_wp.lane_width * 0.5
                    norm_dist_from_center = signed_dist_from_center / lane_hwidth
                    return norm_dist_from_center,not_eq_len_road,lane_heading_errors

        closest_wp = min(closest_wps, key=lambda wp: wp.dist_to(ego_state.position))
        signed_dist_from_center = closest_wp.signed_lateral_error(ego_state.position)
        lane_hwidth = closest_wp.lane_width * 0.5
        norm_dist_from_center = signed_dist_from_center / lane_hwidth

        print('norm_dist_from_center:',norm_dist_from_center)

        return norm_dist_from_center,not_eq_len_road,lane_heading_errors

    else:
        not_eq_len_road = 1
        # 道路不一样长 存在断头路
        # is_all_same_length = False
        # 防止切换道路时出现与该道路前车发生碰撞
        print('断头路')
        if max_len_lane_index not in not_eq_len_behind_index and lane_dist[max_len_lane_index] > 0.05:
            closest_wp = closest_wps[max_len_lane_index]
            # print('closet_wp:',closest_wp)
            signed_dist_from_center = closest_wp.signed_lateral_error(ego_state.position)
            lane_hwidth = closest_wp.lane_width * 0.5
            norm_dist_from_center = signed_dist_from_center / lane_hwidth
            return norm_dist_from_center,not_eq_len_road,lane_heading_errors
        else:
            closest_wp = min(closest_wps, key=lambda wp: wp.dist_to(ego_state.position))
            signed_dist_from_center = closest_wp.signed_lateral_error(ego_state.position)
            lane_hwidth = closest_wp.lane_width * 0.5
            norm_dist_from_center = signed_dist_from_center / lane_hwidth

            return norm_dist_from_center,not_eq_len_road,lane_heading_errors

def empty_lane_index(env_obs,lane_dist,behind_nv_index):

    ego_state = env_obs.ego_vehicle_state
    wp_paths = env_obs.waypoint_paths

    # solve case that wps are not enough, then assume the left heading to be same with the last valid.
    wps_len = [len(path) for path in wp_paths]
    max_len_lane_index = np.argmax(wps_len)

    closest_wps = [path[0] for path in wp_paths]
    lane_num = len(wp_paths)
    lane_index_list = [wp.lane_index for wp in closest_wps]
    # distance of vehicle from center of lane
    closest_wp = min(closest_wps, key=lambda wp: wp.dist_to(ego_state.position))
    ego_lane_index = closest_wp.lane_index

    empty_lane_index = ego_lane_index
    # 当所有道路一样长度时奖励行驶在最空直道
    if len(set(wps_len))<=1:
        if lane_dist[ego_lane_index] < 1:
            # 找到最空的一条有效道路
            # 当存在多条最优时在任意一条上既可以给奖励
            max_dist_lane_indexs = [i for i,x in enumerate(lane_dist) if x==max(lane_dist)]
            if ego_lane_index not in max_dist_lane_indexs:
                for index in max_dist_lane_indexs:
                    if  index not in behind_nv_index and closest_wps[ego_lane_index] != closest_wps[index]:
                        empty_lane_index = index
                        print('new empty lane index:',empty_lane_index)
                        return empty_lane_index
    
    print('ego_lane_index:',empty_lane_index)
    return empty_lane_index

# ==================================================
# obs function
# ==================================================
def observation_adapter(env_obs):
    """
    Transform the environment's observation into something more suited for your model
    """
    ego_state = env_obs.ego_vehicle_state
    wp_paths = env_obs.waypoint_paths
    closest_wps = [path[0] for path in wp_paths]

    # distance of vehicle from center of lane
    closest_wp = min(closest_wps, key=lambda wp: wp.dist_to(ego_state.position))
    ego_lane_index = closest_wp.lane_index
    # signed_dist_from_center = closest_wp.signed_lateral_error(ego_state.position)
    # lane_hwidth = closest_wp.lane_width * 0.5


    # wp heading errors in current lane in front of vehicle
    indices = np.array([0, 1, 2, 3, 5, 8, 13, 21, 34, 50])

    # solve case that wps are not enough, then assume the left heading to be same with the last valid.
    wps_len = [len(path) for path in wp_paths]
    max_len_lane_index = np.argmax(wps_len)
    max_len = np.max(wps_len)
    sample_wp_path = [wp_paths[max_len_lane_index][i] for i in indices]


    last_wp_index = 0
    for i, wp_index in enumerate(indices):
        if wp_index > max_len - 1:
            indices[i:] = last_wp_index
            break
        last_wp_index = wp_index
    
    ego_lane_index = closest_wp.lane_index

    # sample_wp_path = None
    # print('')
    # print('wp_paths num:',len(wp_paths))
    # print('ego_lane_index:',ego_lane_index)
    # print('max_len_lane_index:',max_len_lane_index)
    # print('')
    # if (closest_wps[ego_lane_index].pos == closest_wps[max_len_lane_index].pos).all():
    #     sample_wp_path = [wp_paths[ego_lane_index][i] for i in indices]
    # else:
    #     sample_wp_path = [wp_paths[max_len_lane_index][i] for i in indices]

    heading_errors = [
        math.sin(wp.relative_heading(ego_state.heading)) for wp in sample_wp_path
    ]


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
        lane_crash,
        intersection_crash,
    ) = ttc_by_path(
        ego_state, wp_paths, env_obs.neighborhood_vehicle_states, closest_wp
    )

    overtake_behind_index = list(set(overtake_behind_index))
    not_eq_len_behind_index = list(set(not_eq_len_behind_index))
    print('overtake behind index:',overtake_behind_index)
    print('not eq len behind index',not_eq_len_behind_index)
    # norm_dist_from_center = signed_dist_from_center / lane_hwidth
    norm_dist_from_center,not_eq_len_road,lane_heading_errors = get_longestpath_or_emptypath_distance_frome_center(env_obs,ego_lane_index,lane_dist,overtake_behind_index,not_eq_len_behind_index)

    lane_ttc, lane_dist = ego_ttc_calc(ego_lane_index, lane_ttc, lane_dist)

    if len(wp_paths) ==2 and (wp_paths[0][0].pos == wp_paths[1][0].pos).all(): 
        # 单一道路即将分为两条情况
        lane_dist[0] = lane_dist[1] = lane_dist[3] = lane_dist[4] = 0
        lane_ttc[0] =  lane_ttc[1] = lane_ttc[3] = lane_ttc[4] = 0
    
    if ego_lane_index in overtake_behind_index:
        overtake_behind_index.remove(ego_lane_index)



    return {
        "distance_from_center": np.array([norm_dist_from_center]),
        "heading_errors": np.array(heading_errors),
        "speed": np.array([ego_state.speed / 120]),
        "steering": np.array([ego_state.steering / (0.5 * math.pi)]),
        "lane_ttc": np.array(lane_ttc),
        "lane_dist": np.array(lane_dist),
        "closest_lane_nv_rel_speed": np.array([closest_lane_nv_rel_speed]),
        "intersection_ttc": np.array([intersection_ttc]),
        "intersection_distance": np.array([intersection_distance]),
        "closest_its_nv_rel_speed": np.array([closest_its_nv_rel_speed]),
        "closest_its_nv_rel_pos": np.array(closest_its_nv_rel_pos),
        "overtake_behind_index": np.array(overtake_behind_index),
        "not_eq_len_road": np.array(not_eq_len_road),
        "not_eq_len_behind_index": np.array(not_eq_len_behind_index),
        "lane_heading_errors": np.array(lane_heading_errors),
        "lane_crash": np.array(lane_crash),
        "intersection_crash": np.array(intersection_crash),
    }


# ==================================================
# reward function
# ==================================================
def reward_adapter(env_obs, env_reward):
    """
    Here you can perform your reward shaping.

    The default reward provided by the environment is the increment in
    distance travelled. Your model will likely require a more
    sophisticated reward function
    """
    global lane_crash_flag
    distance_from_center = get_distance_from_center(env_obs)

    center_penalty = -np.abs(distance_from_center)

    # penalise close proximity to lane cars
    if lane_crash_flag:
        crash_penalty = -5
    else:
        crash_penalty = 0

    # penalise close proximity to intersection cars
    if intersection_crash_flag:
        crash_penalty -= 5

    total_reward = np.sum([1.0 * env_reward])
    total_penalty = np.sum([0.1 * center_penalty, 1 * crash_penalty])

    return (total_reward + total_penalty) / 200.0


def action_adapter(model_action):
    assert len(model_action) == 3
    return np.asarray(model_action)


def info_adapter(reward, info):
    return info


agent_interface = AgentInterface(
    max_episode_steps=None,
    waypoints=True,
    # neighborhood < 60m
    neighborhood_vehicles=NeighborhoodVehicles(radius=60),
    # OGM within 64 * 0.25 = 16
    ogm=OGM(64, 64, 0.25),
    action=ActionSpaceType.Continuous,
)

agent_spec = AgentSpec(
    interface=agent_interface,
    observation_adapter=observation_adapter,
    reward_adapter=reward_adapter,
    action_adapter=action_adapter,
    info_adapter=info_adapter,
)

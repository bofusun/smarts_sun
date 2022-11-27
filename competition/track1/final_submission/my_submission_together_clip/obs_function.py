import math

import gym
import numpy as np

def signed_dist_to_line(point, line_point, line_dir_vec):
    p = vec_2d(point)
    p1 = line_point
    p2 = line_point + line_dir_vec

    u = abs(line_dir_vec[1] * p[0] - line_dir_vec[0] * p[1] + p2[0] * p1[1] - p2[1] * p1[0])
    d = u / np.linalg.norm(line_dir_vec)

    line_normal = np.array([-line_dir_vec[1], line_dir_vec[0]])
    _sign = np.sign(np.dot(p - p1, line_normal))
    return d * _sign

def vec_2d(v):
    assert len(v) >= 2
    return np.array(v[:2])

def radians_to_vec(radians):
    angle = (radians + math.pi * 0.5) % (2 * math.pi)
    return np.array((math.cos(angle), math.sin(angle)))

def heading_to_degree(heading):
    # +y = 0 rad. Note the 0 means up direction
    return np.degrees((heading + math.pi) % (2 * math.pi))


def heading_to_vec(heading):
    # axis x: right, y:up
    angle = (heading + math.pi * 0.5) % (2 * math.pi)
    return np.array([math.cos(angle), math.sin(angle)])

def same_edge_behind_nv(ego,neighborhood_vehicle_states,close_nv_index):
    '''input: 主车(主要使用主车的lane_id 主车的位置) 环境车辆状态 
              close_nv_index 根据waypoint得到的与主车距离很近的邻居车辆的索引
              (np.where(nv_closest_distance < 2)[0])
        output: 和主车同edge的在主车后面的邻居车辆的索引
        
        Q:我这里还不太理解为什么用的是二范数但是能够识别是后车
          这边还是需要使用lane_id的信息
    '''
    ego_edge_id = ego.lane_id[:-2]
    # 同edge 同向车道
    same_edge_nv = [nv for nv in neighborhood_vehicle_states if nv.lane_id[:-2] == ego_edge_id]
    nv_poses = np.array([nv['position'] for nv in same_edge_nv])
    # print('nv_pos:',nv_poses)
    ego_pos = np.array([ego['position'][:2]])
    # print('ego_pos:',ego_pos)
    if nv_poses.size:
        nv_ego_distance = np.linalg.norm(nv_poses[:, :2][:, np.newaxis] - ego_pos, axis=2)
        # print('nv_ego_distance:',nv_ego_distance)
        overtake_all_close_nv_index = np.where(nv_ego_distance < 25)[0]
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
    '''input: ego车的状态 周围车的状态 所有可能的waypoints_path, 距离最近的waypiont
       output: lane_ttc, lane_dist, closest_lan_nv_rel_speed 
               intesection_ttc, intesction_dist 等状态
    '''
    # 先设定一些 核心的变量 检测是否碰撞
    global lane_crash_flag
    global intersection_crash_flag

    # 针对input的特殊情况进行考虑，同时设定output的默认值
    # init flag, dist, ttc, headings
    lane_crash_flag = False
    intersection_crash_flag = False

    """ DAI中是5个lane 这里则是4个lane """
    # default 10s
    lane_ttc = np.array([1] * 10, dtype=float)
    # default 100m
    lane_dist = np.array([1] * 10, dtype=float)
    # default 120km/h
    closest_lane_nv_rel_speed = 1

    intersection_ttc = 1
    intersection_distance = 1
    closest_its_nv_rel_speed = 1
    # default 100m
    closest_its_nv_rel_pos = np.array([1, 1])

    # here to set invalid value to 0
    # 一般来讲我们的 wp_paths个数是4 因此 wp_paths_num<=4
    wp_paths_num = len(wp_paths)
    
    # 如果该场景的车道 lane 数量不足 4 便把lane_ttc,lane_dist的默认值设为0 无效值
    lane_ttc[wp_paths_num:] = 0
    lane_dist[wp_paths_num:] = 0
    """ 是不是和 lane_id 相关的变量最终就是 下面这两个变量那 to be done """
    overtake_behind_index = []
    not_eq_len_behind_index = []

    ''' case1: 如果在ego车的检测范围内没有相应的环境车辆 那么相应的距离 时间 相对速度都都会比较大-达到默认值 
        case2: 车辆偏离路径 也就是说没有相应的path供ego车辆选择 此时也设为默认值
    '''
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
        )

    ''' 将所有的waypoints合并到一个列表中merge_waypoint_paths'''
    # merge waypoint paths (consider might not the same length)
    merge_waypoint_paths = []
    for wp_path in wp_paths:
        merge_waypoint_paths += wp_path

    wp_poses = np.array([wp['pos'] for wp in merge_waypoint_paths])
    # 讲position的三维变成二维
    wp_poses = wp_poses[:, :2]

    # compute neighbour vehicle closest wp
    """ 不是很明白为什么要把nv 和所有的 waypoints 进行比较"""
    nv_poses = np.array([nv['pos'] for nv in neighborhood_vehicle_states])
    nv_wp_distance = np.linalg.norm(nv_poses[:, :2][:, np.newaxis] - wp_poses, axis=2)
    # (10,320) (nv_poses_len, wp_poses_len)
    
    
    # 存在一个核心问题 如何将nv_poses和wp_poses的不同个数处理好 再进行距离计算
    nv_closest_wp_index = np.argmin(nv_wp_distance, axis=1)
    nv_closest_distance = np.min(nv_wp_distance, axis=1)
    # filter sv not close to the waypoints including behind the ego or ahead past the end of the waypoints
    close_nv_index = np.where(nv_closest_distance < 2)[0]

    ''' 开始根据邻居车辆 close_nv_index来得到 lane_ttc等相应的状态 '''
    if not close_nv_index.size:
        pass
    else:
        close_nv = [neighborhood_vehicle_states[i] for i in close_nv_index]

        # calculate waypoints distance to ego car along the routes
        ''' 对wps_with_lane_dist_list不是很理解'''
        wps_with_lane_dist_list = []
        for wp_path in wp_paths:
            path_wp_poses = np.array([wp['pos'] for wp in wp_path])
            path_wp_poses = path_wp_poses[ :, :2]
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
            [merge_waypoint_paths[i]['lane_index'] for i in nv_closest_wp_index]
        )

        # get wp path lane index
        lane_index_list = [wp_path[0]['lane_index'] for wp_path in wp_paths]

        # 考虑 wp_num 数量的 lane_index
        for i, lane_index in enumerate(lane_index_list):
            # get same lane vehicle
            same_lane_nv_index = np.where(nv_lane_index == lane_index)[0]
            # print("same_lane_nv_index {}".format(same_lane_nv_index))
            if not same_lane_nv_index.size:
                continue
            same_lane_nv_distance = ego_nv_distance[same_lane_nv_index]
            # print("same_lane_nv_distance {}".format(same_lane_nv_distance))
            closest_nv_index = same_lane_nv_index[np.argmin(same_lane_nv_distance)]
            closest_nv = close_nv[closest_nv_index]
            closest_nv_speed = closest_nv['speed']
            closest_nv_heading = closest_nv['heading']
            # radius to degree
            closest_nv_heading = heading_to_degree(closest_nv_heading)

            closest_nv_pos = closest_nv['pos'][:2]
            bounding_box = closest_nv['box']
            bounding_box_length = bounding_box[0]
            bounding_box_width = bounding_box[1]

            # map the heading to make it consistent with the position coordination
            # 为什么要进行下面的操作那？
            map_heading = (closest_nv_heading + 90) % 360
            map_heading_radius = np.radians(map_heading)
            nv_heading_vec = np.array(
                [np.cos(map_heading_radius), np.sin(map_heading_radius)]
            )
            nv_heading_vertical_vec = np.array([-nv_heading_vec[1], nv_heading_vec[0]])

            # get four edge center position (consider one vehicle take over two lanes when change lane)
            # maybe not necessary
            closest_nv_front = closest_nv_pos + bounding_box_length * nv_heading_vec
            closest_nv_behind = closest_nv_pos - bounding_box_length * nv_heading_vec
            closest_nv_left = (
                closest_nv_pos + bounding_box_width * nv_heading_vertical_vec
            )
            closest_nv_right = (
                closest_nv_pos - bounding_box_width * nv_heading_vertical_vec
            )
            edge_points = np.array(
                [closest_nv_front, closest_nv_behind, closest_nv_left, closest_nv_right]
            )
           # #  print("edge_points: {}".format(edge_points))

            ep_wp_distance = np.linalg.norm(
                edge_points[:, np.newaxis] - wp_poses, axis=2
            )
            # print("ep_wp_distance_shape: {}".format(ep_wp_distance.shape))
            ep_closed_wp_index = np.argmin(ep_wp_distance, axis=1)
            # print("ep_closed_wp_index: {}".format(ep_closed_wp_index))
            ep_closed_wp_lane_index = set(
                [merge_waypoint_paths[i]['lane_index'] for i in ep_closed_wp_index]
                + [lane_index]
            )
            # print("ep_closed_wp_lane_index: {}".format(ep_closed_wp_lane_index))

            min_distance = np.min(same_lane_nv_distance)

            if ego_closest_wp['lane_index'] in ep_closed_wp_lane_index:
                if min_distance < 6:
                    lane_crash_flag = True
                    print('!!! lane crash !!!')

                nv_wp_heading = (
                    closest_nv_heading
                    - heading_to_degree(
                        merge_waypoint_paths[
                            nv_closest_wp_index[closest_nv_index]
                        ]['heading']
                    )
                ) % 360

                # find those car just get from intersection lane into ego lane
                if nv_wp_heading > 30 and nv_wp_heading < 330:
                    relative_close_nv_heading = closest_nv_heading - heading_to_degree(
                        ego['heading']
                    )
                    # map nv speed to ego car heading
                    map_close_nv_speed = closest_nv_speed * np.cos(
                        np.radians(relative_close_nv_heading)
                    )
                    closest_lane_nv_rel_speed = min(
                        closest_lane_nv_rel_speed,
                        (map_close_nv_speed - ego['speed']) * 3.6 / 120,
                    )
                else:
                    closest_lane_nv_rel_speed = min(
                        closest_lane_nv_rel_speed,
                        (closest_nv_speed - ego['speed']) * 3.6 / 120,
                    )

            relative_speed_m_per_s = ego['speed'] - closest_nv_speed

            if abs(relative_speed_m_per_s) < 1e-5:
                relative_speed_m_per_s = 1e-5

            ttc = min_distance / relative_speed_m_per_s
            # normalized into 10s
            ttc /= 10

            for j in ep_closed_wp_lane_index:
                if j < 4:
                    if min_distance / 100 < lane_dist[j]:
                        # normalize into 100m
                        lane_dist[j] = min_distance / 100

                if ttc <= 0:
                    continue

                if j == ego_closest_wp['lane_index']:
                    if ttc < 0.1:
                        lane_crash_flag = True

                if j < 4:
                    if ttc < lane_ttc[j]:
                        lane_ttc[j] = ttc

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
    )


def ego_ttc_calc(ego_lane_index, ttc_by_path, lane_dist):
    ''' 
    没看懂这个函数 为什么要有这个zero_index啊
    '''
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
    closest_wp = min(closest_wps, key=lambda wp: wp.dist_to(ego_state['pos']))
    signed_dist_from_center = closest_wp.signed_lateral_error(ego_state['pos'])
    lane_hwidth = closest_wp.lane_width * 0.5
    norm_dist_from_center = signed_dist_from_center / lane_hwidth

    return norm_dist_from_center

def get_longestpath_or_emptypath_distance_frome_center(env_obs,ego_lane_index,lane_dist,overtake_behind_index,not_eq_len_behind_index):
    not_eq_len_road = 0
    ego_state = env_obs['ego']
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
            if ego_lane_index not in max_dist_lane_indexs and lane_dist[max_dist_lane_indexs[0]] - lane_dist[ego_lane_index] > 0.5 and lane_dist[ego_lane_index] <= 0.3 and ego_state.speed <8:   
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
                    signed_dist_from_center = closest_wp.signed_lateral_error(ego_state['pos'])
                    lane_hwidth = closest_wp.lane_width * 0.5
                    norm_dist_from_center = signed_dist_from_center / lane_hwidth
                    return norm_dist_from_center,not_eq_len_road

        closest_wp = min(closest_wps, key=lambda wp: wp.dist_to(ego_state['pos']))
        signed_dist_from_center = closest_wp.signed_lateral_error(ego_state['pos'])
        lane_hwidth = closest_wp.lane_width * 0.5
        norm_dist_from_center = signed_dist_from_center / lane_hwidth

        print('norm_dist_from_center:',norm_dist_from_center)

        return norm_dist_from_center,not_eq_len_road

    else:
        not_eq_len_road = 1
        # 道路不一样长 存在断头路
        # is_all_same_length = False
        # 防止切换道路时出现与该道路前车发生碰撞
        print('断头路')
        if max_len_lane_index not in not_eq_len_behind_index:
            closest_wp = closest_wps[max_len_lane_index]
            # print('closet_wp:',closest_wp)
            signed_dist_from_center = closest_wp.signed_lateral_error(ego_state['pos'])
            lane_hwidth = closest_wp.lane_width * 0.5
            norm_dist_from_center = signed_dist_from_center / lane_hwidth
            return norm_dist_from_center,not_eq_len_road
        else:
            closest_wp = min(closest_wps, key=lambda wp: wp.dist_to(ego_state['pos']))
            signed_dist_from_center = closest_wp.signed_lateral_error(ego_state['pos'])
            lane_hwidth = closest_wp.lane_width * 0.5
            norm_dist_from_center = signed_dist_from_center / lane_hwidth

            return norm_dist_from_center,not_eq_len_road

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
    closest_wp = min(closest_wps, key=lambda wp: wp.dist_to(ego_state['pos']))
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






###############################################################################################################
# 目标图像观测处理
###############################################################################################################
# 输入为目标的x,y坐标；当前x,y坐标和当前方向，将相对于原点的坐标转换为原点位置到当前位置方向为y轴的坐标系下的坐标
## 这里还需要再推导以下
def get_trans_coor(goal_x, goal_y, cur_x, cur_y, cur_heading):
    cur_heading = (cur_heading + np.pi) % (2 * np.pi) - np.pi
    # 如果方向向左
    if 0 < cur_heading < math.pi:
        theta = cur_heading
    # 如果方向向右
    elif -(math.pi) < cur_heading < 0:
        theta = 2 * math.pi + cur_heading
    # 如果方向为正北
    elif cur_heading == 0:
        theta = 0
    # 如果方向为正南
    elif (cur_heading == math.pi) or (cur_heading == -(math.pi)):
        theta = 2 * math.pi + cur_heading

    # 转移矩阵
    trans_matrix = np.array([[math.cos(theta), math.sin(theta)], [-math.sin(theta), math.cos(theta)]])
    # 转移后的当前位置和目标位置
    cur_pos = np.array([[cur_x], [cur_y]])
    goal_pos = np.array([[goal_x], [goal_y]])
    ## np.matmul 表示的是矩阵乘法
    ## np.round 表示的是取整（采取的是四舍五入的方法） 这里表示的是对小数点后第五位进行取整
    trans_cur = np.round(np.matmul(trans_matrix, cur_pos), 5)
    trans_goal = np.round(np.matmul(trans_matrix, goal_pos), 5)
    # 返回转换后的当前位置和目标位置
    return [trans_cur, trans_goal]


# 获取车辆对目标的观测，为112*112的大小，目标位置为112像素
def get_goal_layer(goal_x, goal_y, cur_x, cur_y, cur_heading):
    # 将当前坐标和目标坐标转换到新坐标系下
    trans_coor = get_trans_coor(goal_x, goal_y, cur_x, cur_y, cur_heading)
    trans_cur = trans_coor[0]
    trans_goal = trans_coor[1]
    ## 判定目标是否在车辆的可视范围内[50,50]的正方形区域内
    if (trans_cur[0, 0] - 56) <= trans_goal[0, 0] <= (trans_cur[0, 0] + 56):
        if (trans_cur[1, 0] - 56) <= trans_goal[1, 0] <= (trans_cur[1, 0] + 56):
            inside = True
        else:
            inside = False
    else:
        inside = False
    # 获得对于目标的图像观测，为112*112，目标像素为112，其他为0
    if inside:
        goal_obs = inside_coor_to_pixel(
            trans_goal[0, 0], trans_goal[1, 0], trans_cur[0, 0], trans_cur[1, 0]
        )
    else:
        goal_obs = outside_coor_to_pixel(
            trans_goal[0, 0], trans_goal[1, 0], trans_cur[0, 0], trans_cur[1, 0]
        )
    return goal_obs


# 目标在50*50的观测里面，则转化为256*256的图像
def inside_coor_to_pixel(goal_x, goal_y, cur_x, cur_y):
    ## 我觉得这里应该是SMARTS数据处理过程中的比例-需要查询图像生成的代码
    ## ratio的作用就是在实际距离和pixel之间进行转化
    ratio = 1  # 256 pixels corresonds to 50 meters
    x_diff = abs(goal_x - cur_x)
    y_diff = abs(goal_y - cur_y)

    # find true condition of first quadrant
    if goal_x > cur_x and goal_y > cur_y:
        x_pixel_loc = min(
            56 + round(x_diff * ratio), 111
        )  # cap on 256 which is the right edge
        y_pixel_loc = max(
            55 - round(y_diff * ratio), 0
        )  # cap on 0 which is the upper edge

    # find second quadrant
    elif goal_x < cur_x and goal_y > cur_y:
        x_pixel_loc = max(
            55 - round(x_diff * ratio), 0
        )  # cap on 0 which is the left edge
        y_pixel_loc = max(
            55 - round(y_diff * ratio), 0
        )  # cap on 0 which is the upper edge

    # To find third quadrant
    elif goal_x < cur_x and goal_y < cur_y:
        x_pixel_loc = max(
            55 - round(x_diff * ratio), 0
        )  # cap on 0 which is the left edge
        y_pixel_loc = min(
            56 + round(y_diff * ratio), 111
        )  # cap on 256 which is the bottom edge

    # To find Fourth quadrant
    elif goal_x > cur_x and goal_y < cur_y:
        x_pixel_loc = min(
            56 + round(x_diff * ratio), 111
        )  # cap on 256 which is the right edge
        y_pixel_loc = min(
            56 + round(y_diff * ratio), 111
        )  # cap on 256 which is the bottom edge

    # To find if goal is at cur (do not change to elif)
    if (abs(cur_x) - 0.05 <= abs(goal_x) <= abs(cur_x) + 0.05) and (
        abs(cur_y) - 0.05 <= abs(goal_y) <= abs(cur_y) + 0.05
    ):
        x_pixel_loc = 56
        y_pixel_loc = 56

    # On x-axis
    elif (abs(cur_y) - 0.05 <= abs(goal_y) <= abs(cur_y) + 0.05) and goal_x != cur_x:
        if goal_x >= cur_x:
            x_pixel_loc = min(56 + round(x_diff * ratio), 111)
        else:
            x_pixel_loc = max(55 - round(x_diff * ratio), 0)
        y_pixel_loc = min(56 + round(y_diff * ratio), 111)

    # On y-axis
    elif (abs(cur_x) - 0.05 <= abs(goal_x) <= abs(cur_x) + 0.05) and goal_y != cur_y:
        if goal_y >= cur_y:
            y_pixel_loc = max(55 - round(y_diff * ratio), 0)
        else:
            y_pixel_loc = min(56 + round(y_diff * ratio), 111)
        x_pixel_loc = min(56 + round(x_diff * ratio), 111)

    goal_obs = np.zeros((1, 112, 112))
    goal_obs[0, y_pixel_loc, x_pixel_loc] = 111
    return goal_obs

# 目标在50*50的观测外面，则转化到256*256的图像观测的边缘外侧
def outside_coor_to_pixel(goal_x, goal_y, cur_x, cur_y):
    ratio = 1  # 256 pixels corresonds to 25 meters
    x_diff = abs(goal_x - cur_x)
    y_diff = abs(goal_y - cur_y)

    # find true condition of first quadrant
    if goal_x > cur_x and goal_y > cur_y:
        theta = math.atan(y_diff / x_diff)
        if 0 < theta < (math.pi / 4):
            x_pixel_loc = 111
            y_pixel_loc = max(55 - round((56 * (y_diff / x_diff)) * ratio), 0)
        elif (math.pi / 4) < theta < (math.pi / 2):
            x_pixel_loc = min(56 + round((56 / (y_diff / x_diff)) * ratio), 111)
            y_pixel_loc = 0
        elif theta == (math.pi / 4):
            x_pixel_loc = 111
            y_pixel_loc = 0

    # find second quadrant
    elif goal_x < cur_x and goal_y > cur_y:
        theta = math.atan(y_diff / x_diff)
        if 0 < theta < (math.pi / 4):
            x_pixel_loc = 0
            y_pixel_loc = max(55 - round((56 * (y_diff / x_diff)) * ratio), 0)
        elif (math.pi / 4) < theta < (math.pi / 2):
            x_pixel_loc = max(55 - round((56 / (y_diff / x_diff)) * ratio), 0)
            y_pixel_loc = 0
        elif theta == (math.pi / 4):
            x_pixel_loc = 0
            y_pixel_loc = 0

    # To find third quadrant
    elif goal_x < cur_x and goal_y < cur_y:
        theta = math.atan(y_diff / x_diff)
        if 0 < theta < (math.pi / 4):
            x_pixel_loc = 0
            y_pixel_loc = min(56 + round((56 * (y_diff / x_diff)) * ratio), 111)
        elif (math.pi / 4) < theta < (math.pi / 2):
            x_pixel_loc = max(55 - round((56 / (y_diff / x_diff)) * ratio), 0)
            y_pixel_loc = 111
        elif theta == (math.pi / 4):
            x_pixel_loc = 0
            y_pixel_loc = 111

    # To find Fourth quadrant
    elif goal_x > cur_x and goal_y < cur_y:
        theta = math.atan(y_diff / x_diff)
        if 0 < theta < (math.pi / 4):
            x_pixel_loc = 111
            y_pixel_loc = min(56 + round((56 * (y_diff / x_diff)) * ratio), 111)
        elif (math.pi / 4) < theta < (math.pi / 2):
            x_pixel_loc = min(56 + round((56 / (y_diff / x_diff)) * ratio), 111)
            y_pixel_loc = 111
        elif theta == (math.pi / 4):
            x_pixel_loc = 111
            y_pixel_loc = 111

    # On x-axis (do not change to elif)
    if (abs(cur_y) - 0.05 <= abs(goal_y) <= abs(cur_y) + 0.05) and goal_x != cur_x:
        if goal_x >= cur_x:
            x_pixel_loc = 111
        else:
            x_pixel_loc = 0
        y_pixel_loc = 111

    # On y-axis
    elif (abs(cur_x) - 0.05 <= abs(goal_x) <= abs(cur_x) + 0.05) and goal_y != cur_y:
        if goal_y >= cur_y:
            y_pixel_loc = 0
        else:
            y_pixel_loc = 111
        x_pixel_loc = 56

    goal_obs = np.zeros((1, 112, 112))
    goal_obs[0, y_pixel_loc, x_pixel_loc] = 111
    return goal_obs

# 目标区域奖励，若车辆接近目标则给10的奖励，否则为0
def goal_region_reward(threshold, goal_x, goal_y, cur_x, cur_y):
    eucl_distance = math.sqrt((goal_x - cur_x) ** 2 + (goal_y - cur_y) ** 2)
    # 判定是否进入限制距离
    if eucl_distance <= threshold:
        return 10
    else:
        return 0


def global_target_pose(action, agent_obs):

    cur_x = agent_obs["ego"]["pos"][0]
    cur_y = agent_obs["ego"]["pos"][1]
    cur_heading = agent_obs["ego"]["heading"]

    if 0 < cur_heading < math.pi:  # Facing Left Half
        theta = cur_heading

    elif -(math.pi) < cur_heading < 0:  # Facing Right Half
        theta = 2 * math.pi + cur_heading

    elif cur_heading == 0:  # Facing up North
        theta = 0

    elif (cur_heading == math.pi) or (cur_heading == -(math.pi)):  # Facing South
        theta = 2 * math.pi + cur_heading

    trans_matrix = np.array(
        [[math.cos(theta), math.sin(theta)], [-math.sin(theta), math.cos(theta)]]
    )
    action_bev = np.array([[action[0]], [action[1]]])
    action_global = np.matmul(np.transpose(trans_matrix), action_bev)
    target_pose = np.array(
        [
            cur_x + action_global[0],
            cur_y + action_global[1],
            action[2] + cur_heading,
            0.1,
        ]
    )

    return target_pose




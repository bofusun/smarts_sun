import math
import numpy as np

# 输入为目标的x,y坐标；当前x,y坐标和当前方向，将相对于原点的坐标转换为原点位置到当前位置方向为y轴的坐标系下的坐标
## 这里还需要再推导以下
def get_trans_coor(goal_x, goal_y, cur_x, cur_y, cur_heading):
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


# 获取车辆对目标的观测，为255*255的大小，目标位置为255像素
def get_goal_layer(goal_x, goal_y, cur_x, cur_y, cur_heading):
    # 将当前坐标和目标坐标转换到新坐标系下
    trans_coor = get_trans_coor(goal_x, goal_y, cur_x, cur_y, cur_heading)
    trans_cur = trans_coor[0]
    trans_goal = trans_coor[1]
    ## 判定目标是否在车辆的可视范围内[50,50]的正方形区域内
    if (trans_cur[0, 0] - 25) <= trans_goal[0, 0] <= (trans_cur[0, 0] + 25):
        if (trans_cur[1, 0] - 25) <= trans_goal[1, 0] <= (trans_cur[1, 0] + 25):
            inside = True
        else:
            inside = False
    else:
        inside = False
    # 获得对于目标的图像观测，为256*256，目标像素为255，其他为0
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
    ratio = 256 / 50  # 256 pixels corresonds to 50 meters
    x_diff = abs(goal_x - cur_x)
    y_diff = abs(goal_y - cur_y)

    # find true condition of first quadrant
    if goal_x > cur_x and goal_y > cur_y:
        x_pixel_loc = min(
            128 + round(x_diff * ratio), 255
        )  # cap on 256 which is the right edge
        y_pixel_loc = max(
            127 - round(y_diff * ratio), 0
        )  # cap on 0 which is the upper edge

    # find second quadrant
    elif goal_x < cur_x and goal_y > cur_y:
        x_pixel_loc = max(
            127 - round(x_diff * ratio), 0
        )  # cap on 0 which is the left edge
        y_pixel_loc = max(
            127 - round(y_diff * ratio), 0
        )  # cap on 0 which is the upper edge

    # To find third quadrant
    elif goal_x < cur_x and goal_y < cur_y:
        x_pixel_loc = max(
            127 - round(x_diff * ratio), 0
        )  # cap on 0 which is the left edge
        y_pixel_loc = min(
            128 + round(y_diff * ratio), 255
        )  # cap on 256 which is the bottom edge

    # To find Fourth quadrant
    elif goal_x > cur_x and goal_y < cur_y:
        x_pixel_loc = min(
            128 + round(x_diff * ratio), 255
        )  # cap on 256 which is the right edge
        y_pixel_loc = min(
            128 + round(y_diff * ratio), 255
        )  # cap on 256 which is the bottom edge

    # To find if goal is at cur (do not change to elif)
    if (abs(cur_x) - 0.05 <= abs(goal_x) <= abs(cur_x) + 0.05) and (
        abs(cur_y) - 0.05 <= abs(goal_y) <= abs(cur_y) + 0.05
    ):
        x_pixel_loc = 128
        y_pixel_loc = 128

    # On x-axis
    elif (abs(cur_y) - 0.05 <= abs(goal_y) <= abs(cur_y) + 0.05) and goal_x != cur_x:
        if goal_x >= cur_x:
            x_pixel_loc = min(128 + round(x_diff * ratio), 255)
        else:
            x_pixel_loc = max(127 - round(x_diff * ratio), 0)
        y_pixel_loc = min(128 + round(y_diff * ratio), 255)

    # On y-axis
    elif (abs(cur_x) - 0.05 <= abs(goal_x) <= abs(cur_x) + 0.05) and goal_y != cur_y:
        if goal_y >= cur_y:
            y_pixel_loc = max(127 - round(y_diff * ratio), 0)
        else:
            y_pixel_loc = min(128 + round(y_diff * ratio), 255)
        x_pixel_loc = min(128 + round(x_diff * ratio), 255)

    goal_obs = np.zeros((1, 256, 256))
    goal_obs[0, y_pixel_loc, x_pixel_loc] = 255
    return goal_obs

# 目标在50*50的观测外面，则转化到256*256的图像观测的边缘外侧
def outside_coor_to_pixel(goal_x, goal_y, cur_x, cur_y):
    ratio = 256 / 50  # 256 pixels corresonds to 25 meters
    x_diff = abs(goal_x - cur_x)
    y_diff = abs(goal_y - cur_y)

    # find true condition of first quadrant
    if goal_x > cur_x and goal_y > cur_y:
        theta = math.atan(y_diff / x_diff)
        if 0 < theta < (math.pi / 4):
            x_pixel_loc = 255
            y_pixel_loc = max(127 - round((25 * (y_diff / x_diff)) * ratio), 0)
        elif (math.pi / 4) < theta < (math.pi / 2):
            x_pixel_loc = min(128 + round((25 / (y_diff / x_diff)) * ratio), 255)
            y_pixel_loc = 0
        elif theta == (math.pi / 4):
            x_pixel_loc = 255
            y_pixel_loc = 0

    # find second quadrant
    elif goal_x < cur_x and goal_y > cur_y:
        theta = math.atan(y_diff / x_diff)
        if 0 < theta < (math.pi / 4):
            x_pixel_loc = 0
            y_pixel_loc = max(127 - round((25 * (y_diff / x_diff)) * ratio), 0)
        elif (math.pi / 4) < theta < (math.pi / 2):
            x_pixel_loc = max(127 - round((25 / (y_diff / x_diff)) * ratio), 0)
            y_pixel_loc = 0
        elif theta == (math.pi / 4):
            x_pixel_loc = 0
            y_pixel_loc = 0

    # To find third quadrant
    elif goal_x < cur_x and goal_y < cur_y:
        theta = math.atan(y_diff / x_diff)
        if 0 < theta < (math.pi / 4):
            x_pixel_loc = 0
            y_pixel_loc = min(128 + round((25 * (y_diff / x_diff)) * ratio), 255)
        elif (math.pi / 4) < theta < (math.pi / 2):
            x_pixel_loc = max(127 - round((25 / (y_diff / x_diff)) * ratio), 0)
            y_pixel_loc = 255
        elif theta == (math.pi / 4):
            x_pixel_loc = 0
            y_pixel_loc = 255

    # To find Fourth quadrant
    elif goal_x > cur_x and goal_y < cur_y:
        theta = math.atan(y_diff / x_diff)
        if 0 < theta < (math.pi / 4):
            x_pixel_loc = 255
            y_pixel_loc = min(128 + round((25 * (y_diff / x_diff)) * ratio), 255)
        elif (math.pi / 4) < theta < (math.pi / 2):
            x_pixel_loc = min(128 + round((25 / (y_diff / x_diff)) * ratio), 255)
            y_pixel_loc = 255
        elif theta == (math.pi / 4):
            x_pixel_loc = 255
            y_pixel_loc = 255

    # On x-axis (do not change to elif)
    if (abs(cur_y) - 0.05 <= abs(goal_y) <= abs(cur_y) + 0.05) and goal_x != cur_x:
        if goal_x >= cur_x:
            x_pixel_loc = 255
        else:
            x_pixel_loc = 0
        y_pixel_loc = 128

    # On y-axis
    elif (abs(cur_x) - 0.05 <= abs(goal_x) <= abs(cur_x) + 0.05) and goal_y != cur_y:
        if goal_y >= cur_y:
            y_pixel_loc = 0
        else:
            y_pixel_loc = 255
        x_pixel_loc = 128

    goal_obs = np.zeros((1, 256, 256))
    goal_obs[0, y_pixel_loc, x_pixel_loc] = 255
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

import torch
from pathlib import Path
from typing import Any, Dict
from network_init_new import *

def global_target_pose(action, pos_and_heading):
    # 获取当前状态
    cur_x = pos_and_heading[0]
    cur_y = pos_and_heading[1]
    cur_heading = pos_and_heading[2]
    speed_limit = pos_and_heading[-1]
    
    # 进行clip切割
    action[0] = np.clip(action[0],-1,1)
    action[1] = np.clip(action[1],-1,1)
    
    # 获取当前动作 最大72km/h
    angle = action[1] * 0.1
    magnitude = (action[0] + 1) * 0.7
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

def submitted_wrappers():
    """Return environment wrappers for wrapping the evaluation environment.
    Each wrapper is of the form: Callable[[env], env]. Use of wrappers is
    optional. If wrappers are not used, return empty list [].

    Returns:
        List[wrappers]: List of wrappers. Default is empty list [].
    """

    from action import Action as DiscreteAction
    from observation import Concatenate, FilterObs, SaveObs
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, SubprocVecEnv

    from smarts.core.controllers import ActionSpaceType
    from smarts.env.wrappers.format_action import FormatAction
    from smarts.env.wrappers.format_obs import FormatObs
    from smarts.env.wrappers.frame_stack import FrameStack
    from smarts.env.wrappers.single_agent import SingleAgent

    # fmt: off
    wrappers = [
        # 设置观测空间格式，满足gym格式
        FormatObs,
        # 设置动作空间格式，满足gym格式 TargetPose Continuous
        lambda env: FormatAction(env=env, space=ActionSpaceType["TargetPose"]),
        # 保存观测设置
        SaveObs,
        # 过滤出需要的观测
        FilterObs,
        # 将接口修改为单代理接口，该接口与gym等库兼容。
        # SingleAgent,
        # lambda env: DummyVecEnv([lambda: env]),
        # lambda env: VecMonitor(venv=env, filename=str(self.logdir), info_keywords=("is_success",))
    ]
    # fmt: on
    return wrappers




class Policy():
    def __init__(self):
        # 设备
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device
        # 模型参数 100/45
        actor_input_dim = 68
        action_dim = 2
        hidden_dim = 256
        heads_num = 2
        key_dim = 64
        value_dim = 64
        layer_num = 2
        mlp_hidden_dim = 512
        output_dim = 6
        # 加载模型
        self.actor_net = All_Mix_Actor_model_action_pose(actor_input_dim, action_dim, heads_num, key_dim, value_dim, layer_num, mlp_hidden_dim, device).to(device=device)
        # 加载参数
        self.load_param()


    # 选择动作
    def select_action(self, all_vector_state, rgb_state, together_vector_state, trans_vector_state, judge_state, action=None, determine=False):
        with torch.no_grad():
            mu, sigma, action, log_prob, entropy, scenario_output = self.actor_net(all_vector_state, rgb_state, together_vector_state, trans_vector_state, judge_state, action, determine=determine)
        return mu, sigma, action, log_prob, entropy

    # 加载参数
    def load_param(self):
        actor_path = Path(__file__).absolute().parents[0] / "first_14_actor_net_110.pth"
        self.actor_net.load_state_dict(torch.load(actor_path, map_location=self.device))

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

    def act(self, obs: Dict[str, Any]):
        wrapped_act = {}
        for agent_id, agent_obs in obs.items():
            all_vector_state = agent_obs["brief_final_obs"]
            rgb_state = agent_obs["rgb"]
            together_vector_state = agent_obs["brief_together_obs"]
            trans_vector_state = agent_obs["brief_transformer_pos_and_heading_2d"]
            judge_state = agent_obs["brief_judge"]
            # 转为torch
            all_vector_state, rgb_state, together_vector_state, trans_vector_state, judge_state = self.numpy_2_torch(all_vector_state, rgb_state, together_vector_state, trans_vector_state, judge_state)
            # 增大维度
            all_vector_state, rgb_state, together_vector_state, trans_vector_state, judge_state = self.state_unsqueeze(all_vector_state, rgb_state, together_vector_state, trans_vector_state, judge_state)
            # 获取 action state
            action_state = agent_obs["pos_and_heading"]
            # 生成动作
            with torch.no_grad():
                mu, sigma, action, log_prob, entropy = self.select_action(all_vector_state, rgb_state, together_vector_state, trans_vector_state, judge_state, determine=True)
            action = global_target_pose(action[0].cpu().numpy(), action_state)
            # 更新action
            wrapped_act.update({agent_id: action})

        return wrapped_act



class RandomPolicy():
    """A sample policy with random actions. Note that only the class named `Policy`
    will be tested during evaluation."""

    def __init__(self):
        """All policy initialization matters, including loading of model, is
        performed here. To be implemented by the user.
        """
        import gym

        self._action_space = gym.spaces.Discrete(4)

    def act(self, obs: Dict[str, Any]):
        """Act function to be implemented by user.

        Args:
            obs (Dict[str, Any]): A dictionary of observation for each ego agent step.

        Returns:
            Dict[str, Any]: A dictionary of actions for each ego agent.
        """
        wrapped_act = {}
        for agent_id, agent_obs in obs.items():
            action = self._action_space.sample()
            wrapped_act.update({agent_id: action})

        return wrapped_act

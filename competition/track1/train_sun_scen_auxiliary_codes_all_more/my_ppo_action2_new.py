"""
连续动作空间三个动作变为两个
"""
import os
import sys
import copy
import time
import torch
import pickle
import logging
import itertools
import numpy as np
from my_env import *
from network_init import *
from normalization import *
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple
from torch.utils.data.sampler import SubsetRandomSampler, BatchSampler

# 记录logger
def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

# 继续加载logger
def get_logger_continue(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "a")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

# 存储状态
def transition(actor_state, critic_state, action, reward, log_prob, done, critic_value, device):
    Transition = namedtuple('transition',['actor_state', 'critic_state', 'action', 'reward', 'log_prob', 'done', 'mask', \
                            'advantage', 'return_value', 'critic_value'])
    return Transition(actor_state, critic_state, action, torch.from_numpy(reward).to(device), log_prob, \
                      done, torch.from_numpy(1 - done).to(device), 0, 0, critic_value)

class Params():
    def __init__(self, actor_input_dim, critic_input_dim, action_dim, hidden_dim, heads_num, key_dim, value_dim, \
                 layer_num, mlp_hidden_dim, seq_len, train_envs, test_env, save_path, buffer_capacity, discount, \
                 use_gae, gae_tau, epoch, clip_param, train_together, target_kl, gradient_clip, entropy_weight, \
                 max_steps, save_interval, log_interval, eval_interval, eval_num, batch_size, device, value_clip, \
                 lr, lr_decay, norm_state, norm_reward, norm_advantage, norm_value_loss, kl_early_stop, schedule_clip, \
                 load_counter, actor_net_type, critic_net_type, env_num):
        self.actor_input_dim = actor_input_dim
        self.critic_input_dim = critic_input_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.heads_num = heads_num
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.layer_num = layer_num
        self.mlp_hidden_dim = mlp_hidden_dim
        self.seq_len = seq_len
        self.train_envs = train_envs
        self.test_env = test_env
        self.save_path = save_path
        self.buffer_capacity = buffer_capacity
        self.discount = discount
        self.use_gae = use_gae
        self.gae_tau = gae_tau
        self.epoch = epoch
        self.clip_param = clip_param
        self.train_together = train_together
        self.target_kl = target_kl
        self.gradient_clip = gradient_clip
        self.entropy_weight = entropy_weight
        self.max_steps = max_steps
        self.save_interval = save_interval
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.eval_num = eval_num
        self.batch_size = batch_size
        self.device = device
        self.value_clip = value_clip
        self.lr = lr
        self.lr_decay = lr_decay
        self.norm_state = norm_state
        self.norm_reward = norm_reward
        self.norm_advantage = norm_advantage
        self.norm_value_loss = norm_value_loss
        self.kl_early_stop = kl_early_stop
        self.schedule_clip = schedule_clip
        self.load_counter = load_counter
        self.actor_net_type = actor_net_type
        self.critic_net_type = critic_net_type
        self.env_num = env_num
        

class PPO():
    def __init__(self, params):
        # 环境设定
        self.scenario_names = params.train_envs[0].scenarios
        self.env_num = params.env_num
        self.train_envs = ParallelEnv(params.train_envs)
        self.test_envs = params.train_envs
        # 网络维度
        self.actor_input_dim = params.actor_input_dim
        self.critic_input_dim = params.critic_input_dim
        self.action_dim = params.action_dim
        self.hidden_dim = params.hidden_dim
        self.heads_num = params.heads_num
        self.key_dim = params.key_dim
        self.value_dim = params.value_dim
        self.layer_num = params.layer_num
        self.mlp_hidden_dim = params.mlp_hidden_dim
        self.seq_len = params.seq_len
        # 保存路径
        self.save_path = params.save_path
        # 是否加载模型
        self.load_counter = params.load_counter
        # 添加Actor网络
        self.actor_net_type = params.actor_net_type
        self.critic_net_type = params.critic_net_type
        if params.actor_net_type == "mlp":
            self.actor_net = Mlp_Actor_model(self.actor_input_dim, self.action_dim, params.device).to(device=params.device)
        elif params.actor_net_type == "cnn1":
            self.actor_net = CNN_Actor_model1(self.action_dim, params.device).to(device=params.device)
        # 添加Critic网络
        if params.critic_net_type == "mlp":
            self.critic_net = Mlp_Critic_net(self.critic_input_dim, params.device).to(params.device)
        elif params.critic_net_type == "cnn1":
            self.critic_net = CNN_Critic_net1(params.device).to(params.device)
        # 回放容器
        self.replay_buffer = []
        # buffer 存储次数
        self.buffer_counter = 0
        # buffer 容量
        self.buffer_capacity = params.buffer_capacity
        # 训练次数
        self.training_step = 0
        # 折扣因子
        self.discount = params.discount
        # 是否使用gae
        self.use_gae = params.use_gae
        # 参数tau
        self.gae_tau = params.gae_tau
        # epoch数量
        self.epoch = params.epoch
        # 是否使用clip
        self.schedule_clip = params.schedule_clip
        # ppo clip参数
        self.clip_param = params.clip_param
        # 是否actor和critic一起训练
        self.train_together = params.train_together
        # kl散度目标值
        self.target_kl = params.target_kl
        # 梯度范围，防止梯度爆炸
        self.gradient_clip = params.gradient_clip
        # 商权重
        self.entropy_weight = params.entropy_weight
        # 优化器
        self.lr = params.lr
        self.lr_decay = params.lr_decay
        # 优化器
        self.optimizer = optim.Adam(itertools.chain(self.actor_net.parameters(), self.critic_net.parameters()), self.lr, betas=(0.9, 0.99))
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), self.lr, betas=(0.9, 0.99))
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), self.lr,betas=(0.9,0.99))
        # 记次数
        self.counter = 0
        # logger初始化
        logger_path = os.path.join(self.save_path, "logger.log")
        if self.load_counter:
            self.logger = get_logger_continue(logger_path)
        else:
            self.logger = get_logger(logger_path)
        # 训练步数
        self.max_steps = params.max_steps
        # 保存间隔
        self.save_interval = params.save_interval
        self.log_interval = params.log_interval
        self.eval_interval = params.eval_interval
        # 加载参数
        self.load_counter = params.load_counter
        # 预测次数
        self.eval_num = params.eval_num
        # batch_size
        self.batch_size = params.batch_size
        # 设备
        self.device = params.device
        # 是否对value做clip
        self.value_clip = params.value_clip
        # kl散度提前停止
        self.kl_early_stop = params.kl_early_stop
        # actor状态标准化
        self.actor_state_norm = Identity()
        # critic状态标准化
        self.critic_state_norm = Identity()
        # reward标准化
        self.reward_norms = [Identity() for i in range(self.env_num)]
        # 是否使用归一化
        self.norm_state = params.norm_state
        self.norm_reward = params.norm_reward
        self.norm_advantage = params.norm_advantage
        self.norm_value_loss = params.norm_value_loss
        if self.norm_state:
            self.actor_state_norm = AutoNormalization(self.actor_state_norm, self.actor_input_dim, clip=10.0)
            self.critic_state_norm = AutoNormalization(self.critic_state_norm, self.critic_input_dim, clip=10.0)        
        # if self.norm_reward:
        #     self.reward_norm = RewardFilter(self.reward_norm, (), clip=10.0)
        if self.norm_reward:
            self.reward_norms = [RewardFilter(self.reward_norms[i], (), clip=10.0) for i in range(self.env_num)]
        # 预处理
        self.return_reward = [0 for i in range(self.env_num)]
        # 状态归一化
        self.actor_state_norm.reset()
        self.critic_state_norm.reset()
        for i in range(len(self.reward_norms)):
            self.reward_norms[i].reset()

    # 选择动作
    def select_action(self, state, action=None, determine=False):
        with torch.no_grad():
            mu, sigma, action, log_prob, entropy = self.actor_net(state, action, determine=determine)
        return mu, sigma, action, log_prob, entropy

    # 资源价值
    def get_value(self, state):
        critic_state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            resource_value = self.activity_critic_net(critic_state)
        return resource_value
    
    # 保存参数
    def save_param(self):
        actor_path = os.path.join(self.save_path,"actor_net_"+str(self.counter)+".pth")
        critic_path = os.path.join(self.save_path, "critic_net_"+str(self.counter)+".pth")
        norm_path = os.path.join(self.save_path, "norm_"+str(self.counter)+".pkl")
        torch.save(self.actor_net.state_dict(), actor_path)
        torch.save(self.critic_net.state_dict(), critic_path)
        norm = {"actor_state_norm": self.actor_state_norm, "critic_state_norm": self.critic_state_norm}
        with open(norm_path, "wb") as f:
            pickle.dump(norm, f, pickle.HIGHEST_PROTOCOL)
            
    # 加载参数
    def load_param(self):
        actor_path = os.path.join(self.save_path, "actor_net_" + str(self.counter) + ".pth")
        critic_path = os.path.join(self.save_path, "critic_net_" + str(self.counter) + ".pth")
        norm_path = os.path.join(self.save_path, "norm_" + str(self.counter) + ".pkl")
        self.actor_net.load_state_dict(torch.load(actor_path, map_location=self.device))
        self.critic_net.load_state_dict(torch.load(critic_path, map_location=self.device))

    def reshape_obs(self, obs):
        return_obs = []
        for i in range(self.env_num):
            if params.actor_net_type == "mlp":
                return_obs.append(list(np.squeeze(obs[i]["final_obs"],axis=0)))
            elif params.actor_net_type == "cnn1" or params.actor_net_type == "cnn2":
                print("rgb shape",obs[i]["rgb"].shape)
                return_obs.append(list(np.squeeze(obs[i]["rgb"],axis=0)))
        return_obs = np.array(return_obs, dtype=np.float32)
        return return_obs

    def clamp_action(self, output_action):
        # 尝试使用clamp切分 clamp0
        action_low = [0, 0, -1]
        action_high = [1, 1, 1]
        output_action = output_action.transpose(0,1)
        for i in range(self.action_dim):
            output_action[i] = torch.clamp(output_action[i], action_low[i], action_high[i])
        output_action = output_action.transpose(0,1)
        return output_action
    
    def change_2_action(self, input_action):
        print("input_action", input_action)
        output_action = torch.zeros([input_action.shape[0], input_action.shape[1]+1],dtype=torch.float)
        output_action = output_action.transpose(0,1)
        input_action = input_action.transpose(0,1)
        output_action[0] = F.relu(input_action[0])
        output_action[1] = F.relu(-1 * input_action[0])
        output_action[2] = input_action[1]
        input_action = input_action.transpose(0,1)
        output_action = output_action.transpose(0,1)
        print("output_action", output_action)
        return output_action

    # 训练步骤
    def train_step(self):
        # 状态归一化
        self.actor_state_norm.reset()
        self.critic_state_norm.reset()
        for i in range(len(self.reward_norms)):
            self.reward_norms[i].reset()
        # 开始
        self.training_step += 1
        self.step_counter = 0
        # 重启环境
        state = self.train_envs.reset()
        # 转换状态
        state = self.reshape_obs(state)
        # 记录
        return_rewards = []
        return_reward = [0 for i in range(self.env_num)]
        # 采样本
        for _ in range(self.buffer_capacity):
            # 复制actor、critic state
            actor_state = copy.deepcopy(state)
            critic_state = copy.deepcopy(state)
            actor_state = self.actor_state_norm(actor_state)
            critic_state = self.critic_state_norm(critic_state)
            actor_state = torch.from_numpy(actor_state).float().to(self.device)
            critic_state = torch.from_numpy(critic_state).float().to(self.device)
            # 生成动作
            with torch.no_grad():
                mu, sigma, action, log_prob, entropy = self.select_action(actor_state)
                critic_value = self.critic_net(critic_state)
            # 动作放入环境
            temp_action = self.change_2_action(action)
            next_state, reward, done, infos  = self.train_envs.step(temp_action.cpu().numpy())
            # 转换状态
            next_state = self.reshape_obs(next_state)
            reward = np.squeeze(np.array(reward))
            done = np.squeeze(np.array(done))
            # 计算reward
            return_reward = [reward[i] + return_reward[i] for i in range(len(reward))]
            reward = np.array([self.reward_norms[i](reward[i]) for i in range(len(reward))])
            for i, done_ in enumerate(done):
                if done_:
                    return_rewards.append(return_reward[i])
                    return_reward[i] = 0
                    self.actor_state_norm.reset()
                    self.critic_state_norm.reset()
                    self.reward_norms[i].reset()
            # 存储技能资源数据
            log_prob = log_prob.view(-1)
            critic_value = critic_value.view(-1)
            # 存储技能资源数据
            trans = transition(actor_state, critic_state, action, reward, log_prob, done, critic_value, self.device)
            # 加入buffer中
            self.replay_buffer.append(trans)
            # 记录次数
            self.step_counter += 1
            # 更新状态
            state = next_state
        print(return_rewards)
        self.counter += 1
        # 复制actor、critic state
        actor_state = copy.deepcopy(state)
        critic_state = copy.deepcopy(state)
        actor_state = self.actor_state_norm(actor_state)
        critic_state = self.critic_state_norm(critic_state)
        actor_state = torch.from_numpy(actor_state).float().to(self.device)
        critic_state = torch.from_numpy(critic_state).float().to(self.device)
        # 生成动作
        with torch.no_grad():
            mu, sigma, action, log_prob, entropy = self.select_action(actor_state)
            return_value = self.critic_net(critic_state)
            return_value = return_value.view(-1)
        advantage = torch.zeros([self.env_num]).to(self.device).detach()
        # 计算return和advantage函数
        for i in reversed(range(self.buffer_capacity)):
            # 更新advantage函数
            if not self.use_gae:
                advantage = return_value - self.replay_buffer[i].critic_value.detach()
            else:
                if i == self.buffer_capacity-1:
                    next_critic_value = return_value
                else:
                    next_critic_value = self.replay_buffer[i + 1].critic_value.detach()
                td_error = self.replay_buffer[i].reward + self.discount * self.replay_buffer[i].mask * next_critic_value \
                            - self.replay_buffer[i].critic_value
                advantage = advantage * self.gae_tau * self.discount * self.replay_buffer[i].mask + td_error
            # 技能返回价值
            return_value = self.replay_buffer[i].reward + self.discount * self.replay_buffer[i].mask * return_value
            self.replay_buffer[i] = self.replay_buffer[i]._replace(advantage=advantage.detach())
            self.replay_buffer[i] = self.replay_buffer[i]._replace(return_value=return_value.detach())
        # 记录损失函数
        action_losses = []
        value_losses = []
        entropies = []
        kls = []
        lrs = []
        # 对活动网络进行更新
        actor_states = torch.cat([t.actor_state for t in self.replay_buffer], axis=0)
        critic_states = torch.cat([t.critic_state for t in self.replay_buffer], axis=0)
        actions = torch.cat([t.action for t in self.replay_buffer], axis=0).to(self.device)
        log_probs = torch.cat([t.log_prob for t in self.replay_buffer], axis=0).view(-1, 1).to(self.device)
        advantages = torch.cat([t.advantage for t in self.replay_buffer], axis=0).view(-1, 1).to(self.device)
        returns = torch.cat([t.return_value for t in self.replay_buffer], axis=0).view(-1,1).to(self.device)
        critic_values = torch.cat([t.critic_value for t in self.replay_buffer], axis=0).view(-1, 1).to(self.device)
        if self.norm_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                # 训练
        for _ in range(self.epoch):
            for index in BatchSampler(SubsetRandomSampler(range(self.buffer_capacity * self.env_num)), self.batch_size, drop_last=False):
                # 经过网络得到当前网络输出
                mu, sigma, action, log_prob, entropy = self.actor_net(actor_states[index], actions[index])
                return_value = self.critic_net(critic_states[index])
                # 得到ratio
                ratio = (log_prob - log_probs[index]).exp()
                # clip_param缩小
                if self.schedule_clip:
                    ep_ratio = 1 - (self.counter / self.max_steps)
                    clip_param = ep_ratio * self.clip_param
                else:
                    clip_param = self.clip_param
                # actor损失
                L1_loss = ratio * advantages[index]
                L2_loss = torch.clamp(ratio, 1 - clip_param, 1 + clip_param) * advantages[index]
                action_loss = -torch.min(L1_loss, L2_loss).mean() - self.entropy_weight * entropy.mean()
                if self.value_clip:
                    return_value_clipped = critic_values[index] + torch.clamp(return_value - critic_values[index],\
                                                                -self.clip_param, self.clip_param)
                    value_loss_1 = (return_value - returns[index]).pow(2)
                    value_loss_2 = (return_value_clipped - returns[index]).pow(2)
                    value_loss = 0.5 * torch.max(value_loss_1, value_loss_2).mean()
                else:
                    value_loss = 0.5 * (return_value - returns[index]).pow(2).mean()
                # 计算kl散度
                # approx_kl = (resource_log_probs[index] - resource_log_prob).mean()
                approx_kl = 0.5*((log_probs[index] - log_prob)**2).mean()
                # approx_kl = (resource_log_probs[index].exp()*(resource_log_probs[index] - resource_log_prob)).mean()
                # KL 提前停止
                if self.kl_early_stop:
                    if abs(approx_kl) > self.target_kl:
                        continue
                # 记录损失
                entropies.append(entropy.mean().cpu().detach().numpy())
                action_losses.append(action_loss.cpu().detach().numpy())
                value_losses.append(value_loss.cpu().detach().numpy())
                kls.append(approx_kl.cpu().detach().numpy())
                # 价值损失归一化
                if self.norm_value_loss:
                    return_6std = 6 * returns[index].std()
                    value_loss = value_loss / (return_6std + 1e-8)
                # 反向传播，若一起训练
                if self.train_together:
                    self.optimizer.zero_grad()
                    torch.cuda.empty_cache()
                    (action_loss + value_loss).backward()
                    nn.utils.clip_grad_norm_(itertools.chain(self.actor_net.parameters(), self.critic_net.parameters()), self.gradient_clip)
                    self.optimizer.step()
                    torch.cuda.empty_cache()
                else:
                    if approx_kl <= 1.5 * self.target_kl:
                        self.actor_optimizer.zero_grad()
                        action_loss.backward()
                        nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.gradient_clip)
                        self.actor_optimizer.step()
                    self.critic_optimizer.zero_grad()
                    value_loss.backward()
                    nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.gradient_clip)
                    self.critic_optimizer.step()
                # 设置是否学习率衰减
                if self.lr_decay == "linear":
                    ep_ratio = 1 - (self.counter / self.max_steps)
                    lr_now = ep_ratio * self.lr
                    for g in self.optimizer.param_groups:
                        g['lr'] = lr_now
                    for g in self.actor_optimizer.param_groups:
                        g['lr'] = lr_now
                    for g in self.critic_optimizer.param_groups:
                        g['lr'] = lr_now
                lrs.append(self.optimizer.param_groups[0]['lr'])
        self.replay_buffer = []
        return np.mean(return_rewards), np.mean(action_losses), np.mean(value_losses), np.mean(entropies), np.mean(kls), np.mean(lrs)
    
    def eval_step(self, obs):
        state = np.squeeze(obs["final_obs"],axis=0)
        actor_state = self.actor_state_norm(state, update=False)
        actor_state = torch.from_numpy(actor_state).float().to(self.device)
        actor_state = torch.unsqueeze(actor_state,0)
        # 生成动作
        with torch.no_grad():
            mu, sigma, action, log_prob, entropy = self.select_action(actor_state, determine=True)
        action = self.change_2_action(action)
        action = action[0]
        return action

    def eval_episode(self, test_env):
        state = test_env.reset()
        all_reward = 0
        while True:
            action = self.eval_step(state)
            state, reward, done, info = test_env.step(action.cpu().numpy())
            all_reward += reward
            if done:
                break
        return all_reward

    def eval_episodes(self, test_env, scenario_name):
        return_rewards = []
        for i in range(self.eval_num):
            all_reward = self.eval_episode(test_env)
            return_rewards.append(all_reward)
        self.logger.info('steps={}\t scenario{:s}\t test_return_reward={:.5f}'.format(self.counter, scenario_name, np.mean(return_rewards)))
        return np.mean(return_rewards)

    def eval_all(self):
        return_rewards = []
        for i in range(len(self.test_envs)):
            reward = self.eval_episodes(self.test_envs[i], self.scenario_names[i])
            return_rewards.append(reward)
        return return_rewards

    def run_steps(self):
        self.logger.info('start training!')
        t0 = time.time()
        if self.load_counter:
            self.counter = self.load_counter
            self.load_param()
            with open(os.path.join(self.save_path, "record_" + str(self.counter) + ".pkl"), "rb") as f:
                data = pickle.load(f)
            test_return_rewards = data['test_return_rewards']
            train_return_rewards = data['train_return_rewards']
            action_losses = data['action_losses']
            value_losses = data['value_losses']
            entropies = data['entropies']
            kls = data['kls']
            lrs = data['lrs']
        else:
            test_return_rewards = [[] for i in range(len(self.test_envs))]
            final_return_rewards = []
            train_return_rewards = []
            action_losses = []
            value_losses = []
            entropies = []
            kls = []
            lrs = []
        while self.counter < self.max_steps:
            # 训练并记录训练数据
            return_reward, action_loss, value_loss, entropy, kl, lr = self.train_step()
            # 记录
            train_return_rewards.append(return_reward)
            action_losses.append(action_loss)
            value_losses.append(value_loss)
            entropies.append(entropy)
            kls.append(kl)
            lrs.append(lr)
            if self.counter % self.save_interval == 0:
                self.save_param()
            if self.counter % self.log_interval:
                self.logger.info('steps %d, %.2f steps/s' % (self.counter, 1 / (time.time() - t0)))
                self.logger.info(
                    'steps={}\t train_return_reward={:.5f}\t action_loss={:.3f}\t '
                    'value_loss={:.3f} \t entropy={:.3f} \t kl={:.5f} \t lr={:.8f}' \
                    .format(self.counter, return_reward, action_loss, value_loss, entropy, kl, lr))
                t0 = time.time()
            if self.counter % self.eval_interval == 0:
                test_return_reward = self.eval_all()
                for i in range(len(self.test_envs)):
                    test_return_rewards[i].append(test_return_reward[i])
                final_return_rewards.append(np.mean(test_return_reward))
            if self.counter % self.log_interval == 0:
                save_info = {"train_return_rewards": train_return_rewards, "test_return_rewards": final_return_rewards,\
                             "action_losses": action_losses, "value_losses": value_losses, "entropies": entropies, \
                             "resource_kl":kls, "resource_lr": lrs}
                eval_info = {"scenario_" + str(i):test_return_rewards[i] for i in range(len(self.test_envs))}
                save_info = dict(save_info, **eval_info)
                with open(os.path.join(self.save_path, "record_" + str(self.counter) + ".pkl"), "wb") as f:
                    pickle.dump(save_info, f, pickle.HIGHEST_PROTOCOL)
        self.logger.info('finish training!')
        return train_return_rewards, test_return_rewards

if __name__ == "__main__":
    # 模型参数
    actor_input_dim = 23
    action_dim = 3
    hidden_dim = 256
    heads_num = 2
    key_dim = 64
    value_dim = 64
    layer_num = 2
    mlp_hidden_dim = 512
    seq_len = 1083
    critic_input_dim = 23
    critic_hidden_dim = 1024
    ###################################
    # 环境参数设置
    ###################################
    # 场景列表
    scenarios = ["1_to_2lane_left_turn_c", "1_to_2lane_left_turn_t", "3lane_merge_single_agent", \
                 "3lane_cruise_single_agent", "3lane_overtake"]
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
    env_class = 0
    #############################################
    # 环境定义
    #############################################
    env_params = ENV_Params(scenarios, sumo_gui, img_meters, num_stack, img_pixels, logdir, visdom, headless)
    train_envs = []
    env_num = 2
    for i in range(env_num):
        train_env = MY_ENV(env_params, i)
        train_envs.append(train_env)
    test_env = 0
    # test_env = MY_ENV(env_params, env_class)
    # 保存文件路径
    path_name = 'new_noclamp_long'
    save_path = os.path.join(os.getcwd(), path_name)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    # 训练参数
    buffer_capacity = 1024
    discount = 0.99
    use_gae = True
    gae_tau = 0.98
    epoch = 10
    clip_param = 0.2
    train_together = True
    target_kl = 0.05
    gradient_clip = 0.5
    entropy_weight = 0.0001
    max_steps = 5000
    save_interval = 50
    log_interval = 10
    eval_interval = 5
    eval_num = 5
    batch_size = 64
    lr = 3e-5
    lr_decay = "linear"
    # gpu设置
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    # ppo策略选项
    value_clip = True
    norm_state = False
    norm_reward = False
    norm_advantage = False
    norm_value_loss = False
    kl_early_stop = True
    schedule_clip = False
    load_counter = False
    # actor_net_type: mlp/cnn
    actor_net_type = "mlp"
    critic_net_type = "mlp"
    params = Params(actor_input_dim, critic_input_dim, action_dim, hidden_dim, heads_num, key_dim, value_dim, layer_num, \
                    mlp_hidden_dim, seq_len, train_envs, test_env, save_path, buffer_capacity, discount, use_gae, gae_tau, \
                    epoch, clip_param, train_together, target_kl, gradient_clip, entropy_weight, max_steps, save_interval, \
                    log_interval, eval_interval, eval_num, batch_size, device, value_clip, lr, lr_decay, norm_state, norm_reward, \
                    norm_advantage, norm_value_loss, kl_early_stop, schedule_clip, load_counter, actor_net_type, critic_net_type, env_num)
    my_ppo = PPO(params)
    train_return_rewards, test_return_rewards = my_ppo.run_steps()





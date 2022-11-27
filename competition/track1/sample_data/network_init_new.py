import torch
import numpy as np
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

##############################################################
# MLP网络模型 targetpose 
##############################################################
class Mlp_Actor_model_action_pose(nn.Module):
    def __init__(self, feature_dim, action_dim, device):
        super(Mlp_Actor_model_action_pose, self).__init__()
        self.action_dim = action_dim
        self.device = device
        self.linear1 = nn.Sequential(nn.Linear(feature_dim, 256),
                                     nn.ReLU())
        self.linear2 = nn.Sequential(nn.Linear(256, 512),
                                     nn.ReLU(),
                                     nn.Linear(512, 256))
        self.layer_norm = nn.LayerNorm(256)
        self.mu_output = nn.Linear(256, self.action_dim)
        self.sigma_output = nn.Linear(256, self.action_dim)
        self.mu_tanh = nn.Tanh()
        # 初始化
        for i in range(len(self.linear1)):
            if type(self.linear1[i]) == nn.Linear:
                init.orthogonal_(self.linear1[i].weight, np.sqrt(2))
                self.linear1[i].bias.data.zero_()
        for i in range(len(self.linear2)):
            if type(self.linear2[i]) == nn.Linear:
                init.orthogonal_(self.linear2[i].weight, np.sqrt(2))
                self.linear2[i].bias.data.zero_()
        if type(self.mu_output) == nn.Linear:
            init.orthogonal_(self.mu_output.weight, 0.01)
            self.mu_output.bias.data.zero_()
        if type(self.sigma_output) == nn.Linear:
            init.orthogonal_(self.sigma_output.weight, 0.01)
            self.sigma_output.bias.data.zero_()

    def forward(self, all_vector_state, rgb_state, together_vector_state, trans_vector_state, judge_state, action=None, determine=None):
        first = self.linear1(all_vector_state)
        residual = first
        second = self.linear2(first)
        second = self.layer_norm(second + residual)
        mu = self.mu_output(second)
        # 使用tanh来限制动作空间
        mu = self.mu_tanh(mu)
        # 获取sigma
        sigma = F.softplus(self.sigma_output(second)) + 0.0001
        var = sigma * sigma
        cov_mat = torch.diag_embed(var)
        dist = MultivariateNormal(mu, cov_mat)
        # 动作选择
        if action == None:
            if determine:
                output_action = mu
            else:
                # 生成动作
                output_action = dist.sample()
        else:
            output_action = action
        # 生成log概率
        log_prob = dist.log_prob(output_action).unsqueeze(-1)
        # 生成熵
        entropy = dist.entropy().unsqueeze(-1)
        return mu, sigma, output_action, log_prob, entropy


# MLP Critic网络
class Mlp_Critic_net(nn.Module):
    def __init__(self, critic_input_dim, device):
        super(Mlp_Critic_net, self).__init__()
        self.device = device
        self.linear1 = nn.Sequential(nn.Linear(critic_input_dim, 256),
                                     nn.ReLU())
        self.linear2 = nn.Sequential(nn.Linear(256, 512),
                                     nn.ReLU(),
                                     nn.Linear(512, 256))
        self.linear3 = nn.Linear(256, 1)
        self.layer_norm = nn.LayerNorm(256)
        # 初始化
        for i in range(len(self.linear1)):
            if type(self.linear1[i]) == nn.Linear:
                init.orthogonal_(self.linear1[i].weight, np.sqrt(2))
                self.linear1[i].bias.data.zero_()
        for i in range(len(self.linear2)):
            if type(self.linear2[i]) == nn.Linear:
                init.orthogonal_(self.linear2[i].weight, np.sqrt(2))
                self.linear2[i].bias.data.zero_()
        if type(self.linear3) == nn.Linear:
            init.orthogonal_(self.linear3.weight, 0.1)
            self.linear3.bias.data.zero_()

    def forward(self, all_vector_state, rgb_state, together_vector_state, trans_vector_state, judge_state):
        first = self.linear1(all_vector_state)
        residual = first
        second = self.linear2(first)
        second = self.layer_norm(second + residual)
        value = self.linear3(second)
        return value


##############################################################
# CNN网络模型 targetpose 
##############################################################
class CNN_Actor_model_action_pose(nn.Module):
    def __init__(self, action_dim, device):
        super(CNN_Actor_model_action_pose, self).__init__()
        self.action_dim = action_dim
        self.device = device
        self.conv1 = nn.Sequential(nn.Conv2d(4,32,kernel_size=8,stride=4,padding=2),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(32,64,kernel_size=4,stride=2,padding=1),
                                   nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(64,64,kernel_size=3,stride=2),
                                   nn.ReLU())
        self.fc = nn.Linear(6*6*64, 256)
        self.linear = nn.Sequential(nn.Linear(256, 256),
                             nn.ReLU())
        self.mu_output = nn.Linear(256, self.action_dim)
        self.sigma_output = nn.Linear(256, self.action_dim)
        self.layer_norm = nn.LayerNorm(256)
        self.relu = nn.ReLU()
        self.mu_tanh = nn.Tanh()
        # 初始化
        for i in range(len(self.conv1)):
            if type(self.conv1[i]) == nn.Conv2d:
                init.orthogonal_(self.conv1[i].weight, np.sqrt(2))
                self.conv1[i].bias.data.zero_()
        for i in range(len(self.conv2)):
            if type(self.conv2[i]) == nn.Conv2d:
                init.orthogonal_(self.conv2[i].weight, np.sqrt(2))
                self.conv2[i].bias.data.zero_()
        for i in range(len(self.conv3)):
            if type(self.conv3[i]) == nn.Conv2d:
                init.orthogonal_(self.conv3[i].weight, np.sqrt(2))
                self.conv3[i].bias.data.zero_()
        for i in range(len(self.linear)):
            if type(self.linear[i]) == nn.Linear:
                init.orthogonal_(self.linear[i].weight, np.sqrt(2))
                self.linear[i].bias.data.zero_()
        if type(self.fc) == nn.Linear:
            init.orthogonal_(self.fc.weight, np.sqrt(2))
            self.fc.bias.data.zero_()
        if type(self.mu_output) == nn.Linear:
            init.orthogonal_(self.mu_output.weight, 0.01)
            self.mu_output.bias.data.zero_()
        if type(self.sigma_output) == nn.Linear:
            init.orthogonal_(self.sigma_output.weight, 0.01)
            self.sigma_output.bias.data.zero_()

    def forward(self, all_vector_state, rgb_state, together_vector_state, trans_vector_state, judge_state, action=None, determine=None):
        # 三层卷积与两层全连接
        first_conv = self.conv1(rgb_state)
        second_conv = self.conv2(first_conv)
        third_conv = self.conv3(second_conv)
        third_conv = third_conv.view(third_conv.size(0), -1)
        fc = self.fc(third_conv)
        feature = self.layer_norm(self.linear(fc) + fc)
        mu = self.mu_output(feature)
        # 使用tanh来限制动作空间
        mu = self.mu_tanh(mu)
        # 获取sigma
        sigma = F.softplus(self.sigma_output(feature)) + 0.0001
        var = sigma * sigma
        cov_mat = torch.diag_embed(var)
        dist = MultivariateNormal(mu, cov_mat)
        # 动作选择
        if action == None:
            if determine:
                output_action = mu
            else:
                # 生成动作
                output_action = dist.sample()
        else:
            output_action = action
        # 生成log概率
        log_prob = dist.log_prob(output_action).unsqueeze(-1)
        # 生成熵
        entropy = dist.entropy().unsqueeze(-1)
        # 将动作转变为
        return mu, sigma, output_action, log_prob, entropy

# CNN Critic网络
class CNN_Critic_net(nn.Module):
    def __init__(self, device):
        super(CNN_Critic_net, self).__init__()
        self.device = device
        self.conv1 = nn.Sequential(nn.Conv2d(4,32,kernel_size=8,stride=4,padding=2),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(32,64,kernel_size=4,stride=2,padding=1),
                                   nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(64,64,kernel_size=3,stride=2),
                                   nn.ReLU())
        self.fc = nn.Linear(6*6*64, 256)
        self.linear = nn.Sequential(nn.Linear(256, 256),
                             nn.ReLU())
        self.output = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(256)
        # 初始化
        for i in range(len(self.conv1)):
            if type(self.conv1[i]) == nn.Conv2d:
                init.orthogonal_(self.conv1[i].weight, np.sqrt(2))
                self.conv1[i].bias.data.zero_()
        for i in range(len(self.conv2)):
            if type(self.conv2[i]) == nn.Conv2d:
                init.orthogonal_(self.conv2[i].weight, np.sqrt(2))
                self.conv2[i].bias.data.zero_()
        for i in range(len(self.conv3)):
            if type(self.conv3[i]) == nn.Conv2d:
                init.orthogonal_(self.conv3[i].weight, np.sqrt(2))
                self.conv3[i].bias.data.zero_()
        for i in range(len(self.linear)):
            if type(self.linear[i]) == nn.Linear:
                init.orthogonal_(self.linear[i].weight, np.sqrt(2))
                self.linear[i].bias.data.zero_()
        if type(self.fc) == nn.Linear:
            init.orthogonal_(self.fc.weight, np.sqrt(2))
            self.fc.bias.data.zero_()
        if type(self.output) == nn.Linear:
            init.orthogonal_(self.output.weight, 0.01)
            self.output.bias.data.zero_()

    def forward(self, all_vector_state, rgb_state, together_vector_state, trans_vector_state, judge_state):
        # 三层卷积与两层全连接
        first_conv = self.conv1(rgb_state)
        # print("critic_first_conv", first_conv.shape)
        second_conv = self.conv2(first_conv)
        # print("critic_second_conv", second_conv.shape)
        third_conv = self.conv3(second_conv)
        # print("critic_third_conv", third_conv.shape)
        third_conv = third_conv.view(third_conv.size(0), -1)
        # print("critic_third_conv", third_conv.shape)
        fc = self.fc(third_conv)
        feature = self.layer_norm(self.linear(fc) + fc)
        value = self.output(feature)
        return value

#######################################################################################
# CNN与MLP结合 Actor网络 targetpose
#######################################################################################
class Mix_Actor_model_action_pose(nn.Module):
    def __init__(self, feature_dim, action_dim, device):
        super(Mix_Actor_model_action_pose, self).__init__()
        self.action_dim = action_dim
        self.device = device
        # 向量特征提取
        self.linear1 = nn.Sequential(nn.Linear(feature_dim, 256),
                                     nn.ReLU())
        self.linear2 = nn.Sequential(nn.Linear(256, 512),
                                     nn.ReLU(),
                                     nn.Linear(512, 256))
        # 图像特征提取
        self.conv1 = nn.Sequential(nn.Conv2d(4,32,kernel_size=8,stride=4,padding=2),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(32,64,kernel_size=4,stride=2,padding=1),
                                   nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(64,64,kernel_size=3,stride=2),
                                   nn.ReLU())
        self.fc = nn.Linear(6*6*64, 256)
        self.cnn_linear = nn.Sequential(nn.Linear(256, 256),
                                        nn.ReLU())
        # 公共决策层
        self.linear = nn.Sequential(nn.Linear(512, 512),
                             nn.ReLU())
        self.mu_output = nn.Linear(512, self.action_dim)
        self.sigma_output = nn.Linear(512, self.action_dim)
        self.layer_norm = nn.LayerNorm(256)
        self.relu = nn.ReLU()
        self.mu_tanh = nn.Tanh()
        # 初始化
        for i in range(len(self.linear1)):
            if type(self.linear1[i]) == nn.Linear:
                init.orthogonal_(self.linear1[i].weight, np.sqrt(2))
                self.linear1[i].bias.data.zero_()
        for i in range(len(self.linear2)):
            if type(self.linear2[i]) == nn.Linear:
                init.orthogonal_(self.linear2[i].weight, np.sqrt(2))
                self.linear2[i].bias.data.zero_()
        for i in range(len(self.conv1)):
            if type(self.conv1[i]) == nn.Conv2d:
                init.orthogonal_(self.conv1[i].weight, np.sqrt(2))
                self.conv1[i].bias.data.zero_()
        for i in range(len(self.conv2)):
            if type(self.conv2[i]) == nn.Conv2d:
                init.orthogonal_(self.conv2[i].weight, np.sqrt(2))
                self.conv2[i].bias.data.zero_()
        for i in range(len(self.conv3)):
            if type(self.conv3[i]) == nn.Conv2d:
                init.orthogonal_(self.conv3[i].weight, np.sqrt(2))
                self.conv3[i].bias.data.zero_()
        for i in range(len(self.linear)):
            if type(self.linear[i]) == nn.Linear:
                init.orthogonal_(self.linear[i].weight, np.sqrt(2))
                self.linear[i].bias.data.zero_()
        if type(self.fc) == nn.Linear:
            init.orthogonal_(self.fc.weight, np.sqrt(2))
            self.fc.bias.data.zero_()
        if type(self.mu_output) == nn.Linear:
            init.orthogonal_(self.mu_output.weight, 0.01)
            self.mu_output.bias.data.zero_()
        if type(self.sigma_output) == nn.Linear:
            init.orthogonal_(self.sigma_output.weight, 0.01)
            self.sigma_output.bias.data.zero_()

    def forward(self, all_vector_state, rgb_state, together_vector_state, trans_vector_state, judge_state, action=None, determine=None):
        # 向量特征提取
        first = self.linear1(all_vector_state)
        residual = first
        second = self.linear2(first)
        vector_feature = self.layer_norm(second + residual)
        # 图像特征提取
        first_conv = self.conv1(rgb_state)
        second_conv = self.conv2(first_conv)
        third_conv = self.conv3(second_conv)
        third_conv = third_conv.view(third_conv.size(0), -1)
        fc = self.fc(third_conv)
        cnn_feature = self.layer_norm(self.cnn_linear(fc) + fc)
        # 合并特征
        feature = torch.cat((vector_feature, cnn_feature), 1)
        feature = self.linear(feature)
        mu = self.mu_output(feature)
        # 使用tanh来限制动作空间
        mu = self.mu_tanh(mu)
        # 获取sigma
        sigma = F.softplus(self.sigma_output(feature)) + 0.0001
        var = sigma * sigma
        cov_mat = torch.diag_embed(var)
        dist = MultivariateNormal(mu, cov_mat)
        # 动作选择
        if action == None:
            if determine:
                output_action = mu
            else:
                # 生成动作
                output_action = dist.sample()
        else:
            output_action = action
        # 生成log概率
        log_prob = dist.log_prob(output_action).unsqueeze(-1)
        # 生成熵
        entropy = dist.entropy().unsqueeze(-1)
        # 将动作转变为
        return mu, sigma, output_action, log_prob, entropy






class Mix_Critic_net(nn.Module):
    def __init__(self, feature_dim, device):
        super(Mix_Critic_net, self).__init__()
        self.device = device
        # 向量特征提取
        self.linear1 = nn.Sequential(nn.Linear(feature_dim, 256),
                                     nn.ReLU())
        self.linear2 = nn.Sequential(nn.Linear(256, 512),
                                     nn.ReLU(),
                                     nn.Linear(512, 256))
        # 图像特征提取
        self.conv1 = nn.Sequential(nn.Conv2d(4,32,kernel_size=8,stride=4,padding=2),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(32,64,kernel_size=4,stride=2,padding=1),
                                   nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(64,64,kernel_size=3,stride=2),
                                   nn.ReLU())
        self.cnn_linear = nn.Sequential(nn.Linear(256, 256),
                                    nn.ReLU())
        self.fc = nn.Linear(6*6*64, 256)
        self.linear = nn.Sequential(nn.Linear(512, 512),
                             nn.ReLU())
        self.output = nn.Linear(512, 1)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(256)
        # 初始化
        for i in range(len(self.linear1)):
            if type(self.linear1[i]) == nn.Linear:
                init.orthogonal_(self.linear1[i].weight, np.sqrt(2))
                self.linear1[i].bias.data.zero_()
        for i in range(len(self.linear2)):
            if type(self.linear2[i]) == nn.Linear:
                init.orthogonal_(self.linear2[i].weight, np.sqrt(2))
                self.linear2[i].bias.data.zero_()
        for i in range(len(self.conv1)):
            if type(self.conv1[i]) == nn.Conv2d:
                init.orthogonal_(self.conv1[i].weight, np.sqrt(2))
                self.conv1[i].bias.data.zero_()
        for i in range(len(self.conv2)):
            if type(self.conv2[i]) == nn.Conv2d:
                init.orthogonal_(self.conv2[i].weight, np.sqrt(2))
                self.conv2[i].bias.data.zero_()
        for i in range(len(self.conv3)):
            if type(self.conv3[i]) == nn.Conv2d:
                init.orthogonal_(self.conv3[i].weight, np.sqrt(2))
                self.conv3[i].bias.data.zero_()
        for i in range(len(self.linear)):
            if type(self.linear[i]) == nn.Linear:
                init.orthogonal_(self.linear[i].weight, np.sqrt(2))
                self.linear[i].bias.data.zero_()
        if type(self.fc) == nn.Linear:
            init.orthogonal_(self.fc.weight, np.sqrt(2))
            self.fc.bias.data.zero_()
        if type(self.output) == nn.Linear:
            init.orthogonal_(self.output.weight, 0.01)
            self.output.bias.data.zero_()

    def forward(self, all_vector_state, rgb_state, together_vector_state, trans_vector_state, judge_state):
        # 向量特征提取
        first = self.linear1(all_vector_state)
        residual = first
        second = self.linear2(first)
        vector_feature = self.layer_norm(second + residual)
        # 图像特征提取
        first_conv = self.conv1(rgb_state)
        second_conv = self.conv2(first_conv)
        third_conv = self.conv3(second_conv)
        third_conv = third_conv.view(third_conv.size(0), -1)
        fc = self.fc(third_conv)
        cnn_feature = self.layer_norm(self.cnn_linear(fc) + fc)
        # 合并特征
        feature = torch.cat((vector_feature, cnn_feature), 1)
        feature = self.linear(feature)
        value = self.output(feature)
        return value


########################################################################
# Transformer
########################################################################
# 获取mask矩阵
def get_attention_pad_mask(seq_q, seq_k):
    '''
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    '''
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # 如果seq_k中有等于0的，则为pad，需要mask掉 [batch_size, 1, len_k], False is masked
    pad_attention_mask = seq_k.data.eq(0).unsqueeze(1)
    return pad_attention_mask.expand(batch_size, len_q, len_k)



class Attention_module(nn.Module):
    def __init__(self):
        super(Attention_module,self).__init__()

    def forward(self, Q, K, V, attention_mask):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        # 计算其他部分对该部分的影响分数 [batch_size, n_heads, len_q, d_k] x [batch_size, n_heads, d_k，len_q] -> [batch_size, n_heads, len_q，len_q]
        scores = torch.matmul(Q, K.transpose(-1,-2)) / np.sqrt(Q.size(-1))
        # 使用负无穷掩码来替代padding标志位
        scores.masked_fill_(attention_mask, -1e9)
        # 注意力softmax分数
        attention = nn.Softmax(dim=-1)(scores)
        # 将softmax分数与价值矩阵相乘 [batch_size, n_heads, len_q，len_q] x [batch_size, n_heads, len_q, value_dim] -> [batch_size, n_heads, len_q, value_dim]
        context = torch.matmul(attention, V)
        return context, attention


class MultiHeadAttention(nn.Module):
    def __init__(self, heads_num, hidden_dim, key_dim, value_dim, device):
        super(MultiHeadAttention, self).__init__()
        self.heads_num = heads_num
        self.hidden_dim = hidden_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.device = device
        # 得到查询向量
        self.W_Q = nn.Linear(hidden_dim, key_dim * heads_num, bias=False)
        # 得到键值向量
        self.W_K = nn.Linear(hidden_dim, key_dim * heads_num, bias=False)
        # 得到值向量
        self.W_V = nn.Linear(hidden_dim, value_dim * heads_num, bias=False)
        # 全连接层
        self.fc = nn.Linear(key_dim * heads_num, hidden_dim, bias=False)
        # 初始化
        for p in self.modules():
            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, 0.01)
                # p.bias.data.zero_()

    def forward(self, input_Q, input_K, input_V, attention_mask):
        '''
        input_Q: [batch_size, len_q, embeding_dim]
        input_K: [batch_size, len_k, embeding_dim]
        input_V: [batch_size, len_v(=len_k), embeding_dim]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        # 获得残差变量和batch_size值
        residual, batch_size = input_Q, input_Q.size(0)
        # 经过运算得到Q、K、V的值
        # [batch_size, seq_length, embeding_dim] -> [batch_size, seq_length, key_dim * heads_num] -> [batch_size, seq_length, heads_num, key_dim]
        # -> [batch_size, heads_num, seq_length, key_dim]
        Q = self.W_Q(input_Q).view(batch_size, -1, self.heads_num, self.key_dim).transpose(1,2)
        K = self.W_K(input_K).view(batch_size, -1, self.heads_num, self.key_dim).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, self.heads_num, self.value_dim).transpose(1, 2)
        # 生成mask矩阵 [batch_size, seq_len, seq_len] -> [batch_size, n_heads, seq_len, seq_len]
        attention_mask = attention_mask.unsqueeze(1).repeat(1, self.heads_num, 1, 1)
        # 通过注意力机制得到对应的语义向量与注意力向量
        # context: [batch_size, n_heads, len_q, value_dim], attention: [batch_size, n_heads, len_q, value_dim]
        context, attention = Attention_module()(Q, K, V, attention_mask)
        # 改变context维度 [batch_size, n_heads, len_q, value_dim] -> [batch_size, len_q, n_heads * value_dim]
        context = context.transpose(1,2).reshape(batch_size, -1, self.heads_num * self.value_dim)
        # 全连接层 [batch_size, len_q, n_heads * value_dim] -> [batch_size, len_q, embeding_dim]
        output = self.fc(context)
        # 经过残差操作和层归一化
        output = nn.LayerNorm(self.hidden_dim).to(self.device)(output + residual)
        return output, attention


class foward_residual_network(nn.Module):
    def __init__(self, hidden_dim, mlp_hidden_dim, device):
        super(foward_residual_network, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim, bias=False),
            nn.Tanh(),
            nn.Linear(mlp_hidden_dim, hidden_dim)
        )
        # 初始化
        for p in self.modules():
            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, 0.01)
                # p.bias.data.zero_()

    def forward(self, input):
        residual = input
        output = self.network(input)
        output = nn.LayerNorm(self.hidden_dim).to(self.device)(output+residual)
        return output


class EncoderLayer(nn.Module):
    def __init__(self, heads_num, hidden_dim, key_dim, value_dim, mlp_hidden_dim, device):
        super(EncoderLayer, self).__init__()
        self.encoder_attention = MultiHeadAttention(heads_num, hidden_dim, key_dim, value_dim, device)
        self.forward_network = foward_residual_network(hidden_dim, mlp_hidden_dim, device)
        # 初始化
        for p in self.modules():
            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, 0.01)
                # p.bias.data.zero_()

    def forward(self, input, encoder_mask):
        '''
        input: [batch_size, src_len, embeding_dim]
        encoder_mask: [batch_size, src_len, src_len]
        '''
        encoder_output, attention = self.encoder_attention(input, input, input, encoder_mask)
        encoder_output = self.forward_network(encoder_output)
        return encoder_output, attention


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, heads_num, key_dim, value_dim, layer_num, mlp_hidden_dim, device):
        super(Encoder, self).__init__()
        # 此处可以选择是否加bias
        self.state_embeding = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList(EncoderLayer(heads_num, hidden_dim, key_dim, value_dim, mlp_hidden_dim, device) for _ in range(layer_num))
        # 初始化
        for p in self.modules():
            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, 0.01)
                # p.bias.data.zero_()

    def forward(self, input, judge_input):
        # 先过一个linear层 [batch_size, src_len, input_dim] -> [batch_size, src_len, hidden_dim]
        output = self.state_embeding(input)
        # 生成掩码 [batch_size, src_len, src_len]
        enc_self_attention_mask = get_attention_pad_mask(judge_input, judge_input)
        # 记录注意力机制
        encoder_self_attentions = []
        for layer in self.layers:
            output, encoder_self_attention = layer(output, enc_self_attention_mask)
            encoder_self_attentions.append(encoder_self_attention)
        # output [batch_size, src_len, hidden_dim]
        return output, encoder_self_attentions



#######################################################################################
# CNN与MLP、Transformer结合 targetpose
#######################################################################################
class All_Mix_Actor_model_action_pose(nn.Module):
    def __init__(self, feature_dim, action_dim, heads_num, key_dim, value_dim, layer_num, mlp_hidden_dim, device):
        super(All_Mix_Actor_model_action_pose, self).__init__()
        self.action_dim = action_dim
        self.device = device
        # transformer 特征提取
        self.encoder = Encoder(5, 256, heads_num, key_dim, value_dim, layer_num, mlp_hidden_dim, device)
        self.trans_fc = nn.Linear(256, 30)
        # 向量特征提取
        self.linear1 = nn.Sequential(nn.Linear(feature_dim, 256),
                                     nn.ReLU())
        self.linear2 = nn.Sequential(nn.Linear(256, 512),
                                     nn.ReLU(),
                                     nn.Linear(512, 256))
        # 图像特征提取
        self.conv1 = nn.Sequential(nn.Conv2d(4,32,kernel_size=8,stride=4,padding=2),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(32,64,kernel_size=4,stride=2,padding=1),
                                   nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(64,64,kernel_size=3,stride=2),
                                   nn.ReLU())
        self.cnn_linear = nn.Sequential(nn.Linear(256, 256),
                                        nn.ReLU())
        self.fc = nn.Linear(6*6*64, 256)
        # 公共决策层
        self.linear = nn.Sequential(nn.Linear(842, 512),
                             nn.ReLU())
        self.mu_output = nn.Linear(512, self.action_dim)
        self.sigma_output = nn.Linear(512, self.action_dim)
        self.layer_norm = nn.LayerNorm(256)
        self.relu = nn.ReLU()
        self.mu_tanh = nn.Tanh()
        # 初始化
        for i in range(len(self.linear1)):
            if type(self.linear1[i]) == nn.Linear:
                init.orthogonal_(self.linear1[i].weight, np.sqrt(2))
                self.linear1[i].bias.data.zero_()
        for i in range(len(self.linear2)):
            if type(self.linear2[i]) == nn.Linear:
                init.orthogonal_(self.linear2[i].weight, np.sqrt(2))
                self.linear2[i].bias.data.zero_()
        for i in range(len(self.conv1)):
            if type(self.conv1[i]) == nn.Conv2d:
                init.orthogonal_(self.conv1[i].weight, np.sqrt(2))
                self.conv1[i].bias.data.zero_()
        for i in range(len(self.conv2)):
            if type(self.conv2[i]) == nn.Conv2d:
                init.orthogonal_(self.conv2[i].weight, np.sqrt(2))
                self.conv2[i].bias.data.zero_()
        for i in range(len(self.conv3)):
            if type(self.conv3[i]) == nn.Conv2d:
                init.orthogonal_(self.conv3[i].weight, np.sqrt(2))
                self.conv3[i].bias.data.zero_()
        for i in range(len(self.linear)):
            if type(self.linear[i]) == nn.Linear:
                init.orthogonal_(self.linear[i].weight, np.sqrt(2))
                self.linear[i].bias.data.zero_()
        if type(self.fc) == nn.Linear:
            init.orthogonal_(self.fc.weight, np.sqrt(2))
            self.fc.bias.data.zero_()
        if type(self.mu_output) == nn.Linear:
            init.orthogonal_(self.mu_output.weight, 0.01)
            self.mu_output.bias.data.zero_()
        if type(self.sigma_output) == nn.Linear:
            init.orthogonal_(self.sigma_output.weight, 0.01)
            self.sigma_output.bias.data.zero_()

    def forward(self, all_vector_state, rgb_state, together_vector_state, trans_vector_state, judge_state, action=None, determine=None):
        # transformer特征提取
        trans_feature, encoder_self_attentions = self.encoder(trans_vector_state, judge_state)
        trans_feature = self.trans_fc(trans_feature)
        # 将output reshape output [batch_size, seq_len]
        trans_feature = trans_feature.view(trans_feature.size(0),-1)
        # 向量特征提取
        first = self.linear1(together_vector_state)
        residual = first
        second = self.linear2(first)
        vector_feature = self.layer_norm(second + residual)
        # 图像特征提取
        first_conv = self.conv1(rgb_state)
        second_conv = self.conv2(first_conv)
        third_conv = self.conv3(second_conv)
        third_conv = third_conv.view(third_conv.size(0), -1)
        fc = self.fc(third_conv)
        cnn_feature = self.layer_norm(self.cnn_linear(fc) + fc)
        # 合并特征
        feature = torch.cat((vector_feature, cnn_feature, trans_feature), 1)
        feature = self.linear(feature)
        mu = self.mu_output(feature)
        # 使用tanh来限制动作空间
        mu = self.mu_tanh(mu)
        # 获取sigma
        sigma = F.softplus(self.sigma_output(feature)) + 0.0001
        var = sigma * sigma
        cov_mat = torch.diag_embed(var)
        dist = MultivariateNormal(mu, cov_mat)
        # 动作选择
        if action == None:
            if determine:
                output_action = mu
            else:
                # 生成动作
                output_action = dist.sample()
        else:
            output_action = action
        # 生成log概率
        log_prob = dist.log_prob(output_action).unsqueeze(-1)
        # 生成熵
        entropy = dist.entropy().unsqueeze(-1)
        # 将动作转变为
        return mu, sigma, output_action, log_prob, entropy


class All_Mix_Critic_net(nn.Module):
    def __init__(self, feature_dim, heads_num, key_dim, value_dim, layer_num, mlp_hidden_dim, device):
        super(All_Mix_Critic_net, self).__init__()
        self.device = device
        # transformer 特征提取
        self.encoder = Encoder(5, 256, heads_num, key_dim, value_dim, layer_num, mlp_hidden_dim, device)
        self.trans_fc = nn.Linear(256, 30)
        # 向量特征提取
        self.linear1 = nn.Sequential(nn.Linear(feature_dim, 256),
                                     nn.ReLU())
        self.linear2 = nn.Sequential(nn.Linear(256, 512),
                                     nn.ReLU(),
                                     nn.Linear(512, 256))
        # 图像特征提取
        self.conv1 = nn.Sequential(nn.Conv2d(4,32,kernel_size=8,stride=4,padding=2),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(32,64,kernel_size=4,stride=2,padding=1),
                                   nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(64,64,kernel_size=3,stride=2),
                                   nn.ReLU())
        self.cnn_linear = nn.Sequential(nn.Linear(256, 256),
                                        nn.ReLU())
        self.fc = nn.Linear(6*6*64, 256)
        self.linear = nn.Sequential(nn.Linear(842, 512),
                             nn.ReLU())
        self.output = nn.Linear(512, 1)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(256)
        # 初始化
        for i in range(len(self.linear1)):
            if type(self.linear1[i]) == nn.Linear:
                init.orthogonal_(self.linear1[i].weight, np.sqrt(2))
                self.linear1[i].bias.data.zero_()
        for i in range(len(self.linear2)):
            if type(self.linear2[i]) == nn.Linear:
                init.orthogonal_(self.linear2[i].weight, np.sqrt(2))
                self.linear2[i].bias.data.zero_()
        for i in range(len(self.conv1)):
            if type(self.conv1[i]) == nn.Conv2d:
                init.orthogonal_(self.conv1[i].weight, np.sqrt(2))
                self.conv1[i].bias.data.zero_()
        for i in range(len(self.conv2)):
            if type(self.conv2[i]) == nn.Conv2d:
                init.orthogonal_(self.conv2[i].weight, np.sqrt(2))
                self.conv2[i].bias.data.zero_()
        for i in range(len(self.conv3)):
            if type(self.conv3[i]) == nn.Conv2d:
                init.orthogonal_(self.conv3[i].weight, np.sqrt(2))
                self.conv3[i].bias.data.zero_()
        for i in range(len(self.linear)):
            if type(self.linear[i]) == nn.Linear:
                init.orthogonal_(self.linear[i].weight, np.sqrt(2))
                self.linear[i].bias.data.zero_()
        if type(self.fc) == nn.Linear:
            init.orthogonal_(self.fc.weight, np.sqrt(2))
            self.fc.bias.data.zero_()
        if type(self.output) == nn.Linear:
            init.orthogonal_(self.output.weight, 0.01)
            self.output.bias.data.zero_()

    def forward(self, all_vector_state, rgb_state, together_vector_state, trans_vector_state, judge_state, action=None, determine=None):
        # transformer特征提取
        trans_feature, encoder_self_attentions = self.encoder(trans_vector_state, judge_state)
        trans_feature = self.trans_fc(trans_feature)
        # 将output reshape output [batch_size, seq_len]
        trans_feature = trans_feature.view(trans_feature.size(0),-1)
        # 向量特征提取
        first = self.linear1(together_vector_state)
        residual = first
        second = self.linear2(first)
        vector_feature = self.layer_norm(second + residual)
        # 图像特征提取
        first_conv = self.conv1(rgb_state)
        second_conv = self.conv2(first_conv)
        third_conv = self.conv3(second_conv)
        third_conv = third_conv.view(third_conv.size(0), -1)
        fc = self.fc(third_conv)
        cnn_feature = self.layer_norm(self.cnn_linear(fc) + fc)
        # 合并特征
        feature = torch.cat((vector_feature, cnn_feature, trans_feature), 1)
        feature = self.linear(feature)
        value = self.output(feature)
        return value





####################################################################################
# 场景分类网络
####################################################################################

class scenario_identify_network(nn.Module):
    def __init__(self, output_dim):
        super(scenario_identify_network, self).__init__()
        # 图像特征提取
        self.conv1 = nn.Sequential(nn.Conv2d(4,32,kernel_size=8,stride=4,padding=2),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(32,64,kernel_size=4,stride=2,padding=1),
                                   nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(64,64,kernel_size=3,stride=2),
                                   nn.ReLU())
        self.fc = nn.Linear(6*6*64, 512)
        self.cnn_linear = nn.Sequential(nn.Linear(512, 512),
                                        nn.ReLU())
        self.output = nn.Linear(512, output_dim)
        self.layer_norm = nn.LayerNorm(512)
        self.softmax = nn.Softmax()
        # 网络初始化
        for i in range(len(self.conv1)):
            if type(self.conv1[i]) == nn.Conv2d:
                init.orthogonal_(self.conv1[i].weight, np.sqrt(2))
                self.conv1[i].bias.data.zero_()
        for i in range(len(self.conv2)):
            if type(self.conv2[i]) == nn.Conv2d:
                init.orthogonal_(self.conv2[i].weight, np.sqrt(2))
                self.conv2[i].bias.data.zero_()
        for i in range(len(self.conv3)):
            if type(self.conv3[i]) == nn.Conv2d:
                init.orthogonal_(self.conv3[i].weight, np.sqrt(2))
                self.conv3[i].bias.data.zero_()
        for i in range(len(self.cnn_linear)):
            if type(self.cnn_linear[i]) == nn.Linear:
                init.orthogonal_(self.cnn_linear[i].weight, np.sqrt(2))
                self.cnn_linear[i].bias.data.zero_()
        if type(self.fc) == nn.Linear:
            init.orthogonal_(self.fc.weight, np.sqrt(2))
            self.fc.bias.data.zero_()
        if type(self.output) == nn.Linear:
            init.orthogonal_(self.output.weight, np.sqrt(2))
            self.fc.bias.data.zero_()

    def forward(self, rgb_input):
        # 图像特征提取
        first_conv = self.conv1(rgb_input)
        second_conv = self.conv2(first_conv)
        third_conv = self.conv3(second_conv)
        third_conv = third_conv.view(third_conv.size(0), -1)
        fc = self.fc(third_conv)
        cnn_feature = self.layer_norm(self.cnn_linear(fc) + fc)
        # 合并特征
        output = self.output(cnn_feature)
        output = self.softmax(output)

        return output





























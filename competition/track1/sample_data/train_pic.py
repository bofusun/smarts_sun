import os
import sys
import copy
import time
import math
import torch
import pickle
import logging
import itertools
import numpy as np
from my_env import *
from network_init_new import *
from normalization import *
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, BatchSampler


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

class GetLoader(torch.utils.data.Dataset):
	# 初始化函数，得到数据
    def __init__(self, data_state,  data_label):
        self.data_state = data_state
        self.data_label = data_label
    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data_state = self.data_state[index]
        data_label = self.data_label[index]
        return data_state, data_label
    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data_state)

class Params():
    def __init__(self, output_dim, device, save_path, data_path, scenarios, lr, epoch_num, batch_size, \
                 log_interval, val_epoch, log_epoch):
        self.output_dim = output_dim
        self.device = device
        self.save_path = save_path
        self.data_path = data_path
        self.scenarios = scenarios
        self.lr = lr
        self.epoch_num = epoch_num
        self.batch_size = batch_size
        self.log_interval = log_interval
        self.val_epoch = val_epoch
        self.log_epoch = log_epoch


class Scenario_Identify_Model():
    def __init__(self, params):
        # 输出维度
        self.output_dim = params.output_dim
        # 辨别网络
        self.identify_net = scenario_identify_network(params.output_dim).to(device=params.device)
        # 保存路径
        self.save_path = params.save_path
        # 数据集路径
        self.data_path = params.data_path
        # 场景
        self.scenarios = params.scenarios
        # 设备
        self.device = params.device
        # 学习参数
        self.lr = params.lr
        self.epoch_num = params.epoch_num
        self.batch_size = params.batch_size
        self.log_interval = params.log_interval
        self.val_epoch = params.val_epoch
        self.log_epoch = params.log_epoch
        # 优化器
        self.optimizer = optim.Adam(self.identify_net.parameters(), self.lr, betas=(0.9, 0.99))
        # logger初始化
        logger_path = os.path.join(self.save_path, "logger.log")
        self.logger = get_logger(logger_path)
        # 获取数据集
        self.train_dataset, self.test_dataset = self.get_data()
        # Data loader
        self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=True)
        # 交叉熵
        self.criterion = torch.nn.CrossEntropyLoss()

    def save_param(self, temp_epoch, train_losses, test_losses, train_accuracy, test_accuracy):
        identify_net_path = os.path.join(self.save_path,"identify_net_"+str(temp_epoch)+".pth")
        record_path = os.path.join(self.save_path,"record_"+str(temp_epoch)+".pkl")
        torch.save(self.identify_net.state_dict(), identify_net_path)
        record = {"train_losses":train_losses,"test_losses":test_losses,"train_accuracy":train_accuracy,"test_accuracy":test_accuracy}
        with open(record_path, "wb") as f:
            pickle.dump(record, f, pickle.HIGHEST_PROTOCOL)

    def get_data(self):
        train_state_data = []
        train_label_data = []
        test_state_data = []
        test_label_data = []
        for i in range(len(self.scenarios)):
            print("scenario:", i)
            state_path = os.path.join(self.data_path, self.scenarios[i], "data.npy")
            temp_data = np.load(state_path)
            temp_label = np.array([[i] for j in range(len(temp_data))])
            temp_train_data = temp_data[:int(len(temp_data) * 0.7)]
            temp_train_label = temp_label[:int(len(temp_label) * 0.7)]
            temp_test_data = temp_data[int(len(temp_data) * 0.7):]
            temp_test_label = temp_label[int(len(temp_label) * 0.7):]
            # print("temp_test_label", temp_label[:10])
            if i == 0:
                train_state_data = temp_train_data
                train_label_data = temp_train_label
                test_state_data = temp_test_data
                test_label_data = temp_test_label
            else:
                train_state_data = np.vstack((train_state_data, temp_train_data))
                train_label_data = np.vstack((train_label_data, temp_train_label))
                test_state_data = np.vstack((test_state_data, temp_test_data))
                test_label_data = np.vstack((test_label_data, temp_test_label))
        train_label_data = np.squeeze(train_label_data)
        test_label_data = np.squeeze(test_label_data)

        # print(train_state_data.shape)
        # print(test_state_data.shape)
        # print(train_label_data.shape)
        # print(test_label_data.shape)

        # 转换为torch形式
        train_state_data = torch.from_numpy(train_state_data).to(device=self.device).to(torch.float32)
        train_label_data = torch.from_numpy(train_label_data).to(device=self.device).to(torch.long)
        test_state_data = torch.from_numpy(test_state_data).to(device=self.device).to(torch.float32)
        test_label_data = torch.from_numpy(test_label_data).to(device=self.device).to(torch.long)
        # 生成数据集
        train_dataset = GetLoader(train_state_data, train_label_data)
        test_dataset = GetLoader(test_state_data, test_label_data)

        return train_dataset, test_dataset


    def train(self):
        train_losses = []
        train_accuracy = []
        test_losses = []
        test_accuracy = []
        for epoch in range(self.epoch_num):
            loss_mean = 0.
            correct = 0.
            total = 0.
            # 遍历 activity_train_loader 取数据
            for i, data in enumerate(self.train_loader):
                # 前向
                state, label = data
                state = state.to(torch.float32)
                output = self.identify_net(state)
                # 反向
                self.identify_net.zero_grad()
                # 无正则化
                loss = self.criterion(output, label)
                loss.backward()
                # 更新权重
                self.optimizer.step()
                # 统计分类情况
                _, predicted = torch.max(output.data, 1)
                total += label.size(0)
                correct += (predicted.cpu() == label.cpu()).squeeze().sum().numpy()
                # 打印训练信息
                loss_mean += loss.item()
                train_losses.append(loss.item())
                train_accuracy.append(correct / total)
                if (i+1) % self.log_interval == 0:
                    loss_mean = loss_mean / self.log_interval
                    print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                        epoch, self.epoch_num, i+1, len(self.train_loader), loss_mean, correct / total))
                    loss_mean = 0.

            if (epoch+1) % self.val_epoch == 0:
                loss_mean_val = 0.
                correct_val = 0.
                total_val = 0.
                with torch.no_grad():
                    # 遍历 activity_test_loader 取数据
                    for j, data in enumerate(self.test_loader):
                        # 前向
                        state, label = data
                        state = state.to(torch.float32)
                        output = self.identify_net(state)
                        loss = self.criterion(output, label)
                        # 统计分类情况
                        _, predicted = torch.max(output.data, 1)
                        total_val += label.size(0)
                        correct_val += (predicted.cpu() == label.cpu()).squeeze().sum().numpy()
                        # 打印测试信息
                        loss_mean_val += loss.item()
                        test_losses.append(loss.item())
                        test_accuracy.append(correct_val / total_val)
                    print("Testing:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                            epoch, self.epoch_num, j+1, len(self.test_loader), loss_mean_val, correct_val / total_val))
                    loss_mean_val = 0.

            if (epoch+1) % self.log_epoch == 0:
                self.save_param(epoch+1, train_losses, test_losses, train_accuracy, test_accuracy)
                

        


if __name__ =="__main__":
    # 输出维度
    output_dim = 6
    # 设备
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    # 模型名字
    path_name = 'identify_model'
    save_path = os.path.join(os.getcwd(), path_name)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    # 数据路径
    data_path = "/root/deep_learning/SMARTS/SMARTS/DATA"
    # 场景列表
    scenarios = ["1_to_2lane_left_turn_c", "1_to_2lane_left_turn_t", "3lane_merge_single_agent", \
                 "3lane_cruise_single_agent", "3lane_overtake", "3lane_cut_in"]
    # 训练参数
    lr = 3e-5
    epoch_num = 100
    batch_size = 256
    log_interval = 50
    log_epoch = 50
    val_epoch = 1
    # 获取辨别模型
    params = Params(output_dim, device, save_path, data_path, scenarios, lr, epoch_num, batch_size, \
                    log_interval, val_epoch, log_epoch)
    my_identify_model = Scenario_Identify_Model(params)
    my_identify_model.train()




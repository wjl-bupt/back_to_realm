#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import torch.nn.functional as F
import copy
import numpy as np
import torch
import torch.nn as nn
from agent_diy.conf.conf import Config
from typing import List



# class Model(nn.Module):
#     def __init__(self, state_shape, action_shape=0, softmax=False):
#         super().__init__()

#         # User-defined network
#         # 用户自定义网络

class CNNLayer(nn.Module):
    def __init__(self):
        super(CNNLayer, self).__init__()

        # 配置参数
        obs_shape = Config.obs_shape
        hidden_size = Config.hidden_size
        use_ReLU = 1 if Config.use_ReLU else 0
        use_orthogonal = 1 if Config.use_orthogonal else 0
        kernel_size = Config.kernel_size
        stride = Config.stride
        
        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        input_channel = obs_shape[0]
        input_width = obs_shape[1]
        input_height = obs_shape[2]

        self.cnn = nn.Sequential(
            init_(nn.Conv2d(in_channels=input_channel,
                            out_channels=hidden_size // 2,
                            kernel_size=kernel_size,
                            stride=stride)
                  ),
            active_func,
            Flatten(),
            init_(nn.Linear(hidden_size // 2 * (input_width - kernel_size + stride) * (input_height - kernel_size + stride),
                            hidden_size)
                  ),
            active_func,
            init_(nn.Linear(hidden_size, hidden_size)), active_func)

    def forward(self, x):
        # x = x / 255.0
        x = self.cnn(x)
        return x

class MLPLayer(nn.Module):
    def __init__(
        self,
        fc_feat_dim_list: List[int],
        name: str,
        non_linearity: nn.Module = nn.ReLU,
        non_linearity_last: bool = False,
    ):
        # Create a MLP object
        # 创建一个 MLP 对象
        super().__init__()
        self.fc_layers = nn.Sequential()
        for i in range(len(fc_feat_dim_list) - 1):
            fc_layer = make_fc_layer(fc_feat_dim_list[i], fc_feat_dim_list[i + 1])
            self.fc_layers.add_module("{0}_fc{1}".format(name, i + 1), fc_layer)
            # no relu for the last fc layer of the mlp unless required
            # 除非有需要，否则 mlp 的最后一个 fc 层不使用 relu
            if i + 1 < len(fc_feat_dim_list) - 1 or non_linearity_last:
                self.fc_layers.add_module("{0}_non_linear{1}".format(name, i + 1), non_linearity())

    def forward(self, data):
        return self.fc_layers(data)


class CNNBase(nn.Module):
    def __init__(self):
        super(CNNBase, self).__init__()
        self.cnn = CNNLayer()
        self.label_mlp = MLPLayer([Config.hidden_size, Config.ACTION_NUM], "label_mlp")
        self.value_mlp = MLPLayer([Config.hidden_size, Config.VALUE_NUM], "value_mlp")

    def process_legal_action(self, label, legal_action):
        label_max, _ = torch.max(label * legal_action, 1, True)
        label = label - label_max
        label = label * legal_action
        label = label + 1e5 * (legal_action - 1)
        return label

    def forward(self, feature, legal_action):
        # Main MLP processing
        # 主MLP处理
        conv_output = self.cnn(feature)
        label_mlp_out = self.label_mlp(conv_output)
        label_out = self.process_legal_action(label_mlp_out, legal_action)
        prob = torch.nn.functional.softmax(label_out, dim=1)
        value = self.value_mlp(conv_output)

        return prob, value

class NetworkModelActor(CNNBase):
    def format_data(self, obs, legal_action):
        return (
            torch.tensor(obs).to(torch.float32),
            torch.tensor(legal_action).to(torch.float32),
        )


class NetworkModelLearner(CNNBase):
    def format_data(self, datas):
        return datas.view(-1, Config.data_len).float().split(Config.DATA_SPLIT_SHAPE, dim=1)

    def forward(self, data_list, inference=False):
        feature = data_list[0].reshape((-1,) + Config.obs_shape)
        legal_action = data_list[-1]
        return super().forward(feature, legal_action)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def check(input):
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output

def make_fc_layer(in_features: int, out_features: int):
    # Wrapper function to create and initialize a linear layer
    # 创建并初始化一个线性层
    fc_layer = nn.Linear(in_features, out_features)

    # initialize weight and bias
    # 初始化权重及偏移量
    nn.init.orthogonal(fc_layer.weight)
    nn.init.zeros_(fc_layer.bias)

    return fc_layer
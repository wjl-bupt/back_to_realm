#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

from agent_diy.conf.conf import Config
from kaiwu_agent.utils.common_func import create_cls, attached
import numpy as np
import math

# The create_cls function is used to dynamically create a class.
# The first parameter of the function is the type name, and the remaining parameters are the attributes of the class.
# The default value of the attribute should be set to None.
# create_cls函数用于动态创建一个类，函数第一个参数为类型名称，剩余参数为类的属性，属性默认值应设为None

# 增加一个rew 统计的数据
ObsData = create_cls(
    "ObsData",
    feature=None,
    legal_action=None,
    reward=None,
    rew_stat=None,
)


ActData = create_cls(
    "ActData",
    probs=None,
    value=None,
    target=None,
    predict=None,
    action=None,
    prob=None,
)

SampleData = create_cls("SampleData", npdata=None)

RelativeDistance = {
    "RELATIVE_DISTANCE_NONE": 0,
    "VerySmall": 1,
    "Small": 2,
    "Medium": 3,
    "Large": 4,
    "VeryLarge": 5,
}


RelativeDirection = {
    "East": 1,
    "NorthEast": 2,
    "North": 3,
    "NorthWest": 4,
    "West": 5,
    "SouthWest": 6,
    "South": 7,
    "SouthEast": 8,
}

DirectionAngles = {
    1: 0,
    2: 45,
    3: 90,
    4: 135,
    5: 180,
    6: 225,
    7: 270,
    8: 315,
}


def reward_process(end_dist, history_dist):
    # step reward
    # 步数奖励
    step_reward = -0.001

    # end reward
    # 终点奖励
    end_reward = -0.02 * end_dist

    # distance reward
    # 距离奖励
    dist_reward = min(0.001, 0.05 * history_dist)

    return [step_reward + dist_reward + end_reward]


class SampleManager:
    def __init__(
        self,
        gamma=0.99,
        tdlambda=0.95,
    ):
        self.gamma = Config.GAMMA
        self.tdlambda = Config.TDLAMBDA

        self.feature = []
        self.probs = []
        self.actions = []
        self.reward = []
        self.value = []
        self.adv = []
        self.tdlamret = []
        self.legal_action = []
        self.count = 0
        self.samples = []

    def add(self, feature, legal_action, prob, action, value, reward):
        self.feature.append(feature)
        self.legal_action.append(legal_action)
        self.probs.append(prob)
        self.actions.append(action)
        self.value.append(value)
        self.reward.append(reward)
        self.adv.append(np.zeros_like(value))
        self.tdlamret.append(np.zeros_like(value))
        self.count += 1

    def add_last_reward(self, reward):
        self.reward.append(reward)
        self.value.append(np.zeros_like(reward))

    def update_sample_info(self):
        last_gae = 0
        for i in range(self.count - 1, -1, -1):
            reward = self.reward[i + 1]
            next_val = self.value[i + 1]
            val = self.value[i]
            delta = reward + next_val * self.gamma - val
            last_gae = delta + self.gamma * self.tdlambda * last_gae
            self.adv[i] = last_gae
            self.tdlamret[i] = last_gae + val

    def sample_process(self, feature, legal_action, prob, action, value, reward):
        self.add(feature, legal_action, prob, action, value, reward)

    def process_last_frame(self, reward):
        self.add_last_reward(reward)
        # 发送前的后向传递更新
        # Backward pass updates before sending
        self.update_sample_info()
        self.samples = self._get_game_data()

    def get_game_data(self):
        ret = self.samples
        self.samples = []
        return ret

    def _get_game_data(self):
        feature = np.array(self.feature).transpose()
        
        probs = np.array(self.probs).transpose()
        actions = np.array(self.actions).transpose()
        reward = np.array(self.reward[:-1]).transpose()
        value = np.array(self.value[:-1]).transpose()
        legal_action = np.array(self.legal_action).transpose()
        adv = np.array(self.adv).transpose()
        tdlamret = np.array(self.tdlamret).transpose()

        data = np.concatenate([feature.reshape(-1,reward.shape[-1]), reward, value, tdlamret, adv, actions, probs, legal_action]).transpose()

        samples = []
        for i in range(0, self.count):
            samples.append(SampleData(npdata=data[i].astype(np.float32)))

        return samples


class ComputeReward:
    def __init__(self):
        self.obstacle_reward = 0.0
        self.buff_reward = 0.0
        self.treasure_reward = 0.0
        self.explore_reward = 0.0
        self.goal_reward = 0.0
    
    def norm(self, v, max_v, min_v=0):
        v = np.maximum(np.minimum(max_v, v), min_v)
        return (v - min_v) / (max_v - min_v)
    
    
    def compute_obstacle_reward(self):
        return 0.0

    def compute_dist_reward(self, cur_pos, target_pos):
        """ buff reward """
        relative_pos = tuple(y - x for x, y in zip(cur_pos, target_pos))
        dist = np.linalg.norm(relative_pos)
        norm_dist = self.norm(dist, 1.41 * 128)
        
        return max(0, 1.0 - norm_dist)
    

    def compute_starting_point_reward(self, cur_pos, starting_pos, step_no):
        relative_pos = tuple(y - x for x, y in zip(cur_pos, starting_pos))
        dist = np.linalg.norm(relative_pos)
        norm_dist = self.norm(dist, 1.41 * 128)
        return 0.0
        
    
    def compute_explore_reward(self, history_pos, threshold = 5, scale = 0.01):
        if len(history_pos) != 10:
            return 0.0
        coords = np.array(history_pos)  # (10, 2)
        diffs = coords[:, None, :] - coords[None, :, :]  # shape (10, 10, 2)
        dists = np.linalg.norm(diffs, axis=-1)  # shape (10, 10)
        i_upper = np.triu_indices(10, k=1)
        unique_dists = dists[i_upper]
        avg_dist = np.mean(unique_dists)
        # print(f"weijun.luo print info: avg dist {avg_dist}")
        if avg_dist < threshold:
            return -scale * (threshold - avg_dist)
        else:
            return 0.0

    def guess_target_pos(self, organ, cur_pos):
        if organ["status"] != -1:
            return  (organ["pos"]["x"], organ["pos"]["z"])
        
        target_pos_dis = RelativeDistance[organ["relative_pos"]["l2_distance"]]
        target_pos_dir = RelativeDirection[organ["relative_pos"]["direction"]]
        distance = target_pos_dis * 20
        theta = DirectionAngles[target_pos_dir]
        delta_x = distance * math.cos(math.radians(theta))
        delta_z = distance * math.sin(math.radians(theta))
        target_pos = (
                    max(0, min(128, round(cur_pos[0] + delta_x))),
                    max(0, min(128, round(cur_pos[1] + delta_z))),
                )
        return target_pos
    
    def reset(self):
        self.obstacle_reward = 0.0
        self.buff_reward = 0.0
        self.treasure_reward = 0.0
        self.explore_reward = 0.0
        self.goal_reward = 0.0
    
    def compute_reward(self, end_dist, history_dist, organs, cur_pos, history_pos, step_no):
        """ 计算奖励 """
        self.reset()
        for organ in organs:
            if organ["status"] == 0:
                continue
            config_id = organ["config_id"]
            target_pos = self.guess_target_pos(organ, cur_pos)
            # buff 距离
            if config_id == 0:
                self.buff_reward += self.compute_dist_reward(cur_pos, target_pos) * 0.1 
            # 起点惩罚
            elif config_id == 21:
                pass
            # 终点
            elif config_id == 22:
                self.goal_reward += min(0.001, 0.05 * history_dist)
            # 宝箱奖励
            else:
                self.treasure_reward += self.compute_dist_reward(cur_pos, target_pos)
        # 步数奖励
        step_reward = -0.001
        end_reward = 1.0 - end_dist
        
        # 探索奖励
        self.explore_reward = self.compute_explore_reward(history_pos)
        
        # 依据时间长度给予奖励
        weight = max(0.1, (1000 - step_no ) / 1000)

        total_reward = [
            step_reward + 
            weight * (self.buff_reward + self.treasure_reward + self.explore_reward) + 
            self.goal_reward + 
            (1 - weight) * end_reward
        ]

        # build dict_
        rew_ = {
            "buff_rew": self.buff_reward,
            "treasure_rew": self.treasure_reward,
            "goal_rew": self.goal_reward,
            "end_rew": end_reward,
            "explore_rew": self.explore_reward,
        }
        
        
        return total_reward, rew_

@attached
def SampleData2NumpyData(g_data):
    return g_data.npdata


@attached
def NumpyData2SampleData(s_data):
    return SampleData(npdata=s_data)

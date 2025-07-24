#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################

"""
Author: Tencent AI Arena Authors

"""

import numpy as np
import math
from agent_diy.feature.definition import RelativeDistance, RelativeDirection, DirectionAngles, reward_process


def norm(v, max_v, min_v=0):
    v = np.maximum(np.minimum(max_v, v), min_v)
    return (v - min_v) / (max_v - min_v)


class Preprocessor:
    def __init__(self) -> None:
        self.move_action_num = 8
        self.reset()
        
        # NOTE(junweiluo):
        self.MyFeatureClass = Build_Feature()

    def reset(self):
        self.step_no = 0
        self.cur_pos = (0, 0)
        self.cur_pos_norm = np.array((0, 0))
        self.end_pos = None
        self.is_end_pos_found = False
        self.history_pos = []
        self.bad_move_ids = set()

    def _get_pos_feature(self, found, cur_pos, target_pos):
        relative_pos = tuple(y - x for x, y in zip(cur_pos, target_pos))
        dist = np.linalg.norm(relative_pos)
        target_pos_norm = norm(target_pos, 128, -128)
        feature = np.array(
            (
                found,
                norm(relative_pos[0] / max(dist, 1e-4), 1, -1),
                norm(relative_pos[1] / max(dist, 1e-4), 1, -1),
                target_pos_norm[0],
                target_pos_norm[1],
                norm(dist, 1.41 * 128),
            ),
        )
        return feature

    def pb2struct(self, frame_state, last_action):
        obs, _ = frame_state
        self.step_no = obs["frame_state"]["step_no"]

        hero = obs["frame_state"]["heroes"][0]
        self.cur_pos = (hero["pos"]["x"], hero["pos"]["z"])

        # History position
        # 历史位置
        self.history_pos.append(self.cur_pos)
        if len(self.history_pos) > 10:
            self.history_pos.pop(0)

        # End position
        # 终点位置
        for organ in obs["frame_state"]["organs"]:
            if organ["sub_type"] == 4:
                end_pos_dis = RelativeDistance[organ["relative_pos"]["l2_distance"]]
                end_pos_dir = RelativeDirection[organ["relative_pos"]["direction"]]
                if organ["status"] != -1:
                    self.end_pos = (organ["pos"]["x"], organ["pos"]["z"])
                    self.is_end_pos_found = True
                # if end_pos is not found, use relative position to predict end_pos
                # 如果终点位置未找到，使用相对位置预测终点位置
                elif (not self.is_end_pos_found) and (
                    self.end_pos is None
                    or self.step_no % 100 == 0
                    or self.end_pos_dir != end_pos_dir
                    or self.end_pos_dis != end_pos_dis
                ):
                    distance = end_pos_dis * 20
                    theta = DirectionAngles[end_pos_dir]
                    delta_x = distance * math.cos(math.radians(theta))
                    delta_z = distance * math.sin(math.radians(theta))

                    self.end_pos = (
                        max(0, min(128, round(self.cur_pos[0] + delta_x))),
                        max(0, min(128, round(self.cur_pos[1] + delta_z))),
                    )

                    self.end_pos_dir = end_pos_dir
                    self.end_pos_dis = end_pos_dis

        self.last_pos_norm = self.cur_pos_norm
        self.cur_pos_norm = norm(self.cur_pos, 128, -128)
        self.feature_end_pos = self._get_pos_feature(self.is_end_pos_found, self.cur_pos, self.end_pos)

        # History position feature
        # 历史位置特征
        self.feature_history_pos = self._get_pos_feature(1, self.cur_pos, self.history_pos[0])

        self.move_usable = True
        self.last_action = last_action

    def process(self, frame_state, last_action):
        self.pb2struct(frame_state, last_action)
        # Legal action
        # 合法动作
        legal_action = self.get_legal_action()

        # Feature
        # 特征
        # feature = np.concatenate([self.cur_pos_norm, self.feature_end_pos, self.feature_history_pos, legal_action])
        
        obs, _ = frame_state
        feature = self.MyFeatureClass.build_feat(obs = obs)

        return (
            feature,
            legal_action,
            reward_process(self.feature_end_pos[-1], self.feature_history_pos[-1]),
        )

    def get_legal_action(self):
        # if last_action is move and current position is the same as last position, add this action to bad_move_ids
        # 如果上一步的动作是移动，且当前位置与上一步位置相同，则将该动作加入到bad_move_ids中
        if (
            abs(self.cur_pos_norm[0] - self.last_pos_norm[0]) < 0.001
            and abs(self.cur_pos_norm[1] - self.last_pos_norm[1]) < 0.001
            and self.last_action > -1
        ):
            self.bad_move_ids.add(self.last_action)
        else:
            self.bad_move_ids = set()

        legal_action = [self.move_usable] * self.move_action_num
        for move_id in self.bad_move_ids:
            legal_action[move_id] = 0

        if self.move_usable not in legal_action:
            self.bad_move_ids = set()
            return [self.move_usable] * self.move_action_num

        return legal_action


class Build_Feature:
    def __init__(self, *args, **kwargs):
        self.obs = None
        self.hero = None
        self.organs = None
        self.view = None
        
        self.destinations = dict()
        self.treasures = dict()
        self.buff = dict()
        self.starting = dict()
        self.map_dict = [ self.buff, self.destinations, self.starting, self.treasures ]
        
        self.visit_map = np.zeros(shape=(128, 128))
        self.map_memory = np.ones(shape=(128, 128))
        self.local_visit = None
        
    
    def reset(self):
        self.obs = None
        self.hero = None
        self.organs = None
        self.view = None
        self.local_visit = None
    
    def cat_var(self, obs):
        self.obs = obs
        self.hero = obs["frame_state"]["heroes"][0]
        self.view = np.array([v["values"] for v in obs["map_info"]])
        # 其中0表示不可通行，1表示可以通行，2表示起点位置，3表示终点位置，4表示宝箱位置，6表示加速增益位置。
        self.organs = obs["frame_state"]["organs"]
        self.update_memory_and_visit(map_obs = self.view, cur_pos = (self.hero['pos']['x'], self.hero['pos']['z']))
        
        def check(type_, organ, config_id):
            if config_id not in self.map_dict[type_]:
                self.map_dict[type_][config_id] = { "status": organ["status"], "pos" : organ["pos"], "relative_pos" : organ["relative_pos"], }
            else:
                self.map_dict[type_][config_id]["status"] = organ["status"]
                if self.map_dict[type_][config_id]["pos"]["x"] == -1:
                    self.map_dict[type_][config_id]["pos"] = organ["pos"]
                self.map_dict[type_][config_id]["relative_pos"] = organ["relative_pos"]
        
        for organ in self.organs:
            config_id = organ["config_id"]
            if config_id == 0:
                check(type_ = 0, organ = organ, config_id = config_id)
            elif config_id == 22:
                check(type_ = 1, organ = organ, config_id = config_id)
            elif config_id == 21:
                check(type_ = 2, organ = organ, config_id = config_id)
            else:
                check(type_ = 3, organ = organ, config_id = config_id)
        
          
    def update_memory_and_visit(self, map_obs, cur_pos):
        x, z = cur_pos
        self.visit_map[x, z] += 1

        # 计算当前局部视野覆盖的全局坐标范围
        gx = np.arange(x - 5, x + 6)
        gz = np.arange(z - 5, z + 6)
        # 生成网格坐标
        gx_grid, gz_grid = np.meshgrid(gx, gz)  # 注意 gz 是行，gx 是列，shape=(11, 11)
        # 合法范围 mask（防止越界）
        valid_mask = (gx_grid >= 0) & (gx_grid < 128) & (gz_grid >= 0) & (gz_grid < 128)
        # 创建不可通行 mask
        block_mask = (map_obs == 0) & valid_mask
        # 将 map_memory 中对应的不可通区域设置为 0
        self.map_memory[gx_grid[block_mask], gz_grid[block_mask]] = 0

    def get_local_visit(self, cen_x, cen_y, size=11, pad_val=2000):
        """
        从全局访问图中裁剪出局部视野访问频率图
        超出边界的部分使用 pad_val（如 2000）填充
        """
        H, W = self.visit_map.shape
        half = size // 2

        # 全局坐标范围
        x_start = cen_x - half
        y_start = cen_y - half
        x_end = cen_x + half + 1
        y_end = cen_y + half + 1

        # 局部图上应填入数据的位置（local patch）
        lx_start = max(0, -x_start)
        ly_start = max(0, -y_start)
        lx_end = size - max(0, x_end - H)
        ly_end = size - max(0, y_end - W)

        # 全局图实际可访问的区域（visit_map）
        vx_start = max(0, x_start)
        vy_start = max(0, y_start)
        vx_end = min(H, x_end)
        vy_end = min(W, y_end)

        # 初始化局部访问图，使用 pad_val 填充
        local = np.full((size, size), fill_value=pad_val, dtype=np.float32)

        # 替换局部访问图中的有效区域
        local[lx_start:lx_end, ly_start:ly_end] = self.visit_map[vx_start:vx_end, vy_start:vy_end]

        return local

    def available_pass_feat(self,):
        """ 生成可通行区域的特征图 """
        pass_feat = np.zeros(shape = self.view.shape)
        cen_x, cen_y = (pass_feat.shape[0] - 1) // 2, (pass_feat.shape[1] - 1) // 2
        pass_feat = (1.0 - self.local_visit / (np.max(self.local_visit) + 1e-6)) * (self.view == 1)
        pass_feat[cen_x][cen_y] = 0.0
        
        return pass_feat
    
    def available_treasure_feat(self):
        """ 生成宝箱的特征图 """
        treasure_feat = np.zeros(shape = self.view.shape)
        cen_x, cen_y = (treasure_feat.shape[0] - 1) // 2, (treasure_feat.shape[1] - 1) // 2
        
        
        # 预测宝箱位置
        for config_id, treasure in self.treasures.items():
            status = treasure["status"]
            # 预测宝箱方位
            direction_angle = DirectionAngles[RelativeDirection[treasure["relative_pos"]["direction"]]]
            dx, dy = int(5 * np.cos(direction_angle)), int(5 * np.sin(direction_angle))
            treasure_feat[cen_x + dx][cen_y + dy] = 1.0
        
        # 没有探测到宝箱，并且也没有返回相对方位
        if np.sum(treasure_feat) == 0:
            pass
        
        return treasure_feat
    
    def available_destination_feat(self):
        """ 生成终点的特征图 """
        dest_feat = np.zeros(shape = self.view.shape)
        cen_x, cen_y = (dest_feat.shape[0] - 1) // 2, (dest_feat.shape[1] - 1) // 2
        # dest_feat = (self.view == 3).astype(float)
        
        for config_id, dest in self.destinations.items():
            direction_angle = DirectionAngles[RelativeDirection[dest["relative_pos"]["direction"]]]
            dx, dy = int(5 * np.cos(direction_angle)), int(5 * np.sin(direction_angle))
            dest_feat[cen_x + dx][cen_y + dy] = 1.0
        
        if np.sum(dest_feat) == 0:
            pass
            
        return dest_feat

    def global_pos_to_local(self, cur_pos, global_size=128, local_size=11):
        """
        将智能体全局位置映射到局部11x11特征图上

        Args:
            cur_pos: (x, y) 全局坐标，范围 [0, global_size-1]
            global_size: 全局地图边长
            local_size: 局部特征图边长
        
        Returns:
            feature_map: (local_size, local_size) numpy数组
        """
        scale = local_size / global_size
        feature_map = np.zeros((local_size, local_size), dtype=np.float32)

        x_scaled = int(cur_pos[0] * scale)
        y_scaled = int(cur_pos[1] * scale)

        # 防止越界
        x_scaled = min(max(x_scaled, 0), local_size - 1)
        y_scaled = min(max(y_scaled, 0), local_size - 1)

        feature_map[x_scaled, y_scaled] = 1.0

        return feature_map
    
    def build_feat(self, obs):
        self.reset()
        self.cat_var(obs)
        self.local_visit = self.get_local_visit(cen_x = self.hero['pos']['x'], cen_y = self.hero['pos']['z'])

        pass_feat = np.expand_dims(self.available_pass_feat(), axis = 0)
        treasure_feat = np.expand_dims(self.available_treasure_feat(), axis = 0)
        destination_feat = np.expand_dims(self.available_destination_feat(), axis = 0)
        curpos_norm_feat = np.expand_dims(self.global_pos_to_local((self.hero['pos']['x'], self.hero['pos']['z'])), axis = 0)

        total_feat = np.concatenate([pass_feat, treasure_feat, destination_feat, curpos_norm_feat])
        
        return total_feat
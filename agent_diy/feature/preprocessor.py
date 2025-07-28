#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################

"""
Author: Tencent AI Arena Authors

"""

import json
import numpy as np
import math
from agent_diy.feature.definition import RelativeDistance, RelativeDirection, DirectionAngles, reward_process, ComputeReward


def norm(v, max_v, min_v=0):
    v = np.maximum(np.minimum(max_v, v), min_v)
    return (v - min_v) / (max_v - min_v)


class Preprocessor:
    def __init__(self) -> None:
        self.move_action_num = 8
        self.reset()
        
        # NOTE(junweiluo):
        self.MyFeatureClass = Build_Feature()
        self.RewCompute = ComputeReward()
    
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
        legal_action = self.get_legal_action(frame_state)

        # Feature
        # 特征
        # feature = np.concatenate([self.cur_pos_norm, self.feature_end_pos, self.feature_history_pos, legal_action])
        
        obs, _ = frame_state
        feature = self.MyFeatureClass.build_feat(obs = obs)
        
        rew, rew_stat = self.RewCompute.compute_reward(end_dist = self.feature_end_pos[-1], 
                                                        history_dist = self.feature_history_pos[-1],
                                                        organs = obs["frame_state"]["organs"],
                                                        cur_pos = (obs["frame_state"]["heroes"][0]["pos"]["x"], obs["frame_state"]["heroes"][0]["pos"]["z"]),
                                                        history_pos = self.history_pos,
                                                        step_no = obs['frame_state']['step_no']
                                                    )

        return (
            feature,
            legal_action,
            rew,
            rew_stat,
            # reward_process(self.feature_end_pos[-1], self.feature_history_pos[-1]),
        )

    def get_legal_action(self, frame_state):
        obs, _ = frame_state
        # MOVING Legal Actions
        # ========= 防止撞墙的逻辑 ============= #
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
            # return [self.move_usable] * self.move_action_num

        # SKILL Legal Actions
        hero = obs["frame_state"]["heroes"][0]
        if hero['talent']['status'] == 0:
            legal_skill_actions = np.zeros(self.move_action_num, dtype=bool)
        else:
            legal_skill_actions = np.zeros(self.move_action_num, dtype=bool)

            cur_x, cur_z = hero["pos"]["x"], hero["pos"]["z"]
            # 初始化全零矩阵
            skill_submap = np.zeros((33, 33), dtype=self.MyFeatureClass.new_memory.dtype)

            # 地图大小
            H, W = self.MyFeatureClass.new_memory.shape

            # 原图中提取的区域范围
            x_start_src = cur_x - 16
            x_end_src   = cur_x + 16 + 1
            z_start_src = cur_z - 16
            z_end_src   = cur_z + 16 + 1

            # 目标矩阵中要放置的位置
            x_start_dst = max(0, -x_start_src)
            z_start_dst = max(0, -z_start_src)

            # 源图中有效提取区域
            x_start_src_clamped = max(0, x_start_src)
            x_end_src_clamped   = min(W, x_end_src)
            z_start_src_clamped = max(0, z_start_src)
            z_end_src_clamped   = min(H, z_end_src)

            # 对应目标矩阵中要放置的位置结束点
            x_end_dst = x_start_dst + (x_end_src_clamped - x_start_src_clamped)
            z_end_dst = z_start_dst + (z_end_src_clamped - z_start_src_clamped)

            # 拷贝有效区域
            skill_submap[z_start_dst:z_end_dst, x_start_dst:x_end_dst] = \
                self.MyFeatureClass.new_memory[z_start_src_clamped:z_end_src_clamped,
                                            x_start_src_clamped:x_end_src_clamped]

            # 8 个方向 (dx,dz)
            dirs = np.array([
                (1, 0),   # →
                (1, 1),   # ↗
                (0, 1),   # ↑
                (-1, 1),  # ↖
                (-1, 0),  # ←
                (-1, -1), # ↙
                (0, -1),  # ↓
                (1, -1),  # ↘
            ], dtype=int)

            # 步长 1..16
            steps = np.arange(1, 17)            # (16,)
            dxs = dirs[:, 0:1] * steps          # (8,16)
            dzs = dirs[:, 1:2] * steps          # (8,16)

            # 子图坐标
            center = 16
            lxs = center + dxs                 # (8,16)
            lzs = center + dzs

            # 有效范围
            valid = (lxs >= 0) & (lxs < 33) & (lzs >= 0) & (lzs < 33)

            # 从 sub 中读取可通行标记
            vals = np.zeros_like(lxs, dtype=int)
            vals[valid] = skill_submap[lzs[valid], lxs[valid]]  # 1=通, 0=阻

            # 计算加权矩阵：对每个方向和步用 step * weight
            # pass_mask: (8,16) bool
            pass_mask = (vals == 1)
            block_mask = valid & ~pass_mask

            # 扩展 steps 到 (8,16) 方便向量运算
            step_mat = np.broadcast_to(steps, vals.shape)

            weighted = 0.7 * (step_mat * pass_mask) \
                    + 0.3 * (step_mat * block_mask)
            # 对每个方向求和
            scores = weighted.sum(axis=1)  # (8,)

            # 选 top4 方向合法
            order = np.argsort(-scores)
            legal_skill_actions[order[:4]] = True

        legal_action = legal_action + legal_skill_actions.tolist()

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
        
        self.fish_mapjson = json.load(open("/data/projects/back_to_the_realm_v2/kaiwu_env/conf/back_to_the_realm_v2/map_data/fish.json", "r"))
        self.visit_map = np.zeros(shape=(128, 128))
        self.map_memory = np.array(self.fish_mapjson["Flags"]).reshape(128, 128)
        self.local_visit = None
        self.new_memory = np.array(self.fish_mapjson["Flags"]).reshape(128, 128)
    
    def reset(self):
        self.obs = None
        self.hero = None
        self.organs = None
        self.view = None
        self.local_visit = None
    
    def cat_var(self, obs):
        self.obs = obs
        self.hero = obs["frame_state"]["heroes"][0]
        # view 要 transpose一下，这样子才能和地图信息对应得上，transpose后
        #                        3   2   1
        #                     4      A      0
        #                        5   6   7
        self.view = np.transpose(np.array([v["values"] for v in obs["map_info"]]))
        # 其中0表示不可通行，1表示可以通行，2表示起点位置，3表示终点位置，4表示宝箱位置，6表示加速增益位置。
        self.organs = obs["frame_state"]["organs"]
        
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
        # pass_feat = (1.0 - self.local_visit / (np.max(self.local_visit) + 1e-6)) * (self.view == 1)
        pass_feat[cen_x][cen_y] = 0.0
        
        return pass_feat
    
    def available_buff_feat(self):
        """ buff  """
        buff_feat = np.zeros(shape = self.view.shape)
        cen_x, cen_z = (buff_feat.shape[0] - 1) // 2, (buff_feat.shape[1] - 1) // 2
        # dest_feat = (self.view == 3).astype(float)
        for config_id, buff in self.buff.items():
            direction_angle = np.deg2rad(DirectionAngles[RelativeDirection[buff["relative_pos"]["direction"]]])
            l2_dist = RelativeDistance[buff['relative_pos']['l2_distance']]
            dx, dz = int(l2_dist * np.cos(direction_angle)), int(l2_dist * np.sin(direction_angle))
            buff_feat[cen_z - dz][cen_x - dx] = 1.0
        
        if np.sum(buff_feat) == 0:
            pass
            
        return buff_feat
        
    
    
    def available_treasure_feat(self):
        """ 生成宝箱的特征图 """
        treasure_feat = np.zeros(shape = self.view.shape)
        cen_x, cen_z = (treasure_feat.shape[1]) // 2, (treasure_feat.shape[0] - 1) // 2
        
        # 预测宝箱位置
        for config_id, treasure in self.treasures.items():
            status = treasure["status"]
            if status == 0:
                continue
            # pos_x, pos_z = treasure['pos']['x'], treasure['pos']['z']
            # 预测宝箱方位
            direction_angle = np.deg2rad(DirectionAngles[RelativeDirection[treasure["relative_pos"]["direction"]]])
            l2_dist = RelativeDistance[treasure['relative_pos']['l2_distance']]
            dx, dz = int(l2_dist * np.cos(direction_angle)), int(l2_dist * np.sin(direction_angle))
            treasure_feat[cen_z - dz][cen_x - dx] += 1.0
        
        
        # 没有探测到宝箱，并且也没有返回相对方位
        if np.sum(treasure_feat) == 0:
            pass
        
        treasure_feat = treasure_feat / np.sum(treasure_feat)
        
        return treasure_feat
    
    def available_destination_feat(self):
        """ 生成终点的特征图 """
        dest_feat = np.zeros(shape = self.view.shape)
        cen_x, cen_y = (dest_feat.shape[0] - 1) // 2, (dest_feat.shape[1] - 1) // 2
        # dest_feat = (self.view == 3).astype(float)
        
        
        for config_id, dest in self.destinations.items():
            direction_angle = np.deg2rad(DirectionAngles[RelativeDirection[dest["relative_pos"]["direction"]]])
            l2_dist = RelativeDistance[dest['relative_pos']['l2_distance']]
            dx, dy = int(l2_dist * np.cos(direction_angle)), int(l2_dist * np.sin(direction_angle))
            dest_feat[cen_x - dx][cen_y - dy] = 1.0
        
        if np.sum(dest_feat) == 0:
            pass
            
        return dest_feat


    def availble_dynamic_obstacle_feat(self):
        """ 好像智能体在fish.json地图上永远不会越界 """
        cur_x, cur_z = self.hero['pos']['x'], self.hero['pos']['z']
        half = self.view.shape[0] // 2
        # 从原始地图中获取没有动态障碍物的地图
        origin_local_view = self.map_memory[cur_z - half : cur_z + half + 1,
                                            cur_x - half : cur_x + half + 1]
        
        obstacle_feat = (origin_local_view != (self.view >= 1)).astype(float)
        
        
        return obstacle_feat


    def global_pos_to_local(self, cur_pos, global_size=128, local_size=11):
        scale = local_size / global_size
        feature_map = np.zeros((local_size, local_size), dtype=np.float32)

        x_scaled = int(cur_pos[0] * scale)
        y_scaled = int(cur_pos[1] * scale)

        # 防止越界
        x_scaled = min(max(x_scaled, 0), local_size - 1)
        y_scaled = local_size - 1 - min(max(y_scaled, 0), local_size - 1)
        feature_map[y_scaled, x_scaled] = 1.0

        return feature_map

    def update_dynamic_obstacles(self):
        """
        local_map_info: np.array shape (11, 11), 0不可通行(含动态障碍物), 其他>0可通行
        cur_x, cur_z: 英雄在全局地图上的坐标
        """

        half = 11 // 2
        cur_x, cur_z = self.hero['pos']['x'], self.hero['pos']['z']
        # 计算全局地图对应局部地图的起点
        x_start, x_end = cur_x - half, cur_x + half + 1
        z_start, z_end = cur_z - half, cur_z + half + 1

        # 取出对应的全局地图区域切片
        global_submap = self.new_memory[z_start:z_end, x_start:x_end]

        # 动态障碍物标记位置（局部地图中值为0，且不改变之前静态墙）
        # 只更新之前可通行位置上的动态障碍物
        dynamic_obstacle_mask = (self.view == 0) & (global_submap != 0)

        # 把这些动态障碍物位置标记为不可通行(0)
        global_submap[dynamic_obstacle_mask] = 0

        # 不改变其他位置，直接写回
        self.new_memory[z_start:z_end, x_start:x_end] = global_submap
    
    
    def build_feat(self, obs):
        self.reset()
        self.cat_var(obs)
        # self.local_visit = self.get_local_visit(cen_x = self.hero['pos']['x'], cen_y = self.hero['pos']['z'])
        self.update_dynamic_obstacles()
        pass_feat = np.expand_dims(self.available_pass_feat(), axis = 0)
        obstacle_feat = np.expand_dims(self.availble_dynamic_obstacle_feat(), axis = 0)
        buff_feat = np.expand_dims(self.available_buff_feat(), axis = 0)
        treasure_feat = np.expand_dims(self.available_treasure_feat(), axis = 0)
        destination_feat = np.expand_dims(self.available_destination_feat(), axis = 0)
        curpos_norm_feat = np.expand_dims(self.global_pos_to_local((self.hero['pos']['x'], self.hero['pos']['z'])), axis = 0)
        
        total_feat = np.concatenate([pass_feat, obstacle_feat, buff_feat, treasure_feat, destination_feat, curpos_norm_feat])
        
        return total_feat
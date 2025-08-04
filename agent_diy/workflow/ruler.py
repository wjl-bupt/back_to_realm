# -*- encoding: utf-8 -*-
'''
@File :ruler.py
@Created-Time :2025-08-04 10:02:24
@Author  :june
@Description   : make ruler.
@Modified-Time : 2025-08-04 10:02:24
'''

import json
import numpy as np
from copy import deepcopy
from dataclasses import dataclass
from agent_diy.feature.definition import RelativeDistance, RelativeDirection, DirectionAngles


class Organ:
    def __init__(self):
        self.status = None
        self.config_id = -1
        self.x = -1
        self.z = -1
        self.alive = True
        self.cost = None
        self.relative_pos = None

    def organ_update(self, organ):
        # buff 会复活，先定死只能吃一次
        self.config_id = organ["config_id"]
        self.status = organ["status"]
        if self.status == 0:
            self.alive = False # 
        if self.x == -1:
            self.x = organ["pos"]["x"]
            self.z = organ["pos"]["z"]
        self.relative_pos = organ["relative_pos"]
    
    



class Ruler():
    def __init__(self, organs):
        self.fish_json = "/data/projects/back_to_the_realm_v2/kaiwu_env/conf/back_to_the_realm_v2/map_data/fish.json"
        self.origin_grid = np.array(json.load(open(self.fish_json,"r"))["Flags"]).reshape(128,128)
        self.known_grid = deepcopy(self.origin_grid)
        self.buff = Organ()
        self.treasures = dict()
        self.denstination = Organ()
        self.start = Organ()
        self.hero = None
        self.treasures_num = -1
        self.step_no = -1
    
        for organ in organs:
            config_id = organ["config_id"]
            # buff 
            if config_id == 0:
                self.buff.organ_update(organ = organ)
            # start point
            elif config_id == 21:
                self.start.organ_update(organ = organ)
            # denstination
            elif config_id == 22:
                self.denstination.organ_update(organ = organ)
            else:
                self.treasures[config_id] = Organ()
                self.treasures[config_id].organ_update(organ = organ)
    
    def update_msg(self, obs):
        self.step_no = obs['frame_state']['step_no']
        organs = obs["frame_state"]["organs"]
        for organ in organs:
            config_id = organ["config_id"]
            # buff 
            if config_id == 0:
                self.buff.organ_update(organ = organ)
                if self.buff.alive == False: print(f'weijun.luo printf: buff eaten!')
            # start point
            elif config_id == 21:
                self.start.organ_update(organ = organ)
            # denstination
            elif config_id == 22:
                self.denstination.organ_update(organ = organ)
            else:
                if config_id in self.treasures:
                    self.treasures[config_id].organ_update(organ = organ)
                    if self.treasures[config_id].alive == False: print(f'weijun.luo printf: treasure eaten! id is {config_id}, location is ({self.treasures[config_id].x}, {self.treasures[config_id].z})')

        self.hero = obs["frame_state"]["heroes"][0]
    
    def remove_null_treasure(self):
        tmp_treasures = dict()
        for config_id, treasure in self.treasures.items():
            if treasure.alive == True:
                tmp_treasures[config_id] = treasure
        
        self.treasures = tmp_treasures
            
    
    def get_action(self, obs):
        curr_view = np.transpose(np.array([v["values"] for v in obs["map_info"]]))
        self.update_msg(obs = obs)
        self.remove_null_treasure()
        
        if 1000 - self.step_no <= 150:
            # 朝终点走
            relative_dir = RelativeDirection[self.denstination.relative_pos['direction']]
        
        elif len(self.treasures) > 0:
            # 收集剩余的宝箱
            nearest_trea_id, nearest_treasure = sorted(self.treasures.items(), key=lambda x:RelativeDistance[x[1].relative_pos['l2_distance']])[0]
            relative_dir = RelativeDirection[nearest_treasure.relative_pos['direction']]
            
        dirs = [
            (1, 0),   # →
            (1, 1),   # ↗
            (0, 1),   # ↑
            (-1, 1),  # ↖
            (-1, 0),  # ←
            (-1, -1), # ↙
            (0, -1),  # ↓
            (1, -1),  # ↘
        ]
        cen_x, cen_z = 5, 5
        
        
        options_choices = [dirs[relative_dir - 1]]
        # 将候选方向二分
        for i in range(1, 4):
            if dirs[(relative_dir - i) % 8] == dirs[(relative_dir + i) % 8]:
                break
            options_choices.append(dirs[(relative_dir + i) % 8])
            options_choices.append(dirs[(relative_dir - i) % 8])
        
        for option in options_choices:
            if curr_view[cen_z][cen_x] > 0:
                return dirs.index(option)
    
        return 0
            # 朝最近的宝箱走
            # relative_dir = np.deg2rad(DirectionAngles[RelativeDirection[nearest_treasures[0]['relative_pos']['direction']]])
            # relative_l2_dis = RelativeDistance[nearest_treasures[0]['relative_pos']['l2_distance']]
            # dx, dz = relative_l2_dis * np.cos(relative_dir), relative_l2_dis * np.sin(relative_dir)
            # 技能可以用
            
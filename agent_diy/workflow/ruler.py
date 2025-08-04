import json
import numpy as np
import math
import heapq
from copy import deepcopy
from agent_diy.feature.definition import RelativeDirection, RelativeDistance, DirectionAngles


class Ruler:
    def __init__(self, logger, step_no, fish_json_path="/data/projects/back_to_the_realm_v2/kaiwu_env/conf/back_to_the_realm_v2/map_data/fish.json"):
        self.grid_size = 128
        if fish_json_path:
            origin_flags = json.load(open(fish_json_path, "r"))['Flags']
            self.origin_grid = np.array(origin_flags).reshape(128, 128).T
        else:
            self.origin_grid = np.zeros((128, 128), dtype=int)
        self.map = deepcopy(self.origin_grid)

        self.position = (0, 0)
        self.speed_timer = 0
        self.blink_cd = 0
        self.blink_range = 16

        self.buff = dict()
        self.chests_memory = {
            i: {'status':-1, 'x':-1, 'z':-1} for i in range(1, 15)
        }
        self.destination = dict()
        self.num_chests = 0
        self.collect_chest = 0

        self.path = []

        self.directions = [
            (1, 0), # 0
            (1, 1), # 45
            (0, 1), # 
            (-1, 1),
            (-1, 0), 
            (-1, -1), 
            (0, -1), 
            (1, -1)
        ]
        
        # game info
        self.logger = logger
        self.step_no = step_no
        
        # target info
        self.target = None

    def update_local_view(self, local_view, top_left):
        for dy in range(len(local_view)):
            for dx in range(len(local_view[0])):
                x, y = top_left[0] + dx, top_left[1] + dy
                if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                    self.map[x][y] = local_view[dx][dy]
        # top_left_x , top_left_z = top_left
        # bottom_rig_x, bottom_rig_z = top_left_x + 11, top_left_z + 11
        # self.map[top_left_x:bottom_rig_x, top_left_z:bottom_rig_z] = local_view

    def update_position(self, pos):
        self.position = pos

    def update_speed_timer(self, buff_time):
        self.speed_timer = buff_time

    def get_neighbors(self, x, y):
        step = int(1.5 if self.speed_timer > 0 else 1)
        neighbors = []
        for dx, dy in self.directions:
            for s in range(1, step+1):
                nx, ny = x + dx * s, y + dy * s
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    if self.map[nx][ny] == 1:
                        neighbors.append((nx, ny))
        return neighbors

    def heuristic(self, a, b):
        dx, dy = abs(a[0]-b[0]), abs(a[1]-b[1])
        return dx + dy + (math.sqrt(2) - 2) * min(dx, dy)

    def a_star(self, start, goal):
        open_list = []
        heapq.heappush(open_list, (0, 0, start))
        came_from = {}
        cost_so_far = {start: 0}
        visited = set()

        closest_node = start
        min_h = self.heuristic(start, goal)

        while open_list:
            _, cost, current = heapq.heappop(open_list)
            if current in visited:
                continue
            visited.add(current)

            h = self.heuristic(current, goal)
            if h < min_h:
                min_h = h
                closest_node = current

            if current == goal:
                break

            for neighbor in self.get_neighbors(*current):
                new_cost = cost + 1
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + self.heuristic(neighbor, goal)
                    heapq.heappush(open_list, (priority, new_cost, neighbor))
                    came_from[neighbor] = current

        # 构造路径
        target_node = goal if goal in came_from else closest_node
        if target_node == start:
            return []

        path = []
        current = target_node
        while current != start:
            path.append(current)
            current = came_from[current]
        path.append(start)
        path.reverse()
        return path

    def blink_target(self, dx, dy):
        for r in range(self.blink_range, 0, -1):
            tx = self.position[0] + dx * r
            ty = self.position[1] + dy * r
            if 0 <= tx < self.grid_size and 0 <= ty < self.grid_size and self.map[ty][tx] == 1:
                return (tx, ty)
        return None

    def update_organ_memory(self, organs):
        
        for organ in organs:
            cid = organ.get('config_id', -1)
            status = organ.get('status', -1)
            pos = organ.get('pos', {'x': -1, 'z': -1})
            if 1 <= cid <= 15:
                if self.num_chests <= cid:
                    self.num_chests += 1
                if status == 0 and cid not in self.chests_memory:
                    self.collect_chest += 1
                    self.logger.info(f"step_no : {self.step_no} | *** chest was collect! config_id is {cid}, pos is {pos}. ***")
                    self.chests_memory.pop(cid, None)
                elif pos['x'] != -1:
                    self.chests_memory[cid] = {
                        'status': status,
                        'x': pos['x'],
                        'z': pos['z'],
                        'relative_pos': organ['relative_pos'],
                    }
            elif cid == 0:
                self.buff = {'x': pos['x'], 'z': pos['z']}
            elif cid == 22 :
                self.destination = {'x': pos['x'], 'z': pos['z'], 'relative_pos': organ['relative_pos']}


    def get_denstination_pos(self):
        if self.destination['x'] != -1:
            return (self.destination['x'], self.destination['z']), False, 22
        rel = self.destination.get('relative_pos', {})
        direction = rel.get('direction', 'East')
        dist_str = rel.get('l2_distance', 'Medium')
        angle = math.radians(DirectionAngles[RelativeDirection[direction]])
        scale = RelativeDistance[dist_str] * 20
        est_x = int(round(self.position[0] + scale * math.cos(angle)))
        est_y = int(round(self.position[1] + scale * math.sin(angle)))
        est_x = max(0, min(127, est_x))
        est_y = max(0, min(127, est_y))
        
        return est_x, est_y

    def get_chest_pos(self, cid):
        
        chest = self.chests_memory[cid]
        if chest['x'] != -1:
            return chest['x'], chest['z']
        
        rel = chest.get('relative_pos', {})
        direction = rel.get('direction', 'East')
        dist_str = rel.get('l2_distance', 'Medium')
        angle = math.radians(DirectionAngles[RelativeDirection[direction]])
        # if RelativeDistance[dist_str] > 0:
        scale = RelativeDistance[dist_str] * 20
        est_x = int(round(self.position[0] + scale * math.cos(angle)))
        est_y = int(round(self.position[1] + scale * math.sin(angle)))
        est_x = max(0, min(127, est_x))
        est_y = max(0, min(127, est_y))
        pos = (est_x, est_y)
        
        return pos
        

    def find_target(self, goal = None):
        if goal != None:
            if goal == 22:
                return self.get_denstination_pos(), False, 22
            else:
                return self.get_chest_pos(cid = goal), False, 22
            
        candidates = []
        if self.collect_chest < self.num_chests:
            for cid, info in self.chests_memory.items():
                # 宝箱不存在
                if cid > self.num_chests:
                    continue
                if info['status'] == 0:
                    continue
                
                pos = self.get_chest_pos(cid = cid)
                dist = self.heuristic(self.position, pos)
                candidates.append((dist, pos, cid))

            if candidates:
                candidates.sort(key=lambda x:x[0])
                return candidates[0][1], candidates[0][0], candidates[0][-1]

        # 若宝箱收集完成，则走到终点
        if self.collect_chest >= self.num_chests or self.step_no >= 900:
            pos = self.get_denstination_pos()
            dist = self.heuristic(self.position, pos)
            return pos, dist, 22

        return None, -1, -1

    def decide_move(self, target):
        path = self.a_star(self.position, target)
        # print(f'path is {len(path)}, {path}')
        if not path or len(path) < 2:
            return 0
        count = 0
        next_step = path[1]
        dx = next_step[0] - self.position[0]
        dz = next_step[1] - self.position[1]
        # for i, (dix, diy) in enumerate(self.directions):
        #     if dx == dix and dy == diy:
        #         return i  # move direction
        act = self.directions.index((dx, dz))
        if act != -1:
            # print(f'cur pos is { self.position}, next pos is {next_step}, act should be {act}')
            return act
        
        return 0

    def decide_blink(self, target):
        if self.blink_cd > 0:
            return -1
        dx = target[0] - self.position[0]
        dy = target[1] - self.position[1]
        norm = math.hypot(dx, dy)
        if norm == 0:
            return -1
        dx = int(round(dx / norm))
        dy = int(round(dy / norm))
        blink_pos = self.blink_target(dx, dy)
        if blink_pos:
            for i, (dix, diy) in enumerate(self.directions):
                if dx == dix and dy == diy:
                    self.position = blink_pos
                    # self.blink_cd = 3
                    return 8 + i
        return -1

    def run_step(self, obs):

        hero = obs['frame_state']['heroes'][0]
        self.step_no = obs['frame_state']['step_no']
        self.update_position((hero['pos']['x'], hero['pos']['z']))
        self.update_speed_timer(hero.get('buff_remain_time', 0))
        self.blink_cd = hero['talent'].get('cooldown', 0)

        self.update_organ_memory(obs['frame_state']['organs'])

        
        
        if self.target  == None:
            self.target = dict()
            target, dist, goal = self.find_target()
            self.target['config_id'] = goal
            self.target['pos'] = target
            self.target['dist'] = dist
        else:
            # 目标是宝箱
            if self.target['config_id'] != 22 :
                if self.target['config_id'] not in self.chests_memory:
                    g = None
                else:
                    g = self.target['config_id']
                target, dist, goal = self.find_target()
                self.target['config_id'] = goal
                self.target['pos'] = target
                self.target['dist'] = dist
            else:
                target, dist, goal = self.find_target(22)
                self.target['config_id'] = goal
                self.target['pos'] = target
                self.target['dist'] = dist
        
        # print(f"target is {self.target['config_id']}, dist is {self.target['dist']}")
        
        if not self.target:
            return 0
        
        if self.target['dist'] >= 12 * 1.41:
            blink_action = self.decide_blink(self.target['pos'])
            if blink_action >= 0:
                # print(f"step_no is {self.step_no}, current pos is {self.position}, act is {blink_action}, current target is {self.target['config_id']}, dist is {self.target['dist']}")
                return blink_action

        move_act = self.decide_move(self.target['pos'])
        # print(f"step_no is {self.step_no}, act is {move_act}, current pos is {self.position}, current target is {self.target['config_id']}, dist is {self.target['dist']}")
        return move_act
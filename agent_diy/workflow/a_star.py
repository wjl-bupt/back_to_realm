# -*- encoding: utf-8 -*-
'''
@File :a_star.py
@Created-Time :2025-08-04 10:02:58
@Author  :june
@Description   : A*
@Modified-Time : 2025-08-04 10:02:58
'''
import numpy as np

import heapq
import math

def astar_with_skills(start, goal, grid):
    """
    A* pathfinding with blink and speed buff skills.

    Parameters:
        start: (x, y)
        goal: (x, y)
        grid: 2D list (0: empty, 1: wall, 2: speed buff)

    Returns:
        path: list of (x, y) or empty list if not found
    """
    rows, cols = len(grid), len(grid[0])

    directions = [
        (1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1),
        (1, 1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
        (1, -1, math.sqrt(2)), (-1, -1, math.sqrt(2))
    ]

    def in_bounds(x, y):
        return 0 <= x < cols and 0 <= y < rows

    def passable(x, y):
        return grid[y][x] != 1

    def heuristic(a, b):
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return dx + dy + (math.sqrt(2) - 2) * min(dx, dy)

    # state: (f, cost, x, y, speed_timer, blink_used, path)
    open_heap = []
    heapq.heappush(open_heap, (0, 0, start[0], start[1], 0, False, [start]))
    visited = set()

    while open_heap:
        f, cost, x, y, speed_timer, blink_used, path = heapq.heappop(open_heap)
        state_id = (x, y, speed_timer, blink_used)

        if state_id in visited:
            continue
        visited.add(state_id)

        if (x, y) == goal:
            return path

        # Speed buff pickup
        if grid[y][x] == 2:
            speed_timer = 40

        # Determine step options
        max_step = 2 if speed_timer > 0 else 1
        new_speed_timer = max(0, speed_timer - 1)

        for dx, dy, base_cost in directions:
            for step in range(1, max_step + 1):
                nx, ny = x + dx * step, y + dy * step
                if not in_bounds(nx, ny) or not passable(nx, ny):
                    continue
                move_cost = cost + base_cost * step
                h = heuristic((nx, ny), goal)
                heapq.heappush(open_heap, (move_cost + h, move_cost, nx, ny, new_speed_timer, blink_used, path + [(nx, ny)]))

        # Blink move
        if not blink_used:
            for dx in range(-16, 17):
                for dy in range(-16, 17):
                    if dx**2 + dy**2 > 16**2:
                        continue
                    nx, ny = x + dx, y + dy
                    if not in_bounds(nx, ny) or not passable(nx, ny):
                        continue
                    h = heuristic((nx, ny), goal)
                    blink_cost = cost + 0.1  # blink is low cost
                    heapq.heappush(open_heap, (blink_cost + h, blink_cost, nx, ny, new_speed_timer, True, path + [(nx, ny)]))

    return []  # No path found


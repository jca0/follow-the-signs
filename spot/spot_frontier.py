import ast
import os
import sys
import math
import time

from typing import Literal
from pydantic import BaseModel, Field

from pprint import pprint
from PIL import Image
import heapq
import numpy as np
import yaml
import re
import random
from queue import Queue

from helpers.env_utils import *
from helpers.agent_maps import *
from helpers.spot_env import *
from helpers.spot_utils import *

def frontier_exploration(seen_grid, agent_pos):
    """
    Find all nearest seen cells ('.') that are adjacent to unexplored cells ('?').
    Returns a list of coordinates [(r1, c1), (r2, c2), ...].
    """
    rows, cols = len(seen_grid.get_grid()), len(seen_grid.get_grid()[0])
    DIRECTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    queue = Queue()
    queue.put((agent_pos, 0))  # (position, distance)
    visited = {agent_pos}
    frontier_cells = []
    found_frontier = False
    min_distance = None

    while not queue.empty():
        (r, c), dist = queue.get()

        # Get current cell and check traversability (free)
        current_cell = seen_grid.get_grid()[r][c]
        current_type = current_cell.get('feature_type') if isinstance(current_cell, dict) else None

        # Check if current cell is a seen free cell that borders an unknown cell
        if current_type == 'free':
            for dr, dc in DIRECTIONS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    neighbor_cell = seen_grid.get_grid()[nr][nc]
                    # Unknown cells are represented as empty dicts
                    if neighbor_cell == {}:
                        if not found_frontier:
                            found_frontier = True
                            min_distance = dist
                        if dist == min_distance:
                            frontier_cells.append((r, c))
                        break  # no need to check other neighbors once confirmed frontier

        # Only continue exploring if frontier not yet found
        if not found_frontier:
            for dr, dc in DIRECTIONS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                    neighbor_cell = seen_grid.get_grid()[nr][nc]
                    neighbor_type = neighbor_cell.get('feature_type') if isinstance(neighbor_cell, dict) else None
                    if neighbor_type == 'free':
                        visited.add((nr, nc))
                        queue.put(((nr, nc), dist + 1))

    return frontier_cells

def run_frontier_agent(env, command_client, grid_loc, seen_occupancy, seen_semantic, start, goal, k, seed=None):
    random.seed(seed)
    found_goal = False
    steps = 0
    complete_path = []
    agent_pos = start
    while True:
        steps += 1

        print(f"STEP {steps}")
        # print(f"STEP {steps}:")
        seen_occupancy.update_with_slice(agent_pos, k)
        seen_semantic.update_with_slice(agent_pos, k)
        # print(seen_occupancy.mark_grid(agent_pos))
        # print(seen_semantic.get_slice(agent_pos, k))
        
        goal_pos = seen_semantic.find_label(goal)
        if goal_pos:
            found_goal = True
            break

        if seen_occupancy.is_fully_explored():
            print("No valid path to goal.")
            break

        # if goal is not in slice, find unexplored cell and move towards it
        # path = frontier_exploration(seen_grid, agent_pos)
        # if path:
        #     agent_pos = path[1]
        # else:
        #     print("No path found")
        #     break       
        frontier_cells = frontier_exploration(seen_semantic, agent_pos)
        random_frontier = None
        if not frontier_cells:
            occ_frontiers = seen_occupancy.find_frontier_cells()
            if not occ_frontiers:
                break  # or return -1
            random_frontier = random.choice(occ_frontiers)
        else:
            random_frontier = random.choice(frontier_cells)
        print(f"Moving to random frontier cell {random_frontier}")
        # find path to frontier
        path = seen_occupancy.astar(agent_pos, random_frontier)
        agent_pos = path[-1] if path else agent_pos
        cr, cc = grid_loc.get_current_cell()
        cx, cy = env.cell_center_xy(cr, cc)
        tx, ty = env.cell_center_xy(agent_pos[0], agent_pos[1])
        yaw = math.atan2(ty - cy, tx - cx)
        move_to_cell(command_client, grid_loc, agent_pos[0], agent_pos[1], yaw)
        time.sleep(0.2)

    if found_goal:
        # A* to the goal
        # goal_pos = seen_semantic.find_label(goal)
        goal_pos = (0, 7)
        cr, cc = grid_loc.get_current_cell()
        cx, cy = env.cell_center_xy(cr, cc)
        tx, ty = env.cell_center_xy(goal_pos[0], goal_pos[1])
        yaw = math.atan2(ty - cy, tx - cx)
        move_to_cell(command_client, grid_loc, goal_pos[0], goal_pos[1], yaw)
        time.sleep(0.2)
        print(agent_pos, goal_pos)
        last_path = seen_occupancy.astar(agent_pos, goal_pos)
        try:
            total_steps = len(last_path) + 2 + steps
        except:
            total_steps = steps + 2
            print(f"Found goal in {total_steps} steps.")
        return total_steps
    else:
        print("No path to goal found.") 
        return -1


if __name__ == "__main__":  
    env = Bldg4()
    occupancy_grid = env.occupancy_grid
    semantic_grid = env.semantic_grid
    seen_occupancy = SeenOccupancyGrid(occupancy_grid)
    seen_semantic = SeenSemanticGrid(semantic_grid)
    k = 5
    result = run_frontier_agent(seen_occupancy, seen_semantic, (0, 0), '1')
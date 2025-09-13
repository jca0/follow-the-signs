import numpy as np
from bosdyn.client import math_helpers

class SpotEnv:
    def __init__(self, resolution_m=1, origin_xy_m=(0, 0), origin_yaw_rad=0.0):
        self.resolution_m = resolution_m
        self.origin_xy_m = origin_xy_m
        self.origin_yaw_rad = origin_yaw_rad

    def cell_center_xy(self, r, c):
        x = (c + 0.5) * self.resolution_m - self.origin_xy_m[0]
        y = (r + 0.5) * self.resolution_m - self.origin_xy_m[1]
        return x, y
    
    def se2_from_cell(self, r, c, yaw_rad):
        x, y = self.cell_center_xy(r, c)
        return math_helpers.SE2Pose(x, y, yaw_rad)
        

class TestEnv(SpotEnv):
    def __init__(self):
        super().__init__(resolution_m=1, origin_xy_m=(0, 0), origin_yaw_rad=0.0)

        self.occupancy_grid = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ])

        rows, cols = self.occupancy_grid.shape

        self.semantic_grid = [[{} for _ in range(cols)] for _ in range(rows)]
        for y in range(rows):
            for x in range(cols):
                if self.occupancy_grid[y][x] == 1:
                    self.semantic_grid[y][x] = {"feature_type": "wall"}
                else:
                    self.semantic_grid[y][x] = {"feature_type": "free"}
    
        self.semantic_grid[1][1]["room_number"] = "1"


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

    def cell_from_xy(self, x, y):
        c = int(np.floor((x + self.origin_xy_m[0]) / self.resolution_m))
        r = int(np.floor((y + self.origin_xy_m[1]) / self.resolution_m))
        return r, c
    
    def se2_from_cell(self, r, c, yaw_rad):
        x, y = self.cell_center_xy(r, c)
        return math_helpers.SE2Pose(x, y, yaw_rad)

    def cell_from_se2(self, se2):
        return self.cell_from_xy(se2.x, se2.y)


class TestEnv(SpotEnv):
    def __init__(self, resolution_m=1, origin_xy_m=(0, 0), origin_yaw_rad=0.0):
        super().__init__(resolution_m=resolution_m, origin_xy_m=origin_xy_m, origin_yaw_rad=origin_yaw_rad)

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


# class RealSchwarz(SpotEnv):
#     def __init__(self, resolution_m=1, origin_xy_m=(0, 0), origin_yaw_rad=0.0):
#         super().__init__(resolution_m=resolution_m, origin_xy_m=origin_xy_m, origin_yaw_rad=origin_yaw_rad)

#         self.occupancy_grid = np.array([
#             [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
#             [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
#             [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
#             [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
#             [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
#         ])
#         row, col = self.occupancy_grid.shape
        
#         self.semantic_grid = [[{} for _ in range(col)] for _ in range(row)]
#         for y in range(row):
#             for x in range(col):
#                 if self.occupancy_grid[y][x] == 1:
#                     self.semantic_grid[y][x] = {"feature_type": "wall"}
#                 else:
#                     self.semantic_grid[y][x] = {"feature_type": "free"}
        
#         self.semantic_grid[8][14]["room_number"] = "621"
#         self.semantic_grid[8][15]["room_number"] = "621"
#         self.semantic_grid[8][16]["room_number"] = "621"
#         self.semantic_grid[8][17]["room_number"] = "621"
#         self.semantic_grid[8][33]["room_number"] = "631"
#         self.semantic_grid[8][34]["room_number"] = "631"
#         self.semantic_grid[8][35]["room_number"] = "631"

#         self.semantic_grid[2][3]["sign"] = "Rooms 621-631 southward."
#         self.semantic_grid[8][6]["sign"] = "Rooms 621-631 to the right."
#         self.semantic_grid[8][14]["sign"] = "Room 631 to the right."


class RealSchwarz(SpotEnv):
    def __init__(self, resolution_m=1, origin_xy_m=(0, 0), origin_yaw_rad=0.0):
        super().__init__(resolution_m=resolution_m, origin_xy_m=origin_xy_m, origin_yaw_rad=origin_yaw_rad)

        # Flip occupancy grid horizontally
        self.occupancy_grid = np.array([
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1][::-1],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1][::-1],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1][::-1],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1][::-1],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1][::-1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][::-1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][::-1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][::-1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0][::-1],
        ])
        row, col = self.occupancy_grid.shape

        # Build semantic grid
        self.semantic_grid = [[{} for _ in range(col)] for _ in range(row)]
        for y in range(row):
            for x in range(col):
                if self.occupancy_grid[y][x] == 1:
                    self.semantic_grid[y][x] = {"feature_type": "wall"}
                else:
                    self.semantic_grid[y][x] = {"feature_type": "free"}

        # Flipped indices: col' = n_cols - 1 - col
        n = col
        # Rooms
        for c in [14, 15, 16, 17]:
            self.semantic_grid[8][n - 1 - c]["room_number"] = "621"
        for c in [33, 34, 35]:
            self.semantic_grid[8][n - 1 - c]["room_number"] = "631"

        # Signs
        self.semantic_grid[2][n - 1 - 3]["sign"] = "Rooms 621-631 southward."
        self.semantic_grid[8][n - 1 - 6]["sign"] = "Rooms 621-631 to the right."
        self.semantic_grid[8][n - 1 - 14]["sign"] = "Room 631 to the right."

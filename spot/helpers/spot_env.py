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


class RealSchwarz(SpotEnv):
    def __init__(self, resolution_m=1, origin_xy_m=(0, 0), origin_yaw_rad=0.0):
        super().__init__(resolution_m=resolution_m, origin_xy_m=origin_xy_m, origin_yaw_rad=origin_yaw_rad)

        # self.occupancy_grid = np.array([
        #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #     [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
        #     [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
        #     [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
        #     [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
        #     [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
        # ])
        self.occupancy_grid = np.array([
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
        ])
        row, col = self.occupancy_grid.shape

        self.semantic_grid = [[{} for _ in range(col)] for _ in range(row)]
        for y in range(row):
            for x in range(col):
                if self.occupancy_grid[y][x] == 1:
                    self.semantic_grid[y][x] = {"feature_type": "wall"}
                else:
                    self.semantic_grid[y][x] = {"feature_type": "free"}

        self.semantic_grid[0][9]["room_number"] = "621"
        self.semantic_grid[0][23]["room_number"] = "631"

        self.semantic_grid[4][19]["sign"] = "Rooms 607–609, 611–615, 621, 631–633, 644–646 upwards."
        self.semantic_grid[4][15]["sign"] = "Rooms 607–609, 611–615, 621 to the right; Rooms 631–633, 641, 646 to the left."


class RealSchwarzExtended(SpotEnv):
    def __init__(self, resolution_m=1, origin_xy_m=(0, 0), origin_yaw_rad=0.0):
        super().__init__(resolution_m=resolution_m, origin_xy_m=origin_xy_m, origin_yaw_rad=origin_yaw_rad)

        self.occupancy_grid = np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
        ])
        row, col = self.occupancy_grid.shape

        self.semantic_grid = [[{} for _ in range(col)] for _ in range(row)]
        for y in range(row):
            for x in range(col):
                if self.occupancy_grid[y][x] == 1:
                    self.semantic_grid[y][x] = {"feature_type": "wall"}
                else:
                    self.semantic_grid[y][x] = {"feature_type": "free"}

        self.semantic_grid[0][20]["room_number"] = "621"
        self.semantic_grid[0][34]["room_number"] = "631"
        self.semantic_grid[0][10]["room_number"] = "615"
        self.semantic_grid[4][8]["room_number"] = "609"
        self.semantic_grid[4][5]["room_number"] = "607"
        self.semantic_grid[0][3]["room_number"] = "611"
        self.semantic_grid[6][1]["room_number"] = "601N"

        self.semantic_grid[4][30]["sign"] = "Rooms 607–609, 611–615, 621, 631–633, 644–646 upwards."
        self.semantic_grid[4][26]["sign"] = "Rooms 607–609, 611–615, 621 to the right; Rooms 631–633, 641, 646 to the left."
        self.semantic_grid[4][10]["sign"] = "Rooms 607-609, 611-615 to the right."
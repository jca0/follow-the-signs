import numpy as np

class TestEnv:
    def __init__(self):
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
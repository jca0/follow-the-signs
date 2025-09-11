import numpy as np
import heapq
from .utils import bresenham_line

class SeenOccupancyGrid:

    def __init__(self, occupancy_grid: np.ndarray):
        self.occupancy_grid = occupancy_grid
        self.grid = np.full(occupancy_grid.shape, -1)
        self.height = occupancy_grid.shape[0]
        self.width = occupancy_grid.shape[1]

    def update_with_slice(self, agent_pos: tuple, k: int):
        if not (0 <= agent_pos[0] < self.height and 0 <= agent_pos[1] < self.width):
            raise ValueError("Agent position is out of bounds.")
        
        r = k // 2
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                x, y = agent_pos[0] + dx, agent_pos[1] + dy
                if 0 <= x < self.height and 0 <= y < self.width:
                    line = bresenham_line(agent_pos[0], agent_pos[1], x, y)
                    visible = True
                    for (ix, iy) in line[:-1]:
                        if self.occupancy_grid[ix][iy] == 1:
                            visible = False
                            break
                    if visible:
                        self.grid[x][y] = self.occupancy_grid[x][y]

    def find_frontier_cells(self):
        rows, cols = self.height, self.width
        frontiers = []
        for r in range(rows):
            for c in range(cols):
                if self.grid[r][c] != 0:
                    continue
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and self.grid[nr][nc] == -1:
                        frontiers.append((r, c))
                        break
        return frontiers
    
    def plan_towards(self, start, target):
        if start != target:
            path = self.astar(start, target)
            if path:
                return path
        
        frontiers = self.find_frontier_cells()
        if not frontiers:
            return None
        
        def manhattan(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
        best_frontier = min(frontiers, key=lambda x: manhattan(x, target))
        return self.astar(start, best_frontier)
    
    def astar(self, start, target):
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
        rows, cols = self.height, self.width

        if not (0 <= target[0] < rows and 0 <= target[1] < cols):
            return None
        open_set = []
        heapq.heappush(open_set, (0, start))

        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, target)}

        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == target:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]
            
            for dr, dc in directions:
                nr, nc = (current[0] + dr, current[1] + dc)
                if 0 <= nr < rows and 0 <= nc < cols:
                    # Only traverse known-free cells, but allow stepping onto the target cell
                    if (nr, nc) != target and self.grid[nr][nc] != 0:
                        continue
                    tentative_g_score = g_score[current] + 1
                    if (nr, nc) not in g_score or tentative_g_score < g_score[(nr, nc)]:
                        came_from[(nr, nc)] = current
                        g_score[(nr, nc)] = tentative_g_score
                        f_score[(nr, nc)] = tentative_g_score + heuristic((nr, nc), target)
                        heapq.heappush(open_set, (f_score[(nr, nc)], (nr, nc)))
        return None
    
    def is_fully_explored(self):
        return np.all(self.grid != -1)
    
    def get_grid(self):
        return self.grid
    
    def mark_grid(self, pos):
        copy = self.grid.copy()
        copy[pos[0]][pos[1]] = 8
        return copy
    
    def __str__(self):
        return str(self.grid)


class SeenSemanticGrid:

    def __init__(self, semantic_grid):
        self.semantic_grid = semantic_grid
        self.grid = [[{} for _ in range(len(semantic_grid[0]))] for _ in range(len(semantic_grid))]
        self.height = len(semantic_grid)
        self.width = len(semantic_grid[0])
    
    def update_with_slice(self, agent_pos: tuple, k: int):
        if not (0 <= agent_pos[0] < self.height and 0 <= agent_pos[1] < self.width):
            raise ValueError("Agent position is out of bounds.")
        
        r = k // 2
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                x, y = agent_pos[0] + dx, agent_pos[1] + dy
                if 0 <= x < self.height and 0 <= y < self.width:
                    line = bresenham_line(agent_pos[0], agent_pos[1], x, y)
                    visible = True
                    for (ix, iy) in line[:-1]:
                        cell_blocker = self.semantic_grid[ix][iy]
                        blocker_type = cell_blocker.get("feature_type") if isinstance(cell_blocker, dict) else None
                        if blocker_type in ["wall", "door"]:
                            visible = False
                            break
                    if visible:
                        self.grid[x][y] = self.semantic_grid[x][y]
    
    def get_slice(self, agent_pos, k):
        r = k // 2
        slice_grid = []
        for dr in range(max(0, agent_pos[0] - r), min(self.height, agent_pos[0] + r + 1)):
            row = []
            for dc in range(max(0, agent_pos[1] - r), min(self.width, agent_pos[1] + r + 1)):
                row.append(self.grid[dr][dc])
            slice_grid.append(row)
        return slice_grid
        

    def update_key_with_pattern(self, pos, key, value):
        self.grid[pos[0]][pos[1]][key] = value

    def find_label(self, goal):
        for r in range(self.height):
            for c in range(self.width):
                cell = self.grid[r][c]
                if cell:
                    for k, v in cell.items():
                        if k == "feature_type":
                            continue
                        if v == goal:
                            return (r, c)
        return None
    
    def get_grid(self):
        return self.grid
    
    def __str__(self):
        return str(self.grid)


class ConfidenceGrid:
    
    def __init__(self, rows, cols, decay_factor=0.90):
        self.rows = rows
        self.cols = cols
        self.decay_factor = decay_factor
        self.confidence_grid = np.zeros((rows, cols))

    def apply_decay(self):
        self.confidence_grid *= self.decay_factor

    def update_frequency(self, agent_pos, region):
        self.apply_decay()
        if region == "up":
            self.confidence_grid[:agent_pos[0], :] += 1
        elif region == "down":
            self.confidence_grid[agent_pos[0] + 1:, :] += 1
        elif region == "left":
            self.confidence_grid[:, :agent_pos[1]] += 1
        elif region == "right":
            self.confidence_grid[:, agent_pos[1]+1:] += 1

    def update_confidence(self, seen_semantic_grid, goal):
        for r in range(self.rows):
            for c in range(self.cols):
                cell = seen_semantic_grid[r][c]
                if cell != {}:
                    feature_type = cell.get("feature_type")
                    room_number = cell.get("room_number")
                    if feature_type == "door" and room_number == goal:
                        self.confidence_grid[r][c] = 1.0
                    else:
                        self.confidence_grid[r][c] = 0.0
        
    def find_max_confidence_pos(self):
        max_val = np.max(self.confidence_grid)
        max_idxs = np.argwhere(self.confidence_grid == max_val)
        centroid = max_idxs.mean(axis=0)
        return (int(centroid[0]), int(centroid[1]))

    def get_grid(self):
        return self.confidence_grid

if __name__ == "__main__":
    env = Stud()
    semantic_grid = env.semantic_grid
    seen_semantic_grid = SeenSemanticGrid(semantic_grid)
    seen_semantic_grid.update_with_slice((6, 3), 10)
    print(seen_semantic_grid)
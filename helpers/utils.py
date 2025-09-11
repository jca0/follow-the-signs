import heapq
from queue import Queue
from collections import Counter

def bresenham_line(x0, y0, x1, y1):
    """
    Generate cells along a line from (x0, y0) to (x1, y1) using Bresenham's algorithm.
    Includes both start and end points.
    """
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1
    if dx > dy:
        err = dx // 2
        while x != x1:
            points.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
        points.append((x1, y1))
    else:
        err = dy // 2
        while y != y1:
            points.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
        points.append((x1, y1))
    return points

def get_kxk_slice(grid, seen_grid, agent_pos, k):
    """
    Get kxk slice of the grid centered at agent_pos.
    Agent can't see past obstacles ('W', 'D').
    If blocked, use the value from seen_grid instead.
    """
    rows, cols = len(grid), len(grid[0])
    cx, cy = agent_pos
    if not (0 <= cx < rows and 0 <= cy < cols):
        raise ValueError("Agent position is out of bounds.")
    
    r = k // 2
    slice = []
    for dx in range(-r, r + 1):
        row = []
        for dy in range(-r, r + 1):
            x, y = cx + dx, cy + dy
            if 0 <= x < rows and 0 <= y < cols:
                line = bresenham_line(cx, cy, x, y)
                visible = True
                # Skip the final point (x,y) itself when checking for obstacles
                for (ix, iy) in line[:-1]:
                    if grid[ix][iy] != '.':
                        visible = False
                        break
                if visible:
                    row.append(grid[x][y])
                else:
                    row.append(seen_grid[x][y])
        slice.append(row)
    return slice

def update_seen_grid_with_slice(grid, seen_grid, agent_pos, k):
    """
    Update seen_grid with what the agent can actually see
    based on line of sight (can't see through 'W' and 'D').
    """
    rows, cols = len(grid), len(grid[0])
    cx, cy = agent_pos
    if not (0 <= cx < rows and 0 <= cy < cols):
        raise ValueError("Agent position is out of bounds.")
    
    r = k // 2
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            x, y = cx + dx, cy + dy
            if 0 <= x < rows and 0 <= y < cols:
                line = bresenham_line(cx, cy, x, y)
                visible = True
                for (ix, iy) in line[:-1]:  # Check all cells before (x, y)
                    if grid[ix][iy] != '.':
                        visible = False
                        break
                if visible:
                    seen_grid[x][y] = grid[x][y]  # Update only if visible
                # Otherwise: leave seen_grid[x][y] unchanged (still '?')

def find_label_in_grid(grid, goal):
    """
    Returns position of goal in grid if it exists.
    """
    rows, cols = len(grid), len(grid[0])
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == goal:
                return (r, c)
    return None

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

def heuristic(a, b):
    """
    Calculate Manhattan distance
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(grid, start, goal):
    """
    Returns path as list of (row, col) from start to goal
    """
    rows, cols = len(grid), len(grid[0])
    open_set = [] # priority queue
    heapq.heappush(open_set, (0, start))

    came_from = {} # parent node
    g_score = {start: 0} # cost from start to current node
    f_score = {start: heuristic(start, goal)} # cost from start to goal through current node

    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    while open_set:
        _, current = heapq.heappop(open_set) # get the node with the lowest f_score

        if current == goal:
            # reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]
        
        for dr, dc in directions:
            neighbor = (current[0] + dr, current[1] + dc)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                cell_value = grid[neighbor[0]][neighbor[1]]
                if neighbor != goal and cell_value != '.':
                    continue

                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
    return None

def is_valid_grid(actual_grid, predicted_grid):
    """
    Check if the predicted grid is valid.
    """
    rows, cols = len(actual_grid), len(actual_grid[0])
    for r in range(rows):
        for c in range(cols):
            if predicted_grid[r][c].isnumeric() and actual_grid[r][c] == 'D':
                continue
            elif predicted_grid[r][c] == actual_grid[r][c]:
                continue
            else:
                return False
    return True

def most_frequent(grids):
    hashable_grids = [tuple(map(tuple, grid)) for grid in grids]
    freq = Counter(hashable_grids)
    most_common = freq.most_common(1)[0][0]
    most_common_grid = [list(row) for row in most_common]
    return most_common_grid


def get_action():
    pass

def apply_action():
    pass
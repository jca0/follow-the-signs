from pprint import pprint
import numpy as np

class Hallway:
    occupancy_grid = np.array([
        [1, 1, 1, 1],
        [1, 0, 1, 1],
        [1, 0, 1, 1],
        [1, 0, 1, 1],
        [1, 0, 1, 1],
        [1, 0, 1, 1],
        [1, 0, 1, 1],
        [1, 0, 1, 1],
        [1, 0, 1, 1],
        [1, 0, 1, 1],
        [1, 1, 1, 1],
    ])

    semantic_grid = [
        [{"feature_type": "wall"} for _ in range(4)] for _ in range(11)
    ]
    # Update free spaces and doors with room numbers
    for y in range(1, 10):
        semantic_grid[y][1] = {"feature_type": "free"}
        semantic_grid[y][2] = {"feature_type": "door", "room_number": str(y)}

class Schwarz:
    occupancy_grid = np.array([
        [1, 0, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 0, 1, 0, 1, 0, 1],
        [1, 1, 1, 1, 1, 1, 1],
    ])

    semantic_grid = [
        [{} for _ in range(7)] for _ in range(11)
    ]

    for y in range(len(occupancy_grid)):
        for x in range(len(occupancy_grid[0])):
            if occupancy_grid[y][x] == 0:
                semantic_grid[y][x] = {"feature_type": "free"}
            else:
                semantic_grid[y][x] = {"feature_type": "wall"}

    # Update doors with room numbers
    semantic_grid[0][2] = {"feature_type": "door", "room_number": "9"}
    semantic_grid[2][2] = {"feature_type": "door", "room_number": "8"}
    semantic_grid[4][2] = {"feature_type": "door", "room_number": "7"}
    semantic_grid[8][1] = {"feature_type": "door", "room_number": "1"}
    semantic_grid[8][3] = {"feature_type": "door", "room_number": "2"}
    semantic_grid[8][5] = {"feature_type": "door", "room_number": "3"}

    semantic_grid[7][0] = {"sign": "Rooms 1-3 to the right. Rooms 7-9 ahead."}
    semantic_grid[5][2] = {"sign": "Rooms 7-9 ahead."}
    semantic_grid[8][6] = {"sign": "Rooms 1-3 to the left."}

class TwoSide:
    occupancy_grid = np.array([
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
    ])

    semantic_grid = [
        [{"feature_type": "wall"} for _ in range(7)] for _ in range(5)
    ]
    # Update free spaces
    for x in range(1, 6):
        semantic_grid[2][x] = {"feature_type": "free"}
    # Update doors with room numbers
    semantic_grid[1][1] = {"feature_type": "door", "room_number": "1"}
    semantic_grid[1][2] = {"feature_type": "door", "room_number": "3"}
    semantic_grid[1][3] = {"feature_type": "door", "room_number": "5"}
    semantic_grid[1][4] = {"feature_type": "door", "room_number": "7"}
    semantic_grid[1][5] = {"feature_type": "door", "room_number": "9"}
    semantic_grid[3][1] = {"feature_type": "door", "room_number": "2"}
    semantic_grid[3][2] = {"feature_type": "door", "room_number": "4"}
    semantic_grid[3][3] = {"feature_type": "door", "room_number": "6"}
    semantic_grid[3][4] = {"feature_type": "door", "room_number": "8"}
    semantic_grid[3][5] = {"feature_type": "door", "room_number": "10"}

class Maseeh:
    occupancy_grid = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1],
        [1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1],
        [1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1],
        [1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1],
        [1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1],
        [1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1],
        [1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1],
        [1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1],
        [1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1],
        [1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ])

    semantic_grid = [
        [{} for _ in range(11)] for _ in range(17)
    ]

    for y in range(len(occupancy_grid)):
        for x in range(len(occupancy_grid[0])):
            if occupancy_grid[y][x] == 0:
                semantic_grid[y][x] = {"feature_type": "free"}
            else:
                semantic_grid[y][x] = {"feature_type": "wall"}

    # Update doors with room numbers
    room_numbers = [
        (2, 1, "12"), (2, 3, "13"), (2, 7, "17"), (2, 9, "18"),
        (4, 1, "11"), (4, 4, "14"), (4, 5, "15"), (4, 6, "16"), (4, 9, "19"),
        (6, 1, "10"), (6, 9, "20"),
        (8, 1, "8"), (8, 3, "9"), (8, 7, "22"), (8, 9, "21"),
        (10, 1, "6"), (10, 3, "7"), (10, 7, "24"), (10, 9, "23"),
        (12, 1, "4"), (12, 3, "5"), (12, 7, "26"), (12, 9, "25"),
        (14, 1, "2"), (14, 3, "3"), (14, 7, "28"), (14, 9, "27"),
        (15, 2, "1"), (15, 8, "29")
    ]
    for y, x, room in room_numbers:
        semantic_grid[y][x] = {"feature_type": "door", "room_number": room}
    # Update signs
    semantic_grid[7][3] = {
        "feature_type": "wall",
        "sign": "Rooms 1-13 to the left. Rooms 14-29 to the right."
    }
    semantic_grid[9][2] = {
        "feature_type": "wall",
        "sign": "Rooms 1-7 down. Rooms 8-13 up."
    }
    semantic_grid[7][8] = {
        "feature_type": "wall",
        "sign": "Rooms 21-29 down. Rooms 17-20 up."
    }

    semantic_grid[4][5] = {
        "feature_type": "door",
        "room_number": "15",
        "sign": "Rooms 16-29 to the right. Rooms 1-14 to the left."
    } 

class Stud:
    occupancy_grid = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1],
        [1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1],
        [1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ])

    semantic_grid = [
        [{} for _ in range(13)] for _ in range(7)
    ]

    for y in range(len(occupancy_grid)):
        for x in range(len(occupancy_grid[0])):
            if occupancy_grid[y][x] == 0:
                semantic_grid[y][x] = {"feature_type": "free"}
            else:
                semantic_grid[y][x] = {"feature_type": "wall"}
    
    room_numbers = [
        (0, 2, "1"), (0, 4, "2"), (0, 6, "3"), (0, 8, "4"), (0, 10, "5"),
        (2, 0, '14'), (2, 3, '15'), (2, 9, '19'), (2, 12, '6'),
        (3, 2, '18'), (3, 4, '16'), (3, 8, '22'), (3, 10, '20'),
        (4, 0, '13'), (4, 3, '17'), (4, 9, '21'), (4, 12, '7'),
        (6, 1, '12'), (6, 3, '11'), (6, 5, '10'), (6, 7, '9'), (6, 9, '8'),
    ]
    for y, x, room in room_numbers:
        semantic_grid[y][x] = {"feature_type": "door", "room_number": room}

    semantic_grid[0][1] = {"sign": "Rooms 1-7 to the right."}
    semantic_grid[0][11] = {"sign": "Rooms 6-12 southward."}
    semantic_grid[6][11] = {"sign": "Rooms 8-14 to the left."}
    semantic_grid[6][1] = {"sign": "Rooms 13-14 ahead."}
    semantic_grid[6][6] = {"sign": "Rooms 8-9 to the right."}
    semantic_grid[3][0] = {"sign": "Rooms 15-22 to the right."}
    semantic_grid[3][12] = {"sign": "Rooms 15-22 to the left."}



if __name__ == "__main__":
    for row in Stud.semantic_grid:
        print(row)
        print()
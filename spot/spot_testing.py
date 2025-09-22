from helpers.spot_utils import *
from helpers.spot_env import *
from helpers.robot_structs import *
from helpers.env_utils import *
from spot_ours import *
from spot_frontier import *

import math

from bosdyn.client.robot_command import blocking_stand
from bosdyn.client.docking import blocking_dock_robot, get_dock_id

# (r, c)
# when r increases, robot moves left
# when c increases, robot moves up

if __name__ == "__main__":
    robot_client = setup_robot()
    command_client = robot_client.command_client
    
    # env = RealSchwarz(resolution_m=1) # for frontier
    env = RealSchwarz(resolution_m=0.9) # for ours
    occupancy_grid = env.occupancy_grid
    semantic_grid = env.semantic_grid
    seen_occupancy = SeenOccupancyGrid(occupancy_grid)
    seen_semantic = SeenSemanticGrid(semantic_grid)
    confidence_grid = ConfidenceGrid(len(occupancy_grid), len(occupancy_grid[0]))

    grid_loc = GridLocalizer(robot_client, env)
    grid_loc.calibrate(8, 20, body_yaw_rad=math.pi)
    # print(render_grid(env, *grid_loc.get_current_cell()))
    agent_pos = grid_loc.get_current_cell()
    # print(f"Agent position: {agent_pos}")

    # trajectory = [(5, 20), (2, 20), (1, 22), (1, 19), (1, 16), (1, 13), (1, 8)] # rotates 90, move forward, move right
    # trajectory = [(6, 20), (4, 20), (1, 20), (1, 22), (1, 20), (1, 24), (2, 24), (2, 26), (2, 28), (2, 30), (2, 28), (2, 30), (2, 28), (2, 30)]
    trajectory = [(6, 20), (4, 20), (3, 22), (3, 24), (2, 25), (1, 27), (1, 29), (1, 31), (3, 31), (1, 31), (1, 27), (1, 29), (1, 31), (1, 27), (1, 29), (1, 31), (1, 27), (1, 29)]
    # TODO: flip env about vertical axis
    # agent_pos = (8, 20)

    # run_frontier_agent(env, command_client, grid_loc, seen_occupancy, seen_semantic, agent_pos, '621', 7, seed=2)

    # run_agent(env, command_client, grid_loc, seen_occupancy, seen_semantic, confidence_grid, agent_pos, '621', 10, path_steps=3, seed=123)
    for r, c in trajectory:
        cr, cc = grid_loc.get_current_cell()
        cx, cy = env.cell_center_xy(cr, cc)
        tx, ty = env.cell_center_xy(r, c)
        yaw = math.atan2(ty - cy, tx - cx)
        move_to_cell(robot_client.command_client, grid_loc, r, c, yaw)
        time.sleep(0.2)
        # print(grid_loc.get_current_cell(), (r, c))
        # print(render_grid(env, *grid_loc.get_current_cell()))
    print("done")

    # cr, cc = grid_loc.get_current_cell()
    # cx, cy = env.cell_center_xy(cr, cc)
    # tx, ty = env.cell_center_xy(0, 7)
    # yaw = math.atan2(ty - cy, tx - cx)
    # move_to_cell(robot_client.command_client, grid_loc, 0, 7, yaw)

    # move_to_cell(robot_client.command_client, grid_loc, trajectory[-1][0], trajectory[-1][1], 1.7)

    # move_relative(1, 0, 0, frame_name=ODOM_FRAME_NAME, command_client=robot_client.command_client, state_client=robot_client.state_client)

    # robot_client.robot.power_off(cut_immediately=False, timeout_sec=20)
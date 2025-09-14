from helpers.spot_utils import *
from helpers.spot_env import *
from helpers.robot_structs import *
from helpers.env_utils import *
from spot_ours import *

import math

from bosdyn.client.robot_command import blocking_stand
from bosdyn.client.docking import blocking_dock_robot, get_dock_id

# (r, c)
# when r increases, robot moves left
# when c increases, robot moves up

if __name__ == "__main__":
    robot_client = setup_robot()
    command_client = robot_client.command_client
    
    env = RealSchwarz(resolution_m=0.7)
    occupancy_grid = env.occupancy_grid
    semantic_grid = env.semantic_grid
    seen_occupancy = SeenOccupancyGrid(occupancy_grid)
    seen_semantic = SeenSemanticGrid(semantic_grid)
    confidence_grid = ConfidenceGrid(len(occupancy_grid), len(occupancy_grid[0]))

    grid_loc = GridLocalizer(robot_client, env)
    grid_loc.calibrate(3, 20)
    # print(render_grid(env, *grid_loc.get_current_cell()))
    agent_pos = grid_loc.get_current_cell()
    print(f"Agent position: {agent_pos}")

    # # run_agent(env, command_client, grid_loc, seen_occupancy, seen_semantic, confidence_grid, agent_pos, '631', 7, path_steps=3)
    for r, c in trajectory:
        # cr, cc = grid_loc.get_current_cell()
        # cx, cy = env.cell_center_xy(cr, cc)
        # tx, ty = env.cell_center_xy(r, c)
        # yaw = math.atan2(ty - cy, tx - cx)
        move_to_cell(robot_client.command_client, grid_loc, r, c, 0)
        print(grid_loc.get_current_cell(), (r, c))
        print(render_grid(env, *grid_loc.get_current_cell()))

    # move_relative(1, 0, 0, frame_name=ODOM_FRAME_NAME, command_client=robot_client.command_client, state_client=robot_client.state_client)

    # robot_client.robot.power_off(cut_immediately=False, timeout_sec=20)
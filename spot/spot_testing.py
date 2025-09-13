from helpers.spot_utils import *
from helpers.spot_env import *
from helpers.robot_structs import *
from helpers.env_utils import *

from bosdyn.client.robot_command import blocking_stand
from bosdyn.client.docking import blocking_dock_robot, get_dock_id

# (r, c)
# when r increases, robot moves left
# when c increases, robot moves up

if __name__ == "__main__":
    robot_client = setup_robot()
    
    env = TestEnv(resolution_m=0.5)
    grid_loc = GridLocalizer(robot_client, env)
    grid_loc.calibrate(0, 0)
    print(render_grid(env, *grid_loc.get_current_cell()))
    
    trajectory = [(1, 0), (2, 0), (2, 1), (2, 2), (0, 0)]
    for r, c in trajectory:
        move_to_cell(robot_client.command_client, grid_loc, r, c, 0.0)
        print(render_grid(env, *grid_loc.get_current_cell()))

    robot_client.robot.power_off(cut_immediately=False, timeout_sec=20)
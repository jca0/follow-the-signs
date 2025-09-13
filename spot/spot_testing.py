from helpers.spot_utils import *
from helpers.spot_env import *
from helpers.robot_structs import *
from helpers.env_utils import *

from bosdyn.client.robot_command import blocking_stand
from bosdyn.client.docking import blocking_dock_robot, get_dock_id

if __name__ == "__main__":
    robot_client = setup_robot()
    
    env = TestEnv()
    grid_loc = GridLocalizer(robot_client, env)
    grid_loc.calibrate(2, 2)
    print(render_grid(env, *grid_loc.get_current_cell()))

    move_to_cell(robot_client.command_client, grid_loc, 0, 0)

    print(render_grid(env, *grid_loc.get_current_cell()))

    robot_client.robot.power_off(cut_immediately=False, timeout_sec=20)
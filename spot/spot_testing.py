from helpers.spot_utils import *
from helpers.spot_env import *
from helpers.robot_structs import *

from bosdyn.client.robot_command import blocking_stand
from bosdyn.client.docking import blocking_dock_robot, get_dock_id

if __name__ == "__main__":
    robot_client = setup_robot()
    env = TestEnv()
    grid_loc = GridLocalizer(robot_client, env)
    grid_loc.calibrate(1, 0)

    pose_now = grid_loc.odom_pose_from_cell(1, 2)

    print(pose_now)
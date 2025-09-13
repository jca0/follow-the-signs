from dataclasses import dataclass

import bosdyn.client
import bosdyn.client.util
import bosdyn.geometry
from bosdyn.client import math_helpers
from bosdyn.client.robot_command import RobotCommandClient
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.robot import Robot
from bosdyn.client.lease import LeaseKeepAlive
from bosdyn.client.frame_helpers import get_se2_a_tform_b, ODOM_FRAME_NAME, BODY_FRAME_NAME
from bosdyn.client import math_helpers

from helpers.spot_env import SpotEnv

@dataclass
class RobotClient:
    robot: Robot
    command_client: RobotCommandClient
    state_client: RobotStateClient
    lease_keepalive: LeaseKeepAlive
    home_pose: math_helpers.SE2Pose


class GridLocalizer:
    def __init__(self, robot: Robot, env: SpotEnv):
        self._robot = robot
        self._env = env
        self._odom_T_grid = math_helpers.SE2Pose(0, 0, 0) # set after calibrate
        self._last_pose = math_helpers.SE2Pose(0, 0, 0)

    def calibrate(self, r, c, body_yaw_rad=None):
        # read current odom_T_body
        transforms = self._robot.state_client.get_robot_state().kinematic_state.transforms_snapshot
        odom_T_body = get_se2_a_tform_b(transforms, ODOM_FRAME_NAME, BODY_FRAME_NAME)

        # pose of cell in unanchored grid grame
        grid_T_body = self._env.se2_from_cell(r, c, body_yaw_rad or odom_T_body.angle)

        self._odom_T_grid = odom_T_body * grid_T_body.inverse()
        self._last_pose = odom_T_body

    def odom_pose_from_cell(self, r, c, yaw_rad=0.0):
        grid_T_body = self._env.se2_from_cell(r, c, yaw_rad)
        odom_T_body = self._odom_T_grid * grid_T_body
        self._last_pose = odom_T_body
        return odom_T_body

    def cell_from_odom_pose(self, odom_pose):
        grid_T_body = self._odom_T_grid.inverse() * odom_pose
        return self._env.cell_from_se2(grid_T_body)

    def get_last_pose(self):
        return self._last_pose

    def get_last_cell(self):
        return self.cell_from_odom_pose(self._last_pose)

    def get_current_cell(self):
        transforms = self._robot.state_client.get_robot_state().kinematic_state.transforms_snapshot
        odom_T_body = get_se2_a_tform_b(transforms, ODOM_FRAME_NAME, BODY_FRAME_NAME)
        return self.cell_from_odom_pose(odom_T_body)

    

# TODO make function to go to a cell in grid
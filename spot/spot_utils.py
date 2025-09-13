import time
from dataclasses import dataclass

import bosdyn.client
import bosdyn.client.util
import bosdyn.geometry
from bosdyn.api import trajectory_pb2
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import GRAV_ALIGNED_BODY_FRAME_NAME, ODOM_FRAME_NAME, get_a_tform_b
from bosdyn.client.image import ImageClient
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient, blocking_stand
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.robot import Robot
from bosdyn.client.lease import LeaseKeepAlive
from bosdyn.util import seconds_to_duration

@dataclass
class RobotClient:
    robot: Robot
    robot_command_client: RobotCommandClient
    robot_state_client: RobotStateClient
    lease_keepalive: LeaseKeepAlive
    home_pose: math_helpers.SE2Pose

def move_to_cell():
    pass

def start_spot(hostname="192.168.80.3"):
    print("Initializing Spot")
    bosdyn.client.util.setup_logging(False)
    sdk = bosdyn.client.create_standard_sdk('RobotClient')
    robot = sdk.create_robot(hostname)
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()
    assert not robot.is_estopped(), 'Robot is estopped. Please use an external E-Stop client, ' \
                                    'such as the estop SDK example, to configure E-Stop.'
    
    
    print("Initializing clients")
    lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
    lease_client.take()

    robot_command_client = robot.ensure_client(RobotCommandClient.default_service_name)
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    lease_keepalive = bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True)
    robot.power_on(timeout_sec=20)

    home_pose = math_helpers.SE2Pose(0, 0, 0)

    return RobotClient(
        robot=robot,
        robot_command_client=robot_command_client,
        robot_state_client=robot_state_client,
        lease_keepalive=lease_keepalive,
        home_pose=home_pose,
    )


from dataclasses import dataclass, field

import bosdyn.client
import bosdyn.client.estop
import bosdyn.client.lease
import bosdyn.client.util
import yaml
from bosdyn.api import geometry_pb2, robot_state_pb2
from bosdyn.api.basic_command_pb2 import RobotCommandFeedbackStatus
from bosdyn.api.geometry_pb2 import SE2Velocity, SE2VelocityLimit, Vec2
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.client import math_helpers
from bosdyn.client.exceptions import ProxyConnectionError, TimedOutError
from bosdyn.client.frame_helpers import (
    BODY_FRAME_NAME,
    ODOM_FRAME_NAME,
    get_se2_a_tform_b,
)
from bosdyn.client.image import ImageClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.robot import Robot
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.sdk import Robot

@dataclass
class RobotClient:
    """All robot clients packaged into one object."""

    robot: Robot
    sdk: Sdk
    state_client: RobotStateClient = None
    command_client: RobotCommandClient = None
    image_client: ImageClient = None
    manipulation_client: ManipulationApiClient = None
    lease_keepalive: LeaseKeepAlive = None
    localizer: SpotLocalizer = None
    home_pose: math_helpers.SE2Pose = None

def setup_robot(graphnav: str, spot_ip: str) -> RobotClient:
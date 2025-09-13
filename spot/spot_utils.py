import time
from dataclasses import dataclass

import bosdyn.client
import bosdyn.client.util
import bosdyn.geometry
from bosdyn.api.basic_command_pb2 import RobotCommandFeedbackStatus
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import BODY_FRAME_NAME, ODOM_FRAME_NAME, get_se2_a_tform_b
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.robot import Robot
from bosdyn.client.lease import LeaseKeepAlive
import bosdyn.client.estop

DEFAULT_SPOT_IP= "192.168.80.3"

@dataclass
class RobotClient:
    robot: Robot
    command_client: RobotCommandClient
    state_client: RobotStateClient
    lease_keepalive: LeaseKeepAlive
    home_pose: math_helpers.SE2Pose

def move_to_cell():
    pass

def relative_move(dx, dy, dyaw, frame_name=ODOM_FRAME_NAME, robot_command_client=None, robot_state_client=None, stairs=False):
    transforms = robot_state_client.get_robot_state().kinematic_state.transforms_snapshot

    body_tform_goal = math_helpers.SE2Pose(x=dx, y=dy, angle=dyaw)
    out_tform_body = get_se2_a_tform_b(transforms, frame_name, BODY_FRAME_NAME)
    out_tform_goal = out_tform_body * body_tform_goal

    robot_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
        goal_x=out_tform_goal.x, goal_y=out_tform_goal.y, goal_heading=out_tform_goal.angle,
        frame_name=frame_name, params=RobotCommandBuilder.mobility_params(stair_hint=stairs))
    end_time = 10.0
    cmd_id = robot_command_client.robot_command(lease=None, command=robot_cmd,
                                                end_time_secs=time.time() + end_time)
    while True:
        feedback = robot_command_client.robot_command_feedback(cmd_id)
        mobility_feedback = feedback.feedback.synchronized_feedback.mobility_command_feedback
        if mobility_feedback.status != RobotCommandFeedbackStatus.STATUS_PROCESSING:
            print('Failed to reach the goal')
            return False
        traj_feedback = mobility_feedback.se2_trajectory_feedback
        if (traj_feedback.status == traj_feedback.STATUS_AT_GOAL and
                traj_feedback.body_movement_status == traj_feedback.BODY_STATUS_SETTLED):
            print('Arrived at the goal.')
            return True
        time.sleep(1)

    return True

def setup_robot(hostname=DEFAULT_SPOT_IP):
    bosdyn.client.util.setup_logging(False)

    sdk = bosdyn.client.create_standard_sdk('RobotClient')
    robot = sdk.create_robot(hostname)

    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()

    assert not robot.is_estopped(), 'Robot is estopped. Please use an external E-Stop client, ' \
                                    'such as the estop SDK example, to configure E-Stop.'
    
    lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
    lease_client.take()
    lease_keepalive = bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True)

    robot_command_client = robot.ensure_client(RobotCommandClient.default_service_name)
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)

    robot.power_on(timeout_sec=20)

    home_pose = math_helpers.SE2Pose(0, 0, 0)

    return RobotClient(
        robot=robot,
        command_client=robot_command_client,
        state_client=robot_state_client,
        lease_keepalive=lease_keepalive,
        home_pose=home_pose,
    )


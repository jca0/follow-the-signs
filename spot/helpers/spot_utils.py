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

from helpers.robot_structs import *


DEFAULT_SPOT_IP= "192.168.80.3"

def move_to_cell(command_client: RobotCommandClient, grid_loc: GridLocalizer, r, c, yaw_rad=0.0, timeout=20.0):
    odom_goal = grid_loc.odom_pose_from_cell(r, c, yaw_rad)
    cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
        goal_x=odom_goal.x,
        goal_y=odom_goal.y,
        goal_heading=odom_goal.angle,
        frame_name=ODOM_FRAME_NAME,
        params=RobotCommandBuilder.mobility_params(stair_hint=False)
        )
    cmd_id = command_client.robot_command(lease=None, command=cmd, end_time_secs=time.time() + timeout)

    while True:
        fb = command_client.robot_command_feedback(cmd_id)
        mfb = fb.feedback.synchronized_feedback.mobility_command_feedback
        if mfb.status != RobotCommandFeedbackStatus.STATUS_PROCESSING:
            print('Failed to reach the goal')
            return False
        tfb = mfb.se2_trajectory_feedback
        if (tfb.status == tfb.STATUS_AT_GOAL and
                tfb.body_movement_status == tfb.BODY_STATUS_SETTLED):
            print('Arrived at the goal.')
            return True
        time.sleep(1)

    return True
    

def move_relative(dx, dy, dyaw, frame_name=ODOM_FRAME_NAME, command_client=None, state_client=None, stairs=False):
    transforms = state_client.get_robot_state().kinematic_state.transforms_snapshot

    body_tform_goal = math_helpers.SE2Pose(x=dx, y=dy, angle=dyaw)
    out_tform_body = get_se2_a_tform_b(transforms, frame_name, BODY_FRAME_NAME)
    out_tform_goal = out_tform_body * body_tform_goal

    cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
        goal_x=out_tform_goal.x, goal_y=out_tform_goal.y, goal_heading=out_tform_goal.angle,
        frame_name=frame_name, params=RobotCommandBuilder.mobility_params(stair_hint=stairs))
    end_time = 10.0
    cmd_id = command_client.robot_command(lease=None, command=cmd,
                                                end_time_secs=time.time() + end_time)
    while True:
        fb = command_client.robot_command_feedback(cmd_id)
        mfb = fb.feedback.synchronized_feedback.mobility_command_feedback
        if mfb.status != RobotCommandFeedbackStatus.STATUS_PROCESSING:
            print('Failed to reach the goal')
            return False
        tfb = mfb.se2_trajectory_feedback
        if (tfb.status == tfb.STATUS_AT_GOAL and
                tfb.body_movement_status == tfb.BODY_STATUS_SETTLED):
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


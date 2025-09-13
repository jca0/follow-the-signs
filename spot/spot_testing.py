from spot_utils import *
from bosdyn.client.robot_command import blocking_stand
from bosdyn.client.docking import blocking_dock_robot, get_dock_id

if __name__ == "__main__":
    robot_client = setup_robot()
    print("command robot to stand")
    blocking_stand(robot_client.command_client, timeout_sec=10)
    time.sleep(1.5)
    relative_move(1.5, 0.0, 0.0, robot_command_client=robot_client.command_client, robot_state_client=robot_client.state_client)
    blocking_dock_robot(robot_client.robot, 549)
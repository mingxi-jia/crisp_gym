"""Example on how to teleoperate a robot using another one."""

import argparse
import logging
import time

import numpy as np

from crisp_gym.manipulator_env import make_env
from crisp_gym.teleop.teleop_robot import make_leader
from crisp_gym.util.setup_logger import setup_logging
from crisp_gym.teleop.gello import Gello
from crisp_py.gripper import Gripper, GripperConfig

# Parse args:
parser = argparse.ArgumentParser(description="Teleoperation of a leader robot.")
parser.add_argument(
    "--use-force-feedback",
    action="store_true",
    help="Use force feedback from the leader robot (default: False)",
)
parser.add_argument(
    "--log-level",
    type=str,
    default="INFO",
    choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    help="Set the logging level (default: INFO)",
)
parser.add_argument(
    "--control-frequency",
    type=float,
    default=100.0,
    help="Control frequency in Hz (default: 100.0)",
)


args = parser.parse_args()

# Set up logging
setup_logging(level=args.log_level)
logger = logging.getLogger(__name__)

# %% Leader setup
logger.info("Setting up leader robot...")
gello = Gello()

# %% Environment setup
logger.info("Setting up environment...")
env = make_env(
        env_type="no_cam_franka", 
        config_path="crisp_gym/config/envs/no_cam_franka.yaml",
        control_type="joint",
        namespace="",
        joint_control_param_config="config/joint_control_custom.yaml",
        gripper_config=GripperConfig.from_yaml("/home/mrping/mingxi_ws/crisp/crisp_gym/config/gripper_config_custom.yaml")
    )
env.robot.home()


obs, _ = env.reset()


print("Switching to joint controller...")
env.switch_controller("joint")
time.sleep(1.0)  # Give controller time to switch

franka_home = obs['joint']

gello.calibrate(franka_home)

previous_pose = gello.get_joint_state()

while True:
    # NOTE: the leader pose and follower pose will drift apart over time but this is
    #       fine assuming that we are just recording the leader's actions and not absolute positions.
    current_pose = gello.get_joint_state()
    action_pose = current_pose[:7] - previous_pose[:7]
    # print(f"action_pose (raw): {action_pose}")
    action_pose = np.clip(action_pose*100, -0.05, 0.05)  # Increased from 0.01
    gripper_action = current_pose[7]
    # print(gripper_action)
    # print(f"action_pose (clipped): {action_pose}")

    # Add gripper action
    action = np.concatenate([action_pose[:7], [gripper_action]])  
    
    obs, *_ = env.step(action, block=False)
    # print(f"robot joints: {obs['joint'][:3]}...")  # Show first 3 joints
    time.sleep(1.0 / args.control_frequency)
    previous_pose = gello.get_joint_state()

"""Script showcasing how to record data in Lerobot Format."""

import argparse
import logging

import numpy as np
import rclpy  # noqa: F401

import crisp_gym  # noqa: F401
from crisp_gym.config.path import CRISP_CONFIG_PATH
from crisp_gym.manipulator_env import make_env
from crisp_gym.manipulator_env_config import list_env_configs
from crisp_gym.record.record_functions import make_gello_fn
from crisp_gym.record.recording_manager import make_recording_manager
from crisp_gym.teleop.gello import Gello
from crisp_py.gripper import GripperConfig
from crisp_gym.util import prompt
from crisp_gym.util.lerobot_features import get_features
from crisp_gym.util.setup_logger import setup_logging
import os
import time
parser = argparse.ArgumentParser(description="Record data in Lerobot Format")
parser.add_argument(
    "--repo-id",
    type=str,
    default="test",
    help="Repository ID for the dataset",
)
parser.add_argument(
    "--tasks",
    type=str,
    nargs="+",
    default=["pick the lego block."],
    help="List of task descriptions to record data for, e.g. 'clean red' 'clean green'",
)
parser.add_argument(
    "--robot-type",
    type=str,
    default="franka",
    help="Type of robot being used.",
)
parser.add_argument(
    "--fps",
    type=int,
    default=15,
    help="Frames per second for recording",
)
parser.add_argument(
    "--num-episodes",
    type=int,
    default=10,
    help="Number of episodes to record",
)
parser.add_argument(
    "--resume",
    action="store_true",
    default=False,
    help="Resume recording of an already existing dataset",
)
parser.add_argument(
    "--push-to-hub",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Whether to push the dataset to the Hugging Face Hub.",
)
parser.add_argument(
    "--recording-manager-type",
    type=str,
    default="keyboard",
    help="Type of recording manager to use. Currently only 'keyboard' and 'ros' are supported.",
)
parser.add_argument(
    "--follower-namespace",
    type=str,
    default="",
    help="Namespace for the follower robot. This is used to identify the robot in the ROS ecosystem.",
)
parser.add_argument(
    "--log-level",
    type=str,
    default="INFO",
    choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    help="Set the logger level.",
)

args = parser.parse_args()

# Set up logger
logger = logging.getLogger(__name__)
setup_logging(level=args.log_level)


logger.info("Arguments:")
for arg, value in vars(args).items():
    logger.info(f"  {arg}: {value}")


# Gello doesn't need configuration selection, so skip this section
logger.info("Using Gello as leader device")


try:
    ctrl_type = "joint"

    base_dir = os.path.dirname(os.path.abspath(__file__))
    joint_control_param_config = os.path.join(base_dir, "../config/joint_control_custom.yaml")
    gripper_config_path = os.path.join(base_dir, "../config/gripper_config_custom.yaml")

    env = make_env(
        env_type="no_cam_franka",
        config_path="crisp_gym/config/envs/no_cam_franka.yaml",
        control_type="joint",
        namespace=args.follower_namespace,
        joint_control_param_config=joint_control_param_config,
        gripper_config=GripperConfig.from_yaml(gripper_config_path)
    )

    # Setup Gello device
    gello = Gello()
    logger.info("Gello device initialized")

    features = get_features(env_config=env.config, ctrl_type=ctrl_type)
    logger.debug(f"Using the features: {features}")

    recording_manager = make_recording_manager(
        recording_manager_type=args.recording_manager_type,
        features=features,
        repo_id=args.repo_id,
        robot_type=args.robot_type,
        num_episodes=args.num_episodes,
        fps=args.fps,
        resume=args.resume,
        push_to_hub=args.push_to_hub,
    )
    recording_manager.wait_until_ready()

    logger.info("Homing follower robot and calibrating Gello before starting recording.")

    # Prepare environment
    env.home()
    obs, _ = env.reset()
    
    # Switch to joint controller
    env.switch_controller("joint")
    
    # Calibrate Gello with current robot position
    franka_home = obs['joint']
    gello.calibrate(franka_home)

    tasks = list(args.tasks)

    def on_start():
        """Hook function to be called when starting a new episode."""
        env.reset()
        # Ensure joint controller is active
        env.switch_controller("joint")

    def on_end():
        """Hook function to be called when stopping the recording."""
        gello.home()
        env.robot.reset_targets()
        env.robot.home(blocking=True)

    with recording_manager:
        while not recording_manager.done():
            logger.info(
                f"→ Episode {recording_manager.episode_count + 1} / {recording_manager.num_episodes}"
            )

            task = tasks[np.random.randint(0, len(tasks))] if tasks else "No task specified."
            logger.info(f"▷ Task: {task}")

            recording_manager.record_episode(
                data_fn=make_gello_fn(env, gello),
                task=task,
                on_start=on_start,
                on_end=on_end,
            )

    logger.info("Homing follower.")
    env.home()

    logger.info("Closing the environment.")
    env.close()

    logger.info("Finished recording.")

except TimeoutError as e:
    logger.exception(f"Timeout error occurred during recording: {e}.")
    logger.error(
        "Please check if the robot container is running and the namespace is correct."
        "\nYou can check the topics using `ros2 topic list` command."
    )

except Exception as e:
    logger.exception(f"An error occurred during recording: {e}.")

finally:
    if rclpy.ok():
        rclpy.shutdown()

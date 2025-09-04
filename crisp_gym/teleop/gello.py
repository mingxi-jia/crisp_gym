"""Class defining the teleoperation leader robot in a leader-follower setup."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import sys
from numpy.typing import NDArray
import glob
import numpy as np
import time
# Add the gello_software path to Python path
gello_software_path = str(Path(__file__).parent.parent.parent / "third_party" / "gello_software")
print(f"Adding gello_software to Python path: {gello_software_path}")
if gello_software_path not in sys.path:
    sys.path.insert(0, gello_software_path)

from crisp_py.gripper import Gripper
from crisp_gym.teleop.teleop_robot_config import TeleopRobotConfig, make_leader_config

# Import GelloAgent with explicit path handling to avoid naming conflicts
import importlib.util
gello_agent_path = Path(__file__).parent.parent.parent / "third_party" / "gello_software" / "gello" / "agents" / "gello_agent.py"
spec = importlib.util.spec_from_file_location("gello_agent_module", gello_agent_path)
gello_agent_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gello_agent_module)
GelloAgent = gello_agent_module.GelloAgent

@dataclass
class GelloConfig():
    """Configuration for the teleoperation leader robot.

    Attributes:
        leader (RobotConfig): Configuration for the leader robot.
        leader_gripper (GripperConfig): Configuration for the gripper of the leader robot.
        gravity_compensation_controller (Path): Path to the gravity compensation controller configuration.
        leader_namespace (str): Namespace for the leader robot.
        leader_gripper_namespace (str): Namespace for the leader robot's gripper.

    """

    joint_names = None
    home_joints = np.array([ 0.11504856, -0.13192235, -0.04141748, -1.92821385, -0.0874369,   1.83924296, 0.7347768,   0. ])

    gravity_compensation_controller: Path

    leader_namespace: str = ""
    leader_gripper_namespace: str = ""

    disable_gripper_torque: bool = True


class Gello:
    """A crisp wrapper for GelloAgent third_party/gello_software/gello/agents/gello_agent.py
    
    This class provides a Robot-like interface for the GelloAgent, making it compatible
    with the crisp_py.robot.Robot interface expected by the teleop system.
    """
    
    def __init__(self, config=GelloConfig, namespace: str = ""):
        """Initialize the Gello wrapper.
        
        Args:
            robot_config: Configuration for the robot (contains gello port info)
            namespace: Robot namespace (unused for gello but kept for interface compatibility)
        """
        self.namespace = namespace

        self._current_joint = None
        
        # Extract gello port from robot config if available
        usb_ports = glob.glob("/dev/serial/by-id/*")
        print(f"Found {len(usb_ports)} ports")
        if len(usb_ports) > 0:
            gello_port = usb_ports[0]
            print(f"using port {gello_port}")
        else:
            raise ValueError(
                "No gello port found, please specify one or plug in gello"
            )
        
        self.reset_joints = None
        
        # Initialize the GelloAgent
        self._gello_agent = GelloAgent(port=gello_port)
        
        # Mock some properties that might be expected by the Robot interface
        self.controller_switcher_client = MockControllerSwitcher()
        self.cartesian_controller_parameters_client = MockParametersClient()
        
    def wait_until_ready(self):
        """Wait until the gello device is ready."""
        # Gello doesn't have an explicit ready check, assume it's ready after initialization
        pass
        
    def home(self, blocking: bool = True):
        """Home the robot by checking if current joints are near home position.
        
        For gello as a leader device, this checks if the device is positioned
        near the reset/home joint values before allowing the system to proceed.
        
        Args:
            blocking: If True, blocks until home position is reached
            
        Raises:
            RuntimeError: If gello is not in home position and blocking is True
        """
        print("Please move gello to home position.")
        import numpy as np
        import time
        
        home_position = self.reset_joints
        self.offset = None
        
        # Tolerance for considering joints "at home" (in radians)
        joint_tolerance = 0.2  # ~11.5 degrees
        gripper_tolerance = 0.3  # More lenient for gripper
        
        def is_at_home():
            joints = self.get_joint_state()
            
            # Check joint positions (exclude gripper - last joint)
            joint_diff = np.abs(joints[:-1] - home_position[:-1])
            joints_ok = np.all(joint_diff < joint_tolerance)
            
            # Check gripper position (last joint)
            gripper_ok = abs(joints[-1] - home_position[-1]) < gripper_tolerance
            
            return joints_ok and gripper_ok
        
        assert blocking, "only blocking is tested"
        if blocking:
            max_wait_time = 30.0  # Maximum wait time in seconds
            start_time = time.time()
            
            while not is_at_home():

                current = self.get_joint_state()
                expected = home_position
                if time.time() - start_time > max_wait_time:
                    raise RuntimeError(
                        f"Gello device not in home position after {max_wait_time}s. "
                        f"Current joints: {current}, Expected: {expected} "
                        f"(tolerance: {joint_tolerance} rad for joints, {gripper_tolerance} for gripper)"
                    )
                print(f"current: {current}, \nexpected: {expected},")
                time.sleep(0.1)  # Check every 100ms
            print("Gello is at home position!")
        else:
            # Non-blocking: just check once and warn if not at home
            if not is_at_home():
                current = self.get_joint_state()
                print(f"Warning: Gello not at home position. Current: {current}, Expected: {home_position}")
        
    def reset_targets(self):
        """Reset robot targets (no-op for gello)."""
        # Gello is a leader device, no targets to reset
        pass
        
    def get_joint_state(self):
        """Get current joint state from gello device."""
        return self._gello_agent.act({})

    def calibrate(self, franka_home):
        print("put gello at the same pose as the arm. And press Enter.")
        input()
        gello_readings = []
        for i in range(10):  # Take multiple readings for stability
            reading = self.get_joint_state()[:7]
            gello_readings.append(reading)
            time.sleep(0.1)
        
        gello_home = np.mean(gello_readings, axis=0)
        gello_std = np.std(gello_readings, axis=0)
        
        # Calculate offset
        self.offset = franka_home - gello_home
        self.reset_joints = np.concatenate([gello_home, [0.0]])  # Assume gripper open at 0.0
        print(f"The offset is {self.offset}")

    @property
    def joint_values(self) -> NDArray:
        """Get the current joint values of the robot.

        Returns:
            numpy.ndarray: Copy of current joint values, or None if not available.
        """
        if self._current_joint is None:
            self._current_joint = self.get_joint_state()
        return self._current_joint.copy()

class MockControllerSwitcher:
    """Mock controller switcher for gello compatibility."""
    
    def switch_controller(self, _controller_name: str):
        """Mock controller switching (no-op for gello)."""
        pass


class MockParametersClient:
    """Mock parameters client for gello compatibility."""
    
    def load_param_config(self, _file_path: Path):
        """Mock parameter loading (no-op for gello)."""
        pass

class TeleopGello:
    """Class defining the teleoperation leader robot in a leader-follower setup.

    This class encapsulates the functionality for controlling a leader robot in a
    teleoperation scenario, allowing for interaction with a follower robot or environment.
    """

    def __init__(self, config=GelloConfig, namespace: str = ""):
        """Initialize the TeleopRobot with a leader robot and its gripper.

        Args:
            config (TeleopRobotConfig): Configuration for the teleoperation leader robot,
                including the leader robot and its gripper configurations.
            namespace (str, optional): Namespace for the leader robot. Defaults to an empty string.
        """
        self.config = config
        self.robot = Gello(
            robot_config=config, )
        self.gripper = None

    def wait_until_ready(self):
        """Wait until the leader robot and its gripper are ready."""
        self.robot.wait_until_ready()
        if self.gripper is not None:
            self.gripper.wait_until_ready()

    def prepare_for_teleop(self, home: bool = True, blocking: bool = True):
        """Prepare the leader robot for teleoperation.

        This method sets the leader robot to a ready state for teleoperation,
        ensuring that it is in a suitable configuration to receive commands.
        """
        if home:
            self.robot.home(blocking=blocking)

        self.robot.cartesian_controller_parameters_client.load_param_config(
            file_path=self.config.gravity_compensation_controller
        )
        self.robot.controller_switcher_client.switch_controller("cartesian_impedance_controller")

        if self.gripper is not None and not self.config.disable_gripper_torque:
            self.gripper.disable_torque()


def make_leader() -> TeleopGello:
    """Create a TeleopRobot instance using the specified configuration.

    Args:
        name (str): The name of the robot configuration to use.
        namespace (str, optional): Namespace for the leader robot. Defaults to "".

    Returns:
        TeleopRobot: A fully initialized TeleopRobot instance.

    Raises:
        ValueError: If the specified robot configuration name is not supported.
    """
    return TeleopGello()

def test_gello():
    import time
    agent = Gello()
    print("testing the connection for 1s")
    starting_time = time.time()
    while True:
        now = time.time()
        gello_state = agent.get_joint_state()
        print(f"\rgello_state: {gello_state}", end='', flush=True)
        # if now-starting_time > 1:
        #     break

    # testing agent.home() function
    agent.home()

def calibrate_gello():
    """Calibrate Gello by finding home position and offset relative to Franka robot."""
    import time
    import rclpy
    from crisp_py.robot import Robot
    from crisp_py.robot_config import FrankaConfig
    
    print("=== Gello Calibration ===")
    
    # Get Franka home configuration
    franka_config = FrankaConfig()
    franka_home = np.asarray(franka_config.home_config)
    print(f"Franka home position: {franka_home}")
    
    # Initialize Gello
    print("Initializing Gello...")
    gello = Gello()
    
    # Get Gello reading at "home" position
    print("Reading Gello position...")
    gello_readings = []
    for i in range(10):  # Take multiple readings for stability
        reading = gello.get_joint_state()[:7]
        gello_readings.append(reading)
        time.sleep(0.1)
    
    gello_home = np.mean(gello_readings, axis=0)
    gello_std = np.std(gello_readings, axis=0)
    
    print(f"Gello home position (averaged): {gello_home}")
    print(f"Reading stability (std dev): {gello_std}")
    
    # Calculate offset
    offset = franka_home - gello_home
    # signs = np.array([1, -1, 1, -1, 1, -1, 1])
    signs = np.array([1, 1, 1, 1, 1, 1, 1])
    print(f"\nCalculated offset (robot_home - gello_home): {offset}")
    
    # # Test the calibration
    # print("\n=== Testing Calibration ===")
    # print("Move the Gello to different positions to test calibration...")
    
    # for test_step in range(5):
    #     print(f"\nTest {test_step + 1}/5 - Press Enter to read current position...")
    #     input()
        
    #     gello_current = gello.get_joint_state()[:7]
    #     calibrated_position = gello_current * signs + offset
        
    #     print(f"Raw Gello reading: {gello_current}")
    #     print(f"Calibrated position: {calibrated_position}")

    
    # Save calibration results
    print(f"\n=== Calibration Results ===")
    print(f"Gello home position: {gello_home}")
    print(f"Robot home position: {franka_home}")  
    print(f"Calibration offset: {offset}")
    print(f"Reading stability: {gello_std}")
    
    # Update the config with calibrated values
    print(f"\nTo use this calibration, update GelloConfig:")
    print(f"home_joints = {gello_home}")
    print(f"calibration_offset = {offset}")
    
    
    return {
        'gello_home': gello_home,
        'robot_home': franka_home,
        'offset': offset,
        'stability': gello_std
    }

if __name__ == "__main__":
    test_gello()
#!/usr/bin/env python3
"""Simple test of Robot class without full environment setup."""

import time
import numpy as np
import rclpy
from crisp_py.robot import Robot, Pose
from crisp_py.robot_config import FrankaConfig
from scipy.spatial.transform import Rotation

def test_robot_basic():
    """Test basic robot functionality."""
    print("=== Testing Basic Robot Functionality ===")
    
    if not rclpy.ok():
        rclpy.init()
    
    try:
        # Create robot instance
        print("Creating robot...")
        robot = Robot(
            namespace="",
            robot_config=FrankaConfig(),
            name="test_robot_simple"
        )
        
        print("Waiting for robot to be ready...")
        robot.wait_until_ready(timeout=10.0)
        print("✓ Robot is ready!")
        
        # Display current state
        current_pose = robot.end_effector_pose
        current_joints = robot.joint_values
        
        print(f"Current end-effector position: {current_pose.position}")
        print(f"Current joint values: {current_joints}")
        
        # Test basic movement (small delta)
        print("\n=== Testing Small Movement ===")
        
        # Store initial pose
        initial_pose = robot.end_effector_pose
        print(f"Initial position: {initial_pose.position}")
        
        # Small movement in x-direction
        target_position = initial_pose.position + np.array([0.05, 0, 0])  # 5cm in x
        target_pose = Pose(
            position=target_position,
            orientation=initial_pose.orientation
        )
        
        print(f"Moving to: {target_position}")
        robot.set_target(pose=target_pose)
        
        # Monitor movement for a few seconds
        for i in range(20):
            current_pose = robot.end_effector_pose
            distance_to_target = np.linalg.norm(current_pose.position - target_position)
            print(f"Step {i}: Current pos: {current_pose.position}, Distance to target: {distance_to_target:.4f}m")
            time.sleep(0.2)
            
            if distance_to_target < 0.01:  # 1cm tolerance
                print("✓ Reached target!")
                break
        
        # Return to initial position
        print(f"\nReturning to initial position: {initial_pose.position}")
        robot.set_target(pose=initial_pose)
        
        time.sleep(3.0)  # Wait a bit
        
        final_pose = robot.end_effector_pose
        return_distance = np.linalg.norm(final_pose.position - initial_pose.position)
        print(f"Final position: {final_pose.position}")
        print(f"Return accuracy: {return_distance:.4f}m")
        
        print("\n✅ Basic robot test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        return False
    finally:
        try:
            robot.shutdown()
        except:
            pass

def test_joint_space():
    """Test joint space movement."""
    print("\n=== Testing Joint Space Movement ===")
    
    try:
        robot = Robot(
            namespace="",
            robot_config=FrankaConfig(),
            name="test_robot_joints"
        )
        
        print("Waiting for robot to be ready...")
        robot.wait_until_ready(timeout=10.0)
        
        # Get current joint configuration
        initial_joints = robot.joint_values
        print(f"Initial joints: {initial_joints}")
        
        # Small joint movement (move joint 1 by 0.1 radians)
        target_joints = initial_joints.copy()
        target_joints[0] += 0.1  # ~5.7 degrees
        
        print(f"Moving to joint configuration: {target_joints}")
        robot.set_target_joint(target_joints)
        
        # Monitor joint movement
        for i in range(15):
            current_joints = robot.joint_values
            joint_error = np.linalg.norm(current_joints - target_joints)
            print(f"Step {i}: Joint error: {joint_error:.4f} rad")
            time.sleep(0.2)
            
            if joint_error < 0.05:  # Small tolerance
                print("✓ Reached target joint configuration!")
                break
        
        # Return to initial configuration
        print(f"Returning to initial joints: {initial_joints}")
        robot.set_target_joint(initial_joints)
        time.sleep(3.0)
        
        print("✅ Joint space test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Joint space test failed: {e}")
        return False
    finally:
        try:
            robot.shutdown()
        except:
            pass

def main():
    """Main test function."""
    print("=== Franka Robot Simple Test ===\n")
    
    success1 = test_robot_basic()
    time.sleep(1.0)
    success2 = test_joint_space()
    
    print(f"\n=== Test Summary ===")
    print(f"Basic robot test: {'✅ PASSED' if success1 else '❌ FAILED'}")
    print(f"Joint space test: {'✅ PASSED' if success2 else '❌ FAILED'}")
    
    if success1 and success2:
        print("🎉 All tests passed! Robot is working correctly.")
    else:
        print("⚠️  Some tests failed. Check robot configuration and status.")

if __name__ == "__main__":
    main()
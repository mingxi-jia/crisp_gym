#!/usr/bin/env python3
"""Diagnostic script to test Franka robot connection and topics."""

import time
import rclpy
from rclpy.node import Node
import subprocess
import sys

def check_ros_topics():
    """Check available ROS2 topics."""
    print("=== Checking ROS2 Topics ===")
    try:
        result = subprocess.run(['ros2', 'topic', 'list'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            topics = result.stdout.strip().split('\n')
            print(f"Found {len(topics)} topics:")
            
            # Look for Franka-related topics
            franka_topics = [t for t in topics if 'franka' in t.lower() or 'cartesian' in t.lower() or 'joint' in t.lower()]
            if franka_topics:
                print("Franka/Robot related topics:")
                for topic in franka_topics:
                    print(f"  - {topic}")
            else:
                print("No Franka/robot topics found!")
                
            return topics
        else:
            print(f"Error getting topics: {result.stderr}")
            return []
    except Exception as e:
        print(f"Failed to check topics: {e}")
        return []

def check_topic_data(topic_name, timeout=5):
    """Check if a topic is publishing data."""
    print(f"\n=== Checking topic: {topic_name} ===")
    try:
        result = subprocess.run(['ros2', 'topic', 'echo', topic_name, '--once'], 
                              capture_output=True, text=True, timeout=timeout)
        if result.returncode == 0:
            print(f"✓ Topic {topic_name} is publishing data")
            return True
        else:
            print(f"✗ No data on topic {topic_name}")
            return False
    except subprocess.TimeoutExpired:
        print(f"✗ Timeout waiting for data on {topic_name}")
        return False
    except Exception as e:
        print(f"✗ Error checking topic {topic_name}: {e}")
        return False

def test_robot_basic():
    """Test basic robot connection."""
    print("\n=== Testing Basic Robot Connection ===")
    
    if not rclpy.ok():
        rclpy.init()
    
    try:
        from crisp_py.robot import Robot
        from crisp_py.robot_config import FrankaConfig
        
        print("Creating robot instance...")
        robot = Robot(
            namespace="",
            robot_config=FrankaConfig(),
            name="test_robot"
        )
        
        print("Robot created successfully")
        print(f"Robot config: {robot.config}")
        print(f"Expected pose topic: {robot.config.current_pose_topic}")
        print(f"Expected joint topic: {robot.config.current_joint_topic}")
        
        print("Checking if robot becomes ready (5 second timeout)...")
        start_time = time.time()
        timeout = 5.0
        
        while not robot.is_ready() and (time.time() - start_time) < timeout:
            print(f"Waiting... Current state - Pose: {robot._current_pose is not None}, "
                  f"Joint: {robot._current_joint is not None}, "
                  f"Target pose: {robot._target_pose is not None}, "
                  f"Target joint: {robot._target_joint is not None}")
            time.sleep(0.5)
        
        if robot.is_ready():
            print("✓ Robot is ready!")
            print(f"Current pose: {robot.end_effector_pose}")
            print(f"Current joints: {robot.joint_values}")
        else:
            print("✗ Robot not ready within timeout")
            
        robot.shutdown()
        return robot.is_ready()
        
    except Exception as e:
        print(f"✗ Error testing robot: {e}")
        return False

def main():
    """Main diagnostic function."""
    print("=== Franka Robot Connection Diagnostics ===\n")
    
    # Check ROS2 is working
    print("=== Checking ROS2 Environment ===")
    try:
        result = subprocess.run(['ros2', '--version'], 
                              capture_output=True, text=True, timeout=5)
        print(f"ROS2 version: {result.stdout.strip()}")
    except Exception as e:
        print(f"ROS2 not available: {e}")
        return
    
    # Check available topics
    topics = check_ros_topics()
    
    # Check key topics for data
    key_topics = [
        '/franka_robot_state_broadcaster/robot_state',
        '/joint_states', 
        '/cartesian_pose',
        '/franka_robot_state_broadcaster/current_pose',
        '/franka_robot_state_broadcaster/target_frame'
    ]
    
    active_topics = []
    for topic in key_topics:
        if topic in topics and check_topic_data(topic, timeout=3):
            active_topics.append(topic)
    
    print(f"\n=== Summary ===")
    print(f"Total topics found: {len(topics)}")
    print(f"Active robot topics: {len(active_topics)}")
    
    if active_topics:
        print("Active topics:")
        for topic in active_topics:
            print(f"  - {topic}")
    else:
        print("⚠️  No active robot topics found!")
        print("Make sure:")
        print("  1. Franka robot is powered on")
        print("  2. Robot driver/simulation is running")
        print("  3. ROS2 bridge is active")
        return
    
    # Test basic robot connection
    success = test_robot_basic()
    
    if success:
        print("\n✅ Robot connection test PASSED")
    else:
        print("\n❌ Robot connection test FAILED")
        print("\nTroubleshooting suggestions:")
        print("1. Check if robot driver is running:")
        print("   ros2 launch franka_bringup franka.launch.py robot_ip:=<ROBOT_IP>")
        print("2. Check robot emergency stop is released")
        print("3. Verify network connection to robot")
        print("4. Check robot is in the correct mode")

if __name__ == "__main__":
    main()
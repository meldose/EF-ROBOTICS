#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import time


class Humanoid_Robot(Node):

    def __init__(self):
        super().__init__('joint_mover')

        # Publisher to command robot joints
        self.publisher_joint = self.create_publisher(
            JointState, '/humanoid_joint_command', 10
        )

        # Subscriber to get current joint states
        self.subscription = self.create_subscription(
            JointState,
            '/humanoid_joint_states',
            self.joint_state_callback,
            10
        )

        self.latest_joint_state = None

        # 10 joints in your humanoid
        self.joint_names = [
            'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
            'left_elbow_pitch_joint', 'left_elbow_roll_joint',
            'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
            'right_elbow_pitch_joint', 'right_elbow_roll_joint'
        ]

    # Store latest joint state
    def joint_state_callback(self, msg):
        self.latest_joint_state = msg

    # Wait for first joint_state message before starting
    def wait_for_joint_state(self, timeout=5.0):
        start = time.time()
        while self.latest_joint_state is None:
            if time.time() - start > timeout:
                self.get_logger().warn("Timeout waiting for joint states")
                return False
            rclpy.spin_once(self, timeout_sec=0.1)
        return True

    # Linear interpolation joint motion
    def move_joints(self, start_positions, end_positions, duration):
        self.get_logger().info("Moving joints...")
        start_time = time.time()

        while time.time() - start_time < duration:
            t = (time.time() - start_time) / duration

            interpolated = [
                s + (e - s) * t
                for s, e in zip(start_positions, end_positions)
            ]

            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = self.joint_names
            msg.position = interpolated
            msg.effort = [0.5] * len(interpolated)

            self.publisher_joint.publish(msg)
            time.sleep(0.01)


def main(args=None):
    rclpy.init(args=args)
    humanoid = Humanoid_Robot()

    # Wait for robot joint feedback
    if not humanoid.wait_for_joint_state():
        humanoid.get_logger().error("Joint states not received. Exiting.")
        rclpy.shutdown()
        return

    # Define all positions (10 joints each)
    start_position = [0.0] * 10

    home_position = [0.5, 0.0, 0.1, -1.0, 0.0,
                     0.0, 0.0, -0.5, 0.0, 0.1]

    move_position_1 = [-0.20, 0.0, 0.0, -1.5, 0.0,
                       0.0, 1.5, 0.20, 0.0, -1.5]

    move_position_2 = [-0.26, -1.0, 0.0, -0.40, 0.0,
                       0.0, 1.5, 0.26, -1.0, -0.40]

    move_position_3 = [-0.24, -1.0, 0.0, -0.6, 0.0,
                       0.0, 1.5, 0.24, -1.0, -0.6]

    move_position_4 = [-0.24, -1.0, 0.0, -0.6, 0.0,
                       0.0, 1.5, 0.24, -1.0, 0.0]

    # Execute full motion sequence
    humanoid.move_joints(start_position, home_position, 5)
    time.sleep(1)

    humanoid.move_joints(home_position, move_position_1, 5)
    time.sleep(1)

    humanoid.move_joints(move_position_1, move_position_2, 5)
    time.sleep(1)

    humanoid.move_joints(move_position_2, move_position_3, 5)
    time.sleep(1)

    humanoid.move_joints(move_position_3, move_position_4, 5)

    humanoid.get_logger().info("Motion sequence completed.")
    rclpy.shutdown()


if __name__ == '__main__':
    main()


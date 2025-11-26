#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import JointState
import time


class HumanoidRobot(object):

    def __init__(self):
        # Initialize ROS1 node
        rospy.init_node('joint_mover', anonymous=True)

        # Publisher (ROS1)
        self.publisher_joint = rospy.Publisher(
            '/humanoid_joint_command', JointState, queue_size=10
        )

        # Subscriber (ROS1)
        self.latest_joint_state = None
        rospy.Subscriber('/humanoid_joint_states', JointState, self.joint_state_callback)

        # 10 joints
        self.joint_names = [
            'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
            'left_elbow_pitch_joint', 'left_elbow_roll_joint',
            'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
            'right_elbow_pitch_joint', 'right_elbow_roll_joint'
        ]

    def joint_state_callback(self, msg):
        self.latest_joint_state = msg

    def wait_for_joint_state(self, timeout=5.0):
        start = time.time()
        rospy.loginfo("Waiting for /humanoid_joint_states...")

        while not rospy.is_shutdown() and self.latest_joint_state is None:
            if time.time() - start > timeout:
                rospy.logwarn("Timeout waiting for /humanoid_joint_states")
                return False
            rospy.sleep(0.1)

        rospy.loginfo("Joint states received.")
        return True

    def move_joints(self, start_positions, end_positions, duration):
        rospy.loginfo("Moving joints...")
        start_time = time.time()

        rate = rospy.Rate(100)  # 100 Hz update

        while not rospy.is_shutdown() and time.time() - start_time < duration:
            t = (time.time() - start_time) / duration

            interpolated = [
                s + (e - s) * t
                for s, e in zip(start_positions, end_positions)
            ]

            msg = JointState()
            msg.header.stamp = rospy.Time.now()
            msg.name = self.joint_names
            msg.position = interpolated
            msg.effort = [0.5] * len(interpolated)

            self.publisher_joint.publish(msg)
            rate.sleep()


def main():
    humanoid = HumanoidRobot()

    # Wait for initial joint state feedback
    if not humanoid.wait_for_joint_state():
        rospy.logerr("Joint states not received. Exiting.")
        return

    # 10-element joint vectors
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

    # Execute sequence in ROS1
    humanoid.move_joints(start_position, home_position, 5)
    rospy.sleep(1)

    humanoid.move_joints(home_position, move_position_1, 5)
    rospy.sleep(1)

    humanoid.move_joints(move_position_1, move_position_2, 5)
    rospy.sleep(1)

    humanoid.move_joints(move_position_2, move_position_3, 5)
    rospy.sleep(1)

    humanoid.move_joints(move_position_3, move_position_4, 5)

    rospy.loginfo("Motion sequence completed.")


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

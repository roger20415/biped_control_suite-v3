import numpy as np
import rclpy
import sys
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String
from .config import LegSide

VALID_LEG_SIDES: tuple[str, ...] = ("left", "right")


class StanceLegControlNode(Node):
    def __init__(self):
        super().__init__('stance_leg_control_node')
        self._leg_side: LegSide = "undefined"

        self._stance_side_subscriber_ = self.create_subscription(
            String,
            '/biped/stance_side',
            self._stance_side_callback,
            10
        )
        self._stance_joint_target_subscriber_ = self.create_subscription(
            Float32MultiArray,
            '/biped/stance_joint_target',
            self._stance_joint_target_callback,
            10
        )  # joint angles in degrees

        self._left_joint_target_publisher_ = self.create_publisher(
            Float32MultiArray,
            '/biped/left_joint_target',
            10
        )  # in rad
        self._right_joint_target_publisher_ = self.create_publisher(
            Float32MultiArray,
            '/biped/right_joint_target',
            10
        )  # in rad

    def _stance_joint_target_callback(self, msg: Float32MultiArray) -> None:
        # joint angles in degrees
        if self._leg_side not in VALID_LEG_SIDES:
            self.get_logger().warn(f"Leg side is invalid: {self._leg_side}")
            return
        leg_side = self._leg_side
        joint_pose = msg.data
        # joint_pose in degrees
        # Convert degrees to radians
        joint_pose_rad = [np.deg2rad(angle) for angle in joint_pose]
        self._pub_joint_pos(joint_pose_rad, leg_side)

    def _stance_side_callback(self, msg: String) -> None:
        if msg.data not in ("left", "right"):
            self.get_logger().error(
                f"Invalid stance side: {msg.data}. Must be 'left' or 'right'.")
            return
        if msg.data != self._leg_side:
            self.get_logger().info(
                f"Switching stance side from {self._leg_side} to {msg.data}.")
            self._leg_side = msg.data

    def _pub_joint_pos(self, joint_pos: list[float], leg_side: str) -> None:
        msg = Float32MultiArray()
        msg.data = [float(i) for i in joint_pos]
        if leg_side == "left":
            self._left_joint_target_publisher_.publish(msg)  # in rad
        elif leg_side == "right":
            self._right_joint_target_publisher_.publish(msg)  # in rad
        else:
            self.get_logger().error(
                f"Invalid leg side: {leg_side}. Cannot publish joint targets.")


def main(args=None):
    rclpy.init(args=args)
    node = StanceLegControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(0)


if __name__ == '__main__':
    main()

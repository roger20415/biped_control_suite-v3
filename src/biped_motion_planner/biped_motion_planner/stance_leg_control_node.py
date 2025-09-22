import numpy as np
import rclpy
import sys
from rclpy.node import Node
from std_msgs.msg import Float32, Float64MultiArray, String
from .config import Config, LegSide

JOINT_NUMS:int = 5 # exclude back, sacrum
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
        self._stance_leg_alpha_subscriber_ = self.create_subscription(
            Float32,
            '/biped/stance_leg_alpha',
            self._stance_leg_alpha_callback,
            10
        ) # leg_alpha in degrees

        self._left_joint_target_publisher_ = self.create_publisher(
            Float64MultiArray,
            '/biped/left_joint_target',
            10
        )
        self._right_joint_target_publisher_ = self.create_publisher(
            Float64MultiArray,
            '/biped/right_joint_target',
            10
        )
    
    def _stance_leg_alpha_callback(self, msg: Float32) -> None:
        # alpha in degrees
        if self._leg_side not in VALID_LEG_SIDES:
            self.get_logger().warn(f"Leg side is invalid: {self._leg_side}")
            return
        leg_side = self._leg_side
        joint_pose = self._compose_joint_pose_for_publish(msg.data, leg_side)
        # joint_pose in degrees
        joint_pose_rad = [np.deg2rad(angle) for angle in joint_pose]  # Convert degrees to radians
        self._pub_joint_pos(joint_pose_rad, leg_side)

    def _stance_side_callback(self, msg: String) -> None:
        if msg.data not in ("left", "right"):
            self.get_logger().error(f"Invalid stance side: {msg.data}. Must be 'left' or 'right'.")
            return
        if msg.data != self._leg_side:
            self.get_logger().info(f"Switching stance side from {self._leg_side} to {msg.data}.")
            self._leg_side = msg.data

    def _pub_joint_pos(self, joint_pos: list[float], leg_side: str) -> None:
        msg = Float64MultiArray()
        msg.data = [float(i) for i in joint_pos]
        if leg_side == "left":
            self._left_joint_target_publisher_.publish(msg)
        elif leg_side == "right":
            self._right_joint_target_publisher_.publish(msg)
        else:
            self.get_logger().error(f"Invalid leg side: {leg_side}. Cannot publish joint targets.")

    def _compose_joint_pose_for_publish(self, leg_alpha: float, leg_side: str) -> list[float]:
        # leg_alpha in degrees
        if leg_alpha > Config.THIGH_MAX_DEG or leg_alpha < Config.THIGH_MIN_DEG:
            self.get_logger().warn(f"Leg alpha {leg_alpha} out of bounds. Clamping to limits.")
            leg_alpha = np.clip(leg_alpha, Config.THIGH_MIN_DEG, Config.THIGH_MAX_DEG)
        if leg_side == "left":
            joint_pose: list[float] = [
                0.0, # hip
                -leg_alpha, # thigh
                0.0, # calf
                -leg_alpha, # ankle
                0.0 # foot
            ]

        elif leg_side == "right":
            joint_pose: list[float] = [
                0.0, # hip
                leg_alpha, # thigh
                0.0, # calf
                leg_alpha, # ankle
                0.0 # foot
            ]
        else:
            self.get_logger().error(f"Invalid leg side: {leg_side}. Cannot compose joint pose.")
            return [0.0]*JOINT_NUMS
        
        if len(joint_pose) != JOINT_NUMS:
            raise ValueError("Invalid swing leg joint pose length.")
        return joint_pose
    

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
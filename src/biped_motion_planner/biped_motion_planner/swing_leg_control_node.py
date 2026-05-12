import rclpy
import sys
from geometry_msgs.msg import Quaternion, Vector3
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float32MultiArray, String
from typing import Optional
from .config import Config, LegSide
from .joint_targets_calculator import JointTargetsCalculator

REQUIRED_P_W_KEYS: tuple[str] = ("baselink", "hip", "foot", "target")
REQUIRED_P_W_RAW_KEYS: tuple[str] = ("l_hip", "l_foot", "r_hip", "r_foot")
VALID_LEG_SIDES: tuple[str, ...] = ("left", "right")


class SwingLegControlNode(Node):
    def __init__(self):
        super().__init__('swing_leg_control_node')
        self._leg_side: LegSide = "undefined"
        self.joint_targets_calculator = JointTargetsCalculator()
        self._q_W_baselink: Optional[Quaternion] = None
        self._p_W_raw: dict[str, Optional[Vector3]] = {
            k: None for k in REQUIRED_P_W_RAW_KEYS}
        self._p_W: dict[str, Optional[Vector3]] = {
            k: None for k in REQUIRED_P_W_KEYS}

        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self._swing_target_subscriber_ = self.create_subscription(
            Vector3,
            '/biped/swing_target',
            self._swing_target_callback,
            10
        )
        self._swing_side_subscriber_ = self.create_subscription(
            String,
            '/biped/swing_side',
            self._swing_side_callback,
            10
        )
        self._baselink_quat_subscriber_ = self.create_subscription(
            Quaternion,
            '/baselink/quat',
            self._baselink_quat_callback,
            qos_sensor
        )
        self._baselink_translate_subscriber_ = self.create_subscription(
            Vector3,
            '/baselink/translate',
            self._baselink_translate_callback,
            qos_sensor
        )
        self._l_hip_translate_subscriber_ = self.create_subscription(
            Vector3,
            '/l_hip/translate',
            self._l_hip_translate_callback,
            qos_sensor
        )
        self._r_hip_translate_subscriber_ = self.create_subscription(
            Vector3,
            '/r_hip/translate',
            self._r_hip_translate_callback,
            qos_sensor
        )
        self._l_foot_translate_subscriber_ = self.create_subscription(
            Vector3,
            '/l_foot/translate',
            self._l_foot_translate_callback,
            qos_sensor
        )
        self._r_foot_translate_subscriber_ = self.create_subscription(
            Vector3,
            '/r_foot/translate',
            self._r_foot_translate_callback,
            qos_sensor
        )
        self._left_joint_target_publisher_ = self.create_publisher(
            Float32MultiArray,
            '/biped/left_joint_target',
            10
        )
        self._right_joint_target_publisher_ = self.create_publisher(
            Float32MultiArray,
            '/biped/right_joint_target',
            10
        )

    def _swing_target_callback(self, msg: Vector3) -> None:
        if self._leg_side not in VALID_LEG_SIDES:
            self.get_logger().warn(f"Leg side is invalid: {self._leg_side}")
            return
        leg_side = self._leg_side
        self._p_W["target"] = msg
        self._compose_leg_p_W(leg_side)
        ready, missing = self._check_state_ready()
        if not ready:
            self.get_logger().warn(
                f"State not ready, missing: {', '.join(missing)}")
            return
        # TODO: check if state is fresh enough
        self.get_logger().info(f"swing_target: {self._p_W['target']}")
        hold_prev_pose, joint_targets = self.joint_targets_calculator.calc_joint_targets(
            self._p_W, self._q_W_baselink, leg_side)
        # joint_targets in rad
        if hold_prev_pose:
            self.get_logger().info("Holding previous pose.")
        else:
            joint_pose = self._compose_joint_pose_for_publish(joint_targets)
            self._pub_joint_pos(joint_pose, leg_side)

    def _swing_side_callback(self, msg: String) -> None:
        if msg.data not in ("left", "right"):
            self.get_logger().error(
                f"Invalid swing side: {msg.data}. Must be 'left' or 'right'.")
            return
        if msg.data != self._leg_side:
            self.get_logger().info(
                f"Switching swing side from {self._leg_side} to {msg.data}.")
            self._leg_side = msg.data

    def _baselink_quat_callback(self, msg: Quaternion) -> None:
        self._q_W_baselink = msg

    def _baselink_translate_callback(self, msg: Vector3) -> None:
        self._p_W["baselink"] = msg
        if msg.z < Config.FALL_DOWN_BASELINK_Z_THRESHOLD:
            self.get_logger().warn("Robot has fallen down! Clear and init p_W_target")
            self._init_p_W_target()

    def _l_hip_translate_callback(self, msg: Vector3) -> None:
        self._p_W_raw["l_hip"] = msg

    def _r_hip_translate_callback(self, msg: Vector3) -> None:
        self._p_W_raw["r_hip"] = msg

    def _l_foot_translate_callback(self, msg: Vector3) -> None:
        self._p_W_raw["l_foot"] = msg

    def _r_foot_translate_callback(self, msg: Vector3) -> None:
        self._p_W_raw["r_foot"] = msg

    def _pub_joint_pos(self, joint_pos: list[float], leg_side: str) -> None:
        msg = Float32MultiArray()
        msg.data = [float(i) for i in joint_pos]
        if leg_side == "left":
            self._left_joint_target_publisher_.publish(msg)
        elif leg_side == "right":
            self._right_joint_target_publisher_.publish(msg)
        else:
            self.get_logger().error(
                f"Invalid leg side: {leg_side}. Cannot publish joint targets.")

    def _compose_joint_pose_for_publish(self, joint_targets) -> list[float]:
        joint_pose: list[float] = [
            joint_targets['hip'],
            joint_targets['thigh'],
            joint_targets['calf'],
            joint_targets['ankle'],
            joint_targets['foot']
        ]
        if len(joint_pose) != Config.JOINT_NUMS:
            raise ValueError("Invalid swing leg joint pose length.")
        return joint_pose

    def _check_state_ready(self) -> tuple[bool, list[str]]:
        missing: list[str] = []
        if self._q_W_baselink is None:
            missing.append("q_W_baselink")
        for k in REQUIRED_P_W_KEYS:
            if self._p_W.get(k) is None:
                missing.append(f"p_W[{k}]")
        return (len(missing) == 0, missing)

    def _compose_leg_p_W(self, leg_side: str) -> None:
        if any(self._p_W_raw.get(k) is None for k in REQUIRED_P_W_RAW_KEYS):
            self.get_logger().error("Raw leg joint positions are incomplete.")
            return
        if leg_side == "left":
            self._p_W["hip"] = self._p_W_raw["l_hip"]
            self._p_W["foot"] = self._p_W_raw["l_foot"]
        elif leg_side == "right":
            self._p_W["hip"] = self._p_W_raw["r_hip"]
            self._p_W["foot"] = self._p_W_raw["r_foot"]


def main(args=None):
    rclpy.init(args=args)
    node = SwingLegControlNode()
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

import numpy as np
import rclpy
import sys
from geometry_msgs.msg import Quaternion, Vector3
from numpy.typing import NDArray
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float64MultiArray, String
from typing import Optional
from .config import Config, SupportSide
from .linear_algebra_utils import LinearAlgebraUtils


_COM_KEYS: tuple[str, ...] = (
    "baselink",
    "back", "sacrum",
    "l_hip", "r_hip",
    "l_thigh", "r_thigh",
    "l_calf", "r_calf",
    "l_ankle", "r_ankle",
    "l_foot", "r_foot",
)
# in meters
# left to right foot distance 0.0065
LEAN_MOVE_THRESHOLD: float = 0.0065/50
LEAN_MAX_ERR: float = 0.018  # error_signed value max limit
LEAN_MIN_STEP: float = float(np.deg2rad(0.04))
LEAN_MAX_STEP: float = float(np.deg2rad(1.0))
LEAN_STEP_ALPHA: float = 0.8  # 0.5~1.5：<1 sensitive；>1 preserve

PUBLISH_PERIOD: float = 0.05  # in seconds
VALID_SUPPORT_SIDES: tuple[str, ...] = ("left", "right", "mid")


class CounterweightControlNode(Node):
    def __init__(self):
        super().__init__('counterweight_control_node')
        self._support_side: SupportSide = "undefined"
        self._p_W_joints_com: dict[str, NDArray[np.float64]] = {
            k: np.zeros(3, dtype=np.float64) for k in _COM_KEYS}
        
        self._lean_target: float = 0.0
        self._if_fall_down: bool = False

        self._p_W_l_foot: Optional[NDArray[np.float64]] = None
        self._p_W_r_foot: Optional[NDArray[np.float64]] = None
        self._q_W_l_foot: Optional[Quaternion] = None
        self._q_W_r_foot: Optional[Quaternion] = None
        self._q_W_baselink: Optional[Quaternion] = None

        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self._support_side_subscriber_ = self.create_subscription(
            String,
            '/biped/support_side',
            self._support_side_callback,
            10
        )
        self._com_subscriber_ = self.create_subscription(
            Float64MultiArray,
            '/com',
            self._com_callback,
            qos_sensor
        )
        self._baselink_quat_subscriber_ = self.create_subscription(
            Quaternion,
            '/baselink/quat',
            self._baselink_quat_callback,
            qos_sensor
        )

        self._counterweight_publisher_ = self.create_publisher(
            Float64MultiArray,
            '/counterweight/joint_targets',
            10
        )

        self._baselink_translate_subscriber_ = self.create_subscription(
            Vector3,
            '/baselink/translate',
            self._baselink_translate_callback,
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
        self._l_foot_quat_subscriber_ = self.create_subscription(
            Quaternion,
            '/l_foot/quat',
            self._l_foot_quat_callback,
            qos_sensor
        )
        self._r_foot_quat_subscriber_ = self.create_subscription(
            Quaternion,
            '/r_foot/quat',
            self._r_foot_quat_callback,
            qos_sensor
        )

        self._left_joint_target_publisher_ = self.create_publisher(
            Float64MultiArray,
            '/biped/left_joint_target',
            10
        )
        self._right_joint_target_publisher_ = self.create_publisher(
            Float64MultiArray,
            '/biped/right_joint_target',
            10
        )  # in rad

        self._timer = self.create_timer(PUBLISH_PERIOD, self._timer_callback)

    def _pub_counterweight_pos(self, counterweight_pos) -> None:
        """Publish the target joint positions for the counterweight (sacrum) - Now kept at 0.0."""
        msg = Float64MultiArray()
        msg.data = [float(i) for i in counterweight_pos]
        self._counterweight_publisher_.publish(msg)

    def _pub_leg_targets(self, lean_angle: float) -> None:
        hip_cmd = lean_angle
        foot_cmd = lean_angle

        left_msg = Float64MultiArray()
        right_msg = Float64MultiArray()
        
        left_msg.data = [hip_cmd, 0.0, 0.0, 0.0, foot_cmd]
        right_msg.data = [hip_cmd, 0.0, 0.0, 0.0, foot_cmd]
        
        self._left_joint_target_publisher_.publish(left_msg)
        self._right_joint_target_publisher_.publish(right_msg)

    def _support_side_callback(self, msg: String) -> None:
        if msg.data != self._support_side:
            self.get_logger().info(
                f"Switching support side from {self._support_side} to {msg.data}.")
            if msg.data in ("left", "right", "mid"):
                self._support_side = msg.data
            else:
                self.get_logger().error(
                    f"Invalid support side: {msg.data}. Keeping previous: {self._support_side}.")

    def _com_callback(self, msg: Float64MultiArray) -> None:
        data = np.asarray(msg.data, dtype=np.float64)
        expected = 3 * len(_COM_KEYS)
        if data.size != expected:
            self.get_logger().error(
                f"Expected {expected} values (got {data.size}).")
            return

        vecs = data.reshape(len(_COM_KEYS), 3)
        p_W_joints_com = {name: vecs[i] for i, name in enumerate(_COM_KEYS)}
        self._p_W_joints_com = p_W_joints_com

    def _calc_p_W_biped_com(self, joints_com: dict[str, NDArray[np.float64]]) -> NDArray[np.float64]:
        total_mass: float = (
            Config.BASELINK_MASS + Config.BACK_MASS + Config.SACRUM_MASS +
            Config.HIP_MASS * 2 +
            Config.THIGH_MASS * 2 +
            Config.CALF_MASS * 2 +
            Config.ANKLE_MASS * 2 +
            Config.FOOT_MASS * 2)
        weighted_sum: NDArray[np.float64] = (
            joints_com["baselink"] * Config.BASELINK_MASS +
            joints_com["back"] * Config.BACK_MASS +
            joints_com["sacrum"] * Config.SACRUM_MASS +
            (joints_com["l_hip"] + joints_com["r_hip"]) * Config.HIP_MASS +
            (joints_com["l_thigh"] + joints_com["r_thigh"]) * Config.THIGH_MASS +
            (joints_com["l_calf"] + joints_com["r_calf"]) * Config.CALF_MASS +
            (joints_com["l_ankle"] + joints_com["r_ankle"]) * Config.ANKLE_MASS +
            (joints_com["l_foot"] + joints_com["r_foot"]) * Config.FOOT_MASS
        )
        p_W_biped_com: NDArray[np.float64] = weighted_sum / total_mass
        return p_W_biped_com

    def _calc_p_S_support(self, support_side: str) -> NDArray[np.float64]:
        if self._p_W_l_foot is None or self._p_W_r_foot is None:
            raise ValueError("Foot positions are not yet received.")
        if self._q_W_l_foot is None or self._q_W_r_foot is None:
            raise ValueError("Foot orientations are not yet received.")

        if support_side == "left":
            xLFOOT_W_norm = self._calc_xFOOT_W_norm(self._q_W_l_foot)
            p_S_support = self._p_W_l_foot - Config.FOOT_LINK_X_SEMI_LENGTH * xLFOOT_W_norm
            p_S_support[2] = 0.0
        elif support_side == "right":
            xRFOOT_W_norm = self._calc_xFOOT_W_norm(self._q_W_r_foot)
            p_S_support = self._p_W_r_foot - Config.FOOT_LINK_X_SEMI_LENGTH * xRFOOT_W_norm
            p_S_support[2] = 0.0
        else:
            raise ValueError("Support side is undefined.")
        return p_S_support

    def _calc_xFOOT_W_norm(self, q_W_foot: Quaternion) -> NDArray[np.float64]:
        R_W_FOOT: NDArray[np.float64] = LinearAlgebraUtils.quaternion_to_rotation_matrix(
            q_W_foot)
        xFOOT_W = R_W_FOOT[:, 0]
        return LinearAlgebraUtils.normalize_vec(xFOOT_W)

    def _timer_callback(self) -> None:
        if self._support_side not in VALID_SUPPORT_SIDES:
            self.get_logger().warn("Support side is invalid.")
            return
        support_side: str = self._support_side
        if self._if_fall_down:
            return
        if self._q_W_baselink is None:
            self.get_logger().warn("Waiting for /baselink/quat ...")
            return

        try:
            vec_S_com_to_support = self._calc_vec_S_com_to_support(support_side)
            vec_S_sacrum_proj_norm = self._calc_vec_S_sacrum_proj_norm()
            
            err_signed = float(np.dot(vec_S_com_to_support[:2], vec_S_sacrum_proj_norm[:2]))

            self._lean_target = self._calc_lean_target(err_signed)

        except Exception as e:
            self.get_logger().error(f"Timer step failed: {e}")
            return
            
        self.get_logger().info(f"Parallelogram Mode | Lean Angle: {np.rad2deg(self._lean_target):.2f}°") 

        self._pub_counterweight_pos([0.0, 0.0])
        self._pub_leg_targets(self._lean_target)

    def _calc_lean_target(self, err_signed: float) -> float:
        """Calculate the target lean angle for the legs (parallelogram) based on CoM error."""
        if abs(err_signed) < LEAN_MOVE_THRESHOLD:
            return self._lean_target

        error_sign = float(np.sign(err_signed))
        mag = abs(err_signed) / LEAN_MAX_ERR
        mag = np.clip(mag, 0.0, 1.0)
        
        step = LEAN_MIN_STEP + (LEAN_MAX_STEP - LEAN_MIN_STEP) * (mag ** LEAN_STEP_ALPHA)

        lean_target = self._lean_target - error_sign * step
        return lean_target

    def _baselink_quat_callback(self, msg: Quaternion) -> None:
        self._q_W_baselink = msg

    def _baselink_translate_callback(self, msg: Vector3) -> None:
        if (msg.z < Config.FALL_DOWN_BASELINK_Z_THRESHOLD) and not self._if_fall_down:
            self.get_logger().warn("Robot has fallen down! Resetting lean target.")
            self._lean_target = 0.0
            self._if_fall_down = True
        elif msg.z >= Config.FALL_DOWN_BASELINK_Z_THRESHOLD and self._if_fall_down:
            self.get_logger().info("Robot is back up.")
            self._lean_target = 0.0
            self._if_fall_down = False

    def _l_foot_translate_callback(self, msg: Vector3) -> None:
        self._p_W_l_foot = np.array([msg.x, msg.y, msg.z], dtype=np.float64)

    def _r_foot_translate_callback(self, msg: Vector3) -> None:
        self._p_W_r_foot = np.array([msg.x, msg.y, msg.z], dtype=np.float64)

    def _l_foot_quat_callback(self, msg: Quaternion) -> None:
        self._q_W_l_foot = msg

    def _r_foot_quat_callback(self, msg: Quaternion) -> None:
        self._q_W_r_foot = msg

    def _calc_vec_S_com_to_support(self, support_side: str) -> NDArray[np.float64]:
        p_W_biped_com: NDArray[np.float64] = self._calc_p_W_biped_com(
            self._p_W_joints_com)
        p_S_biped_com: NDArray[np.float64] = np.array(
            [p_W_biped_com[0], p_W_biped_com[1], 0.0], dtype=np.float64)
        p_S_support: NDArray[np.float64] = self._calc_p_S_support(support_side)
        vec_S_com_to_support: NDArray[np.float64] = p_S_support - p_S_biped_com
        return vec_S_com_to_support

    def _calc_vec_S_sacrum_proj_norm(self) -> NDArray[np.float64]:
        if self._q_W_baselink is None:
            raise ValueError("Baselink quaternion is not yet received.")
        R_WB = LinearAlgebraUtils.quaternion_to_rotation_matrix(
            self._q_W_baselink)
        vec_W_yB = R_WB[:, 1]
        vec_S_yB = np.array([vec_W_yB[0], vec_W_yB[1], 0.0], dtype=np.float64)
        vec_S_yB_length = np.linalg.norm(vec_S_yB)
        if vec_S_yB_length < 1e-10:
            return vec_S_yB
        return vec_S_yB / vec_S_yB_length


def main(args=None):
    rclpy.init(args=args)
    node = CounterweightControlNode()
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
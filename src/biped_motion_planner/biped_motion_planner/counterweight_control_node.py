import numpy as np
import rclpy
import sys
from geometry_msgs.msg import Quaternion, Vector3
from numpy.typing import NDArray
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float64MultiArray, String, Float64
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
ERR_MAX_ERR: float = 45/10000  # error_signed value max limit
LEAN_MOVE_THRESHOLD: float = 0.1/10000
SACRUM_MOVE_THRESHOLD: float = 0.1/10000
LEAN_MIN_STEP: float = float(np.deg2rad(0.04))
LEAN_MAX_STEP: float = float(np.deg2rad(1.0))
LEAN_STEP_ALPHA: float = 0.8  # 0.5~1.5：<1 sensitive；>1 preserve

SACRUM_MIN_STEP: float = float(np.deg2rad(0.04))
SACRUM_MAX_STEP: float = float(np.deg2rad(3.0))
SACRUM_STEP_ALPHA: float = 1.5  # 0.5~1.5：<1 sensitive；>1 preserve

TRANSITION_ALPHA_INCREMENT: float = 0.02  # Increment for transition alpha per timer tick
MAX_ALPHA = 1.0

PUBLISH_PERIOD: float = 0.05  # in seconds
VALID_SUPPORT_SIDES: tuple[str, ...] = ("left", "right", "mid")


SWING_LIFT_HIP_PITCH_MAG: float = float(np.deg2rad(-25.0))   # Thigh lift magnitude
SWING_LIFT_KNEE_MAG: float = float(np.deg2rad(50.0))        # Calf bend magnitude
SWING_LIFT_ANKLE_PITCH_MAG: float = float(np.deg2rad(-25.0)) # Ankle compensation magnitude
SWING_ABDUCT_HIP_ROLL_MAG: float = float(np.deg2rad(35.0))  # Hip roll abduction magnitude
PHASE3_TICKS_PER_STAGE: int = int(2.0 / PUBLISH_PERIOD)     # 10 ticks (0.5s) per stage

class CounterweightControlNode(Node):
    def __init__(self):
        super().__init__('counterweight_control_node')
        self._support_side: SupportSide = "undefined"
        self._p_W_joints_com: dict[str, NDArray[np.float64]] = {
            k: np.zeros(3, dtype=np.float64) for k in _COM_KEYS}
        
        self._lean_target: float = 0.0
        self._if_fall_down: bool = False

        self._phase: int = 1  # 1: Parallelogram, 2: Transition, 3: Pelvic Tilt
        self._transition_alpha: float = 0.0  # 0.0 full parallelogram，1.0 full pelvic tilt(MAX ALPHA)
        self._stable_count: int = 0
        
        self._sacrum_target: float = 0.0

        self._p_W_l_foot: Optional[NDArray[np.float64]] = None
        self._p_W_r_foot: Optional[NDArray[np.float64]] = None
        self._q_W_l_foot: Optional[Quaternion] = None
        self._q_W_r_foot: Optional[Quaternion] = None
        self._q_W_baselink: Optional[Quaternion] = None

        self._phase3_tick: int = 0
        self._phase3_cycle: int = 0
        self._swing_offsets: list[float] = [0.0, 0.0, 0.0, 0.0]  # [hip_roll, hip_pitch, knee, ankle_pitch]

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

        self._phase_num_publisher_ = self.create_publisher(
            Float64,
            '/biped/phase_num',
            10
        )

        self._timer = self.create_timer(PUBLISH_PERIOD, self._timer_callback)

    def _pub_phase_num(self) -> None:
        msg = Float64()
        if self._phase == 1:
            msg.data = 0.0 / 6.0
        elif self._phase == 2:
            msg.data = 1.0 / 6.0
        elif self._phase == 3:
            stage = self._phase3_tick // PHASE3_TICKS_PER_STAGE
            msg.data = (2.0 + stage) / 6.0
        else:
            msg.data = 0.0

        self._phase_num_publisher_.publish(msg)

    def _pub_counterweight_pos(self, sacrum_angle: float) -> None:
        msg = Float64MultiArray()
        msg.data = [0.0, float(sacrum_angle)] 
        self._counterweight_publisher_.publish(msg)

    def _pub_leg_targets(self, lean_angle: float, support_side: str, alpha: float) -> None:
        """
        Calculates and publishes the joint targets for both legs.
        Applies swing leg animation offsets to the non-supporting leg during Phase 3.
        """
        left_msg = Float64MultiArray()
        right_msg = Float64MultiArray()
        
        left_hip, left_foot = 0.0, 0.0
        right_hip, right_foot = 0.0, 0.0

        if support_side in ("left", "right"):
            left_hip = lean_angle * (1.0 - 2.0 * alpha)
            left_foot = lean_angle * (1.0 - alpha)
            right_hip = lean_angle * (1.0 - 2.0 * alpha)
            right_foot = lean_angle * (1.0 - alpha)
        else:
            left_hip = lean_angle * (1.0 - alpha)
            left_foot = lean_angle * (1.0 - alpha)
            right_hip = lean_angle * (1.0 - alpha)
            right_foot = lean_angle * (1.0 - alpha)

        l_data = [left_hip, 0.0, 0.0, 0.0, left_foot]
        r_data = [right_hip, 0.0, 0.0, 0.0, right_foot]

        if support_side == "right":
            l_data[0] += self._swing_offsets[0]  # hip roll
            l_data[1] += self._swing_offsets[1]  # hip pitch
            l_data[2] += self._swing_offsets[2]  # knee
            l_data[3] += self._swing_offsets[3]  # ankle pitch
        elif support_side == "left":
            r_data[0] += self._swing_offsets[0]  
            r_data[1] += self._swing_offsets[1]  
            r_data[2] += self._swing_offsets[2]  
            r_data[3] += self._swing_offsets[3]  

        left_msg.data = l_data
        right_msg.data = r_data
        
        self._left_joint_target_publisher_.publish(left_msg)
        self._right_joint_target_publisher_.publish(right_msg)

    def _support_side_callback(self, msg: String) -> None:
        if msg.data != self._support_side:
            self.get_logger().info(
                f"Switching support side from {self._support_side} to {msg.data}.")
            if msg.data in ("left", "right", "mid"):
                self._support_side = msg.data
                self._phase = 1
                self._transition_alpha = 0.0
                self._stable_count = 0
                self._sacrum_target = 0.0
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
            self.get_logger().info(f"err_signed: {err_signed*10000:.6f}")
            if support_side not in ("left", "right"):
                self._phase = 1
                self._transition_alpha = 0.0
                self._sacrum_target = 0.0
                self._reset_phase3_anim()

            if self._phase == 1:
                self._sacrum_target = 0.0
                self._lean_target = self._calc_lean_target(err_signed)
                if abs(err_signed) < LEAN_MOVE_THRESHOLD:
                    self._stable_count += 1
                    if self._stable_count >= 10:
                        self.get_logger().info("Entering transition phase.")
                        self._phase = 2
                else:
                    self._stable_count = 0
            
            elif self._phase == 2:
                self._transition_alpha += TRANSITION_ALPHA_INCREMENT
                self._sacrum_target = self._calc_sacrum_target(err_signed)
                if self._transition_alpha >= MAX_ALPHA:
                    self._transition_alpha = MAX_ALPHA
                    self.get_logger().info("Entering pelvic tilt phase.")
                    self._phase = 3

            elif self._phase == 3:
                self._sacrum_target = self._calc_sacrum_target(err_signed)
                self._update_swing_leg_animation(support_side)

        except Exception as e:
            self.get_logger().error(f"Counterweight control Node timer step failed: {e}")
            return

        self._pub_counterweight_pos(self._sacrum_target)
        self._pub_leg_targets(self._lean_target, support_side, self._transition_alpha)
        self._pub_phase_num()

    def _calc_lean_target(self, err_signed: float) -> float:
        if abs(err_signed) < LEAN_MOVE_THRESHOLD:
            return self._lean_target
        err_sign = float(np.sign(err_signed))
        mag = abs(err_signed) / ERR_MAX_ERR
        mag = np.clip(mag, 0.0, 1.0)
        step = LEAN_MIN_STEP + (LEAN_MAX_STEP - LEAN_MIN_STEP) * (mag ** LEAN_STEP_ALPHA)
        lean_target = self._lean_target - step*err_sign
        return lean_target

    def _calc_sacrum_target(self, err_signed: float) -> float:
        if abs(err_signed) < SACRUM_MOVE_THRESHOLD:
            return self._sacrum_target
        err_sign = float(np.sign(err_signed))
        mag = abs(err_signed) / ERR_MAX_ERR
        mag = np.clip(mag, 0.0, 1.0)
        step = SACRUM_MIN_STEP + (SACRUM_MAX_STEP - SACRUM_MIN_STEP) * (mag ** SACRUM_STEP_ALPHA)
        sacrum_target = self._sacrum_target + step*err_sign
        sacrum_target = np.clip(sacrum_target, -np.deg2rad(Config.SACRUM_MAX_DEG), np.deg2rad(Config.SACRUM_MAX_DEG))

        return float(sacrum_target)

    def _baselink_quat_callback(self, msg: Quaternion) -> None:
        self._q_W_baselink = msg

    def _baselink_translate_callback(self, msg: Vector3) -> None:
        if (msg.z < Config.FALL_DOWN_BASELINK_Z_THRESHOLD) and not self._if_fall_down:
            self.get_logger().warn("Robot has fallen down! Resetting lean target.")
            self._lean_target = 0.0
            self._if_fall_down = True
            self._phase = 1
            self._transition_alpha = 0.0
            self._sacrum_target = 0.0
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

    def _reset_phase3_anim(self) -> None:
        """
        Resets the swing leg animation states and offsets.
        """
        self._phase3_tick = 0
        self._phase3_cycle = 0
        self._swing_offsets = [0.0, 0.0, 0.0, 0.0]

    def _update_swing_leg_animation(self, support_side: str) -> None:
        """
        Updates the swing leg trajectory offsets during Phase 3 based on hardware joint limits and directions.
        Generates a 4-stage smooth motion: Lift -> Abduct -> Return -> Put down.

        Args:
            support_side: The current supporting side ('left' or 'right').
        """
        stage = self._phase3_tick // PHASE3_TICKS_PER_STAGE
        progress = (self._phase3_tick % PHASE3_TICKS_PER_STAGE) / float(PHASE3_TICKS_PER_STAGE)

        if support_side == "right":
            p_sign, k_sign, a_sign = -1.0, 1.0, 1.0  # Thigh(-), Calf(+), Ankle(+)
            r_sign = 1.0                             # Hip Roll(+)
        elif support_side == "left":
            p_sign, k_sign, a_sign = 1.0, -1.0, -1.0 # Thigh(+), Calf(-), Ankle(-)
            r_sign = -1.0                            # Hip Roll(-)
        else:
            return

        target_hip_pitch = p_sign * SWING_LIFT_HIP_PITCH_MAG
        target_knee = k_sign * SWING_LIFT_KNEE_MAG
        target_ankle_pitch = a_sign * SWING_LIFT_ANKLE_PITCH_MAG
        target_hip_roll = r_sign * SWING_ABDUCT_HIP_ROLL_MAG

        hip_roll, hip_pitch, knee, ankle_pitch = 0.0, 0.0, 0.0, 0.0

        if stage == 0:
            hip_pitch = target_hip_pitch * progress
            knee = target_knee * progress
            ankle_pitch = target_ankle_pitch * progress
            
        elif stage == 1:
            hip_pitch = target_hip_pitch
            knee = target_knee
            ankle_pitch = target_ankle_pitch
            hip_roll = target_hip_roll * progress
            
        elif stage == 2:
            hip_pitch = target_hip_pitch
            knee = target_knee
            ankle_pitch = target_ankle_pitch
            hip_roll = target_hip_roll * (1.0 - progress)
            
        elif stage == 3:
            hip_pitch = target_hip_pitch * (1.0 - progress)
            knee = target_knee * (1.0 - progress)
            ankle_pitch = target_ankle_pitch * (1.0 - progress)

        self._swing_offsets = [hip_roll, hip_pitch, knee, ankle_pitch]

        self._phase3_tick += 1
        if self._phase3_tick >= PHASE3_TICKS_PER_STAGE * 4:
            self._phase3_tick = 0
            self._phase3_cycle += 1

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
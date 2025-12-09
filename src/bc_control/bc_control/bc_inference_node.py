import math
import os
import rclpy
import torch
import numpy as np
from typing import Optional, List
from collections import deque

from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Quaternion, Twist, Vector3
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState

from .actor_bc import ActorBC
from .preprocess_cfg import PreprocessCfg

# Configs (Keep consistent with Collector)
TIMER_PERIOD_SEC = 0.05  # 20 Hz
FOOT_CONTACT_THRESHOLD = 0.0014

# Model Architecture Configs (Must match training!)
NET_ARCH_PI = [64, 64]
ACTIVATION_FN_STR = 'nn.ELU'

OBS_HISTORY_LEN = 3
ACTION_HISTORY_LEN = 2

# Path Configs
try:
    SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()
BODY_WEIGHTS_PATH  = os.path.join(SCRIPT_DIR, "model/bc_actor_body_weights.pth")
HEAD_WEIGHTS_PATH  = os.path.join(SCRIPT_DIR, "model/bc_actor_head_weights.pth")
ACTION_SCALES_PATH = os.path.join(SCRIPT_DIR, "action_scales.npy")

class BcInferenceNode(Node):
    def __init__(self):
        super().__init__('bc_inference_node')
        self._step_counter = 0

        # === 1. State Variables (Same as Collector) ===
        self._p_W_baselink_z: Optional[float] = None
        self._q_W_baselink: Optional[Quaternion] = None
        self._twist_W_baselink: Optional[list[float]] = None
        self._joint_positions: Optional[List[float]] = None
        self._joint_velocities: Optional[List[float]] = None
        self._p_W_l_foot_z: Optional[float] = None
        self._p_W_r_foot_z: Optional[float] = None

        # === 2. Load Normalization Parameters ===
        self._obs_mean = np.array(PreprocessCfg.OBS_MEAN, dtype=np.float32)
        self._obs_std = np.array(PreprocessCfg.OBS_STD, dtype=np.float32)
        self._obs_std[self._obs_std < 1e-8] = 1.0

        self._default_joint_pos = np.array(PreprocessCfg.DEFAULT_JOINT_POSITIONS, dtype=np.float32)
        self._action_scales = self._load_action_scales()

        # --- Load Model ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Inference Device: {self.device}")

        # Determine dimensions from Config
        self._obs_dim = self._obs_mean.shape[0]
        self._act_dim = self._action_scales.shape[0]
        self._input_dim = (self._obs_dim * OBS_HISTORY_LEN) + (self._act_dim * ACTION_HISTORY_LEN)
        self._model = self._load_model(input_dim=self._input_dim)

        self._obs_history = deque(maxlen=OBS_HISTORY_LEN)
        self._act_history = deque(maxlen=ACTION_HISTORY_LEN)

        # QoS Setting
        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # === 2. Subscribers (Observation Sources) ===
        self._baselink_translate_subscriber_ = self.create_subscription(
            Vector3, '/baselink/translate', self._baselink_translate_callback, qos_sensor)
        
        self._baselink_quat_subscriber_ = self.create_subscription(
            Quaternion, '/baselink/quat', self._baselink_quat_callback, qos_sensor)
        
        self._baselink_twist_subscriber_ = self.create_subscription(
            Twist, '/baselink/twist', self._baselink_twist_callback, qos_sensor)
        
        self._joint_state_subscriber_ = self.create_subscription(
            JointState, '/biped/jointState', self._joint_state_callback, qos_sensor)
        
        self._l_foot_translate_subscriber_ = self.create_subscription(
            Vector3, '/l_foot/translate', self._l_foot_translate_callback, qos_sensor)
        
        self._r_foot_translate_subscriber_ = self.create_subscription(
            Vector3, '/r_foot/translate', self._r_foot_translate_callback, qos_sensor)

        # === 3. Publishers (Action Targets) ===
        self._left_joint_target_pub_ = self.create_publisher(
            Float64MultiArray, '/biped/left_joint_target', 10)
        
        self._right_joint_target_pub_ = self.create_publisher(
            Float64MultiArray, '/biped/right_joint_target', 10)
        
        self._counterweight_joint_targets_pub_ = self.create_publisher(
            Float64MultiArray, '/counterweight/joint_targets', 10)

        # === 4. Inference Timer ===
        self._timer = self.create_timer(TIMER_PERIOD_SEC, self.on_inference_step)

    def _load_action_scales(self) -> np.ndarray:
        if not os.path.exists(ACTION_SCALES_PATH):
            self.get_logger().error(f"Action scales file not found at: {ACTION_SCALES_PATH}")
            raise FileNotFoundError(f"Action scales file missing: {ACTION_SCALES_PATH}")
        try:
            scales = np.load(ACTION_SCALES_PATH)
            return scales.astype(np.float32)
        except Exception as e:
            self.get_logger().error(f"Failed to load action scales: {e}")
            raise e

    def _load_model(self, input_dim: int) -> ActorBC:
        model = ActorBC(
            obs_dim=input_dim,
            act_dim=self._act_dim,
            net_arch_pi=NET_ARCH_PI,
            activation_fn_str=ACTIVATION_FN_STR
        ).to(self.device)

        if not os.path.exists(BODY_WEIGHTS_PATH) or not os.path.exists(HEAD_WEIGHTS_PATH):
            self.get_logger().error(f"Weights not found at {BODY_WEIGHTS_PATH} or {HEAD_WEIGHTS_PATH}")
            raise FileNotFoundError("Model weights missing.")

        try:
            body_sd = torch.load(BODY_WEIGHTS_PATH, map_location=self.device)
            head_sd = torch.load(HEAD_WEIGHTS_PATH, map_location=self.device)
            
            model.policy_net.load_state_dict(body_sd, strict=True)
            model.action_net.load_state_dict(head_sd, strict=True)
            model.eval() # Set to evaluation mode
            self.get_logger().info("Model weights loaded successfully.")
        except Exception as e:
            self.get_logger().error(f"Failed to load weights: {e}")
            raise e
        return model

    # ================= Callbacks (State Updates) =================
    def _baselink_translate_callback(self, msg: Vector3) -> None:
        self._p_W_baselink_z = msg.z

    def _baselink_quat_callback(self, msg: Quaternion) -> None:
        self._q_W_baselink = msg

    def _baselink_twist_callback(self, msg: Twist) -> None:
        # linear(3) + angular(3)
        self._twist_W_baselink = [
            msg.linear.x, msg.linear.y, msg.linear.z,
            msg.angular.x, msg.angular.y, msg.angular.z
        ]

    def _joint_state_callback(self, msg: JointState) -> None:
        self._joint_positions = list(msg.position) if msg.position else []
        self._joint_velocities = list(msg.velocity) if msg.velocity else []

    def _l_foot_translate_callback(self, msg: Vector3) -> None:
        self._p_W_l_foot_z = msg.z

    def _r_foot_translate_callback(self, msg: Vector3) -> None:
        self._p_W_r_foot_z = msg.z

    # ================= Main Inference Loop =================
    def on_inference_step(self) -> None:
        # 1. Check if all sensor data is ready
        if not self._check_states_ready():
            return
        # 2. Compose Observation
        raw_obs_t = self._get_current_raw_state()
        # 3. Normalize Observation
        norm_obs_t = self._normalize_obs(raw_obs_t)
        
        # 4. Update History Buffer
        self._update_history_buffers(norm_obs_t)
        
        # 5. Construct Stacked Input Vector [S_t-2, S_t-1, S_t, A_t-2, A_t-1]
        input_vector = self._construct_model_input()
        
        # 6. Model Inference
        # input_vector shape is (1, input_dim)
        obs_tensor = torch.as_tensor(input_vector, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            norm_action_pred = self._model(obs_tensor).squeeze(0).cpu().numpy()
        self._act_history.append(norm_action_pred)

        # 7. Scale Actions to Real World
        real_actions = self._scale_actions(norm_action_pred)
        self._log_predict_actions(real_actions)
        
        # 8. Publish
        self._publish_actions(real_actions)

    # ================= Helper Functions =================
    def _get_current_raw_state(self) -> np.ndarray:
        # 2. Construct Observation (MUST match training data order exactly)
        obs_list: list[float] = []
        ## (1) baselink z position
        obs_list.append(float(self._p_W_baselink_z))
        
        ## (2) baselink euler angles (Roll, Pitch, Yaw)
        euler_angle_W_baselink = self._quaternion_to_euler(self._q_W_baselink)
        obs_list.extend([float(x) for x in euler_angle_W_baselink])
        
        ## (3) baselink twist (linear vel & angular vel)
        obs_list.extend([float(x) for x in self._twist_W_baselink])
        
        ## (4) 11 joints positions
        obs_list.extend([float(x) for x in self._joint_positions])
        
        ## (5) 11 joints velocities
        obs_list.extend([float(x) for x in self._joint_velocities])
        
        ## (6) left foot contact
        obs_list.append(self._check_foot_contact(self._p_W_l_foot_z))
        
        ## (7) right foot contact
        obs_list.append(self._check_foot_contact(self._p_W_r_foot_z))
        
        # Convert to numpy array
        raw_obs = np.asarray(obs_list, dtype=np.float32)
        return raw_obs

    def _update_history_buffers(self, current_norm_obs: np.ndarray) -> None:
        if len(self._obs_history) == 0:
            self._obs_history.append(current_norm_obs)
            self._obs_history.append(current_norm_obs)
            zeros_act = np.zeros(self._act_dim, dtype=np.float32)
            self._act_history.append(zeros_act)
            self._act_history.append(zeros_act)
            
        self._obs_history.append(current_norm_obs)

    def _construct_model_input(self) -> np.ndarray:
        s_list = list(self._obs_history)
        a_list = list(self._act_history)
        input_vector = np.concatenate(s_list + a_list, axis=0)
        
        return input_vector

    def _check_states_ready(self) -> bool:
        if (self._p_W_baselink_z is None or
            self._q_W_baselink is None or  # Added check for Quaternion
            self._p_W_l_foot_z is None or
            self._p_W_r_foot_z is None or
            not self._twist_W_baselink or
            not self._joint_positions or
            not self._joint_velocities):
            return False
        return True

    def _quaternion_to_euler(self, q: Quaternion) -> list[float]:
        x, y, z, w = q.x, q.y, q.z, q.w
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)
            
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return [roll, pitch, yaw]

    def _check_foot_contact(self, foot_z: float) -> float:
        return 1.0 if foot_z < FOOT_CONTACT_THRESHOLD else -1.0
    
    def _normalize_obs(self, raw_obs: np.ndarray) -> np.ndarray:
        if raw_obs.shape[0] != self._obs_mean.shape[0]:
            raise ValueError(
                f"Observation dimension mismatch! "
                f"Received {raw_obs.shape[0]}, expected {self._obs_mean.shape[0]} (from cfg)."
            )
        normalized_obs = (raw_obs - self._obs_mean) / self._obs_std
        
        return normalized_obs.astype(np.float32)

    def _scale_actions(self, model_output: np.ndarray) -> np.ndarray:
            action_delta = model_output * self._action_scales
            real_target = action_delta + self._default_joint_pos          
            return real_target

    def _publish_actions(self, targets: np.ndarray) -> None:
        if targets.shape[0] != 11:
            self.get_logger().error(f"Action dim mismatch. Expected 11, got {targets.shape[0]}")
            return
        
        msg_cw = Float64MultiArray()
        msg_cw.data = [0.0, float(targets[0])] 
        self._counterweight_joint_targets_pub_.publish(msg_cw)

        msg_left = Float64MultiArray()
        msg_left.data = [float(x) for x in targets[1:6]]
        self._left_joint_target_pub_.publish(msg_left)

        msg_right = Float64MultiArray()
        msg_right.data = [float(x) for x in targets[6:11]]
        self._right_joint_target_pub_.publish(msg_right)
    
    def _log_predict_actions(self, real_actions: np.ndarray) -> None:
        self._step_counter += 1
        if self._step_counter % 100 == 0:
            action_str = np.array2string(real_actions, precision=3, suppress_small=True)
            self.get_logger().info(f"[Step {self._step_counter}] Real Actions: {action_str}")

def main(args=None):
    rclpy.init(args=args)
    node = BcInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
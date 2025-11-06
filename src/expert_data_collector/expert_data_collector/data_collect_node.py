import math
import os
from typing import Optional, List

import numpy as np
import rclpy
from geometry_msgs.msg import Quaternion, Twist, Vector3
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState

TIMER_PERIOD_SEC = 0.05  #20 Hz

BASELINK_HEIGHT_BOUND = (0.0137, 0.022) # must be consistent with IsaaclabRlEnvCfg
FOOT_CONTACT_THRESHOLD = 0.00125 # must be consistent with IsaaclabRlEnvCfg

DATA_BUFFER_SIZE = 1000
SAVE_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/expert_data.npz")


class DataCollectNode(Node):
    def __init__(self):
        super().__init__('data_collect_node')
        
        # for states
        self._p_W_baselink_z: Optional[float] = None
        self._q_W_baselink: Optional[Quaternion] = None
        self._p_W_l_foot_z: Optional[float] = None
        self._p_W_r_foot_z: Optional[float] = None
        self._twist_W_baselink: Optional[list[float]] = None
        self._joint_positions: Optional[List[float]] = None
        self._joint_velocities: Optional[List[float]] = None

        # for actions
        self._sacrum_joint_target: Optional[float] = None
        self._left_joint_targets: Optional[List[float]] = None
        self._right_joint_targets: Optional[List[float]] = None

        # === BC data buffer & saving config ===
        self._obs_dim: Optional[int] = None
        self._act_dim: Optional[int] = None
        self._obs_buffer: List[np.ndarray] = []
        self._act_buffer: List[np.ndarray] = []

        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self._baselink_translate_subscriber_ = self.create_subscription(
            Vector3,
            '/baselink/translate',
            self._baselink_translate_callback,
            qos_sensor
        )
        self._baselink_quat_subscriber_ = self.create_subscription(
            Quaternion,
            '/baselink/quat',
            self._baselink_quat_callback,
            qos_sensor
        )
        self._baselink_twist_subscriber_ = self.create_subscription(
            Twist,
            '/baselink/twist',
            self._baselink_twist_callback,
            qos_sensor
        )
        self._joint_state_subscriber_ = self.create_subscription(
            JointState,
            '/biped/jointState', # 11 joints expected back
            self._joint_state_callback, 
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
        self._left_joint_target_subscriber_ = self.create_subscription(
            Float64MultiArray,
            '/biped/left_joint_target',# 5 joints
            self._left_joint_target_callback,
            qos_sensor
        )
        self._right_joint_target_subscriber_ = self.create_subscription(
            Float64MultiArray,
            '/biped/right_joint_target',# 5 joints
            self._right_joint_target_callback,
            qos_sensor
        )
        self._counterweight_joint_targets_subscriber_ = self.create_subscription(
            Float64MultiArray,
            '/counterweight/joint_targets', # [back, sacrum]
            self._counterweight_joint_targets_callback,
            qos_sensor
        )
        self._timer = self.create_timer(TIMER_PERIOD_SEC, self.on_timer)

    def _baselink_translate_callback(self, msg: Vector3) -> None:
        self._p_W_baselink_z = msg.z

    def _baselink_quat_callback(self, msg: Quaternion) -> None:
        self._q_W_baselink = msg

    def _baselink_twist_callback(self, msg: Twist) -> None:
        self._twist_W_baselink = [
            msg.linear.x,
            msg.linear.y,
            msg.linear.z,
            msg.angular.x,
            msg.angular.y,
            msg.angular.z
        ]

    def _joint_state_callback(self, msg: JointState) -> None:
        self._joint_positions = list(msg.position) if msg.position else []
        self._joint_velocities = list(msg.velocity) if msg.velocity else []

    def _l_foot_translate_callback(self, msg: Vector3) -> None:
        self._p_W_l_foot_z = msg.z

    def _r_foot_translate_callback(self, msg: Vector3) -> None:
        self._p_W_r_foot_z = msg.z

    def _left_joint_target_callback(self, msg: Float64MultiArray) -> None:
        self._left_joint_targets = list(msg.data) if msg.data else []
    def _right_joint_target_callback(self, msg: Float64MultiArray) -> None:
        self._right_joint_targets = list(msg.data) if msg.data else []
    def _counterweight_joint_targets_callback(self, msg: Float64MultiArray) -> None:
        self._sacrum_joint_target = msg.data[1] if msg.data else None

    def on_timer(self) -> None:
        if not self._check_states_ready():
            print("Waiting for all state data to be ready...")
            return
        if not self._check_actions_ready():
            print("Waiting for all action data to be ready...")
            return
        if not self._check_data_in_episode(self._p_W_baselink_z):
            print("Not in episode, skipping data collection...")
            return

        # compose observations
        obs_list: list[float] = []
        ## 1 baselink z position
        obs_list.append(float(self._p_W_baselink_z))
        ## 2 baselink eular angles
        eular_angle_W_baselink = self._quaternion_to_euler(self._q_W_baselink)
        obs_list.extend([float(x) for x in eular_angle_W_baselink])
        ## 3 baselink twist (linear vel & angular vel)
        obs_list.extend([float(x) for x in self._twist_W_baselink])
        ## 4 11 joints positions
        obs_list.extend([float(x) for x in self._joint_positions])
        ## 5 11 joints velocities
        obs_list.extend([float(x) for x in self._joint_velocities])
        ## 6 left foot contact
        obs_list.append(self._check_foot_contact(self._p_W_l_foot_z))
        ## 7 right foot contact
        obs_list.append(self._check_foot_contact(self._p_W_r_foot_z))
        obs = np.asarray(obs_list, dtype=np.float32)

        # compose actions
        act_list: list[float] = []
        ## 1 sacrum joint target
        act_list.append(float(self._sacrum_joint_target))
        ## 2 left 5 joints targets
        act_list.extend([float(x) for x in self._left_joint_targets])
        ## 3 right 5 joints targets
        act_list.extend([float(x) for x in self._right_joint_targets])
        acts = np.asarray(act_list, dtype=np.float32)

        # record dimensions if first time
        if self._obs_dim is None:
            self._obs_dim = obs.shape[0]
        if self._act_dim is None:
            self._act_dim = acts.shape[0]

        # protect against dimension mismatch
        if obs.shape[0] != self._obs_dim or acts.shape[0] != self._act_dim:
            return

        # add in buffer
        self._obs_buffer.append(obs)
        self._act_buffer.append(acts)

        # if buffer full, flush to NPZ
        if len(self._obs_buffer) >= DATA_BUFFER_SIZE:
            self._flush_npz()

    def _flush_npz(self) -> None:
        if not self._obs_buffer:
            return
        obs_arr = np.stack(self._obs_buffer, axis=0).astype(np.float32)
        act_arr = np.stack(self._act_buffer, axis=0).astype(np.float32)

        if os.path.exists(SAVE_FILE_PATH):
            try:
                old = np.load(SAVE_FILE_PATH)
                old_obs = old['obs']
                old_act = old['actions']
                obs_out = np.concatenate([old_obs, obs_arr], axis=0)
                act_out = np.concatenate([old_act, act_arr], axis=0)
            except Exception:
                obs_out, act_out = obs_arr, act_arr
        else:
            obs_out, act_out = obs_arr, act_arr

        np.savez(SAVE_FILE_PATH, obs=obs_out, actions=act_out)
        self._obs_buffer.clear()
        self._act_buffer.clear()

    def destroy_node(self):
        try:
            self._flush_npz()
        finally:
            super().destroy_node()

    def _check_states_ready(self) -> bool:
        if (self._p_W_baselink_z is None or
            self._p_W_l_foot_z is None or
            self._p_W_r_foot_z is None or
            not self._twist_W_baselink or
            not self._joint_positions or
            not self._joint_velocities):
            return False
        return True
    
    def _check_actions_ready(self) -> bool:
        if (self._sacrum_joint_target is None or
            not self._left_joint_targets or
            not self._right_joint_targets):
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
    
    def _check_data_in_episode(self, baselink_z: float) -> bool:
        if BASELINK_HEIGHT_BOUND[0] < baselink_z < BASELINK_HEIGHT_BOUND[1]:
            return True
        return False

def main(args=None):

    rclpy.init(args=args)
    node = DataCollectNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
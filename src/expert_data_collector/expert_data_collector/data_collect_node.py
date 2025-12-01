import math
import os
from typing import Optional, List

import numpy as np
import rclpy
from collections import deque
from geometry_msgs.msg import Quaternion, Twist, Vector3
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState

TIMER_PERIOD_SEC = 0.05  #20 Hz

BASELINK_HEIGHT_BOUND = (0.0198, 0.0212) # must be consistent with IsaaclabRlEnvCfg
FOOT_CONTACT_THRESHOLD = 0.0014 # must be consistent with IsaaclabRlEnvCfg

PRE_STATE_QUEUE_LEN = 2
DIRTY_DATA_ROLLBACK_N = 30
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

        self._state_history: deque = deque(maxlen=PRE_STATE_QUEUE_LEN)

        # === episode state ===
        self._is_in_episode: bool = False

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
            if self._is_in_episode:
                print("State data lost, ending episode and rolling back.")
                self._rollback_buffers(DIRTY_DATA_ROLLBACK_N)
                self._is_in_episode = False
            self._clean_actions_data()
            return
        
        if not self._check_actions_ready():
            print("Waiting for all action data to be ready...")
            if self._is_in_episode:
                print("Action data lost, ending episode and rolling back.")
                self._rollback_buffers(DIRTY_DATA_ROLLBACK_N)
                self._is_in_episode = False
            return

        current_in_episode = self._check_data_in_episode(self._p_W_baselink_z)
        if not current_in_episode:
            if self._is_in_episode:
                print("rollback buffers...")
                self._rollback_buffers(DIRTY_DATA_ROLLBACK_N)
            self._is_in_episode = False
            self._clean_actions_data()
            self._state_history.clear()
            print("Not in episode, skipping data collection...")
            return
        
        if not self._is_in_episode:
            print("Starting new episode data collection...")
            self._is_in_episode = True
            self._state_history.clear()

        # compose observations
        current_state_list: list[float] = []
        ## 1 baselink z position
        current_state_list.append(float(self._p_W_baselink_z))
        ## 2 baselink eular angles
        eular_angle_W_baselink = self._quaternion_to_euler(self._q_W_baselink)
        current_state_list.extend([float(x) for x in eular_angle_W_baselink])
        ## 3 baselink twist (linear vel & angular vel)
        current_state_list.extend([float(x) for x in self._twist_W_baselink])
        ## 4 11 joints positions
        current_state_list.extend([float(x) for x in self._joint_positions])
        ## 5 11 joints velocities
        current_state_list.extend([float(x) for x in self._joint_velocities])
        ## 6 left foot contact
        current_state_list.append(self._check_foot_contact(self._p_W_l_foot_z))
        ## 7 right foot contact
        current_state_list.append(self._check_foot_contact(self._p_W_r_foot_z))
        current_state = np.asarray(current_state_list, dtype=np.float32)
        state_dim = current_state.shape[0]

        # update state history
        prev_states = list(self._state_history)
        missing_frames = 2 - len(prev_states)

        obs_parts = []
        if missing_frames > 0:
            zeros = np.zeros(state_dim, dtype=np.float32)
            for _ in range(missing_frames):
                obs_parts.append(zeros)

        obs_parts.extend(prev_states)
        obs_parts.append(current_state)
        final_obs = np.concatenate(obs_parts, axis=0)

        self._state_history.append(current_state)

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
            self._obs_dim = final_obs.shape[0]
        if self._act_dim is None:
            self._act_dim = acts.shape[0]

        # protect against dimension mismatch
        if final_obs.shape[0] != self._obs_dim or acts.shape[0] != self._act_dim:
            return

        # add in buffer
        self._obs_buffer.append(final_obs)
        self._act_buffer.append(acts)

        # if buffer full, flush to NPZ
        if len(self._obs_buffer) >= DATA_BUFFER_SIZE + DIRTY_DATA_ROLLBACK_N:
            print("flush into npz...")
            self._flush_npz()

    def _flush_npz(self) -> None:
        # check buffer validity
        if not self._obs_buffer:
            return
        if len(self._obs_buffer) <= DIRTY_DATA_ROLLBACK_N:
            print("Not enough clean data, skip flushing.")
            return
        
        # split dirty data and clean data
        data_to_flush_obs = self._obs_buffer[: -DIRTY_DATA_ROLLBACK_N]
        data_to_flush_act = self._act_buffer[: -DIRTY_DATA_ROLLBACK_N]
        data_to_keep_obs = self._obs_buffer[-DIRTY_DATA_ROLLBACK_N:]
        data_to_keep_act = self._act_buffer[-DIRTY_DATA_ROLLBACK_N:]

        obs_arr = np.stack(data_to_flush_obs, axis=0).astype(np.float32)
        act_arr = np.stack(data_to_flush_act, axis=0).astype(np.float32)

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
        print(f"Saved {obs_out.shape[0]} samples to {SAVE_FILE_PATH}")

        # reset buffers
        self._obs_buffer = data_to_keep_obs
        self._act_buffer = data_to_keep_act
        print(f"Flushed {len(obs_arr)} samples. "
              f"Kept {len(self._obs_buffer)} (n={DIRTY_DATA_ROLLBACK_N}) in buffer.")

    def destroy_node(self):
        try:
            self._flush_npz()
        finally:
            super().destroy_node()

    def _check_states_ready(self) -> bool:
        if (self._p_W_baselink_z is None or
            self._q_W_baselink is None or
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

    def _clean_actions_data(self) -> None:
        self._sacrum_joint_target = None
        self._left_joint_targets = None
        self._right_joint_targets = None

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

    def _rollback_buffers(self, n: int) -> None:
            if n <= 0:
                return
            num_in_buffer = len(self._obs_buffer)
            if num_in_buffer == 0:
                return
            if num_in_buffer < n:
                self._obs_buffer.clear()
                self._act_buffer.clear()
            else:
                self._obs_buffer = self._obs_buffer[:-n]
                self._act_buffer = self._act_buffer[:-n]

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
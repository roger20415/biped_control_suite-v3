from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Dict, Optional

import numpy as np
import rclpy
from geometry_msgs.msg import Quaternion, Vector3
from numpy.typing import NDArray
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float64MultiArray, String

from .config import Config, LegSide, SupportSide, VALID_LEG_SIDES, VALID_SUPPORT_SIDES, REQUIRED_P_W_KEYS, REQUIRED_Q_W_KEYS
from .ds_to_ss_manager import DSToSSManager
from .init_to_ss_manager import InitToSSManager
from .ss_to_ds_manager import SSToDSManager

TIMER_PERIOD: float = 0.05  # in seconds
INIT_TO_SS_DURATION: float = 1.0  # in seconds
SS_TO_DS_DURATION: float = 1.0  # in seconds
DS_TO_SS_DURATION: float = 1.0  # in seconds
WAIT_SACRUM_TIME: float = 1.3  # in seconds


class Phase(Enum):
    INIT_TO_SS = auto()
    SS_TO_DS = auto()
    DS_TO_SS = auto()


@dataclass
class PhaseHandlers:
    on_enter: Callable[[], None]
    on_step:  Callable[[], Optional[Phase]]


class BipedMotionPlannerNode(Node):
    def __init__(self):
        super().__init__('biped_motion_planner')
        self._on_timer_impl = self._on_timer_bootstrap
        self._timer = self.create_timer(TIMER_PERIOD, self._on_timer)
        self.init_to_ss_manager = InitToSSManager()
        self.ss_to_ds_manager = SSToDSManager()
        self.ds_to_ss_manager = DSToSSManager()
        self.stance_side: LegSide = "right"
        self.swing_side: LegSide = "left"
        self.support_side: SupportSide = "right"
        self._phase_start_time: float = 0.0
        self._clock_sin: float = np.sin(2.0 * np.pi * 0.75)
        self._clock_cos: float = np.cos(2.0 * np.pi * 0.75)

        self._current_phase: Phase = Phase.INIT_TO_SS
        self._total_step_idx: int = 0
        self._phase_step_idx: int = 0
        self._phase_duration_time: float = 0.0
        self._phase_time_budget: Dict[Phase, float] = {
            Phase.INIT_TO_SS: INIT_TO_SS_DURATION,
            Phase.SS_TO_DS: SS_TO_DS_DURATION,
            Phase.DS_TO_SS: DS_TO_SS_DURATION,
        }
        self._handlers: Dict[Phase, PhaseHandlers] = {
            Phase.INIT_TO_SS: PhaseHandlers(
                on_enter=self._enter_init_to_ss,
                on_step=self._step_init_to_ss
            ),
            Phase.SS_TO_DS: PhaseHandlers(
                on_enter=self._enter_ss_to_ds,
                on_step=self._step_ss_to_ds
            ),
            Phase.DS_TO_SS: PhaseHandlers(
                on_enter=self._enter_ds_to_ss,
                on_step=self._step_ds_to_ss
            ),
        }
        # in degrees
        self._next_stance_joint_pose: Optional[list[float]] = None
        # point in world frame
        self._next_swing_position: Optional[NDArray[np.float64]] = None

        self._p_W: dict[str, Optional[Vector3]] = {
            k: None for k in REQUIRED_P_W_KEYS}
        self._q_W: dict[str, Optional[Quaternion]] = {
            k: None for k in REQUIRED_Q_W_KEYS}
        # in degrees
        self._left_joint_targets: Optional[NDArray[np.float64]] = None
        # in degrees
        self._right_joint_targets: Optional[NDArray[np.float64]] = None
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
        self._baselink_quat_subscriber_ = self.create_subscription(
            Quaternion,
            '/baselink/quat',
            self._baselink_quat_callback,
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
        self._left_joint_targets_subscriber_ = self.create_subscription(
            Float64MultiArray,
            '/biped/left_joint_target',
            self._left_joint_targets_callback,
            qos_sensor  # in rad
        )
        self._right_joint_targets_subscriber_ = self.create_subscription(
            Float64MultiArray,
            '/biped/right_joint_target',
            self._right_joint_targets_callback,
            qos_sensor  # in rad
        )
        self._support_side_publisher_ = self.create_publisher(
            String,
            '/biped/support_side',
            10
        )
        self._stance_side_publisher_ = self.create_publisher(
            String,
            '/biped/stance_side',
            10
        )
        self._swing_side_publisher_ = self.create_publisher(
            String,
            '/biped/swing_side',
            10
        )
        self._stance_joint_targets_publisher_ = self.create_publisher(
            Float64MultiArray,
            '/biped/stance_joint_target',
            10
        )  # joint angles in degrees
        self._swing_target_publisher_ = self.create_publisher(
            Vector3,
            '/biped/swing_target',
            10
        )
        self._clock_sin_cos_publisher_ = self.create_publisher(
            Float64MultiArray,
            '/biped/clock_sin_cos',
            10
        )

    def _baselink_translate_callback(self, msg: Vector3) -> None:
        self._p_W["baselink"] = msg

    def _l_foot_translate_callback(self, msg: Vector3) -> None:
        self._p_W["l_foot"] = msg

    def _r_foot_translate_callback(self, msg: Vector3) -> None:
        self._p_W["r_foot"] = msg

    def _baselink_quat_callback(self, msg: Quaternion) -> None:
        self._q_W["baselink"] = msg

    def _l_foot_quat_callback(self, msg: Quaternion) -> None:
        self._q_W["l_foot"] = msg

    def _r_foot_quat_callback(self, msg: Quaternion) -> None:
        self._q_W["r_foot"] = msg

    def _left_joint_targets_callback(self, msg: Float64MultiArray) -> None:
        self._left_joint_targets = np.rad2deg(
            np.array(msg.data, dtype=np.float64))

    def _right_joint_targets_callback(self, msg: Float64MultiArray) -> None:
        self._right_joint_targets = np.rad2deg(
            np.array(msg.data, dtype=np.float64))

    def _pub_support_side(self) -> None:
        if self.support_side not in VALID_SUPPORT_SIDES:
            self.get_logger().warn(
                f"Support side is invalid: {self.support_side}")
            return
        msg = String()
        msg.data = self.support_side
        self._support_side_publisher_.publish(msg)

    def _pub_stance_side(self) -> None:
        if self.stance_side not in VALID_LEG_SIDES:
            self.get_logger().warn(
                f"Stance side is invalid: {self.stance_side}")
            return
        msg = String()
        msg.data = self.stance_side
        self._stance_side_publisher_.publish(msg)

    def _pub_swing_side(self) -> None:
        if self.swing_side not in VALID_LEG_SIDES:
            self.get_logger().warn(f"Swing side is invalid: {self.swing_side}")
            return
        msg = String()
        msg.data = self.swing_side
        self._swing_side_publisher_.publish(msg)

    def _pub_stance_joint_targets(self) -> None:
        if self._next_stance_joint_pose is None:
            self.get_logger().warn("next_stance_joint_pose is None")
            return
        msg = Float64MultiArray()
        msg.data = self._next_stance_joint_pose  # in degrees
        self._stance_joint_targets_publisher_.publish(msg)

    def _pub_swing_target(self) -> None:
        if self._next_swing_position is None:
            self.get_logger().warn("Next swing position is not set.")
            return
        msg = Vector3()
        msg.x = float(self._next_swing_position[0])
        msg.y = float(self._next_swing_position[1])
        msg.z = float(self._next_swing_position[2])
        self._swing_target_publisher_.publish(msg)

    def _pub_clock_sin_cos(self) -> None:
        msg = Float64MultiArray()
        msg.data = [self._clock_sin, self._clock_cos]
        self._clock_sin_cos_publisher_.publish(msg)

    def _on_timer(self) -> None:
        self._on_timer_impl()

    def _on_timer_bootstrap(self) -> None:
        self._pub_support_side()
        time.sleep(WAIT_SACRUM_TIME/2)
        self._handlers[self._current_phase].on_enter()
        self._on_timer_impl = self._on_timer_main

    def _on_timer_main(self) -> None:
        next_phase = self._handlers[self._current_phase].on_step()
        self._pub_support_side()
        self._pub_stance_side()
        self._pub_swing_side()
        self._pub_stance_joint_targets()
        self._pub_swing_target()
        self._total_step_idx += 1
        self._phase_step_idx += 1
        if next_phase is not None or self._phase_budget_reached():
            self._transition_to(next_phase or self._default_next_phase())

    def _phase_budget_reached(self) -> bool:
        budget = self._phase_time_budget.get(self._current_phase, None)
        return budget is not None and self._phase_duration_time >= budget

    def _default_next_phase(self) -> Phase:
        if self._current_phase == Phase.INIT_TO_SS:
            return Phase.SS_TO_DS
        if self._current_phase == Phase.SS_TO_DS:
            return Phase.DS_TO_SS
        if self._current_phase == Phase.DS_TO_SS:
            return Phase.SS_TO_DS

    def _transition_to(self, next_phase: Phase) -> None:
        if next_phase == self._current_phase:
            return
        self.get_logger().info(
            f'Transition: {self._current_phase.name} -> {next_phase.name} '
            f'(total_step={self._total_step_idx})'
        )
        if (self._current_phase, next_phase) == (Phase.SS_TO_DS, Phase.DS_TO_SS):
            self._switch_side()
            self._pub_support_side()
            time.sleep(WAIT_SACRUM_TIME)
        self._current_phase = next_phase
        self._phase_step_idx = 0
        self._handlers[self._current_phase].on_enter()

    def _enter_init_to_ss(self) -> None:
        self.get_logger().info('[ENTER] INIT_TO_SS')
        self.init_to_ss_manager.clear_phase_state()
        self.init_to_ss_manager.set_support_side(self.support_side)
        self.init_to_ss_manager.set_swing_side(self.swing_side)
        self.init_to_ss_manager.build_swing_func(self._p_W)
        self._update_clock(0.0)
        self._start_phase_timer()

    def _step_init_to_ss(self) -> Optional[Phase]:
        self._update_phase_timer()
        s_value = max(0.0, min(self._phase_duration_time /
                      self._phase_time_budget[Phase.INIT_TO_SS], 1.0))
        self._next_stance_joint_pose = [0.0]*Config.JOINT_NUMS
        self._next_swing_position = self.init_to_ss_manager.calc_swing_position(
            s_value)
        self._update_clock(s_value)
        # TODO check if reached the target
        return None

    def _enter_ss_to_ds(self) -> None:
        self.get_logger().info('[ENTER] SS_TO_DS')
        self.ss_to_ds_manager.clear_phase_state()
        self.ss_to_ds_manager.set_stance_side(self.stance_side)
        self.ss_to_ds_manager.set_swing_side(self.swing_side)
        self.ss_to_ds_manager.set_support_side(self.support_side)
        self.ss_to_ds_manager.build_stance_func()
        self.ss_to_ds_manager.build_swing_func(self._p_W, self._q_W)
        self._update_clock(0.0)
        self._start_phase_timer()

    def _step_ss_to_ds(self) -> Optional[Phase]:
        self._update_phase_timer()
        s_value = max(0.0, min(self._phase_duration_time /
                      self._phase_time_budget[Phase.SS_TO_DS], 1.0))
        self._next_stance_joint_pose = self.ss_to_ds_manager.calc_stance_joint_pose(
            s_value)
        self._next_swing_position = self.ss_to_ds_manager.calc_swing_position(
            s_value)
        self._update_clock(s_value)
        # TODO check if reached the target
        return None

    def _enter_ds_to_ss(self) -> None:
        self.get_logger().info('[ENTER] DS_TO_SS')
        self.ds_to_ss_manager.clear_phase_state()
        self.ds_to_ss_manager.set_stance_side(self.stance_side)
        self.ds_to_ss_manager.set_swing_side(self.swing_side)
        self.ds_to_ss_manager.set_support_side(self.support_side)
        if self._left_joint_targets is None or self._right_joint_targets is None:
            raise ValueError("Left or Right joint target is not yet received.")
        if self.stance_side == "left":
            self.ds_to_ss_manager.build_stance_func(self._left_joint_targets)
        elif self.stance_side == "right":
            self.ds_to_ss_manager.build_stance_func(self._right_joint_targets)
        self.ds_to_ss_manager.build_swing_func(self._p_W, self._q_W)
        self._update_clock(0.0)
        self._start_phase_timer()

    def _step_ds_to_ss(self) -> Optional[Phase]:
        self._update_phase_timer()
        s_value = max(0.0, min(self._phase_duration_time /
                      self._phase_time_budget[Phase.DS_TO_SS], 1.0))
        self._next_stance_joint_pose = self.ds_to_ss_manager.calc_stance_joint_pose(
            s_value)
        self._next_swing_position = self.ds_to_ss_manager.calc_swing_position(
            s_value)
        self._update_clock(s_value)
        # TODO check if reached the target
        return None

    def _start_phase_timer(self) -> None:
        self._phase_start_time = self.get_clock().now().nanoseconds * 1e-9
        self._phase_duration_time = 0.0

    def _update_phase_timer(self) -> None:
        now = self.get_clock().now()
        self._phase_duration_time = now.nanoseconds * 1e-9 - self._phase_start_time

    def _switch_side(self) -> None:
        self.stance_side, self.swing_side = self.swing_side, self.stance_side
        # TODO support side switch logic
        self.support_side = self.stance_side

    def _update_clock(self, s_value: float) -> None:
        phase_offset = self._cal_s_value_phase_offset()
        clock = phase_offset + s_value / 4.0
        self._clock_sin = np.sin(2.0 * np.pi * clock)
        self._clock_cos = np.cos(2.0 * np.pi * clock)

    def _cal_s_value_phase_offset(self) -> float:
        phase_offset: float = 0.0
        if self._current_phase == Phase.INIT_TO_SS:
            phase_offset = 0.75
        elif self._current_phase == Phase.SS_TO_DS and self.swing_side == "left":
            phase_offset = 0.0
        elif self._current_phase == Phase.DS_TO_SS and self.swing_side == "right":
            phase_offset = 0.25
        elif self._current_phase == Phase.SS_TO_DS and self.swing_side == "right":
            phase_offset = 0.5
        elif self._current_phase == Phase.DS_TO_SS and self.swing_side == "left":
            phase_offset = 0.75
        return phase_offset
        

def main(args=None):
    rclpy.init(args=args)
    node = BipedMotionPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

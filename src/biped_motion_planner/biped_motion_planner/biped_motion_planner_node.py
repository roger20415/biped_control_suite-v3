from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Dict, Optional

import numpy as np
import rclpy
from numpy.typing import NDArray
from rclpy.node import Node

from .config import Config, LegSide, SupportSide, VALID_LEG_SIDES, VALID_SUPPORT_SIDES
from .ss_to_ds_manager import SSTODSManager

TIMER_PERIOD: float = 0.05 # in seconds
INIT_TO_SS_DURATION: float = 5.0 # in seconds
SS_TO_DS_DURATION: float = 5.0 # in seconds
DS_TO_SS_DURATION: float = 5.0 # in seconds


class Phase(Enum):
    INIT_TO_SS = auto()
    SS_TO_DS   = auto()
    DS_TO_SS   = auto()


@dataclass
class PhaseHandlers:
    on_enter: Callable[['BipedMotionPlannerNode'], None]
    on_step:  Callable[['BipedMotionPlannerNode'], Optional[Phase]]


class BipedMotionPlannerNode(Node):
    def __init__(self):
        super().__init__('biped_motion_planner')
        self._timer = self.create_timer(TIMER_PERIOD, self._on_timer)
        self.ss_to_ds_manager = SSTODSManager()
        self.stance_side: LegSide = "undefined"
        self.swing_side: LegSide = "undefined"
        self.support_side: SupportSide = "undefined"
        self._phase_start_time: float = 0.0

        self._current_phase: Phase = Phase.INIT_TO_SS
        self._total_step_idx: int = 0
        self._phase_step_idx: int = 0
        self._phase_duration_time: float = 0.0
        self._phase_time_budget: Dict[Phase, float] =  {
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

        self._next_stance_alpha: float = 0.0
        self._next_swing_position: Optional[NDArray[np.float64]] = None

        self._handlers[self._current_phase].on_enter(self)

    def _on_timer(self) -> None:
        next_phase = self._handlers[self._current_phase].on_step(self)
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
        self._current_phase = next_phase
        self._phase_step_idx = 0
        self._handlers[self._current_phase].on_enter(self)

    def _enter_init_to_ss(self) -> None:
        self.get_logger().info('[ENTER] INIT_TO_SS')

    def _step_init_to_ss(self) -> Optional[Phase]:
        return None

    def _enter_ss_to_ds(self) -> None:
        self.get_logger().info('[ENTER] SS_TO_DS')
        self._switch_stance_and_swing()
        self.ss_to_ds_manager.clear_phase_state()
        self.ss_to_ds_manager.set_stance_side(self.stance_side)
        self.ss_to_ds_manager.set_swing_side(self.swing_side)
        self.ss_to_ds_manager.set_support_side(self.support_side)
        self.ss_to_ds_manager.build_stance_of_s()
        self.ss_to_ds_manager.build_swing_of_s(self._p_W, self._q_W)
        self._start_phase_timer()

    def _step_ss_to_ds(self) -> Optional[Phase]:
        s_value = max(0.0, min(self._phase_duration_time / self._phase_time_budget[Phase.SS_TO_DS], 1.0))
        self._next_stance_alpha = self.ss_to_ds_manager.calc_stance_alpha(s_value)
        self._next_swing_position = self.ss_to_ds_manager.calc_swing_position(s_value)
        # TODO publish
        # TODO check if reached the target
        return None

    def _enter_ds_to_ss(self) -> None:
        self.get_logger().info('[ENTER] DS_TO_SS')

    def _step_ds_to_ss(self) -> Optional[Phase]:
        return None
    
    def _start_phase_timer(self) -> None:
        self._phase_start_time = self.get_clock().now().nanoseconds * 1e-9
        self._phase_duration_time = 0.0

    def _update_phase_timer(self) -> None:
        now = self.get_clock().now()
        self._phase_duration_time = now.nanoseconds * 1e-9 - self._phase_start_time
    
    def _switch_stance_and_swing(self) -> None:
        self.stance_side, self.swing_side = self.swing_side, self.stance_side

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

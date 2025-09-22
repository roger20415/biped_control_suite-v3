from typing import Mapping, Optional

import numpy as np
import sympy as sp
from geometry_msgs.msg import Quaternion, Vector3
from numpy.typing import NDArray

from .config import Config, SupportSide, VALID_SUPPORT_SIDES


class InitToSSManager:
    def __init__(self):
        self._support_side: SupportSide = "undefined"
        self._swing_of_s: Optional[NDArray[object]] = None
    
    def set_support_side(self, side: SupportSide) -> None:
        if side not in VALID_SUPPORT_SIDES:
            raise ValueError("Invalid support side.")
        self._support_side = side

    def build_swing_of_s(self, p_W: Mapping[str, Vector3], q_W: Mapping[str, Quaternion]) -> None:
        if not self._if_side_defined():
            raise ValueError("support side is undefined.")
        self._swing_of_s = self._build_swing_of_s()

    def calc_swing_position(self, s_value: float) -> NDArray[np.float64]:
        if self._swing_of_s is None:
            raise ValueError("Swing trajectory is not yet built.")
        if s_value < 0.0 or s_value > 1.0:
            raise ValueError("s_value must be between 0 and 1.")
        position = np.empty(3, dtype=np.float64)
        for i in range(3):
            position[i] = float(self._swing_of_s[i].evalf(subs={'s': s_value}))
        return position

    def clear_phase_state(self) -> None:
        self._support_side = "undefined"
        self._swing_of_s = None
    
    def _if_side_defined(self) -> bool:
        return self._support_side in VALID_SUPPORT_SIDES

    def _build_swing_of_s(self) -> NDArray[object]:
        s = sp.symbols('s', real=True)
        swing_of_s = np.array([
            sp.Integer(0), 
            sp.Integer(0), 
            sp.simplify(Config.SS_SWING_FOOT_HEIGHT * s)
        ], dtype=object)
        return swing_of_s
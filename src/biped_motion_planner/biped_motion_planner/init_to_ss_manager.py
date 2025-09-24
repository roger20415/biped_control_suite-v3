import numpy as np
import sympy as sp
from geometry_msgs.msg import Vector3
from numpy.typing import NDArray

from .config import Config, LegSide, SupportSide, VALID_SUPPORT_SIDES, VALID_LEG_SIDES


class InitToSSManager:
    def __init__(self):
        self._support_side: SupportSide = "undefined"
        self._swing_side: LegSide = "undefined"
        self._swing_func = None
        self._s = sp.symbols('s', real=True)

    def set_support_side(self, side: SupportSide) -> None:
        if side not in VALID_SUPPORT_SIDES:
            raise ValueError("Invalid support side.")
        self._support_side = side

    def set_swing_side(self, side: LegSide) -> None:
        if side not in VALID_LEG_SIDES:
            raise ValueError("Invalid leg side.")
        self._swing_side = side

    def build_swing_func(self, p_W: dict[str, Vector3]) -> None:
        if self._swing_side == "left":
            p_W_foot = p_W["l_foot"]
        elif self._swing_side == "right":
            p_W_foot = p_W["r_foot"]
        swing_of_s = self._build_swing_of_s(p_W_foot)
        self._swing_func = [sp.lambdify(self._s, e, 'numpy')
                            for e in swing_of_s]

    def calc_swing_position(self, s_value: float) -> np.ndarray:
        if self._swing_func is None:
            raise ValueError("Swing trajectory is not yet built.")
        s_val = float(s_value)
        return np.array([float(f(s_val)) for f in self._swing_func], dtype=np.float64)

    def clear_phase_state(self) -> None:
        self._support_side = "undefined"
        self._swing_side = "undefined"
        self._swing_func = None

    def _if_side_defined(self) -> bool:
        return self._support_side in VALID_SUPPORT_SIDES and self._swing_side in VALID_LEG_SIDES

    def _build_swing_of_s(self, p_W_foot: Vector3) -> NDArray[object]:
        height = sp.Float(float(Config.SS_SWING_FOOT_HEIGHT))
        return np.array([sp.Float(p_W_foot.x), sp.Float(p_W_foot.y), height * self._s], dtype=object)

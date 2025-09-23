from typing import Mapping, Optional

import numpy as np
import sympy as sp
from geometry_msgs.msg import Quaternion, Vector3
from numpy.typing import NDArray

from .config import Config, LegSide, SupportSide, VALID_LEG_SIDES, VALID_SUPPORT_SIDES, REQUIRED_P_W_KEYS, REQUIRED_Q_W_KEYS
from .linear_algebra_utils import LinearAlgebraUtils

class DSToSSManager:
    def __init__(self):
        self._s = sp.symbols('s', real=True)
        self._stance_side: LegSide = "undefined"
        self._swing_side: LegSide = "undefined"
        self._support_side: SupportSide = "undefined"
        self._stance_func = None
        self._swing_func = None

    def set_stance_side(self, side: LegSide) -> None:
        if side not in VALID_LEG_SIDES:
            raise ValueError("Invalid leg side.")
        self._stance_side = side

    def set_swing_side(self, side: LegSide) -> None:
        if side not in VALID_LEG_SIDES:
            raise ValueError("Invalid leg side.")
        self._swing_side = side
    
    def set_support_side(self, side: SupportSide) -> None:
        if side not in VALID_SUPPORT_SIDES:
            raise ValueError("Invalid support side.")
        self._support_side = side
        # TODO add support side logic

    def build_stance_func(self, current_joint_target: NDArray[np.float64]) -> None:
        raw_stance_of_s = (1 - self._s) * current_joint_target
        stance_of_s = np.empty(Config.JOINT_NUMS, dtype=object)
        for i in range(Config.JOINT_NUMS):
            stance_of_s[i] = sp.simplify(raw_stance_of_s[i])
        self._stance_func = [sp.lambdify(self._s, expr, 'numpy') for expr in stance_of_s]

    def calc_stance_joint_pose(self, s_value: float) -> list[float]:
        if self._stance_func is None:
            raise ValueError("Stance trajectory is not yet built.")
        s_value = float(np.clip(s_value, 0.0, 1.0))
        next_joint_pose = [float(f(s_value)) for f in self._stance_func]
        return next_joint_pose

    def build_swing_func(self, p_W: Mapping[str, Vector3], q_W: Mapping[str, Quaternion]) -> None:
        if not self._if_subscribe_data_ready(p_W, q_W):
            raise ValueError("Subscribed data is not ready.")
        if not self._if_side_defined():
            raise ValueError("Leg sides are not properly defined.")
        if self._swing_side == "left":
            p_W_swingFoot = p_W["l_foot"]
            q_W_swingFoot = q_W["l_foot"]
            p_W_stanceFoot = p_W["r_foot"]
        elif self._swing_side == "right":
            p_W_swingFoot = p_W["r_foot"]
            q_W_swingFoot = q_W["r_foot"]
            p_W_stanceFoot = p_W["l_foot"]
        else:
            raise ValueError("Swing side is undefined.")
        zSwingFoot_W_norm = self._calc_zFOOT_W_norm(q_W_swingFoot)
        yB_W_norm = self._calc_yB_W_norm(q_W["baselink"])
        swing_start = self._calc_swing_start(p_W_swingFoot, zSwingFoot_W_norm)
        swing_end = self._calc_swing_end(p_W_stanceFoot, yB_W_norm)
        swing_of_s = self._build_swing_of_s(swing_start, swing_end)
        self._swing_func = [sp.lambdify(self._s, expr, 'numpy') for expr in swing_of_s]
        
    def calc_swing_position(self, s_value: float) -> NDArray[np.float64]:
        if self._swing_func is None:
            raise ValueError("Swing trajectory is not yet built.")
        if not (0.0 <= s_value <= 1.0):
            raise ValueError("s_value must be between 0 and 1.")
        return np.array([float(f(s_value)) for f in self._swing_func], dtype=np.float64)

    def clear_phase_state(self) -> None:
        self._stance_side = "undefined"
        self._swing_side = "undefined"
        self._support_side = "undefined"
        self._stance_func = None
        self._swing_func = None

    def _if_subscribe_data_ready(self, p_W: Mapping[str, Vector3], q_W: Mapping[str, Quaternion]) -> bool:
        for key in REQUIRED_P_W_KEYS:
            if p_W.get(key) is None:
                return False
        for key in REQUIRED_Q_W_KEYS:
            if q_W.get(key) is None:
                return False
        return True
    
    def _if_side_defined(self) -> bool:
        return self._stance_side in VALID_LEG_SIDES and self._swing_side in VALID_LEG_SIDES and self._support_side in VALID_SUPPORT_SIDES

    def _calc_zFOOT_W_norm(self, q_W_foot: Quaternion) -> NDArray[np.float64]:
        R_W_FOOT: NDArray[np.float64] = LinearAlgebraUtils.quaternion_to_rotation_matrix(q_W_foot)
        zFOOT_W = R_W_FOOT[:, 2]
        return LinearAlgebraUtils.normalize_vec(zFOOT_W)

    def _calc_yB_W_norm(self, q_W_baselink: Quaternion) -> NDArray[np.float64]:
        R_WB: NDArray[np.float64] = LinearAlgebraUtils.quaternion_to_rotation_matrix(q_W_baselink)
        yB_W = R_WB[:, 1]
        return LinearAlgebraUtils.normalize_vec(yB_W)

    def _calc_swing_start(self, p_W_swingFoot: Vector3, zSwingFoot_W_norm: NDArray[np.float64]) -> NDArray[np.float64]:
        p_W_swingFoot = np.array([p_W_swingFoot.x, p_W_swingFoot.y, p_W_swingFoot.z], dtype=float)
        swing_start = p_W_swingFoot - Config.FOOT_LEN*zSwingFoot_W_norm
        return swing_start # foot bottom
    
    def _calc_swing_end(self, p_W_stanceFoot: Vector3, yB_W_norm: NDArray[np.float64]) -> NDArray[np.float64]:
        p_W_stanceFoot = np.array([p_W_stanceFoot.x, p_W_stanceFoot.y, p_W_stanceFoot.z], dtype=float)
        foot_to_center_distance = Config.ORIGIN_L_TARGET.y
        if self._swing_side == "left":
            swing_end = p_W_stanceFoot + yB_W_norm*(2*foot_to_center_distance)
        elif self._swing_side == "right":
            swing_end = p_W_stanceFoot - yB_W_norm*(2*foot_to_center_distance)
        else:
            raise ValueError("Swing side is undefined.")
        swing_end[2] = Config.SS_SWING_FOOT_HEIGHT
        return swing_end # foot bottom
    
    def _build_swing_of_s(self, swing_start: NDArray[np.float64], swing_end: NDArray[np.float64]) -> NDArray[object]:
        raw_swing_of_s = swing_start*(1 - self._s) + swing_end*self._s
        swing_of_s = np.empty(3, dtype=object)
        for i in range(3):
            swing_of_s[i] = sp.simplify(raw_swing_of_s[i])
        return swing_of_s
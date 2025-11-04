import numpy as np
import warnings
from geometry_msgs.msg import Quaternion, Vector3
from numpy.typing import NDArray
from typing import Literal, Mapping, Optional
from .config import Config
from .linear_algebra_utils import LinearAlgebraUtils
from .trigonometric_utils import TrigonometricUtils

LegSide = Literal["left", "right", "undefined"]


class JointTargetsCalculator():
    def __init__(self):
        self.leg_side: LegSide = "undefined"
        self.p_W: dict[str, Vector3] = {}  # points in the World frame
        self.p_B: dict[str, Vector3] = {}  # points in the Baselink frame
        self.p_L: dict[str, Vector3] = {}  # points in the Leg frame
        # points in the (u,w) coordinates in the Leg frame
        self.p_uw: dict[str, NDArray[np.float64]] = {}
        # +u-axis rotation in the Leg frame (in degrees)
        self.joint_phi: dict[str, float] = {}
        # -v-axis rotation in the Leg frame (in degrees)
        self.joint_theta: dict[str, float] = {}
        # final joint angle commands (in radians)
        self.joint_targets_rad: dict[str, float] = {}

    def calc_joint_targets(self, p_W: Mapping[str, Vector3], q_W_baselink: Quaternion, leg_side: str) -> tuple[bool, dict[str, float]]:
        self.leg_side: LegSide = leg_side
        self.p_W = dict(p_W)
        self.p_W["foot"] = Vector3(x=p_W["target"].x,
                                   y=p_W["target"].y,
                                   z=p_W["target"].z + Config.FOOT_LEN)
        R_WB, T_BW = self._calc_WB_transforms(
            q_W_baselink, self.p_W["baselink"])
        self.p_B.update(self._transform_points_World_to_Baselink(T_BW))
        R_BL, T_LB = self._calc_BL_transforms()
        self.p_L.update(self._transform_points_Baselink_to_Leg(T_LB))
        self.p_uw.update(self._transform_points_Leg_to_uw())
        self.joint_phi["leg"] = self._calc_phi_BL()
        e_L_proj = self._project_gravity_to_uw_plane(R_WB.T, R_BL.T)
        self.p_uw["ankle"] = self._calc_p_uw_ankle(e_L_proj)
        self.p_uw["thigh"] = np.array([0.0, -Config.HIP_LEN])
        self.joint_theta["calf"], thigh_to_ankle_vec_uw, self.p_uw["ankle"], hold_prev_pose = self._calc_theta_calf()
        if hold_prev_pose is True:
            return hold_prev_pose, None
        self.joint_theta["thigh"] = self._calc_theta_thigh(
            thigh_to_ankle_vec_uw)
        self.joint_theta["ankle"] = self._calc_theta_ankle(e_L_proj)
        self.joint_phi["foot"], hold_prev_pose = self._calc_phi_foot(
            R_WB, R_BL, e_L_proj)
        if hold_prev_pose is True:
            return hold_prev_pose, None
        self.joint_targets_rad.update(self._calc_and_clamp_joint_targets_rad())
        return hold_prev_pose, self.joint_targets_rad

    def _transform_points_World_to_Baselink(self, T_BW: NDArray[np.float64]) -> dict[str, Vector3]:
        p_B_hip = LinearAlgebraUtils.transform_point(T_BW, self.p_W["hip"])
        p_B_foot = LinearAlgebraUtils.transform_point(T_BW, self.p_W["foot"])
        return {"hip": p_B_hip, "foot": p_B_foot}

    def _calc_WB_transforms(self,
                            q_W_baselink: Quaternion,
                            p_W_baselink: Vector3) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        R_WB = LinearAlgebraUtils.quaternion_to_rotation_matrix(q_W_baselink)
        T_WB = LinearAlgebraUtils.combine_transformation_matrix(
            R_WB, p_W_baselink)
        T_BW = LinearAlgebraUtils.invert_transformation_matrix(T_WB)
        return R_WB, T_BW

    def _transform_points_Baselink_to_Leg(self, T_LB: NDArray[np.float64]) -> dict[str, Vector3]:
        p_L_foot = LinearAlgebraUtils.transform_point(T_LB, self.p_B["foot"])
        return {"foot": p_L_foot}

    def _calc_BL_transforms(self) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        R_BL = self._calc_R_BL()
        T_BL = LinearAlgebraUtils.combine_transformation_matrix(
            R_BL, self.p_B["hip"])
        T_LB = LinearAlgebraUtils.invert_transformation_matrix(T_BL)
        return R_BL, T_LB

    def _calc_R_BL(self) -> NDArray[np.float64]:
        """
        The Leg frame must follow the following three rules:
        1. The origin is located at the hip joint.
        2. The u-axis is parallel to the x-axis of the Baselink frame.
        3. The uw-plane passes through the foot point.
        """
        foot_to_hip_vec = Vector3(x=self.p_B["hip"].x - self.p_B["foot"].x,
                                  y=self.p_B["hip"].y - self.p_B["foot"].y,
                                  z=self.p_B["hip"].z - self.p_B["foot"].z)
        # length of the original w vector
        norm_w = np.linalg.norm(
            np.array([foot_to_hip_vec.y, foot_to_hip_vec.z]))
        if norm_w < 1e-12:
            raise ValueError("Cannot normalize zero-length vector")
        u = np.array([1.0, 0.0, 0.0])
        w = np.array([0.0, foot_to_hip_vec.y/norm_w, foot_to_hip_vec.z/norm_w])
        v = -np.cross(u, w)

        return np.column_stack((u, v, w))

    def _transform_points_Leg_to_uw(self) -> dict[str, NDArray[np.float64]]:
        p_uw_foot = np.array([self.p_L["foot"].x, self.p_L["foot"].z])
        return {"foot": p_uw_foot}

    def _calc_phi_BL(self) -> float:
        delta_y = self.p_B["hip"].y - self.p_B["foot"].y
        delta_z = self.p_B["hip"].z - self.p_B["foot"].z
        phi_BL = np.degrees(np.arctan2(delta_y, delta_z))
        phi_BL = TrigonometricUtils.normalize_angle_to_180(phi_BL)
        return phi_BL

    def _project_gravity_to_uw_plane(self, R_BW: NDArray[np.float64], R_LB: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Returns the projection of World -Z (gravity) onto the Leg's uw-plane,
        expressed as (u, v, w) scalar components.
        """
        e_W = np.array([0.0, 0.0, -1.0])  # gravity vector in World frame
        e_B = R_BW @ e_W
        R_BL = R_LB.T
        nhat_B_uw = -R_BL[:, 1]

        P = (np.eye(3) - np.outer(nhat_B_uw, nhat_B_uw)) @ e_B
        norm_P = np.linalg.norm(P)

        if norm_P < 1e-12:
            warnings.warn(
                "Gravity is parallel to the uw-plane normal; projection is degenerated. Returning -w_L.",
                category=RuntimeWarning,
                stacklevel=2,
            )
            return np.array([0, 0, -1])

        e_B_proj = P / norm_P
        e_L_proj = R_LB @ e_B_proj
        return e_L_proj

    def _calc_p_uw_ankle(self, e_L_proj: NDArray[np.float64]) -> NDArray[np.float64]:
        e_uw_proj: NDArray[np.float64] = e_L_proj[[0, 2]]
        e_uw_proj_norm = LinearAlgebraUtils.normalize_vec(e_uw_proj)
        d_uw_ankle = Config.ANKLE_LEN * e_uw_proj_norm
        return self.p_uw["foot"] - d_uw_ankle

    def _calc_theta_calf(self) -> tuple[Optional[float], Optional[NDArray[np.float64]], Optional[NDArray[np.float64]], bool]:
        """
        Compute the calf (knee) angle in degrees and (optionally) a clamped ankle point.

        Returns: (theta_calf_deg, p_uw_ankle_new, hold_prev_pose)
        - theta_calf_deg: knee angle [deg] (None if caller should hold previous pose)
        - thigh_to_ankle_vec_uw: vector from thigh to ankle in uw-plane (None if holding previous pose)
        - p_uw_ankle_new: ankle (u,w) after clamping (None if holding previous pose)
        - hold_prev_pose: True → do not update this cycle; reuse previous command

        Behavior:
        • Too far (‖ankle-thigh‖ ≥ L1+L2 - EPS):
            warn; return (0.0, ankle clamped to radius L1+L2 along thigh→ankle, False)
        • Too close (‖ankle-thigh‖ ≤ |L1-L2| + EPS):
            warn; return (None, None, True)
        • Reachable:
            use law of cosines; clamp cos to [-1,1]; map to human-knee sign; normalize to [-180,180].
            If angle outside [CALF_MIN_DEG - EPS, CALF_MAX_DEG + EPS]:
                warn; return (None, None, True)
            else:
                return (theta_calf, original ankle, False)

        Notes: angles are degrees; EPS is a small tolerance; human-like knee uses negative bend.
        """
        # sign = -1 for forward-facing knee (human-like); sign = 1 for backward-facing knee (dog-like)
        EPS = 1e-12
        SIGN = np.sign(-1)
        thigh_to_ankle_vec_uw = self.p_uw["ankle"] - self.p_uw["thigh"]
        thigh_to_ankle_distance = np.linalg.norm(thigh_to_ankle_vec_uw)

        if thigh_to_ankle_distance >= (Config.THIGH_LEN + Config.CALF_LEN)-EPS:
            warnings.warn(
                "Ankle is too far from hip. Cannot reach target",
                category=RuntimeWarning,
                stacklevel=2,
            )
            theta_calf = 0.0
            thigh_to_ankle_vec_uw_norm = thigh_to_ankle_vec_uw / thigh_to_ankle_distance
            p_uw_ankle_new = self.p_uw["thigh"] + thigh_to_ankle_vec_uw_norm * (
                Config.THIGH_LEN + Config.CALF_LEN)
            thigh_to_ankle_vec_uw = p_uw_ankle_new - self.p_uw["thigh"]
            return theta_calf, thigh_to_ankle_vec_uw, p_uw_ankle_new, False

        elif thigh_to_ankle_distance <= abs(Config.THIGH_LEN - Config.CALF_LEN)+EPS:
            warnings.warn(
                "Ankle is too close to hip. Cannot reach target",
                category=RuntimeWarning,
                stacklevel=2,
            )
            return None, None, None, True

        # gamma is the angle between thigh link and calf link
        cos_gamma = (Config.THIGH_LEN**2 + Config.CALF_LEN**2 -
                     thigh_to_ankle_distance**2) / (2 * Config.THIGH_LEN * Config.CALF_LEN)
        cos_gamma = TrigonometricUtils.clamp_cos(cos_gamma)
        gamma = SIGN*np.degrees(np.arccos(cos_gamma))
        theta_calf = 180.0 - gamma
        theta_calf = TrigonometricUtils.normalize_angle_to_180(theta_calf)

        if (theta_calf < Config.CALF_MIN_DEG-EPS) or (theta_calf > Config.CALF_MAX_DEG+EPS):
            warnings.warn(
                f"theta_calf={theta_calf} exceeds +{Config.CALF_MAX_DEG} to -{Config.CALF_MIN_DEG} degree. Hold previous pose.",
                category=RuntimeWarning,
                stacklevel=2,
            )
            return None, None, None, True

        return theta_calf, thigh_to_ankle_vec_uw, self.p_uw["ankle"], False

    def _calc_theta_thigh(self, thigh_to_ankle_vec_uw: NDArray[np.float64]) -> float:
        alpha_thigh_to_ankle_vec_uw_rad = np.arctan2(
            thigh_to_ankle_vec_uw[1], thigh_to_ankle_vec_uw[0])
        theta_calf_rad = np.deg2rad(self.joint_theta["calf"])
        alpha_thigh_link_to_thigh_to_ankle_vec_uw_rad = np.arctan2(Config.CALF_LEN*np.sin(theta_calf_rad),
                                                                   Config.THIGH_LEN + Config.CALF_LEN*np.cos(theta_calf_rad))
        alpha_thigh_to_ankle_vec_uw = np.degrees(
            alpha_thigh_to_ankle_vec_uw_rad)
        alpha_thigh_link_to_thigh_to_ankle_vec_uw = np.degrees(
            alpha_thigh_link_to_thigh_to_ankle_vec_uw_rad)
        alpha_thigh = alpha_thigh_to_ankle_vec_uw - \
            alpha_thigh_link_to_thigh_to_ankle_vec_uw
        theta_thigh = alpha_thigh - Config.HIP_THETA_UW
        theta_thigh = TrigonometricUtils.normalize_angle_to_180(theta_thigh)

        return theta_thigh

    def _calc_theta_ankle(self, e_L_proj: NDArray[np.float64]) -> float:
        e_uw_proj = e_L_proj[[0, 2]]
        alpha_thigh = self.joint_theta["thigh"] + Config.HIP_THETA_UW
        alpha_calf = alpha_thigh + self.joint_theta["calf"]
        alpha_ankle_rad = np.arctan2(e_uw_proj[1], e_uw_proj[0])
        alpha_ankle = np.degrees(alpha_ankle_rad)
        theta_ankle = alpha_ankle - alpha_calf
        theta_ankle = TrigonometricUtils.normalize_angle_to_180(theta_ankle)
        return theta_ankle

    def _calc_phi_foot(self, R_WB: NDArray[np.float64], R_BL: NDArray[np.float64], e_L_proj: NDArray[np.float64]) -> tuple[float, bool]:
        z_W = np.array([0, 0, 1], dtype=np.float64)
        w_B = R_BL[:, 2]
        w_W = R_WB @ w_B
        if e_L_proj[2] > 0:
            raise ValueError("e_L_proj.w must < 0")
        elif e_L_proj[2] == 0.0:
            warnings.warn(
                "e_L_proj.w is zero; foot yaw is degenerated. Hold previous pose.",
                category=RuntimeWarning,
                stacklevel=2,
            )
            hold_prev_pose = True
            return None, hold_prev_pose
        a_L_foot = np.array([-e_L_proj[2], 0, e_L_proj[0]], dtype=np.float64)
        a_W_foot = R_WB @ R_BL @ a_L_foot

        # When the uw plane rotates in the negative direction relative to the Baselink xz-plane
        # Then the foot joint should rotate in the positive direction.
        sign = np.sign(-np.dot(np.cross(z_W, w_W), a_W_foot))
        phi_foot_rad = sign * \
            np.arccos(np.dot(w_W, z_W) /
                      (np.linalg.norm(w_W)*np.linalg.norm(z_W)))
        phi_foot = np.degrees(phi_foot_rad)
        phi_foot = TrigonometricUtils.normalize_angle_to_180(phi_foot)
        return phi_foot, False

    def _calc_and_clamp_joint_targets_rad(self) -> dict[str, float]:
        joint_targets_deg: dict[str, float] = {}
        joint_targets_rad: dict[str, float] = {}
        if self.leg_side == "left":
            joint_targets_deg["hip"] = TrigonometricUtils.clip_deg(
                self.joint_phi["leg"], Config.L_HIP_MIN_DEG, Config.L_HIP_MAX_DEG, "hip", "left")
            joint_targets_deg["thigh"] = TrigonometricUtils.clip_deg(
                self.joint_theta["thigh"], Config.THIGH_MIN_DEG, Config.THIGH_MAX_DEG, "thigh", "left")
            joint_targets_deg["calf"] = TrigonometricUtils.clip_deg(
                -self.joint_theta["calf"], Config.CALF_MIN_DEG, Config.CALF_MAX_DEG, "calf", "left")
            joint_targets_deg["ankle"] = TrigonometricUtils.clip_deg(
                -self.joint_theta["ankle"], Config.ANKLE_MIN_DEG, Config.ANKLE_MAX_DEG, "ankle", "left")
            joint_targets_deg["foot"] = TrigonometricUtils.clip_deg(
                -self.joint_phi["foot"], Config.FOOT_MIN_DEG, Config.FOOT_MAX_DEG, "foot", "left")

        elif self.leg_side == "right":
            joint_targets_deg["hip"] = TrigonometricUtils.clip_deg(
                self.joint_phi["leg"], Config.R_HIP_MIN_DEG, Config.R_HIP_MAX_DEG, "hip", "right")
            joint_targets_deg["thigh"] = TrigonometricUtils.clip_deg(
                -self.joint_theta["thigh"], Config.THIGH_MIN_DEG, Config.THIGH_MAX_DEG, "thigh", "right")
            joint_targets_deg["calf"] = TrigonometricUtils.clip_deg(
                self.joint_theta["calf"], Config.CALF_MIN_DEG, Config.CALF_MAX_DEG, "calf", "right")
            joint_targets_deg["ankle"] = TrigonometricUtils.clip_deg(
                self.joint_theta["ankle"], Config.ANKLE_MIN_DEG, Config.ANKLE_MAX_DEG, "ankle", "right")
            joint_targets_deg["foot"] = TrigonometricUtils.clip_deg(
                -self.joint_phi["foot"], Config.FOOT_MIN_DEG, Config.FOOT_MAX_DEG, "foot", "right")

        else:
            raise ValueError(f"Leg side is undefined. Got '{self.leg_side}'")

        for joint, target_deg in joint_targets_deg.items():
            joint_targets_rad[joint] = np.deg2rad(target_deg)

        return joint_targets_rad

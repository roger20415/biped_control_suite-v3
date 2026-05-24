from geometry_msgs.msg import Vector3
from typing import Literal, TypeAlias

LegSide: TypeAlias = Literal["left", "right", "undefined"]
SupportSide: TypeAlias = Literal["left", "right", "mid", "undefined"]
VALID_LEG_SIDES: tuple[str, ...] = ("left", "right")
VALID_SUPPORT_SIDES: tuple[str, ...] = ("left", "right", "mid")
REQUIRED_P_W_KEYS: tuple[str, ...] = ("baselink", "l_foot", "r_foot")
REQUIRED_Q_W_KEYS: tuple[str, ...] = ("baselink", "l_foot", "r_foot")


class Config:
    JOINT_NUMS: int = 5  # exclude back, sacrum
    # foot length (in meters)
    HIP_LEN: float = 0.0043  # hip to thigh joint
    THIGH_LEN: float = 0.006  # thigh to calf joint
    CALF_LEN: float = 0.0053  # calf to ankle joint
    ANKLE_LEN: float = 0.0043  # ankle to foot joint
    FOOT_LEN: float = 0.0011  # foot joint to ground

    # joint angle limits (in degrees)
    L_HIP_MAX_DEG: float = 80.0
    L_HIP_MIN_DEG: float = -50.0
    R_HIP_MAX_DEG: float = 50.0
    R_HIP_MIN_DEG: float = -80.0
    THIGH_MAX_DEG: float = 90.0
    THIGH_MIN_DEG: float = -90.0
    CALF_MAX_DEG: float = 90.0
    CALF_MIN_DEG: float = -90.0
    ANKLE_MAX_DEG: float = 90.0
    ANKLE_MIN_DEG: float = -90.0
    FOOT_MAX_DEG: float = 90.0
    FOOT_MIN_DEG: float = -90.0

    SACRUM_MAX_DEG: float = 60.0
    SACRUM_MIN_DEG: float = -60.0

    # default joint angles (in degrees)
    HIP_THETA_UW: float = 270.0

    # origin target point
    ORIGIN_L_TARGET = Vector3(x=0.002, y=0.00325, z=0.0)
    ORIGIN_R_TARGET = Vector3(x=0.002, y=-0.00325, z=0.0)

    # link mass
    BASELINK_MASS: float = 0.0012584
    BACK_MASS: float = 0.00333526 #TODO real physic robot change to  the same as sacrum mass
    SACRUM_MASS: float = 0.00142664
    HIP_MASS: float = 0.00009
    THIGH_MASS: float = 0.0005
    CALF_MASS: float = 0.00029
    ANKLE_MASS: float = 0.00009
    FOOT_MASS: float = 0.00046

    FALL_DOWN_BASELINK_Z_THRESHOLD: float = 0.017  # in meters

    FOOT_LINK_X_SEMI_LENGTH: float = 0.002  # in meters

    SS_SWING_FOOT_HEIGHT: float = 0.0011  # in meters
    SWING_TRAJECTORY_MID_HEIGHT: float = 0.0015  # in meters

    SACRUM_MAX_NOISE_RAD: float = 0.0008
    HIP_MAX_NOISE_RAD: float = 0.0025
    THIGH_MAX_NOISE_RAD: float = 0.0008
    CALF_MAX_NOISE_RAD: float = 0.0008
    ANKLE_MAX_NOISE_RAD: float = 0.0008
    FOOT_MAX_NOISE_RAD: float = 0.0025
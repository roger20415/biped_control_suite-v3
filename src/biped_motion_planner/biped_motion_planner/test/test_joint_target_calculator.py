import numpy as np
import pytest
from biped_motion_planner.joint_targets_calculator import Config
from biped_motion_planner.joint_targets_calculator import JointTargetsCalculator
from geometry_msgs.msg import Quaternion, Vector3
from pytest import approx


@pytest.fixture
def joint_targets_calculator():
    return JointTargetsCalculator()

def test_calc_joint_targets_adds_FOOT_LEN_to_target_z(joint_targets_calculator):
    p_W = {
        "target": Vector3(x=1.5, y=1.5, z=0.0),
        "baselink": Vector3(x=1.5, y=1.5, z=0.04),
        "hip": Vector3(x=1.5, y=1.5, z=0.02)
    }
    q_W_baselink = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0) # no rotation
    joint_targets_calculator.calc_joint_targets(p_W, q_W_baselink, "left")
    assert np.isclose(
        joint_targets_calculator.p_W["foot"].z - joint_targets_calculator.p_W["target"].z, 
        Config.FOOT_LEN)
    
def test_transform_points_World_to_Baselink(joint_targets_calculator):
    T_BW = np.array([
        [ 0.0,                0.0,               -1.0,                2.5],
        [ 0.7071067811865476, -0.7071067811865476, 0.0,               0.7071067811865476],
        [-0.7071067811865476, -0.7071067811865476, 0.0,               1.4142135623730951],
        [ 0.0,                0.0,                0.0,                1.0],
    ], dtype=np.float32)
    
    joint_targets_calculator.p_W = {
        "foot": Vector3(x=1.5, y=2.5, z=3.5),
        "baselink": Vector3(x=0.5, y=1.5, z=2.5),
        "hip": Vector3(x=0.5, y=2.0, z=3.0)
    }

    joint_targets_calculator.p_B.update(joint_targets_calculator._transform_points_World_to_Baselink(T_BW))
    assert np.allclose(
        [joint_targets_calculator.p_B["hip"].x, joint_targets_calculator.p_B["hip"].y, joint_targets_calculator.p_B["hip"].z],
        [-0.5, -0.35355339, -0.35355339],
        atol=1e-12
    )
    assert np.allclose(
        [joint_targets_calculator.p_B["foot"].x, joint_targets_calculator.p_B["foot"].y, joint_targets_calculator.p_B["foot"].z],
        [-1.0, 0.0, -1.41421356],
        atol=1e-12
    )

def test_calc_WB_transform(joint_targets_calculator):
    q_W_baselink = Quaternion(x=0.6532814824381882, 
                            y=0.27059805007309845, 
                            z=-0.6532814824381883, 
                            w=0.27059805007309856)
    p_W_baselink = Vector3(x=0.5, y=1.5, z=2.5)
    R_WB, T_BW = joint_targets_calculator._calc_WB_transforms(q_W_baselink, p_W_baselink)
    R_WB_expected = np.array([
        [ 0.0,  0.7071067811865476, -0.7071067811865476],
        [ 0.0, -0.7071067811865476, -0.7071067811865476],
        [-1.0,  0.0,                0.0],
    ], dtype=np.float32)
    assert np.allclose(R_WB, R_WB_expected, atol=1e-12)

    T_BW_expected = np.array([
        [ 0.0,                0.0,               -1.0,                2.5],
        [ 0.7071067811865476, -0.7071067811865476, 0.0,               0.7071067811865476],
        [-0.7071067811865476, -0.7071067811865476, 0.0,               1.4142135623730951],
        [ 0.0,                0.0,                0.0,                1.0],
    ], dtype=np.float32)
    assert np.allclose(T_BW, T_BW_expected, atol=1e-12)

def test_transform_points_Baselink_to_Leg(joint_targets_calculator):
    T_LB = np.array([
        [1.0, 0.0, 0.0, 0.5],
        [0.0, 0.948683298050514,  0.316227766016838, 0.447213595499958],
        [0.0, -0.316227766016838, 0.948683298050514, 0.223606797749979],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float32)

    joint_targets_calculator.p_B = {
        "foot": Vector3(x=-1.0, y=0.0, z=-1.41421356)
    }
    joint_targets_calculator.p_L.update(joint_targets_calculator._transform_points_Baselink_to_Leg(T_LB))

    assert np.allclose(
        [joint_targets_calculator.p_L["foot"].x, joint_targets_calculator.p_L["foot"].y, joint_targets_calculator.p_L["foot"].z],
        [-0.5, 0.0, -1.118033989],
        atol=1e-9
    )

def test_calc_BL_transforms(joint_targets_calculator):
    joint_targets_calculator.p_B = {
        "foot": Vector3(x=-1.0, y=0.0, z=-1.41421356),
        "hip": Vector3(x=-0.5, y=-0.35355339, z=-0.35355339)
    }

    R_BL, T_LB = joint_targets_calculator._calc_BL_transforms()
    R_BL_expected = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.948683298050514, -0.316227766016838],
        [0.0, 0.316227766016838,  0.948683298050514],
    ], dtype=np.float32)

    T_LB_expected = np.array([
        [1.0, 0.0, 0.0, 0.5],
        [0.0, 0.948683298050514,  0.316227766016838, 0.447213595499958],
        [0.0, -0.316227766016838, 0.948683298050514, 0.223606797749979],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float32)

    assert np.allclose(R_BL, R_BL_expected, atol=1e-12)
    assert np.allclose(T_LB, T_LB_expected, atol=1e-12)


def test_calc_R_BL(joint_targets_calculator):
    joint_targets_calculator.p_B = {
        "foot": Vector3(x=-1.0, y=0.0, z=-1.41421356),
        "hip": Vector3(x=-0.5, y=-0.35355339, z=-0.35355339)
    }
    R_BL = joint_targets_calculator._calc_R_BL()
    R_BL_expected = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.948683298050514, -0.316227766016838],
        [0.0, 0.316227766016838,  0.948683298050514],
    ], dtype=np.float32)
    assert np.allclose(R_BL, R_BL_expected, atol=1e-12)

def test_calc_R_BL_norm_zero(joint_targets_calculator):
    joint_targets_calculator.p_B = {
        "foot": Vector3(x=0.0, y=0.0, z=0.0),
        "hip": Vector3(x=5.0, y=0.0, z=0.0)
    }
    with pytest.raises(ValueError, match="zero-length"):
        joint_targets_calculator._calc_R_BL()

def test_transform_points_Leg_to_uw(joint_targets_calculator):
    joint_targets_calculator.p_L = {
        "foot": Vector3(x=-0.5, y=0.0, z=-1.118033989)
    }
    joint_targets_calculator.p_uw.update(joint_targets_calculator._transform_points_Leg_to_uw())

    assert np.allclose(
        [joint_targets_calculator.p_uw["foot"][0], joint_targets_calculator.p_uw["foot"][1]],
        [-0.5, -1.118033989],
        atol=1e-12
    )

def test_calc_phi_BL(joint_targets_calculator):

    # foot at 3rd quadrant
    joint_targets_calculator.p_B = {
        "foot": Vector3(x=5.0, y=-1.73205081, z=-1.0),
        "hip": Vector3(x=0.0, y=0.0, z=-0.0)
    }
    phi_BL = joint_targets_calculator._calc_phi_BL()
    assert np.allclose(phi_BL, -60, atol=1e-12)

    # foot at 4th quadrant
    joint_targets_calculator.p_B = {
        "foot": Vector3(x=5.0, y=1.73205081, z=-1.0),
        "hip": Vector3(x=0.0, y=0.0, z=-0.0)
    }
    phi_BL = joint_targets_calculator._calc_phi_BL()
    assert np.isclose(phi_BL, 60, atol=1e-12)


def test_project_gravity_to_uw_plane(joint_targets_calculator):
    R_BW = np.eye(3, dtype=np.float32)
    R_LB = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.948683298050514,  0.316227766016838],
        [0.0, -0.316227766016838, 0.948683298050514],
    ], dtype=np.float32)
    e_L_proj = joint_targets_calculator._project_gravity_to_uw_plane(R_BW, R_LB)
    e_L_proj_expected = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    assert np.allclose(e_L_proj, e_L_proj_expected, atol=1e-12)

    R_BW = np.array([
        [ 0.0, 0.0, 1.0],
        [ 0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
    ], dtype=np.float32)

    R_LB = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.948683298050514,  0.316227766016838],
        [0.0, -0.316227766016838, 0.948683298050514],
    ], dtype=np.float32)

    e_L_proj = joint_targets_calculator._project_gravity_to_uw_plane(R_BW, R_LB)
    e_L_proj_expected = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
    assert np.allclose(e_L_proj, e_L_proj_expected, atol=1e-12)

def test_project_gravity_to_uw_plane_degenerate_projection_warns(joint_targets_calculator):
    R_BW = np.eye(3, dtype=np.float32)
    R_LB = np.array([[1., 0., 0.],
                     [0., 0., -1.],
                     [0., 1.,  0.]], dtype=np.float32)
    with pytest.warns(RuntimeWarning, match="Gravity is parallel to the uw-plane normal; projection is degenerated. Returning -w_L."):
        e_L_proj = joint_targets_calculator._project_gravity_to_uw_plane(R_BW, R_LB)
    e_L_proj_expected = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    assert np.allclose(e_L_proj, e_L_proj_expected, atol=1e-12)

def test_calc_p_uw_ankle(joint_targets_calculator):
    joint_targets_calculator.p_uw = {
        "foot": np.array([-0.5, -1.118033989], dtype=np.float32)
    }
    e_L_proj = np.array([0.6, 0.0, 0.8], dtype=np.float32)
    p_uw_ankle = joint_targets_calculator._calc_p_uw_ankle(e_L_proj)

    # The vector of ankle to foot
    delta = joint_targets_calculator.p_uw["foot"] - p_uw_ankle
    assert np.isclose(np.linalg.norm(delta), Config.ANKLE_LEN, atol=1e-12)

    e_uw_proj = np.array([e_L_proj[0], e_L_proj[2]], dtype=np.float32)
    e_uw_proj_norm = e_uw_proj / np.linalg.norm(e_uw_proj)
    delta_norm = delta / np.linalg.norm(delta)
    assert np.allclose(delta_norm, e_uw_proj_norm, atol=1e-12)

def test_calc_theta_calf(monkeypatch, joint_targets_calculator):
    # test1: ankle at 4th quadrant
    monkeypatch.setattr(Config, "THIGH_LEN", 1.0)
    monkeypatch.setattr(Config, "CALF_LEN", 1.0)
    joint_targets_calculator.p_uw = {
        "thigh": np.array([0.0, -1.0], dtype=np.float32),
        "ankle": np.array([1.0, -2.2], dtype=np.float32)
    }

    theta_calf, thigh_to_ankle_vec_uw, p_uw_ankle_new, hold_prev_pose = joint_targets_calculator._calc_theta_calf()
    assert np.isclose(theta_calf, -77.291, atol=1e-3)
    assert np.allclose(thigh_to_ankle_vec_uw, np.array([1.0, -1.2]), atol=1e-12)
    assert np.allclose(p_uw_ankle_new, joint_targets_calculator.p_uw["ankle"], atol=1e-12)
    assert hold_prev_pose is False

    # test2: ankle at 3rd quadrant
    monkeypatch.setattr(Config, "THIGH_LEN", 1.0)
    monkeypatch.setattr(Config, "CALF_LEN", 3.0)
    joint_targets_calculator.p_uw = {
        "thigh": np.array([0.0, -1.0], dtype=np.float32),
        "ankle": np.array([0.4639, -4.4691], dtype=np.float32)
    }

    theta_calf, thigh_to_ankle_vec_uw, p_uw_ankle_new, hold_prev_pose = joint_targets_calculator._calc_theta_calf()
    assert np.isclose(theta_calf, -67.976, atol=1e-3)
    assert np.allclose(thigh_to_ankle_vec_uw, np.array([0.4639, -3.4691]), atol=1e-12)
    assert np.allclose(p_uw_ankle_new, joint_targets_calculator.p_uw["ankle"], atol=1e-12)
    assert hold_prev_pose is False

    # test3: theta_calf equals -90
    monkeypatch.setattr(Config, "THIGH_LEN", 1.0)
    monkeypatch.setattr(Config, "CALF_LEN", 1.0)
    joint_targets_calculator.p_uw = {
        "thigh": np.array([0.0, -1.0], dtype=np.float32),
        "ankle": np.array([1.0, -2.0], dtype=np.float32)
    }

    theta_calf, thigh_to_ankle_vec_uw, p_uw_ankle_new, hold_prev_pose = joint_targets_calculator._calc_theta_calf()
    assert np.isclose(theta_calf, -90.0, atol=1e-3)
    assert np.allclose(thigh_to_ankle_vec_uw, np.array([1.0, -1.0]), atol=1e-12)
    assert np.allclose(p_uw_ankle_new, joint_targets_calculator.p_uw["ankle"], atol=1e-12)
    assert hold_prev_pose is False

def test_calc_theta_calf_ankle_too_far(monkeypatch, joint_targets_calculator):
    monkeypatch.setattr(Config, "THIGH_LEN", 1.0)
    monkeypatch.setattr(Config, "CALF_LEN", 1.0)
    joint_targets_calculator.p_uw = {
        "thigh": np.array([0.0, -1.0], dtype=np.float32),
        "ankle": np.array([5.0, -6.0], dtype=np.float32)
    }
    p_uw_ankle_new_expect = np.array([2/np.sqrt(2), -1-(2/np.sqrt(2))])
    with pytest.warns(RuntimeWarning, match="Ankle is too far from hip"):
        theta_calf, thigh_to_ankle_vec_uw, p_uw_ankle_new, hold_prev_pose = joint_targets_calculator._calc_theta_calf()
    assert np.isclose(theta_calf, 0.0, atol=1e-12)
    assert np.allclose(thigh_to_ankle_vec_uw, np.array([2/np.sqrt(2), -(2/np.sqrt(2))]), atol=1e-12)
    assert np.allclose(p_uw_ankle_new, p_uw_ankle_new_expect, atol=1e-12)
    assert hold_prev_pose is False

def test_calc_theta_calf_ankle_too_close(monkeypatch, joint_targets_calculator):
    # Hold the previous pose if the ankle is too close to the thigh and unreachable.
    monkeypatch.setattr(Config, "THIGH_LEN", 1.0)
    monkeypatch.setattr(Config, "CALF_LEN", 3.0)
    joint_targets_calculator.p_uw = {
        "thigh": np.array([0.0, -1.0], dtype=np.float32),
        "ankle": np.array([0.0, -2.0], dtype=np.float32)
    }

    with pytest.warns(RuntimeWarning, match="Ankle is too close to hip"):
        theta_calf, thigh_to_ankle_vec_uw, p_uw_ankle_new, hold_prev_pose = joint_targets_calculator._calc_theta_calf()
    assert hold_prev_pose is True
    assert thigh_to_ankle_vec_uw is None
    assert theta_calf is None
    assert p_uw_ankle_new is None

def test_calc_theta_calf_exceed(monkeypatch, joint_targets_calculator):
    # Hold the previous pose if the calf joint angle (theta_calf) exceeds ±90°.
    monkeypatch.setattr(Config, "THIGH_LEN", 1.0)
    monkeypatch.setattr(Config, "CALF_LEN", 1.0)
    joint_targets_calculator.p_uw = {
        "thigh": np.array([0.0, -1.0], dtype=np.float32),
        "ankle": np.array([0.0, -2.0], dtype=np.float32)
    }

    with pytest.warns(RuntimeWarning, match="exceeds"):
        theta_calf, thigh_to_ankle_vec_uw, p_uw_ankle_new, hold_prev_pose = joint_targets_calculator._calc_theta_calf()
    assert hold_prev_pose is True
    assert thigh_to_ankle_vec_uw is None
    assert theta_calf is None
    assert p_uw_ankle_new is None

def test_calc_theta_thigh(monkeypatch, joint_targets_calculator):
    # test1: 3rd quadrant, human-like knee
    monkeypatch.setattr(Config, "THIGH_LEN", 3.0)
    monkeypatch.setattr(Config, "CALF_LEN", 4.0)
    thigh_to_ankle_vec_uw = np.array([-4.5809694797, -4.5809694797], dtype=np.float32)
    joint_targets_calculator.joint_theta = {
        "calf": -45.0
    }

    theta_thigh = joint_targets_calculator._calc_theta_thigh(thigh_to_ankle_vec_uw)
    assert np.isclose(theta_thigh, -19.1135647, atol=1e-6)

    # test2: 4th quadrant, dog-like knee
    monkeypatch.setattr(Config, "THIGH_LEN", 1.0)
    monkeypatch.setattr(Config, "CALF_LEN", 1.0)
    thigh_to_ankle_vec_uw = np.array([1.3660254038, -1.3660254038], dtype=np.float32)
    joint_targets_calculator.joint_theta = {
        "calf": 30.0
    }

    theta_thigh = joint_targets_calculator._calc_theta_thigh(thigh_to_ankle_vec_uw)
    assert np.isclose(theta_thigh, 30.0, atol=1e-12)

def test_calc_theta_ankle(joint_targets_calculator):
    # test1: 3rd quadrant, human-like knee
    joint_targets_calculator.joint_theta = {
        "calf": -45.0,
        "thigh": -30.0
    }
    e_L_proj = np.array([-1.0, 0.0, -1.0])
    theta_ankle = joint_targets_calculator._calc_theta_ankle(e_L_proj)
    assert np.isclose(theta_ankle, 30.0, atol=1e-12)

    # test2: 4th quadrant, dog-like knee
    joint_targets_calculator.joint_theta = {
        "calf": 30.0,
        "thigh": 30.0
    }
    e_L_proj = np.array([1.0, 0.0, -1.7320508075688772])
    theta_ankle = joint_targets_calculator._calc_theta_ankle(e_L_proj)
    assert np.isclose(theta_ankle, -30.0, atol=1e-12)

def test_calc_phi_foot(joint_targets_calculator):
    # test1: R_WB +z=45deg, R_BL +x=0deg
    R_WB = np.array([[0.70710678, -0.70710678, 0.],
                    [0.70710678,  0.70710678, 0.],
                    [0.,          0.,         1.]], dtype=np.float32)
    R_BL = np.array([[1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0]], dtype=np.float32)
    e_L_proj = np.array([1.0, 0.0, -1.0], dtype=np.float32)
    phi_foot, hold_prev_pose = joint_targets_calculator._calc_phi_foot(R_WB, R_BL, e_L_proj)
    assert np.isclose(phi_foot, 0.0, atol=1e-12)
    assert hold_prev_pose is False

    # test2: R_WB +z=0deg, R_BL +x=-30deg
    R_WB = np.array([[1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0]], dtype=np.float32)
    R_BL = np.array([[1.0, 0.0, 0.0],
                    [0.0, 0.8660254, 0.5],
                    [0.0, -0.5, 0.8660254]], dtype=np.float32)
    e_L_proj = np.array([-1.0, 0.0, -1.0], dtype=np.float32)
    phi_foot, hold_prev_pose = joint_targets_calculator._calc_phi_foot(R_WB, R_BL, e_L_proj)
    assert np.isclose(phi_foot, 30.0, atol=1e-12)
    assert hold_prev_pose is False

    # test3: R_WB +z=180deg, R_BL +x=-30deg
    R_WB = np.array([[-1.0,  0.0,  0.0],
                    [ 0.0, -1.0,  0.0],
                    [ 0.0,  0.0,  1.0]], dtype=np.float32)

    R_BL = np.array([[1.0, 0.0, 0.0],
                    [0.0, 0.8660254, 0.5],
                    [0.0, -0.5, 0.8660254]], dtype=np.float32)

    e_L_proj = np.array([1.0, 0.0, -1.0], dtype=np.float32)
    phi_foot, hold_prev_pose = joint_targets_calculator._calc_phi_foot(R_WB, R_BL, e_L_proj)
    assert np.isclose(phi_foot, 30.0, atol=1e-12)
    assert hold_prev_pose is False

    # test4: R_WB +x=20deg/+y=20deg/z=-250deg, R_BL +x=-30deg
    R_WB = np.array([
        [-0.32139380, -0.88302222,  0.34202014],
        [ 0.84301347, -0.43131696, -0.32139380],
        [ 0.43131696,  0.18503361,  0.88302222]
    ], dtype=np.float32)

    R_BL = np.array([
        [1.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.86602540, 0.50000000],
        [0.00000000,-0.50000000, 0.86602540]
    ], dtype=np.float32)
    e_L_proj = np.array([1806.0, 0.0, -100.0], dtype=np.float32)
    phi_foot, hold_prev_pose = joint_targets_calculator._calc_phi_foot(R_WB, R_BL, e_L_proj)
    assert np.isclose(phi_foot, 30.992303, atol=1e-12)
    assert hold_prev_pose is False

def test_calc_phi_foot_e_L_proj_w_zero_raises(joint_targets_calculator):
    R_WB = np.eye(3, dtype=np.float32)
    R_BL = np.eye(3, dtype=np.float32)
    e_L_proj = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    with pytest.warns(RuntimeWarning, match="Hold previous pose"):
        phi_foot, hold_prev_pose = joint_targets_calculator._calc_phi_foot(R_WB, R_BL, e_L_proj)
    assert phi_foot is None
    assert hold_prev_pose is True

def test_calc_phi_foot_e_L_proj_w_positive_raises(joint_targets_calculator):
    R_WB = np.eye(3, dtype=np.float32)
    R_BL = np.eye(3, dtype=np.float32)
    e_L_proj = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    with pytest.raises(ValueError, match="must < 0"):
        joint_targets_calculator._calc_phi_foot(R_WB, R_BL, e_L_proj)

def test_calc_and_clamp_joint_targets_rad(joint_targets_calculator):
    # Test for left leg
    joint_targets_calculator.leg_side = "left"
    joint_targets_calculator.joint_theta = {
        "thigh": 10.0,
        "calf": 20.0,
        "ankle": 30.0,
    }
    joint_targets_calculator.joint_phi = {
        "leg": -10.0,
        "foot": -20.0
    }
    joint_targets = joint_targets_calculator._calc_and_clamp_joint_targets_rad()
    joint_targets_expected = {
        "hip" : np.deg2rad(-10.0),
        "thigh": np.deg2rad(10.0),
        "calf": np.deg2rad(-20.0),
        "ankle": np.deg2rad(-30.0),
        "foot": np.deg2rad(20.0)
    }
    assert set(joint_targets) == set(joint_targets_expected)
    assert joint_targets == approx(joint_targets_expected, rel=1e-6, abs=1e-12)

    # Test for right leg
    joint_targets_calculator.leg_side = "right"
    joint_targets_calculator.joint_theta = {
        "thigh": -10.0,
        "calf": -20.0,
        "ankle": -30.0,
    }
    joint_targets_calculator.joint_phi = {
        "leg": 10.0,
        "foot": 20.0
    }
    joint_targets = joint_targets_calculator._calc_and_clamp_joint_targets_rad()
    joint_targets_expected = {
        "hip" : np.deg2rad(10.0),
        "thigh": np.deg2rad(10.0),
        "calf": np.deg2rad(-20.0),
        "ankle": np.deg2rad(-30.0),
        "foot": np.deg2rad(-20.0)
    }
    assert set(joint_targets) == set(joint_targets_expected)
    assert joint_targets == approx(joint_targets_expected, rel=1e-6, abs=1e-12)

def test_calc_and_clamp_joint_targets_rad_exceed_limits(monkeypatch, joint_targets_calculator):
    monkeypatch.setattr(Config, "L_HIP_MAX_DEG", 80.0)
    monkeypatch.setattr(Config, "L_HIP_MIN_DEG", -50.0)
    monkeypatch.setattr(Config, "R_HIP_MAX_DEG", 50.0)
    monkeypatch.setattr(Config, "R_HIP_MIN_DEG", -80.0)
    monkeypatch.setattr(Config, "THIGH_MAX_DEG", 90.0)
    monkeypatch.setattr(Config, "THIGH_MIN_DEG", -90.0)
    monkeypatch.setattr(Config, "CALF_MAX_DEG", 90.0)
    monkeypatch.setattr(Config, "CALF_MIN_DEG", -90.0)
    monkeypatch.setattr(Config, "ANKLE_MAX_DEG", 90.0)
    monkeypatch.setattr(Config, "ANKLE_MIN_DEG", -90.0)
    monkeypatch.setattr(Config, "FOOT_MAX_DEG", 90.0)
    monkeypatch.setattr(Config, "FOOT_MIN_DEG", -90.0)

    # Test for left leg
    joint_targets_calculator.leg_side = "left"
    joint_targets_calculator.joint_theta = {
        "thigh": -100,
        "calf": -100,
        "ankle": -100,
    }
    joint_targets_calculator.joint_phi = {
        "leg": 100.0,
        "foot": 100.0
    }
    with pytest.warns(RuntimeWarning) as record:
        joint_targets = joint_targets_calculator._calc_and_clamp_joint_targets_rad()
    joint_targets_expected = {
        "hip" : np.deg2rad(80.0),
        "thigh": np.deg2rad(-90.0),
        "calf": np.deg2rad(90.0),
        "ankle": np.deg2rad(90.0),
        "foot": np.deg2rad(-90.0)
    }
    assert set(joint_targets) == set(joint_targets_expected)
    assert len(record) == 5

    # Test for right leg
    joint_targets_calculator.leg_side = "right"
    joint_targets_calculator.joint_theta = {
        "thigh": 100,
        "calf": 100,
        "ankle": 100,
    }
    joint_targets_calculator.joint_phi = {
        "leg": -100.0,
        "foot": -100.0
    }
    with pytest.warns(RuntimeWarning) as record:
        joint_targets = joint_targets_calculator._calc_and_clamp_joint_targets_rad()
    joint_targets_expected = {
        "hip" : np.deg2rad(-80.0),
        "thigh": np.deg2rad(-90.0),
        "calf": np.deg2rad(90.0),
        "ankle": np.deg2rad(90.0),
        "foot": np.deg2rad(90.0)
    }
    assert set(joint_targets) == set(joint_targets_expected)
    assert len(record) == 5

def test_calc_joint_targets_leg_side_undefined(joint_targets_calculator):
    joint_targets_calculator.leg_side = "undefined"
    with pytest.raises(ValueError, match="Leg side is undefined."):
        joint_targets_calculator._calc_and_clamp_joint_targets_rad()

import numpy as np
from geometry_msgs.msg import Quaternion, Vector3
from numpy.typing import NDArray
from typing import Sequence


class LinearAlgebraUtils():

    @staticmethod
    def quaternion_to_rotation_matrix(q: Quaternion) -> NDArray[np.float32]:
        q_norm = LinearAlgebraUtils.normalize_vec([q.x, q.y, q.z, q.w])
        x, y, z, w = q_norm[0], q_norm[1], q_norm[2], q_norm[3]
        R = np.array([
            [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
            [2*(x*y + z*w),     1 - 2*(x*x + z*z),     2*(y*z - x*w)],
            [2*(x*z - y*w),         2*(y*z + x*w), 1 - 2*(x*x + y*y)]
        ], dtype=np.float32)
        LinearAlgebraUtils._check_valid_R(R)
        return R

    @staticmethod
    def combine_transformation_matrix(R: NDArray[np.float32], t: Vector3) -> NDArray[np.float32]:
        LinearAlgebraUtils._check_valid_R(R)
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        T[:3,  3] = np.array([t.x, t.y, t.z], dtype=np.float32)
        return T

    @staticmethod
    def normalize_vec(vec: Sequence[float]) -> NDArray[np.float32]:
        arr = np.array(vec, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm < 1e-12:
            raise ValueError("Cannot normalize zero-length vector")
        return arr / norm

    @staticmethod
    def invert_transformation_matrix(T: NDArray[np.float32]) -> NDArray[np.float32]:
        LinearAlgebraUtils._check_valid_T(T)
        R = T[:3, :3]
        t = T[:3,  3]
        T_inv = np.eye(4, dtype=np.float32)
        Rt = R.T
        T_inv[:3, :3] = Rt
        T_inv[:3,  3] = -Rt @ t
        return T_inv

    @staticmethod
    def transform_point(T: NDArray[np.float32], p: Vector3) -> Vector3:
        LinearAlgebraUtils._check_valid_T(T)
        p_homo = np.array([p.x, p.y, p.z, 1.0],
                          dtype=np.float32)  # shape = (4,)
        p_tranformed = T @ p_homo  # shape = (4,)
        return Vector3(x=float(p_tranformed[0]), y=float(p_tranformed[1]), z=float(p_tranformed[2]))

    @staticmethod
    def _check_valid_T(T: NDArray[np.float32]) -> None:
        if T.shape != (4, 4):
            raise ValueError(f"T must be 4x4, got shape {T.shape}")
        if not np.allclose(T[3, :], [0, 0, 0, 1]):
            raise ValueError(
                f"T last row must be [0. 0. 0. 1.], got {T[3, :]}")

        R = T[:3, :3]
        LinearAlgebraUtils._check_valid_R(R)
        return

    @staticmethod
    def _check_valid_R(R: NDArray[np.float32]) -> None:
        if R.shape != (3, 3):
            raise ValueError(f"R must be 3x3, got shape {R.shape}")
        if not np.allclose(R.T @ R, np.eye(3), atol=1e-6):
            raise ValueError("R must be orthogonal")
        if not np.isclose(np.linalg.det(R), 1.0, atol=1e-6):
            raise ValueError("det(R) must be 1")
        return

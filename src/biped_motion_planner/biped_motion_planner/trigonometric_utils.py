import numpy as np
import warnings


class TrigonometricUtils():
    @staticmethod
    def clamp_cos(c: float, tol: float = 1e-9) -> float:
        if c > 1.0 + tol or c < -1.0 - tol:
            raise ValueError("cosine out of [-1, 1] by more than tol")
        if c > 1.0 or c < -1.0:
            warnings.warn(
                "cosine slightly outside [-1,1]; clamped", RuntimeWarning)
            c = np.clip(c, -1.0, 1.0)
        return c

    @staticmethod
    def normalize_angle_to_180(angle: float) -> float:
        """
        Normalize an angle to the range [-180, 180].
        """
        angle = (angle + 180) % 360 - 180
        return angle

    @staticmethod
    def clip_deg(value: float, low: float, high: float, joint: str, side: str) -> float:
        """Clamp value to [low, high]; warn if clipping happened."""
        v = float(value)
        clipped = float(np.clip(v, low, high))
        if clipped != v:
            bound = 'min' if v < low else 'max'
            limit = low if v < low else high
            warnings.warn(
                f"{side} {joint}: {v:.6f}° -> {clipped:.6f}° (hit {bound}={limit:.6f}°)",
                category=RuntimeWarning,
                stacklevel=2,
            )
        return clipped

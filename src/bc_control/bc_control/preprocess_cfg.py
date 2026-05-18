import numpy as np

class PreprocessCfg:
    """
    Configuration for observation preprocessing and action clipping.
    """
    # 將 OBS_MEAN 全改成 0 (長度維持 34)
    OBS_MEAN = np.zeros(34, dtype=np.float32)

    # 將 OBS_STD 全改成 1.0 (長度維持 34)
    OBS_STD = np.ones(34, dtype=np.float32)

    DEFAULT_JOINT_POSITIONS = np.array([
        0.0,  # sacrum
        0.0, 0.0, 0.0, 0.0, 0.0,  # left 5 joints
        0.0, 0.0, 0.0, 0.0, 0.0   # right 5 joints
    ], dtype=np.float32)

    # 將 ACTION_MAX 全改成 1 (長度維持 11)
    ACTION_MAX = np.ones(11, dtype=np.float32)

    # 將 ACTION_MIN 全改成 -1 (長度維持 11)
    ACTION_MIN = np.ones(11, dtype=np.float32) * -1.0
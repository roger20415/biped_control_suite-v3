import numpy as np
action_header="sacrum,l_hip,l_thigh,l_calf,l_ankle,l_foot,r_hip,r_thigh,r_calf,r_ankle,r_foot"
obs_header = (
    "b_z,b_pitch,b_yaw,b_raw,b_lin_vel_x,b_lin_vel_y,b_lin_vel_z,b_ang_vel_x,b_ang_vel_y,b_ang_vel_z,"
    "sacrum_pos,l_hip_pos,l_thigh_pos,l_calf_pos,l_ankle_pos,l_foot_pos,"
    "r_hip_pos,r_thigh_pos,r_calf_pos,r_ankle_pos,r_foot_pos,"
    "sacrum_vel,l_hip_vel,l_thigh_vel,l_calf_vel,l_ankle_vel,l_foot_vel,"
    "r_hip_vel,r_thigh_vel,r_calf_vel,r_ankle_vel,r_foot_vel,"
    "l_foot_contact,r_foot_contact"
)

f = np.load("expert_data.npz", allow_pickle=True)
print("keys:", f.files)

actions = f["actions"]   # shape=(1324, 11)
obs = f["obs"]           # shape=(1324, 34)
f.close()

np.savetxt("actions.csv", actions, delimiter=",", fmt="%.6f",
           header=action_header, comments="", encoding="utf-8-sig")
np.savetxt("obs.csv", obs, delimiter=",", fmt="%.6f",
           header=obs_header, comments="", encoding="utf-8-sig")

print("export actions.csv & obs.csv")
# biped_control_suite-v3
A ROS2-based framework for controlling bipedal robots.

## Data Collection

### 0. Prerequisites
Start the [Biped-Simulation-Isaac-Sim-stage-v3](https://github.com/roger20415/Biped-Simulation-Isaac-Sim-stage-v3.git) environment in Isaac Sim.

### 1. Launch the biped_control_suite-v3 Environment
```bash
cd biped_control_suite-v3
./run.sh
r
```

### 2. Generate Data via Rule-based Control
```bash
ros2 launch biped_motion_planner biped_motion.launch.py
```

### 3. Collect Simulation Data
```bash
ros2 run expert_data_collector data_collect_node
```
> **Note:** The simulation data will be saved in `src/expert_data_collector/expert_data_collector/data`

### 4. Export and View Simulation Data
```bash
python export_data_csv.py
```
> **Note:** This generates a CSV file associated with the `ros2 run expert_data_collector data_collect_node` execution.

---

## MLP Inference

### 0. Prerequisites
Start the [Biped-Simulation-Isaac-Sim-stage-v3](https://github.com/roger20415/Biped-Simulation-Isaac-Sim-stage-v3.git) environment in Isaac Sim.

### 1. Load MLP Model Weights
Place the trained models from [biped-rl-isaac-lab](https://github.com/roger20415/biped-rl-isaac-lab.git) into the specified directories:

* Move `bc_actor_body_weights.pth` and `bc_actor_head_weights.pth` to:
  `src/bc_control/bc_control/model`
* Move `action_scales.npy` to:
  `src/bc_control/bc_control`

### 2. Start Inference
```bash
ros2 run bc_control bc_inference_node
```
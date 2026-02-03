# Neural Path Planning & LQR Control for Autonomous Vehicles

This project implements a hybrid autonomous driving stack that combines **Deep Learning** for path planning and **Optimal Control (LQR)** for trajectory tracking. The system processes Bird's Eye View (BEV) images to predict a safe trajectory and computes the optimal steering angle to follow it using a linearized bicycle model.

## 📂 Project Structure

```bash
.
├── bev_data/              # Dataset containing BEV images (.png) and waypoints (.npy)
├── model/
│   ├── model.py           # Neural Network architecture (ResNet18 backbone + MLP Head)
│   ├── dataset.py         # Custom PyTorch Dataset with dynamic padding/truncating logic
│   ├── train.py           # Training loop implementation with validation split
│   └── inference_vis.py   # Visual debugging tool for model predictions
├── control/
│   └── lqr.py             # Linear Quadratic Regulator (LQR) controller implementation
├── full_system_test.py    # End-to-end integration demo (Perception -> Planning -> Control)
├── generate_data.py       # Script to generate BEV data from nuScenes
└── README.md
```

## 🚀 How It Works

The pipeline consists of three main stages:

1. **Perception & Planning (AI):**

- **Input:** A $400 \times 400$ pixel BEV image. The image channels encode Lidar obstacles, ego-vehicle history, and map layers.
- **Processing:** The **Neural Planner** uses a ResNet18 backbone to extract spatial features, followed by a Fully Connected (MLP) head.
- **Output:** A sequence of **10 future waypoints** $(x,y)$ in the vehicle's local coordinate frame.

2. **Control (LQR):**

- The **LQR Controller** receives the predicted trajectory.
- It calculates the **Cross-Track Error** (lateral distance) and **Heading Error** (angle difference) relative to the target path.
- It minimizes a quadratic cost function $J = \sum (x^T Q x + u^T R u)$ to compute the optimal steering gain $K$.

3. **Actuation:**

- The system outputs the precise steering command $\delta$, clamped to the vehicle's physical limits (e.g., $\pm 35^\circ$).

## 🛠 File Descriptions & Functionality

### `model/model.py`

Defines the `NeuralPlanner` class.

- **Architecture:** Uses a pre-trained **ResNet18** as a feature extractor, followed by fully connected layers (MLP) to regress the 20 coordinates (10 points $ \times$ 2).
- **Why ResNet18?** It offers an excellent trade-off between inference speed and feature extraction capability for single-frame spatial reasoning.

### `model/dataset.py`

Handles data loading and preprocessing.

- **Key Feature:** Implements robust logic to handle inconsistent data lengths. It automatically truncates trajectories longer than 10 points and pads shorter ones (or fills empty data with zeros) to ensure valid tensor stacking during batch training.

### `control/lqr.py`

Implements the **Linear Quadratic Regulator**.

- **Dynamics:** Based on the linearized Kinematic Bicycle Model.
- **Solver:** Solves the Discrete Algebraic Riccati Equation (DARE) to find the optimal gain matrix $K$.
- **Tuning:** The control cost matrix $R$ is tuned to **300.0**. This value ensures smooth steering and prevents actuator saturation/oscillation.

### `full_system_test.py`

The integration script for demonstration.

- Loads the trained model weights and the LQR class.
- Runs the full stack on random samples.
- **Visualization:** Generates a dual-view plot showing the AI's planned path (BEV) and the resulting steering action (Virtual Dashboard).

### `generate_data.py`

Data generation utility.

- Interacts with the `nuscenes-devkit` to extract scenes.
- Renders Lidar point clouds and map layers into BEV images.
- Saves the future trajectory of the ego-vehicle as `.npy` files for training.

## 💡 Design Choices & Rationale

| Component         | Choice            | Rationale                                                                                                                                                                  |
| ----------------- | ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Input Data**    | **BEV Images**    | Removes perspective distortion typical of frontal cameras, simplifying distance estimation to a 2D mapping problem.                                                        |
| **Loss Function** | **MSE**           | Mean Squared Error is the standard for regression tasks, directly minimizing the Euclidean distance between predicted and ground truth points.                             |
| **Controller**    | **LQR**           | Unlike geometric methods (e.g., Pure Pursuit), LQR considers the vehicle dynamics and minimizes both lateral and angular errors, ensuring stability at higher speeds.      |
| **Backbone**      | **CNN (No LSTM)** | Since the input is a single frame containing history (as a channel) and static obstacles, a Feed-Forward CNN is faster and sufficient compared to recurrent architectures. |

## ⚠️ Known Limitations

1. **Semantic Blindness:** The current model relies heavily on Lidar geometry (obstacles). In scenarios with wide open roads (no walls), the model may fail to follow lane curvature if semantic map inputs (HD Map/Lane markings) are missing or occluded.
2. **Data Dependency:** The LQR controller is optimal _given_ the trajectory. If the Neural Network predicts a wrong path, the controller will execute it faithfully.

---

## 🏁 Usage

### 1. Prerequisites

Ensure you have the following libraries installed:

```bash
pip install torch torchvision matplotlib numpy pillow

```

### 2. Training the Model

To train the neural network from scratch:

```bash
python model/train.py

```

_The model weights will be saved as `neural_planner_weights.pth`. An instance of the trained model is already created and ready for inference._

### 3. Running the Full System Demo

To visualize the AI planning and LQR control in action:

```bash
python full_system_test.py

```

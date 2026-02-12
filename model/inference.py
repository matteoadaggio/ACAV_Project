import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import time
from torch.utils.data import DataLoader
from model import NeuralPlanner
from dataset import NuScenesBEVDataset

def visualize_prediction():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Enable MPS if available (for Apple Silicon hardware)
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    
    # Graphic Parameters
    IMG_SIZE = 400
    SCALE = 4.0
    CENTER_X = 200     
    CENTER_Y = 200     
    
    # Load paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'best_neural_planner.pth')
    data_root = os.path.join(current_dir, '../bev_data')
    
    #Load Model
    model = NeuralPlanner(in_channels=3, num_waypoints=10).to(device)
    
    if not os.path.exists(model_path):
        print(f"Error: Weights not found at {model_path}")
        return

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Weights loaded.")
    except Exception as e:
        print(f"Error load_state_dict: {e}")
        return

    model.eval()

    # Load Dataset
    try:
        dataset = NuScenesBEVDataset(data_root=data_root)
    except FileNotFoundError:
        print("Error: Dataset not found.")
        return

    # Compute ADE, FDE and Success Rate over the entire dataset
    print("Computing ADE, FDE and Success Rate over the full dataset...")
    data_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    total_ade = 0.0
    total_fde = 0.0
    total_samples = 0
    collision_free_count = 0
    total_inference_time = 0.0  # seconds spent in model forward passes

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)    # [B, 3, H, W], values in [0, 1]
            targets = targets.to(device)  # [B, 10, 2]

            # Measure pure model inference time (per BEV)
            start_time = time.perf_counter()
            outputs = model(inputs)  # [B, 10, 2]
            end_time = time.perf_counter()

            batch_inference_time = end_time - start_time
            diffs = outputs - targets  # [B, 10, 2]
            dists = torch.norm(diffs, dim=2)  # [B, 10], L2 per timestep

            ade_batch = dists.mean(dim=1)  # [B]
            fde_batch = dists[:, -1]       # [B]

            batch_size = inputs.size(0)
            total_ade += ade_batch.sum().item()
            total_fde += fde_batch.sum().item()
            total_samples += batch_size
            total_inference_time += batch_inference_time

            # --- Collision checking for Success Rate ---
            # Dynamic objects are stored in the GREEN channel in the original BGR image.
            # After saving with OpenCV (BGR) and loading with PIL (RGB), the channels become:
            #   R -> Static / map
            #   G -> Dynamic objects (vehicles, pedestrians, cyclists)
            #   B -> Lidar points
            # Here we treat a collision as any predicted waypoint landing on a pixel
            # where the dynamic-object channel is non-zero.
            dynamic_channel = inputs[:, 1, :, :]  # [B, H, W]

            preds_np = outputs.detach().cpu().numpy()  # [B, 10, 2] in meters

            for b in range(batch_size):
                pts_meters = preds_np[b]  # [10, 2]

                x_meters = pts_meters[:, 0]
                y_meters = pts_meters[:, 1]

                # Convert to pixels using same convention as to_pixel()
                u = CENTER_X + (x_meters * SCALE)
                v = CENTER_Y - (y_meters * SCALE)

                # Round and clamp to valid image indices
                u_idx = np.clip(np.round(u).astype(np.int64), 0, IMG_SIZE - 1)
                v_idx = np.clip(np.round(v).astype(np.int64), 0, IMG_SIZE - 1)

                # Check dynamic-object occupancy along the predicted path
                dyn_map = dynamic_channel[b].detach().cpu().numpy()  # [H, W], in [0, 1]
                collision = np.any(dyn_map[v_idx, u_idx] > 0.1)

                if not collision:
                    collision_free_count += 1

    if total_samples > 0:
        mean_ade = total_ade / total_samples
        mean_fde = total_fde / total_samples
        success_rate = (collision_free_count / total_samples) * 100.0
        avg_inference_time_ms = (total_inference_time / total_samples) * 1000.0

        print(f"ADE over dataset: {mean_ade:.4f} m")
        print(f"FDE over dataset: {mean_fde:.4f} m")
        print(f"Success Rate (no collision with dynamic objects): {success_rate:.2f}%")
        print(f"Average model inference time per BEV: {avg_inference_time_ms:.3f} ms")
        print(f"Computed on {total_samples} samples.")
    else:
        print("Warning: no samples available to compute ADE/FDE.")

    # Coordinate Conversion Function
    def to_pixel(pts_meters):
        """
        Converts meters to pixels according to the user's specification:
        - X (Longitudinal/Forward) -> Image X Axis (To the Right)
        - Y (Lateral/Left)         -> Image Y Axis (Upward)
        
        """
        x_meters = pts_meters[:, 0]
        y_meters = pts_meters[:, 1]
        
        # X: Right = Forward (Add to center)
        u = CENTER_X + (x_meters * SCALE)
        
        # Y: Up = Left (Subtract from center because image Y goes down)
        v = CENTER_Y - (y_meters * SCALE)
        
        return u, v

    # Visualization Loop
    print("Generating images...")
    for i in range(5): 
        idx = random.randint(0, len(dataset)-1)
        image_tensor, target_waypoints = dataset[idx]
        
        # Prepare background image
        bg_image = image_tensor.permute(1, 2, 0).numpy()
        
        # Inference
        with torch.no_grad():
            input_tensor = image_tensor.unsqueeze(0).to(device)
            predicted_waypoints = model(input_tensor)
            
        # Data in meters
        pred_points = predicted_waypoints[0].cpu().numpy()
        true_points = target_waypoints.numpy()

        # Conversion to pixels
        pred_u, pred_v = to_pixel(pred_points)
        true_u, true_v = to_pixel(true_points)

        # Plotting
        plt.figure(figsize=(10, 10))
        plt.imshow(bg_image)
        
        # Draw Trajectories
        plt.plot(true_u, true_v, 'g-o', linewidth=3, label='Ground Truth', alpha=0.7)
        plt.plot(pred_u, pred_v, 'r-x', linewidth=3, label='Model Prediction')
        
        # Draw the Car (Center and Direction)
        plt.plot(CENTER_X, CENTER_Y, 'bo', markersize=12, label='EGO Vehicle')
        
        # Arrow to indicate the presumed "Forward" direction
        plt.arrow(CENTER_X, CENTER_Y, 40, 0, head_width=10, head_length=10, fc='yellow', ec='yellow', label='X Direction (Forward)')

        plt.legend(loc='upper right')
        plt.title(f"Sample {idx} - Axis Verification")
        plt.axis('off') # Hide axis numbers for cleanliness
        
        # Saving
        plt.savefig(f"inference_corrected_{i}.png")
        plt.close()
        print(f"Saved inference_corrected_{i}.png")

if __name__ == "__main__":
    visualize_prediction()
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from model import NeuralPlanner
from dataset import NuScenesBEVDataset

def visualize_prediction():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference running on: {device}")
    
    # Paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'neural_planner_model.pth')
    data_root = os.path.join(current_dir, '../bev_data')
    
    # Load the model
    # IMPORTANT: Must have the same parameters as during training!
    model = NeuralPlanner(in_channels=3, num_waypoints=10).to(device)
    
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}")
        return

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Weights loaded successfully!")
    except Exception as e:
        print(f"Error loading weights: {e}")
        return

    model.eval() # Turns off Dropout and Batchnorm for inference

    # Load a random sample
    try:
        dataset = NuScenesBEVDataset(data_root=data_root)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {data_root}")
        return
        
    print(f"Dataset loaded: {len(dataset)} samples.")
    
    # Loop to view more than one if needed
    for i in range(3):
        idx = random.randint(0, len(dataset)-1)
        image_tensor, target_waypoints = dataset[idx]
        
        # Prepare the image for visualization (C,H,W) -> (H,W,C)
        bg_image = image_tensor.permute(1, 2, 0).numpy()

        # Prediction
        with torch.no_grad():
            # Add batch dimension [1, 3, 400, 400]
            input_tensor = image_tensor.unsqueeze(0).to(device)
            predicted_waypoints = model(input_tensor) # Output [1, 10, 2]
            
        pred_points = predicted_waypoints[0].cpu().numpy()
        true_points = target_waypoints.numpy()

        # Plotting
        # Conversion parameters (must match generate_data.py)
        # BEV 400x400, Range +/- 50m.
        # Center (200, 200). Resolution 0.25 m/px.
        scale = 4.0
        center_x, center_y = 200, 200
        
        # Conversion Meters -> Pixels for visualization
        # Usually in BEV plots:
        # X (forward 50m) -> Y pixel (up, 0)
        # Y (left 50m) -> X pixel (left, 0)
        
        def to_pixel(pts_meters):
            # pts_meters is [N, 2] (x, y)
            
            # Verify the transformation used in generate_data.py if possible
            
            # If X=forward, Y=left
            px = 200 - (pts_meters[:, 1] * 4)
            py = 200 - (pts_meters[:, 0] * 4)
            return px, py

        pred_px, pred_py = to_pixel(pred_points)
        true_px, true_py = to_pixel(true_points)

        plt.figure(figsize=(8, 8))
        plt.imshow(bg_image)
        plt.plot(true_px, true_py, 'g-o', linewidth=2, label='Ground Truth')
        plt.plot(pred_px, pred_py, 'r-x', linewidth=2, label='Model Prediction')
        plt.plot(center_x, center_y, 'bo', markersize=8, label='EGO')
        
        plt.legend()
        plt.title(f"Sample {idx} - Visual Check")
        plt.tight_layout()
        
        save_path = f"inference_test_{i}.png"
        plt.savefig(save_path)
        print(f"Saved plot: {save_path}")
        plt.show()

if __name__ == "__main__":
    visualize_prediction()
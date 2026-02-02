import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from model import NeuralPlanner
from dataset import NuScenesBEVDataset

def visualize_prediction():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference running on: {device}")
    
    # Graphic Parameters
    IMG_SIZE = 400
    SCALE = 4.0
    CENTER_X = 200     
    CENTER_Y = 200     
    
    # Load paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'neural_planner_model.pth')
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
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import random

from model.model import NeuralPlanner
from model.dataset import NuScenesBEVDataset
from control.lqr import LQRController

# --- Complete System ---
def run_autonomous_stack():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Autonomous Driving Stack on: {device}")
    
    # Init Modules
    planner = NeuralPlanner(in_channels=3, num_waypoints=10).to(device)
    controller = LQRController(wheelbase=2.84)
    
    # Load Weights
    model_path = 'model/best_neural_planner.pth'
    if os.path.exists(model_path):
        planner.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Planner AI loaded from {model_path}")
    else:
        print("Error: Model weights not found!")
        return

    planner.eval()
    dataset = NuScenesBEVDataset(data_root='bev_data')
    
    # Filter for Validation Set (if available)
    val_indices_path = 'model/val_indices.npy'
    if os.path.exists(val_indices_path):
        val_indices = np.load(val_indices_path)
        dataset = torch.utils.data.Subset(dataset, val_indices)
        print(f"Testing on {len(dataset)} UNSCEEN validation samples.")
    else:
        print("Warning: 'val_indices.npy' not found. Testing on RANDOM mix (may include training data).")
    
    # Visualization Parameters
    SCALE = 4.0
    CENTER_X, CENTER_Y = 200, 200

    # Test Loop
    for i in range(5): # Generate 5 examples
        idx = random.randint(0, len(dataset)-1)
        image_tensor, velocity, target_waypoints = dataset[idx] # Subset handles remapping index to original
        
        # Add batch dimension [1, C, H, W] and [1, 1]
        image_tensor = image_tensor.unsqueeze(0).to(device)
        velocity = velocity.unsqueeze(0).to(device)
        
        # A. PERCEPTION & PLANNING (AI)
        with torch.no_grad():
            pred_waypoints = planner(image_tensor, velocity) # Now takes velocity
            pred_waypoints = pred_waypoints.squeeze(0).cpu().numpy() # [10, 2]
            
        # B. CONTROL (LQR)
        # Calculate steering based on predicted trajectory
        # Use actual velocity from dataset for controller (clamp to min 1.0 m/s to avoid div/0)
        v_ego = max(velocity.item(), 1.0)
        steer_rad = controller.compute_steering(pred_waypoints, velocity=v_ego)
        tire_steer_deg = np.degrees(steer_rad)
        steering_wheel_deg= tire_steer_deg * 16.0 # Assuming a steering ratio of 16:1
        
        # --- C. VISUALIZATION "DASHBOARD" ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot 1: BEV Scenario
        bg_image = image_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
        ax1.imshow(bg_image)
        
        # Convert and plot trajectories
        # Ground Truth
        gt = target_waypoints.numpy()
        ax1.plot(CENTER_X + gt[:,0]*SCALE, CENTER_Y - gt[:,1]*SCALE, 'g-o', label='Ground Truth', alpha=0.6)
        # Prediction
        pred_u = CENTER_X + pred_waypoints[:,0]*SCALE
        pred_v = CENTER_Y - pred_waypoints[:,1]*SCALE
        ax1.plot(pred_u, pred_v, 'r-x', linewidth=2, label='AI Plan')
        
        ax1.set_title("Perception & Planning")
        ax1.legend()
        ax1.axis('off')
        
        # Plot 2: Control (Steering Wheel)
        ax2.set_xlim(-1, 1)
        ax2.set_ylim(-1, 1)
        ax2.set_aspect('equal')
        ax2.set_title(f"Control Output\nSteering: {tire_steer_deg:.1f}° | Wheel: {steering_wheel_deg:.1f}°")
        ax2.axis('off')
        
        # Steering Wheel Circle
        circle = patches.Circle((0, 0), 0.8, fill=False, linewidth=5, color='black')
        ax2.add_patch(circle)
        
        # Steering Wheel Spokes (Rotate based on steer_deg)
        # Note: positive steer = left. In trigonometric plot, positive angle = counterclockwise (left).
        # So the rotation is consistent.
        rotation = steering_wheel_deg 
        
        # Central line (direction indicator)
        ax2.plot([0, 0.8 * np.sin(np.radians(-rotation))], 
                 [0, 0.8 * np.cos(np.radians(-rotation))], 'b-', linewidth=4)
        
        # Action Text
        action_text = "STRAIGHT"
        if tire_steer_deg > 2: action_text = "LEFT"
        elif tire_steer_deg < -2: action_text = "RIGHT"
        
        ax2.text(0, -1.2, action_text, ha='center', fontsize=16, fontweight='bold', 
                 color='blue' if abs(tire_steer_deg) > 2 else 'gray')

        plt.tight_layout()
        plt.savefig(f"full_stack_result_{i}.png")
        print(f"Saved full_stack_result_{i}.png -> Steering: {tire_steer_deg:.1f}° | Wheel: {steering_wheel_deg:.1f}°")
        plt.show()

if __name__ == "__main__":
    run_autonomous_stack()
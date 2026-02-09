import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

from torch.utils.data import DataLoader, random_split
from dataset import NuScenesBEVDataset
from model import NeuralPlanner

def train():
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Training parameters
    LR = 1e-4
    EPOCHS = 100
    BATCH_SIZE = 16 
    
    # Model creation
    model = NeuralPlanner(in_channels=3, num_waypoints=10).to(device)
    
    # Instantiate the Dataset
    try:
        full_dataset = NuScenesBEVDataset(data_root='../bev_data')
    except FileNotFoundError:
        print("Error: Folder '../bev_data' not found.")
        return

    # Train/Validation Split (80% / 20%)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Save validation indices for testing
    val_indices = val_dataset.indices
    np.save('val_indices.npy', val_indices)
    print(f"Saved {len(val_indices)} validation indices to 'val_indices.npy'")
    
    # DataLoader creation
    # Increased num_workers for efficiency (set to 0 if Windows causes issues, but 4 is usually safe)
    # pin_memory=True for faster host-to-device transfer
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Data: {len(train_dataset)} training, {len(val_dataset)} validation")
    
    # Optimizer and Loss Function
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    
    # Learning Rate Scheduler
    # Reduces LR by factor of 0.5 if validation loss doesn't improve for 5 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    print("Starting Training...")
    
    train_loss_history = []
    val_loss_history = []
    best_val_loss = float('inf')

    # Teacher Forcing Schedule
    # Start at 1.0 (always use GT) to learn the patterns first.
    # Decay to 0.0 over 50 epochs to gradually introduce autonomy.
    teacher_forcing_ratio = 1.0

    for epoch in range(EPOCHS):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        
        # Decay Teacher Forcing
        if epoch < 50:
             teacher_forcing_ratio = 1.0 * (1 - epoch / 50)
        else:
             teacher_forcing_ratio = 0.0
        
        for i, (inputs, velocity, targets) in enumerate(train_loader):
            inputs, velocity, targets = inputs.to(device), velocity.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            # Pass targets and ratio for Teacher Forcing
            outputs = model(inputs, velocity, teacher_forcing_targets=targets, teacher_forcing_ratio=teacher_forcing_ratio)
            
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient Clipping (Optional but good for stability)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            
        avg_train_loss = running_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)
        
        # --- Validation Phase ---
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, velocity, targets in val_loader:
                inputs, velocity, targets = inputs.to(device), velocity.to(device), targets.to(device)
                outputs = model(inputs, velocity)
                loss = criterion(outputs, targets)
                running_val_loss += loss.item()
        
        avg_val_loss = running_val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)
        
        # Scheduler Step
        scheduler.step(avg_val_loss)
        
        # Checkpointing (Save Best Model)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_neural_planner.pth')
            saved_msg = "(*)" # Indicator for saved model
        else:
            saved_msg = ""
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.2e} {saved_msg}")

    # Loss Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_history, label='Training Loss')
    plt.plot(val_loss_history, label='Validation Loss')
    plt.title("Learning Curve")
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    
    # Save final model
    torch.save(model.state_dict(), 'neural_planner_final.pth')
    print(f"Training Complete. Best Validation Loss: {best_val_loss:.6f}")
    print("Models saved: 'best_neural_planner.pth' (best) and 'neural_planner_final.pth' (final)")
    
    plt.show()

if __name__ == "__main__":
    train()
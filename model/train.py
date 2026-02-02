import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

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
    
    # DataLoader creation
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Data: {len(train_dataset)} training, {len(val_dataset)} validation")
    
    # Optimizer and Loss Function
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    
    print("Starting Training...")
    
    loss_history = []

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        avg_loss = running_loss / len(train_loader)
        
        loss_history.append(avg_loss)
        
        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.6f}")

    # Loss Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Training Loss')
    plt.title("Learning Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Mean Error (MSE in meters^2)")
    plt.legend()
    plt.grid(True)
    
    # Model saving
    plt.show()
    torch.save(model.state_dict(), 'neural_planner_model.pth')
    print("Model saved as neural_planner_model.pth")

if __name__ == "__main__":
    train()
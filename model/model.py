import torch
import torch.nn as nn
from torchvision.models import resnet18

class NeuralPlanner(nn.Module):
    def __init__(self, in_channels=3, num_waypoints=10):
        """
        Initialize the model (ResNet Encoder + GRU Decoder).
        Args:
            in_channels: Number of channels in the BEV image.
            num_waypoints: How many future points we want to predict.
        """
        super(NeuralPlanner, self).__init__()
        
        # --- ENCODER (ResNet18) ---
        self.backbone = resnet18(weights=None)
        
        # Modify input layer (3 channels -> 64)
        self.backbone.conv1 = nn.Conv2d(
            in_channels, 
            64,            
            kernel_size=7,  
            stride=2, 
            padding=3, 
            bias=False
        )
        
        # Remove original FC
        self.encoder_features = self.backbone.fc.in_features # 512
        self.backbone.fc = nn.Identity() # Pass through raw features
        
        # --- DECODER (GRU RNN) ---
        self.hidden_size = 512
        self.num_waypoints = num_waypoints
        
        # GRU Cell: Input is (x,y) = 2 inputs. Hidden state is features = 512.
        self.gru_cell = nn.GRUCell(input_size=2, hidden_size=self.hidden_size)
        
        # Output Head: Hidden State -> Next Waypoint (dx, dy)
        self.regressor = nn.Sequential(
            nn.Linear(self.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 2) # Output: x, y
        )

    def forward(self, x, teacher_forcing_targets=None, teacher_forcing_ratio=0.0):
        """
        Forward Pass with Autoregression.
        
        Args:
            x: Input images [Batch, Channels, H, W]
            teacher_forcing_targets: Ground Truth waypoints [Batch, 10, 2] (optional)
            teacher_forcing_ratio: Probability of using GT input (0.0 to 1.0)
        """
        batch_size = x.size(0)
        
        # 1. Encode Image -> Hidden State
        # [Batch, 512, 1, 1] -> [Batch, 512]
        features = self.backbone(x) 
        hidden = features # Initialize GRU hidden state with Image Features
        
        # 2. Decoding Loop
        predictions = []
        
        # Start at (0,0) (Ego relative position)
        current_input = torch.zeros(batch_size, 2).to(x.device) 
        
        for t in range(self.num_waypoints):
            # GRU Step
            hidden = self.gru_cell(current_input, hidden)
            
            # Predict next point (absolute position in ego frame)
            # We predict the position directly. 
            # (Alternative: predict displacement and add to previous)
            next_point = self.regressor(hidden)
            
            predictions.append(next_point)
            
            # Prepare input for next step
            # Teacher Forcing Logic:
            # If training and lucky roll, use Ground Truth as next input.
            # Otherwise use model's own prediction.
            if teacher_forcing_targets is not None and torch.rand(1).item() < teacher_forcing_ratio:
                 current_input = teacher_forcing_targets[:, t, :] # Use GT for NEXT step input
            else:
                 current_input = next_point # Autoregression
        
        # Stack predictions: [Batch, num_waypoints, 2]
        return torch.stack(predictions, dim=1)

# --- TEST BLOCK ---
"""if __name__ == "__main__":
    # 1. Create the model
    model = NeuralPlanner(in_channels=3, num_waypoints=10)
    print("Model created")
    
    # 2. Create dummy data (Batch=2)
    dummy_input = torch.randn(2, 3, 200, 200)
    
    # 3. Try to make a prediction
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Dimension check
    if output.shape == (2, 10, 2):
        print("Dimensions are correct.")
    else:
        print("Error: Dimensions are not correct!")"""
import torch
import torch.nn as nn
from torchvision.models import resnet18

class NeuralPlanner(nn.Module):
    def __init__(self, in_channels=3, num_waypoints=10):
        """
        Initialize the model.
        Args:
            in_channels: Number of channels in the BEV image (Map, Ego, Prediction).
            num_waypoints: How many future points we want to predict.
        """
        super(NeuralPlanner, self).__init__()
        
        # Uploading backbone ResNet18
        # weights=None (random weights)
        self.backbone = resnet18(weights=None)
        
        # Input Modification
        # ResNet relys on 3 input channels by default
        self.backbone.conv1 = nn.Conv2d(
            in_channels, 
            64,            
            kernel_size=7,  
            stride=2, 
            padding=3, 
            bias=False
        )
        
        # Output Modification
        # We need to reach a level that returns 10 coordinates.
        
        num_features = self.backbone.fc.in_features
        
        # Replace the original 'fc' (Fully Connected) layer
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),                    
            nn.Dropout(0.3),              
            nn.Linear(512, num_waypoints * 2)
        )
        
        self.num_waypoints = num_waypoints

    def forward(self, x):
        """
        Forward Pass.
        The input passes through the layers and the prediction comes out.
        """
        # x input shape: [Batch, Channels, H, W]
        x = self.backbone(x)
        
        # x shape: [Batch, num_waypoints * 2]
        
        # Reshape:
        # -1 does not change batch size
        # self.num_waypoints is the number of points (10)
        # 2 are the coordinates (x, y)
        return x.view(-1, self.num_waypoints, 2)

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
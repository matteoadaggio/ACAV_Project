import torch
import matplotlib.pyplot as plt

def get_mock_batch(batch_size=4, image_size=200, num_waypoints=10):
    """
    Emulates a batch coming from the dataloader.
    
    Args:
        batch_size: How many examples to process together.
        image_size: Resolution of the BEV.
        num_waypoints: How many future points we want to predict.
        
    Returns:
        inputs: Tensor [Batch, 3, H, W] -> The BEV with 3 channels.
        targets: Tensor [Batch, num_waypoints, 2] -> The true (x, y) coordinates.
    """
    
    # --- 1. SIMULATED INPUT CREATION (BEV) ---
    # Channels: 0=Map, 1=Ego, 2=Future Obstacles
    # We use randn to generate noise, but in reality they will be 0 and 1.
    inputs = torch.randn(batch_size, 3, image_size, image_size)
    
    # --- 2. SIMULATED TARGET CREATION (Trajectory) ---
    # We simulate 10 points (x, y). 
    # We assume the car is at the center or bottom of the image.
    # For now, we generate random numbers, but ideally these are coordinates in meters.
    targets = torch.randn(batch_size, num_waypoints, 2)
    
    return inputs, targets

# --- VISUAL TEST ---
if __name__ == "__main__":

    BATCH_SIZE = 2
    fake_inputs, fake_targets = get_mock_batch(batch_size=BATCH_SIZE)
    
    print(f"Input Shape: {fake_inputs.shape}")  
    print(f"Target Shape: {fake_targets.shape}")
    
    # Visualize the first example in the batch
    first_image = fake_inputs[0] 
    
    # To visualize with Matplotlib, we need to change the dimension order:
    # From [Channels, H, W] to [H, W, Channels]
    img_to_show = first_image.permute(1, 2, 0) 
    
    # Normalize between 0 and 1 for visualization (only for the plotting purpose)
    img_to_show = (img_to_show - img_to_show.min()) / (img_to_show.max() - img_to_show.min())

    plt.figure(figsize=(5,5))
    plt.title("Mock BEV Visualization (Random Noise)")
    plt.imshow(img_to_show.numpy())
    plt.axis('off')
    plt.show()
    
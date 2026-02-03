import torch
from torch.utils.data import Dataset
import numpy as np
import os
import glob
from PIL import Image
import torchvision.transforms as transforms

class NuScenesBEVDataset(Dataset):
    def __init__(self, data_root='../bev_data', transform=None):
        """
        Args:
            data_root (string): path to the bev_data folder.
        """
        self.data_root = data_root
        self.images_dir = os.path.join(data_root, 'images')
        self.waypoints_dir = os.path.join(data_root, 'waypoints')
        
        # Find all image files sorted
        self.image_files = sorted(glob.glob(os.path.join(self.images_dir, '*.png')))
        
        # Check data presence
        if len(self.image_files) == 0:
            raise RuntimeError(f"No image found in {self.images_dir}")
            
        print(f"Dataset loaded: found {len(self.image_files)} samples.")

        # PIL to Tensor transformation
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(), 
            ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 1. Load image
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB') 
        image_tensor = self.transform(image)

        # 2. Load waypoints
        base_name = os.path.basename(img_path).replace('.png', '_waypoints.npy')
        waypoint_path = os.path.join(self.waypoints_dir, base_name)
        
        # Alternative naming handling
        if not os.path.exists(waypoint_path):
            waypoint_path = os.path.join(self.waypoints_dir, base_name.replace('_sample', ''))

        if not os.path.exists(waypoint_path):
             raise FileNotFoundError(f"Waypoint not found for {base_name}")

        waypoints = np.load(waypoint_path)
        
        # The model expects 10 points, handle edge cases
        target_num_points = 10
        current_num_points = waypoints.shape[0]
        
        if current_num_points > target_num_points:
            waypoints = waypoints[:target_num_points]
            
        elif current_num_points < target_num_points:
            if current_num_points == 0:
                waypoints = np.zeros((target_num_points, 2), dtype=np.float32)
            else:
                missing = target_num_points - current_num_points
                last_point = waypoints[-1]
                padding = np.tile(last_point, (missing, 1))
                waypoints = np.vstack((waypoints, padding))

        waypoints_tensor = torch.from_numpy(waypoints).float()
        return image_tensor, waypoints_tensor

# --- DATASET TEST ---
"""if __name__ == "__main__":
    try:
        ds = NuScenesBEVDataset(data_root='../bev_data')
        img, wp = ds[0]
        print(f"Input Image Shape: {img.shape}") # Expected: [3, 400, 400]
        print(f"Target Waypoints Shape: {wp.shape}") # Expected: [10, 2]
        print("Dataset Test Passed")
    except Exception as e:
        print(f"Error in Dataset: {e}")"""
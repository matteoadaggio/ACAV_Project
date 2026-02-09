import torch
from torch.utils.data import Dataset
import numpy as np
import os
import glob
from PIL import Image
import torchvision.transforms as transforms

import json
import random

class NuScenesBEVDataset(Dataset):
    def __init__(self, data_root='../bev_data', transform=None, augment=False):
        """
        Args:
            data_root (string): path to the bev_data folder.
            augment (bool): If True, applies random horizontal flips.
        """
        self.data_root = data_root
        self.augment = augment
        self.images_dir = os.path.join(data_root, 'images')
        self.waypoints_dir = os.path.join(data_root, 'waypoints')
        self.metadata_path = os.path.join(data_root, 'metadata', 'dataset.json') # New metadata file
        
        self.samples = []
        
        # Try to load from metadata JSON (Preferred)
        if os.path.exists(self.metadata_path):
            print(f"Loading metadata from {self.metadata_path}...")
            with open(self.metadata_path, 'r') as f:
                self.samples = json.load(f)
            print(f"Dataset loaded: {len(self.samples)} samples found in metadata.")
        else:
            # Fallback to file globbing (Legacy)
            print("Metadata not found. Falling back to file globbing (velocity will be 0.0)")
            image_files = sorted(glob.glob(os.path.join(self.images_dir, '*.png')))
            for img_path in image_files:
                self.samples.append({
                    'image_path': img_path,
                    'waypoint_path': img_path.replace('images', 'waypoints').replace('.png', '_waypoints.npy'),
                    'ego_velocity': 0.0 # Default
                })
            
            if len(self.samples) == 0:
                raise RuntimeError(f"No image found in {self.images_dir}")
            print(f"Dataset loaded: found {len(self.samples)} samples via glob.")

        # PIL to Tensor transformation
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(), 
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        
        # 1. Load image
        # Metadata might store absolute or relative path, handle both
        img_path = item['image_path']
        if not os.path.isabs(img_path) and not os.path.exists(img_path):
             # Fix for Windows paths on Linux: replace backslash with forward slash
             filename = os.path.basename(img_path.replace('\\', '/'))
             img_path = os.path.join(self.data_root, 'images', filename)
             
        if not os.path.exists(img_path):
             # Fallback logic for colab/local path mismatch
             filename = os.path.basename(item['image_path'].replace('\\', '/'))
             img_path = os.path.join(self.images_dir, filename)
             
        image = Image.open(img_path).convert('RGB') 
        
        # --- Augmentation (Random Horizontal Flip) ---
        is_flipped = False
        if self.augment and random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            is_flipped = True
            
        image_tensor = self.transform(image)

        # 2. Load waypoints
        # Metadata might store absolute or relative path, handle both
        wp_path = item['waypoint_path']
        if not os.path.isabs(wp_path) and not os.path.exists(wp_path):
             # Fix for Windows paths on Linux
             filename = os.path.basename(wp_path.replace('\\', '/'))
             wp_path = os.path.join(self.data_root, 'waypoints', filename)
             
        if not os.path.exists(wp_path):
             # Fallback logic for colab/local path mismatch
             filename = os.path.basename(item['waypoint_path'].replace('\\', '/'))
             wp_path = os.path.join(self.waypoints_dir, filename)
        
        waypoints = np.load(wp_path)
        
        # Load Velocity (Input)
        # Default to 0.0 if not found (backward compatibility)
        curr_velocity = item.get('ego_velocity', 0.0) 
        velocity_tensor = torch.tensor([curr_velocity], dtype=torch.float32)

        # Ensure target_num_points = 10
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

        # Apply augmentation directly to numpy array
        if is_flipped:
            waypoints[:, 0] *= -1 # Negate X coordinate (Left <-> Right)

        waypoints_tensor = torch.from_numpy(waypoints).float()
        
        return image_tensor, velocity_tensor, waypoints_tensor

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
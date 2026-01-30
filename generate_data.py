"""
BEV Generator for nuScenes Dataset
Generates Bird's Eye View representations for waypoint prediction training
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility
from pyquaternion import Quaternion
from typing import Tuple, List, Dict
import cv2
from tqdm import tqdm


class BEVGenerator:
    """Generate Bird's Eye View representations from nuScenes data"""
    
    def __init__(self, nusc: NuScenes, 
                 bev_size: Tuple[int, int] = (400, 400),
                 bev_range: Tuple[float, float, float, float] = (-50, 50, -50, 50),
                 output_dir: str = './bev_data'):
        """
        Initialize BEV Generator
        
        Args:
            nusc: NuScenes instance
            bev_size: Size of BEV image (height, width) in pixels
            bev_range: Range of BEV in meters (x_min, x_max, y_min, y_max)
            output_dir: Directory to save BEV images
        """
        self.nusc = nusc
        self.bev_size = bev_size
        self.bev_range = bev_range
        self.output_dir = output_dir
        
        # Calculate resolution (meters per pixel)
        self.x_range = bev_range[1] - bev_range[0]
        self.y_range = bev_range[3] - bev_range[2]
        self.resolution_x = self.x_range / bev_size[1]
        self.resolution_y = self.y_range / bev_size[0]
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'waypoints'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'metadata'), exist_ok=True)
        
    def world_to_bev(self, points: np.ndarray) -> np.ndarray:
        """
        Convert world coordinates to BEV image coordinates
        
        Args:
            points: Nx3 array of points in world coordinates (x, y, z)
            
        Returns:
            Nx2 array of points in BEV image coordinates (u, v)
        """
        # Extract x, y coordinates
        x = points[:, 0]
        y = points[:, 1]
        
        # Convert to pixel coordinates
        u = ((x - self.bev_range[0]) / self.resolution_x).astype(np.int32)
        v = ((self.bev_range[3] - y) / self.resolution_y).astype(np.int32)
        
        # Filter points outside BEV range
        mask = (u >= 0) & (u < self.bev_size[1]) & (v >= 0) & (v < self.bev_size[0])
        
        return np.stack([u, v], axis=1), mask
    
    def get_lidar_data(self, sample_data_token: str) -> np.ndarray:
        """
        Get lidar point cloud in global coordinates
        
        Args:
            sample_data_token: Token for LIDAR_TOP sample_data
            
        Returns:
            Nx4 array of points (x, y, z, intensity) in global coordinates
        """
        # Get sample data record
        sd_rec = self.nusc.get('sample_data', sample_data_token)
        
        # Get lidar point cloud
        pcl_path = os.path.join(self.nusc.dataroot, sd_rec['filename'])
        pc = LidarPointCloud.from_file(pcl_path)
        
        # Get calibrated sensor and ego pose
        cs_rec = self.nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
        pose_rec = self.nusc.get('ego_pose', sd_rec['ego_pose_token'])
        
        # Transform to global coordinates
        # First, rotate and translate from sensor to ego vehicle frame
        pc.rotate(Quaternion(cs_rec['rotation']).rotation_matrix)
        pc.translate(np.array(cs_rec['translation']))
        
        # Then, rotate and translate from ego vehicle to global frame
        pc.rotate(Quaternion(pose_rec['rotation']).rotation_matrix)
        pc.translate(np.array(pose_rec['translation']))
        
        return pc.points.T
    
    def get_ego_pose(self, sample_data_token: str) -> Tuple[np.ndarray, Quaternion]:
        """
        Get ego vehicle pose
        
        Args:
            sample_data_token: Token for sample_data
            
        Returns:
            translation: 3D position in global coordinates
            rotation: Quaternion rotation
        """
        sd_rec = self.nusc.get('sample_data', sample_data_token)
        pose_rec = self.nusc.get('ego_pose', sd_rec['ego_pose_token'])
        
        translation = np.array(pose_rec['translation'])
        rotation = Quaternion(pose_rec['rotation'])
        
        return translation, rotation
    
    def extract_future_waypoints(self, sample_token: str, num_waypoints: int = 10) -> np.ndarray:
        """
        Extract future waypoints from subsequent samples
        
        Args:
            sample_token: Current sample token
            num_waypoints: Number of future waypoints to extract
            
        Returns:
            Nx2 array of waypoints in ego vehicle frame (x, y)
        """
        sample = self.nusc.get('sample', sample_token)
        current_ego_pos, current_ego_rot = self.get_ego_pose(sample['data']['LIDAR_TOP'])
        
        waypoints = []
        current_token = sample_token
        
        for _ in range(num_waypoints):
            # Get next sample
            current_sample = self.nusc.get('sample', current_token)
            next_token = current_sample['next']
            
            if next_token == '':
                break
                
            next_sample = self.nusc.get('sample', next_token)
            next_ego_pos, _ = self.get_ego_pose(next_sample['data']['LIDAR_TOP'])
            
            # Transform to current ego frame
            global_offset = next_ego_pos - current_ego_pos
            rot_matrix = current_ego_rot.inverse.rotation_matrix
            ego_offset = rot_matrix @ global_offset
            
            waypoints.append(ego_offset[:2])  # Only x, y
            current_token = next_token
        
        return np.array(waypoints) if waypoints else np.zeros((0, 2))
    
    def create_bev_image(self, sample_token: str, 
                         use_lidar: bool = True,
                         use_annotations: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Create BEV image for a sample
        
        Args:
            sample_token: Sample token
            use_lidar: Whether to include lidar points
            use_annotations: Whether to include object annotations
            
        Returns:
            bev_image: BEV image as numpy array
            metadata: Dictionary with sample metadata
        """
        sample = self.nusc.get('sample', sample_token)
        
        # Initialize BEV image (3 channels in BGR order for OpenCV)
        # Channel 0 (B): Static objects
        # Channel 1 (G): Dynamic objects  
        # Channel 2 (R): Lidar points
        bev_image = np.zeros((self.bev_size[0], self.bev_size[1], 3), dtype=np.uint8)
        
        # Get ego pose for this sample
        ego_pos, ego_rot = self.get_ego_pose(sample['data']['LIDAR_TOP'])
        
        # Channel 2 (R): Lidar points
        if use_lidar:
            lidar_token = sample['data']['LIDAR_TOP']
            points_global = self.get_lidar_data(lidar_token)
            
            # Transform to ego frame
            points_ego = points_global[:, :3] - ego_pos
            # Apply rotation using rotation matrix instead of quaternion.rotate()
            rot_matrix = ego_rot.inverse.rotation_matrix
            points_ego = (rot_matrix @ points_ego.T).T
            
            # Convert to BEV coordinates
            bev_coords, mask = self.world_to_bev(points_ego)
            bev_coords = bev_coords[mask]
            
            # Draw lidar points in RED channel (index 2 for BGR)
            if len(bev_coords) > 0:
                bev_image[bev_coords[:, 1], bev_coords[:, 0], 2] = 255
        
        # Channels 1 & 2: Object annotations
        if use_annotations:
            for ann_token in sample['anns']:
                ann = self.nusc.get('sample_annotation', ann_token)
                
                # Get box in global coordinates
                box = Box(ann['translation'], ann['size'], Quaternion(ann['rotation']))
                
                # Transform to ego frame
                box.translate(-ego_pos)
                box.rotate(ego_rot.inverse)
                
                # Get box corners
                corners = box.corners()  # 3x8 array
                
                # Project to BEV (use bottom corners)
                bottom_corners = corners[:2, [0, 1, 2, 3]].T  # 4x2 array
                bev_corners, mask = self.world_to_bev(
                    np.hstack([bottom_corners, np.zeros((4, 1))])
                )
                
                if mask.sum() > 0:
                    bev_corners = bev_corners[mask]
                    
                    # Determine if dynamic or static
                    category = ann['category_name']
                    is_vehicle = category.startswith('vehicle')
                    is_pedestrian = category.startswith('human')
                    
                    # Channel assignment in BGR:
                    # Green (index 1) = Dynamic (vehicles, pedestrians)
                    # Blue (index 0) = Static (barriers, cones, etc)
                    channel = 1 if (is_vehicle or is_pedestrian) else 0
                    
                    # Draw filled polygon
                    if len(bev_corners) >= 3:
                        # Reshape to the format OpenCV expects: array of shape (n_points, 1, 2)
                        pts = bev_corners.astype(np.int32).reshape((-1, 1, 2))
                        # Create a temporary single-channel image
                        temp_img = bev_image[:, :, channel].copy()
                        cv2.fillPoly(temp_img, [pts], 255)
                        bev_image[:, :, channel] = temp_img
        
        # Create metadata
        metadata = {
            'sample_token': sample_token,
            'scene_token': sample['scene_token'],
            'timestamp': sample['timestamp'],
            'ego_position': ego_pos.tolist(),
            'ego_rotation': ego_rot.elements.tolist(),
        }
        
        return bev_image, metadata
    
    def process_scene(self, scene_token: str) -> List[Dict]:
        """
        Process all samples in a scene
        
        Args:
            scene_token: Scene token
            
        Returns:
            List of dictionaries with sample information
        """
        scene = self.nusc.get('scene', scene_token)
        scene_name = scene['name']
        
        print(f"\nProcessing scene: {scene_name}")
        
        # Get all samples in scene
        sample_tokens = []
        current_token = scene['first_sample_token']
        
        while current_token != '':
            sample_tokens.append(current_token)
            current_sample = self.nusc.get('sample', current_token)
            current_token = current_sample['next']
        
        # Process each sample
        results = []
        for idx, sample_token in enumerate(tqdm(sample_tokens, desc=f"Samples in {scene_name}")):
            # Generate BEV image
            bev_image, metadata = self.create_bev_image(sample_token)
            
            # Extract future waypoints
            waypoints = self.extract_future_waypoints(sample_token, num_waypoints=10)
            
            # Save BEV image (already in BGR format for cv2.imwrite)
            image_filename = f"{scene_name}_sample_{idx:04d}.png"
            image_path = os.path.join(self.output_dir, 'images', image_filename)
            cv2.imwrite(image_path, bev_image)
            
            # Save waypoints
            waypoint_filename = f"{scene_name}_sample_{idx:04d}_waypoints.npy"
            waypoint_path = os.path.join(self.output_dir, 'waypoints', waypoint_filename)
            np.save(waypoint_path, waypoints)
            
            # Store result
            result = {
                'scene_name': scene_name,
                'sample_idx': idx,
                'sample_token': sample_token,
                'image_path': image_path,
                'waypoint_path': waypoint_path,
                'num_waypoints': len(waypoints),
                **metadata
            }
            results.append(result)
        
        return results
    
    def process_all_scenes(self) -> List[Dict]:
        """
        Process all scenes in the dataset
        
        Returns:
            List of all sample information
        """
        all_results = []
        
        for scene in self.nusc.scene:
            scene_results = self.process_scene(scene['token'])
            all_results.extend(scene_results)
        
        # Save dataset metadata
        import json
        metadata_path = os.path.join(self.output_dir, 'metadata', 'dataset.json')
        with open(metadata_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n✓ Processed {len(all_results)} samples from {len(self.nusc.scene)} scenes")
        print(f"✓ Saved to: {self.output_dir}")
        
        return all_results
    
    def create_channel_debug_image(self, sample_token: str, save_path: str = None):
        """
        Create a debug image showing each channel separately with statistics
        
        Args:
            sample_token: Sample token
            save_path: Path to save debug image
        """
        # Generate BEV
        bev_image, metadata = self.create_bev_image(sample_token)
        
        # Create figure with 4 subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 16))
        
        # Combined RGB (convert BGR to RGB for display)
        bev_rgb = cv2.cvtColor(bev_image, cv2.COLOR_BGR2RGB)
        axes[0, 0].imshow(bev_rgb)
        axes[0, 0].set_title('Combined RGB BEV')
        axes[0, 0].axis('off')
        
        # Red channel (Lidar) - index 2 in BGR
        red_pixels = np.sum(bev_image[:, :, 2] > 0)
        axes[0, 1].imshow(bev_image[:, :, 2], cmap='Reds', vmin=0, vmax=255)
        axes[0, 1].set_title(f'RED: Lidar Points\n({red_pixels} pixels, {red_pixels/(self.bev_size[0]*self.bev_size[1])*100:.2f}% coverage)')
        axes[0, 1].axis('off')
        
        # Green channel (Dynamic objects) - index 1 in BGR
        green_pixels = np.sum(bev_image[:, :, 1] > 0)
        axes[1, 0].imshow(bev_image[:, :, 1], cmap='Greens', vmin=0, vmax=255)
        axes[1, 0].set_title(f'GREEN: Dynamic Objects\n({green_pixels} pixels, {green_pixels/(self.bev_size[0]*self.bev_size[1])*100:.2f}% coverage)')
        axes[1, 0].axis('off')
        
        # Blue channel (Static objects) - index 0 in BGR
        blue_pixels = np.sum(bev_image[:, :, 0] > 0)
        axes[1, 1].imshow(bev_image[:, :, 0], cmap='Blues', vmin=0, vmax=255)
        axes[1, 1].set_title(f'BLUE: Static Objects\n({blue_pixels} pixels, {blue_pixels/(self.bev_size[0]*self.bev_size[1])*100:.2f}% coverage)')
        axes[1, 1].axis('off')
        
        plt.suptitle(f'Channel Debug - Sample: {sample_token[:8]}...', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved channel debug image: {save_path}")
        else:
            plt.show()
        
        plt.close()
        
        # Print statistics
        print(f"\nChannel Statistics for sample {sample_token[:8]}:")
        print(f"  RED (Lidar):          {red_pixels:6d} pixels ({red_pixels/(self.bev_size[0]*self.bev_size[1])*100:5.2f}%)")
        print(f"  GREEN (Dynamic):      {green_pixels:6d} pixels ({green_pixels/(self.bev_size[0]*self.bev_size[1])*100:5.2f}%)")
        print(f"  BLUE (Static):        {blue_pixels:6d} pixels ({blue_pixels/(self.bev_size[0]*self.bev_size[1])*100:5.2f}%)")
        print(f"  Total non-zero:       {red_pixels + green_pixels + blue_pixels:6d} pixels")
    
    def visualize_sample(self, sample_token: str, save_path: str = None):
        """
        Visualize a sample with BEV and waypoints
        
        Args:
            sample_token: Sample token
            save_path: Path to save visualization (optional)
        """
        # Generate BEV
        bev_image, _ = self.create_bev_image(sample_token)
        
        # Extract waypoints
        waypoints = self.extract_future_waypoints(sample_token, num_waypoints=10)
        
        # Create visualization
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        # Show BEV image (convert BGR to RGB for matplotlib)
        bev_rgb = cv2.cvtColor(bev_image, cv2.COLOR_BGR2RGB)
        ax.imshow(bev_rgb)
        
        # Plot waypoints
        if len(waypoints) > 0:
            waypoint_pixels, mask = self.world_to_bev(
                np.hstack([waypoints, np.zeros((len(waypoints), 1))])
            )
            waypoint_pixels = waypoint_pixels[mask]
            
            if len(waypoint_pixels) > 0:
                ax.plot(waypoint_pixels[:, 0], waypoint_pixels[:, 1], 
                       'yo-', markersize=8, linewidth=2, label='Future Waypoints')
                ax.plot(waypoint_pixels[0, 0], waypoint_pixels[0, 1], 
                       'go', markersize=12, label='Next Waypoint')
        
        # Mark ego vehicle position (center)
        center = (self.bev_size[1] // 2, self.bev_size[0] // 2)
        ax.plot(center[0], center[1], 'r*', markersize=20, label='Ego Vehicle')
        
        ax.set_title(f'BEV Visualization\nSample: {sample_token[:8]}...')
        ax.legend()
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to: {save_path}")
        
        plt.show()


def main():
    """Main function to generate BEV data"""
    
    # Configuration
    DATAROOT = './data/v1.0-mini'
    OUTPUT_DIR = './bev_data'
    
    # BEV parameters
    BEV_SIZE = (400, 400)  # pixels
    BEV_RANGE = (-50, 50, -50, 50)  # meters (x_min, x_max, y_min, y_max)
    
    print("=" * 60)
    print("nuScenes BEV Generator")
    print("=" * 60)
    
    # Load nuScenes
    print(f"\nLoading nuScenes dataset from: {DATAROOT}")
    nusc = NuScenes(version='v1.0-mini', dataroot=DATAROOT, verbose=True)
    
    # Create BEV generator
    print(f"\nInitializing BEV Generator")
    print(f"  BEV Size: {BEV_SIZE}")
    print(f"  BEV Range: {BEV_RANGE} meters")
    print(f"  Output Directory: {OUTPUT_DIR}")
    
    bev_gen = BEVGenerator(
        nusc=nusc,
        bev_size=BEV_SIZE,
        bev_range=BEV_RANGE,
        output_dir=OUTPUT_DIR
    )
    
    # Process all scenes
    results = bev_gen.process_all_scenes()
    
    # Visualize a few samples
    print("\n" + "=" * 60)
    print("Creating Sample Visualizations")
    print("=" * 60)
    
    vis_dir = os.path.join(OUTPUT_DIR, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Visualize first sample from each scene
    for i, scene in enumerate(nusc.scene[:3]):  # First 3 scenes
        sample_token = scene['first_sample_token']
        
        # Regular visualization
        save_path = os.path.join(vis_dir, f'scene_{i}_sample_0.png')
        bev_gen.visualize_sample(sample_token, save_path=save_path)
        
        # Debug channel visualization
        debug_path = os.path.join(vis_dir, f'scene_{i}_channel_debug.png')
        bev_gen.create_channel_debug_image(sample_token, save_path=debug_path)
    
    print("\n" + "=" * 60)
    print("BEV Generation Complete!")
    print("=" * 60)
    print(f"\nGenerated {len(results)} BEV samples")
    print(f"Saved to: {OUTPUT_DIR}")
    print("\nDirectory structure:")
    print(f"  {OUTPUT_DIR}/images/          - BEV images (PNG)")
    print(f"  {OUTPUT_DIR}/waypoints/       - Future waypoints (NPY)")
    print(f"  {OUTPUT_DIR}/metadata/        - Dataset metadata (JSON)")
    print(f"  {OUTPUT_DIR}/visualizations/  - Sample visualizations (PNG)")


if __name__ == '__main__':
    main()

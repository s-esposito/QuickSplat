from typing import Optional, Dict, Any, List
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2

from pycolmap.pycolmap.scene_manager import SceneManager

from dataset.utils import (
    focal2fov,
    getWorld2View2,
    getProjectionMatrix,
    farthest_point_sampling,
    voxelize,
)


# Target max extent after normalization (matches typical ScanNetPP scene scale)
TARGET_SCENE_EXTENT = 10.0


def compute_scene_normalization(source_path: Path, scene_id: str):
    """Compute normalization parameters (center, scale) from COLMAP point cloud.

    Returns (center, scale_factor) such that:
        xyz_normalized = (xyz - center) * scale_factor
    After normalization, the max extent of the point cloud is TARGET_SCENE_EXTENT.
    """
    sm = SceneManager(str(source_path / scene_id / "sparse" / "0"))
    sm.load()
    xyz = sm.points3D.astype(np.float64)

    bbox_min = xyz.min(axis=0)
    bbox_max = xyz.max(axis=0)
    center = (bbox_min + bbox_max) / 2.0
    extent = bbox_max - bbox_min
    max_extent = extent.max()

    if max_extent < 1e-6:
        scale_factor = 1.0
    else:
        scale_factor = TARGET_SCENE_EXTENT / max_extent

    return center, scale_factor, sm


def normalize_c2w(c2w: np.ndarray, center: np.ndarray, scale_factor: float) -> np.ndarray:
    """Apply scene normalization to a camera-to-world matrix (only translates + scales position)."""
    c2w_norm = c2w.copy()
    c2w_norm[:3, 3] = (c2w[:3, 3] - center) * scale_factor
    return c2w_norm


class ColmapDataset(Dataset):
    """Single-scene dataset that loads camera poses and intrinsics from COLMAP binary files.

    Frame selection is driven by a views_split dict with 'context' (train) and 'target' (test) indices
    into the sorted list of image filenames on disk.

    Scene coordinates are normalized so the point cloud fits within ~TARGET_SCENE_EXTENT meters,
    matching the scale expected by the model (trained on ScanNetPP indoor scenes).
    """

    def __init__(
        self,
        source_path: str,
        scene_id: str,
        split: str,
        views_split: Dict[str, List[int]],
        image_dir: str = "images_4",
        downsample: int = 1,
        num_train_frames: int = -1,
        subsample_randomness: bool = False,
        load_depth: bool = False,
    ):
        self.source_path = Path(source_path)
        self.scene_id = scene_id
        self.split = split
        self.downsample = downsample
        self.load_depth = load_depth

        scene_dir = self.source_path / scene_id
        image_dir_path = scene_dir / image_dir

        # 1. List available images sorted by name
        all_image_names = sorted(os.listdir(image_dir_path))

        # 2. Load COLMAP reconstruction and compute normalization
        center, scale_factor, sm = compute_scene_normalization(self.source_path, scene_id)

        # 3. Build name -> COLMAP image lookup
        name_to_colmap_img = {}
        for img_id, img in sm.images.items():
            name_to_colmap_img[img.name] = img

        # 4. Map split indices to image names
        train_names = [all_image_names[i] for i in views_split["context"]]
        test_names = [all_image_names[i] for i in views_split["target"]]

        # 5. Build frame data for all frames (train + test for AABB computation)
        def _build_frame_data(image_name):
            colmap_img = name_to_colmap_img[image_name]
            cam = sm.cameras[colmap_img.camera_id]

            # Build world-to-camera from COLMAP (already in COLMAP/OpenCV convention)
            w2c = np.eye(4, dtype=np.float64)
            w2c[:3, :3] = colmap_img.R()
            w2c[:3, 3] = colmap_img.tvec
            c2w = np.linalg.inv(w2c).astype(np.float64)

            # Normalize camera position to match point cloud normalization
            c2w = normalize_c2w(c2w, center, scale_factor)
            w2c = np.linalg.inv(c2w).astype(np.float64)

            return {
                "file_path": image_dir_path / image_name,
                "depth_path": scene_dir / "depth" / image_name.replace(".png", "_depth.png"),
                "camera_to_world": c2w,
                "world_to_camera": w2c,
                "fx": cam.fx,
                "fy": cam.fy,
                "cx": cam.cx,
                "cy": cam.cy,
                "height": cam.height,
                "width": cam.width,
            }

        train_frames = [_build_frame_data(n) for n in train_names]
        test_frames = [_build_frame_data(n) for n in test_names]

        # Compute AABB from all camera positions
        all_xyz = []
        for frame in train_frames + test_frames:
            all_xyz.append(frame["camera_to_world"][:3, 3])
        all_xyz = np.stack(all_xyz, axis=0)
        self.all_xyz = all_xyz

        # Subsample training frames if needed
        if num_train_frames > 0 and len(train_frames) > num_train_frames:
            invalid_mask = np.ones(all_xyz.shape[0], dtype=bool)
            invalid_mask[:len(train_frames)] = False

            if subsample_randomness:
                generator = None
            else:
                generator = np.random.RandomState(0)

            sample_train_indices = farthest_point_sampling(
                all_xyz,
                num_train_frames,
                invalid_mask=invalid_mask,
                randomness=subsample_randomness,
                generator=generator,
            )
            train_frames = [train_frames[i] for i in sample_train_indices]

        self.num_train_frames = len(train_frames)
        self.num_test_frames = len(test_frames)

        if split == "train":
            all_frames = train_frames
        else:
            all_frames = test_frames

        self.data = all_frames
        self.scene_scale = self._get_global_scale(self.data)

    def _get_global_scale(self, data) -> float:
        camera_centers = []
        for x in data:
            camera_centers.append(x["camera_to_world"][:3, 3])
        camera_centers = np.array(camera_centers)
        avg_cam_center = np.mean(camera_centers, axis=0)
        dists = np.linalg.norm(camera_centers - avg_cam_center, axis=1)
        radius = np.max(dists) * 1.1
        return radius

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        frame = self._load_frame(self.data, idx, bbox=None)
        frame["file_name"] = self.data[idx]["file_path"].name
        frame["sample_idx"] = idx
        frame["scene_id"] = self.scene_id
        return frame

    def _load_frame(self, frame_data, frame_idx: int, bbox: Optional[torch.Tensor]) -> Dict[str, Any]:
        x = frame_data[frame_idx]
        world_to_camera = x["world_to_camera"]
        image = Image.open(x["file_path"])
        width, height = image.size
        fx = x["fx"]
        fy = x["fy"]

        if self.downsample > 1:
            image = image.resize(
                (x["width"] // self.downsample, x["height"] // self.downsample),
                resample=Image.BILINEAR,
            )

        if self.load_depth and "depth_path" in x:
            if os.path.exists(x["depth_path"]):
                depth = cv2.imread(str(x["depth_path"]), cv2.IMREAD_UNCHANGED)
                depth = depth.astype(np.float32) / 1000.0
                if self.downsample > 1:
                    depth = cv2.resize(
                        depth,
                        (width // self.downsample, height // self.downsample),
                        interpolation=cv2.INTER_NEAREST,
                    )
            else:
                # Depth not available for this dataset — return zeros
                dh = x["height"] // self.downsample if self.downsample > 1 else height
                dw = x["width"] // self.downsample if self.downsample > 1 else width
                depth = np.zeros((dh, dw), dtype=np.float32)

        image = np.array(image, dtype=np.float32) / 255.0
        height, width = image.shape[:2]
        fx = x["fx"] / self.downsample
        fy = x["fy"] / self.downsample
        cx = x["cx"] / self.downsample
        cy = x["cy"] / self.downsample
        fov_x = focal2fov(fx, width)
        fov_y = focal2fov(fy, height)
        intrinsic = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1],
        ], dtype=np.float32)

        outputs = {
            "rgb": torch.from_numpy(image).permute(2, 0, 1).float(),
            "intrinsic": torch.from_numpy(intrinsic).float(),
            "camera_to_world": torch.from_numpy(x["camera_to_world"]).float(),
            "world_to_camera": torch.from_numpy(world_to_camera).float(),
            "fov_x": torch.tensor(fov_x),
            "fov_y": torch.tensor(fov_y),
            "height": torch.tensor(height),
            "width": torch.tensor(width),
        }
        if self.load_depth:
            outputs["depth"] = torch.from_numpy(depth)

        rot = np.transpose(world_to_camera[:3, :3])
        trans = world_to_camera[:3, 3]
        world_view_transform = torch.from_numpy(getWorld2View2(rot, trans)).transpose(0, 1)
        projection_matrix = getProjectionMatrix(
            znear=0.01,
            zfar=100.0,
            fovX=fov_x,
            fovY=fov_y,
        ).transpose(0, 1)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        outputs["world_view_transform"] = world_view_transform
        outputs["projection_matrix"] = projection_matrix
        outputs["full_proj_transform"] = full_proj_transform
        outputs["camera_center"] = world_view_transform.inverse()[3, :3]

        if bbox is not None:
            outputs["bbox"] = bbox
        return outputs


class ColmapPointDataset:
    """Multi-scene point cloud dataset that loads points directly from COLMAP reconstructions.

    Scene list and views split are derived from a views split JSON file.
    Points are normalized per-scene to match ScanNetPP scale (~TARGET_SCENE_EXTENT meters).
    """

    def __init__(
        self,
        source_path: str,
        views_split_path: str,
        voxel_size: float = 0.02,
        max_points: int = 600_000,
    ):
        self.source_path = Path(source_path)
        self.voxel_size = voxel_size
        self.max_points = max_points

        with open(views_split_path, "r") as f:
            self.views_split_data = json.load(f)

        self.scene_list = sorted(self.views_split_data.keys())
        self.scene_id_to_idx = {sid: i for i, sid in enumerate(self.scene_list)}

    def __len__(self):
        return len(self.scene_list)

    def load_colmap_points(self, scene_id: str):
        """Load 3D points from COLMAP and normalize to target scene scale."""
        center, scale_factor, sm = compute_scene_normalization(self.source_path, scene_id)
        xyz = sm.points3D.astype(np.float32)
        rgb = sm.point3D_colors.astype(np.float32) / 255.0

        # Remove zero-color noise points (common in COLMAP SfM reconstructions)
        valid_mask = ~(rgb == 0).all(axis=1)
        xyz = xyz[valid_mask]
        rgb = rgb[valid_mask]

        # Apply normalization: center + scale to match ScanNetPP extent
        xyz = (xyz - center.astype(np.float32)) * np.float32(scale_factor)

        return xyz, rgb

    def load_voxelized_colmap_points(self, scene_id: str, voxel_size: float):
        """Load and voxelize COLMAP points — same interface as MultiScannetppPointDataset."""
        xyz, rgb = self.load_colmap_points(scene_id)
        xyz_world = xyz.copy()

        bbox_min = np.min(xyz, axis=0)
        bbox_max = np.max(xyz, axis=0)

        xyz, rgb, xyz_voxel, xyz_offset, bbox, world_to_voxel, xyz_world = voxelize(
            xyz,
            rgb,
            voxel_size=voxel_size,
            xyz_world=xyz_world,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
        )

        bbox_voxel_min = np.min(xyz_voxel, axis=0)
        bbox_voxel_max = np.max(xyz_voxel, axis=0)
        bbox_voxel = np.stack([bbox_voxel_min, bbox_voxel_max], axis=0).astype(np.int32)
        return xyz_world, rgb, xyz_voxel, xyz_offset, bbox, bbox_voxel, world_to_voxel

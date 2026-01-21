import argparse
import os
import sys
from pathlib import Path
import json

import torch
import imageio
import numpy as np
import cv2
import trimesh
from tqdm import tqdm
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRasterizer,
    PerspectiveCameras,
)

from submodules.scannetpp.common.utils.colmap import read_model
from submodules.scannetpp.common.scene_release import ScannetppScene_Release


def parse_args():
    parser = argparse.ArgumentParser(description="Render depth maps from scene files using PyTorch3D.")
    parser.add_argument('--data_root', type=str, required=True, help='Root directory of the dataset.')
    parser.add_argument('--scene_file', type=str, required=True, help='Path to the scene file.')
    parser.add_argument('--output_root', type=str, required=True, help='Directory to save the output data.')
    return parser.parse_args()


def render_depth_maps(
    data_root: str, output_root: str, scene_id: str,
):
    scene = ScannetppScene_Release(
        scene_id=scene_id,
        data_root=data_root,
    )

    transforms_path = Path(output_root) / "data" / scene_id / "transforms.json"
    with open(transforms_path, 'r') as f:
        transform = json.load(f)

    # Create a mapping from image name to transform matrix
    image_name_to_transform = {}
    for frame in transform["test_frames"]:
        image_name = frame["file_path"].split("/")[-1]
        image_name_to_transform[image_name] = np.array(frame["transform_matrix"])

    cameras, images, points3D = read_model(scene.dslr_colmap_dir, ".txt")
    assert len(cameras) == 1, "Multiple cameras not supported"
    camera = next(iter(cameras.values()))

    fx = transform["fl_x"]
    fy = transform["fl_y"]
    cx = transform["cx"]
    cy = transform["cy"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    mesh_path = str(scene.scan_mesh_path)
    verts = None
    faces = None
    
    if mesh_path.lower().endswith('.obj'):
        try:
            mesh = load_obj(mesh_path)
            verts = mesh[0]
            faces = mesh[1].verts_idx
        except Exception as e:
            print(f"Failed to load OBJ mesh: {e}")
            verts = None
    else:
        try:
            import trimesh
            tm = trimesh.load(mesh_path, process=False)
            verts = torch.from_numpy(np.asarray(tm.vertices)).to(device).float()
            faces = torch.from_numpy(np.asarray(tm.faces)).to(device).long()
        except Exception as e:
            print(f"Failed to load mesh with trimesh: {e}")
            verts = None

    if verts is None or faces is None:
        print(f"ERROR: Failed to load mesh from {mesh_path}")
        sys.exit(1)

    if not isinstance(verts, torch.Tensor):
        verts = verts.to(device).float()
    if not isinstance(faces, torch.Tensor):
        faces = faces.to(device).long()
    
    mesh_p3d = Meshes(verts=[verts], faces=[faces])

    cameras_p3d = PerspectiveCameras(
        focal_length=[(fx, fy)],
        principal_point=[(cx, cy)],
        image_size=[(camera.height, camera.width)],
        device=device,
        in_ndc=False,
    )

    raster_settings = RasterizationSettings(
        image_size=(camera.height, camera.width),
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    rasterizer = MeshRasterizer(cameras=cameras_p3d, raster_settings=raster_settings)

    depth_dir = Path(output_root) / "data" / scene_id / "depth"
    depth_dir.mkdir(parents=True, exist_ok=True)
    
    for image_id, image in tqdm(images.items(), "Rendering images"):
        if image.name not in image_name_to_transform:
            continue
        
        # Use the transform from transforms.json (already in correct coordinate system)
        camera_to_world = image_name_to_transform[image.name]
        
        # Apply coordinate system transformations (same as in dataset/scannetpp.py)
        camera_to_world[2, :] *= -1
        camera_to_world = camera_to_world[np.array([1, 0, 2, 3]), :]
        camera_to_world[0:3, 1:3] *= -1
        
        # Convert to world_to_camera for PyTorch3D
        world_to_camera = np.linalg.inv(camera_to_world)
        extrinsic = world_to_camera.astype(np.float64)
        
        # PyTorch3D expects R as the rotation matrix (not transposed) and T as translation
        # The extrinsic matrix format is: [R | t] where a point p_cam = R * p_world + t
        R = torch.from_numpy(extrinsic[:3, :3].T.astype(np.float32)).unsqueeze(0).to(device)
        T = torch.from_numpy(extrinsic[:3, 3].astype(np.float32)).unsqueeze(0).to(device)

        fragments = rasterizer(mesh_p3d, R=R, T=T)
        zbuf = fragments.zbuf[0, ..., 0].cpu().numpy()
        zbuf[np.isinf(zbuf)] = 0
        
        # PyTorch3D renders with Y-axis pointing up, but images have Y pointing down
        # Also flip horizontally to match the coordinate system
        zbuf = np.flipud(zbuf)
        zbuf = np.fliplr(zbuf)
        
        # Convert depth to millimeters and save as single-channel uint16 PNG
        # 0 indicates invalid depth
        depth = (zbuf.astype(np.float32) * 1000).clip(0, 65535).astype(np.uint16)
        
        depth_name = image.name.split(".")[0] + ".png"
        # Save as single-channel uint16 PNG (imageio handles this correctly)
        imageio.imwrite(depth_dir / depth_name, depth)
        
        # Save visualized depth with colormap
        depth_vis_name = image.name.split(".")[0] + ".jpg"
        depth_float = zbuf.astype(np.float32)
        
        # Normalize depth for visualization
        valid_mask = (depth_float > 0) & np.isfinite(depth_float)
        if valid_mask.any():
            min_val = depth_float[valid_mask].min()
            max_val = depth_float[valid_mask].max()
            depth_normalized = np.where(valid_mask, 
                                       (depth_float - min_val) / (max_val - min_val + 1e-6) * 255, 
                                       0)
        else:
            depth_normalized = np.zeros_like(depth_float)
        
        depth_normalized = depth_normalized.astype(np.uint8)
        depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_VIRIDIS)
        cv2.imwrite(str(depth_dir / depth_vis_name), depth_colormap)


def build_symlinks(
    data_root: str, output_root: str, scene_id: str,
):
    scene = ScannetppScene_Release(
        scene_id=scene_id,
        data_root=data_root,
    )

    dslr_undistorted_dir = scene.dslr_resized_undistorted_dir
    output_image_dir = Path(output_root) / "data" / scene_id / "images"
    
    if not output_image_dir.exists():
        os.makedirs(output_image_dir.parent, exist_ok=True)
        os.symlink(dslr_undistorted_dir, output_image_dir)

    output_transform_path = Path(output_root) / "data" / scene_id / "transforms.json"
    if not output_transform_path.exists():
        os.symlink(scene.dslr_nerfstudio_transform_undistorted_path, output_transform_path)
    else:
        print("Transform file already exists:", output_transform_path)


def main():
    args = parse_args()

    # Load the scene file
    with open(args.scene_file, 'r') as f:
        scene_ids = [line.strip() for line in f if line.strip()]

    # Create output directory if it doesn't exist
    os.makedirs(args.output_root, exist_ok=True)

    # go through each scene
    for scene_id in tqdm(scene_ids, desc="scene"):
        build_symlinks(
            data_root=args.data_root,
            output_root=args.output_root,
            scene_id=scene_id,
        )
        render_depth_maps(
            data_root=args.data_root,
            output_root=args.output_root,
            scene_id=scene_id,
        )


if __name__ == "__main__":
    main()

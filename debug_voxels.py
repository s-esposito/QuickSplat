"""Debug script: Compare voxel distributions between DL3DV and ScanNetPP scenes.

Visualizes why the initializer fails on DL3DV (all voxels pruned or OOM).
"""
import json
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from dataset.colmap_dataset import ColmapPointDataset
from dataset.scannetpp import MultiScannetppPointDataset
from dataset.utils import voxelize


VOXEL_SIZE = 0.04  # Same as config


def analyze_voxels(name, xyz, rgb, xyz_voxel, bbox):
    """Print detailed voxel grid statistics."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    # World-space stats
    print(f"\n  World-space point cloud:")
    print(f"    Num points (after voxelization): {xyz.shape[0]}")
    print(f"    BBox min: {bbox[0]}")
    print(f"    BBox max: {bbox[1]}")
    extent = bbox[1] - bbox[0]
    print(f"    Extent: {extent}")
    print(f"    Max extent: {extent.max():.3f}")

    # Voxel-space stats
    vmin = xyz_voxel.min(axis=0)
    vmax = xyz_voxel.max(axis=0)
    vrange = vmax - vmin
    print(f"\n  Voxel-space grid:")
    print(f"    Voxel min: {vmin}")
    print(f"    Voxel max: {vmax}")
    print(f"    Voxel range: {vrange}")
    print(f"    Num unique voxels: {xyz_voxel.shape[0]}")

    # Compute theoretical grid volume vs occupied ratio
    grid_volume = np.prod(vrange + 1)
    occupancy = xyz_voxel.shape[0] / grid_volume if grid_volume > 0 else 0
    print(f"    Grid volume (product of ranges): {grid_volume:,.0f}")
    print(f"    Occupancy ratio: {occupancy:.6f} ({occupancy*100:.4f}%)")

    # Distribution of voxels per axis
    for axis, label in enumerate(["X", "Y", "Z"]):
        unique_vals = np.unique(xyz_voxel[:, axis])
        print(f"    {label}: {len(unique_vals)} unique values, range [{unique_vals.min()}, {unique_vals.max()}]")

    # Density: how many non-empty voxels exist per unit volume of bounding box
    bbox_volume = np.prod(extent) if np.prod(extent) > 0 else 1
    density = xyz_voxel.shape[0] / bbox_volume
    print(f"\n  Density metrics:")
    print(f"    Points per world m³: {density:.1f}")
    print(f"    Points per voxel axis unit: {xyz_voxel.shape[0] / np.mean(vrange):.1f}")

    return {
        "name": name,
        "num_voxels": xyz_voxel.shape[0],
        "extent": extent,
        "voxel_range": vrange,
        "grid_volume": grid_volume,
        "occupancy": occupancy,
        "vmin": vmin,
        "vmax": vmax,
        "xyz_voxel": xyz_voxel,
        "xyz": xyz,
        "rgb": rgb,
    }


def estimate_decoder_expansion(stats):
    """Estimate how many voxels each decoder level would produce.

    The encoder downsamples by 2x at each level (stride 2).
    The decoder upsamples by 2x at each level (transposed conv stride 2).

    At each decoder level, a transposed convolution with stride 2 and kernel 3
    roughly doubles the coordinate range, potentially creating up to 8x more
    coordinates (2^3 for 3D). But in practice, the expansion depends on sparsity.
    """
    num_input = stats["num_voxels"]
    vrange = stats["voxel_range"]

    print(f"\n  Estimated encoder/decoder flow (voxel_size={VOXEL_SIZE}):")
    print(f"    Input: {num_input:,} voxels, grid range {vrange}")

    # Encoder downsample levels
    enc_ranges = [vrange.copy()]
    enc_counts = [num_input]

    for level in range(4):
        prev_range = enc_ranges[-1]
        new_range = prev_range // 2  # stride 2 downsample
        # Rough count: occupancy stays similar but grid shrinks
        new_count = max(enc_counts[-1] // 4, 1)  # rough: sparse conv reduces ~4x at each level
        enc_ranges.append(new_range)
        enc_counts.append(new_count)
        print(f"    Encoder L{level+1} (stride 2): ~{new_count:,} voxels, range ~{new_range}")

    # Decoder upsample levels
    print(f"    --- Bottleneck ---")
    dec_count = enc_counts[-1]
    dec_range = enc_ranges[-1]

    for level in range(4):
        # Transposed conv stride 2: each voxel can create up to 2^3 = 8 new coordinates
        # But with kernel 3 and overlapping, the growth can be more
        # The key insight: sparser grids expand MORE because there's less overlap
        prev_count = dec_count
        dec_range = dec_range * 2 + 2  # transposed conv with kernel 3 approximately doubles + adds border

        # Estimate: each existing voxel generates ~8 new coords, minus overlap
        # With very sparse data (low occupancy), overlap is minimal → closer to 8x
        # With dense data, overlap reduces this significantly
        occupancy_at_level = prev_count / np.prod(dec_range / 2 + 1)
        if occupancy_at_level > 0.01:
            expansion = 4  # dense: significant overlap
        else:
            expansion = 8  # sparse: minimal overlap

        dec_count = prev_count * expansion
        # Also add skip connection contribution
        skip_count = enc_counts[3 - level]
        dec_count = dec_count + skip_count  # Union with skip coords

        print(f"    Decoder L{level+1} (upsample 2x): ~{dec_count:,} voxels (expansion ~{expansion}x + skip {skip_count:,})")

    print(f"    Final output: ~{dec_count:,} estimated voxels")
    return dec_count


def create_visualization(stats_list, save_path="debug_voxels.png"):
    """Create side-by-side comparison plots."""
    fig, axes = plt.subplots(2, len(stats_list), figsize=(8 * len(stats_list), 12))
    if len(stats_list) == 1:
        axes = axes.reshape(-1, 1)

    for col, stats in enumerate(stats_list):
        xyz_voxel = stats["xyz_voxel"]

        # Top: 2D projection (XY plane) of voxel positions
        ax = axes[0, col]
        ax.scatter(xyz_voxel[:, 0], xyz_voxel[:, 1], s=0.1, alpha=0.3, c='blue')
        ax.set_title(f"{stats['name']}\nXY Projection ({stats['num_voxels']:,} voxels)")
        ax.set_xlabel("X voxel coord")
        ax.set_ylabel("Y voxel coord")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        # Bottom: Histogram of voxel coordinates per axis
        ax = axes[1, col]
        for axis, label, color in [(0, "X", "red"), (1, "Y", "green"), (2, "Z", "blue")]:
            vals = xyz_voxel[:, axis]
            bins = min(100, int(vals.max() - vals.min() + 1))
            ax.hist(vals, bins=bins, alpha=0.4, label=f"{label} [{vals.min()}-{vals.max()}]", color=color)
        ax.set_title(f"Voxel coordinate distribution")
        ax.set_xlabel("Voxel coordinate")
        ax.set_ylabel("Count")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"\nSaved visualization to {save_path}")


def run_initializer_verbose(xyz_voxel, rgb, device="cuda"):
    """Run the actual initializer with verbose logging to see exact counts at each level."""
    import MinkowskiEngine as ME
    from configs.config import get_cfg_defaults
    from utils.sparse import xyz_list_to_bxyz
    from models.initializer import Initializer18
    from models.unet_base import ResUNetConfig

    # Load the checkpoint to get model weights
    config = get_cfg_defaults()
    config.merge_from_file("configs/phase2_eval_long_no_adam.yaml")
    config.freeze()

    # Build initializer (same as Phase2Trainer.setup_decoders)
    out_channels = 2 + 3 + 4  # scale(2) + rgb(3) + rotation(4) for 2d gaussians
    initializer = Initializer18(
        in_channels=3,
        out_channels=out_channels,
        config=ResUNetConfig(),
        use_time_emb=False,
        dense_bottleneck=True,
        num_dense_blocks=config.MODEL.INIT.num_dense_blocks,
    ).to(device).eval()

    # Load weights
    ckpt = torch.load("checkpoints/phase2.ckpt", map_location="cpu")
    if "initializer" in ckpt:
        initializer.load_state_dict(ckpt["initializer"])
        print("  Loaded initializer weights from checkpoint")

    # Prepare input
    xyz_voxel_t = torch.from_numpy(xyz_voxel).float().to(device)
    rgb_t = torch.from_numpy(rgb).float().to(device)

    bxyz, _ = xyz_list_to_bxyz([xyz_voxel_t])
    sparse_input = ME.SparseTensor(features=rgb_t, coordinates=bxyz)

    print(f"\n  Running initializer (threshold=0.0, default)...")
    try:
        outputs = initializer(
            sparse_input,
            gt_coords=bxyz,
            verbose=True,
        )
        out_sparse = outputs["out"]
        xyz_out, feat_out = out_sparse.decomposed_coordinates_and_features
        print(f"  Output: {xyz_out[0].shape[0]} voxels with {feat_out[0].shape[1]} features")

        # Also check the occupancy logits
        last_prob = outputs["last_prob"]
        logits = last_prob.features_at(batch_index=0)
        print(f"  Last layer logits: min={logits.min():.3f}, max={logits.max():.3f}, mean={logits.mean():.3f}")
        print(f"  Logits > 0: {(logits > 0).sum().item()} / {logits.shape[0]}")
        print(f"  Sigmoid > 0.5: {(torch.sigmoid(logits) > 0.5).sum().item()} / {logits.shape[0]}")

        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def main():
    # ---- DL3DV ----
    dl3dv_dir = "/mnt/lustre/work/geiger/gwb929/datasets/dl3dv-colmap-sfm"
    views_split_path = "/mnt/lustre/work/geiger/gwb929/projects/optgs-unified/assets/dl3dv_evaluation/dl3dv_start_0_distance_40_ctx_8v_tgt_8v.json"

    dl3dv_dataset = ColmapPointDataset(
        source_path=dl3dv_dir,
        views_split_path=views_split_path,
        voxel_size=VOXEL_SIZE,
    )
    first_dl3dv_scene = dl3dv_dataset.scene_list[0]
    print(f"DL3DV scene: {first_dl3dv_scene}")

    # Load raw points first (before voxelization) to compare
    dl3dv_xyz_raw, dl3dv_rgb_raw = dl3dv_dataset.load_colmap_points(first_dl3dv_scene)
    print(f"DL3DV raw points: {dl3dv_xyz_raw.shape[0]}")
    print(f"DL3DV raw extent: {dl3dv_xyz_raw.max(axis=0) - dl3dv_xyz_raw.min(axis=0)}")

    # Check for zero-color noise points
    zero_color_mask = (dl3dv_rgb_raw == 0).all(axis=1)
    num_zero = zero_color_mask.sum()
    print(f"\nDL3DV zero-color points: {num_zero} / {dl3dv_rgb_raw.shape[0]} ({100*num_zero/dl3dv_rgb_raw.shape[0]:.1f}%)")
    print(f"DL3DV non-zero color points: {(~zero_color_mask).sum()}")
    if num_zero > 0:
        # Compare extent with and without zero-color points
        valid_xyz = dl3dv_xyz_raw[~zero_color_mask]
        print(f"Extent (all): {dl3dv_xyz_raw.max(axis=0) - dl3dv_xyz_raw.min(axis=0)}")
        print(f"Extent (no zero-color): {valid_xyz.max(axis=0) - valid_xyz.min(axis=0)}")
        print(f"Points with zero color - extent: {dl3dv_xyz_raw[zero_color_mask].max(axis=0) - dl3dv_xyz_raw[zero_color_mask].min(axis=0)}")
        # Check color distribution
        print(f"\nRGB stats (all points): mean={dl3dv_rgb_raw.mean(axis=0)}, std={dl3dv_rgb_raw.std(axis=0)}")
        print(f"RGB stats (non-zero): mean={dl3dv_rgb_raw[~zero_color_mask].mean(axis=0)}, std={dl3dv_rgb_raw[~zero_color_mask].std(axis=0)}")

    dl3dv_xyz, dl3dv_rgb, dl3dv_voxel, dl3dv_offset, dl3dv_bbox, dl3dv_bbox_voxel, dl3dv_w2v = \
        dl3dv_dataset.load_voxelized_colmap_points(first_dl3dv_scene, VOXEL_SIZE)

    dl3dv_stats = analyze_voxels(f"DL3DV: {first_dl3dv_scene}", dl3dv_xyz, dl3dv_rgb, dl3dv_voxel, dl3dv_bbox)
    estimate_decoder_expansion(dl3dv_stats)

    # ---- ScanNetPP ----
    spp_source = "/home/geiger/gwb987/work/codebase/QuickSplat/quicksplat_spp_data_processed/data"
    spp_ply = "/home/geiger/gwb987/work/codebase/QuickSplat/quicksplat_spp_data_processed/colmap"
    spp_gt_ply = "/home/geiger/gwb987/work/codebase/QuickSplat/quicksplat_spp_data_processed/mesh"
    spp_test_split = "/home/geiger/gwb987/work/codebase/QuickSplat/quicksplat_spp_data_processed/splits/test_scene_ids.txt"

    if Path(spp_test_split).exists():
        spp_dataset = MultiScannetppPointDataset(
            spp_source, spp_ply, spp_gt_ply, spp_test_split,
            voxel_size=VOXEL_SIZE,
        )
        first_spp_scene = spp_dataset.scene_list[0]
        print(f"\nScanNetPP scene: {first_spp_scene}")

        spp_xyz, spp_rgb, spp_voxel, spp_offset, spp_bbox, spp_bbox_voxel, spp_w2v = \
            spp_dataset.load_voxelized_colmap_points(first_spp_scene, VOXEL_SIZE)

        spp_stats = analyze_voxels(f"ScanNetPP: {first_spp_scene}", spp_xyz, spp_rgb, spp_voxel, spp_bbox)
        estimate_decoder_expansion(spp_stats)

        stats_list = [dl3dv_stats, spp_stats]
    else:
        print(f"\nScanNetPP test split not found at {spp_test_split}, skipping comparison")
        stats_list = [dl3dv_stats]

    # Create visualization
    create_visualization(stats_list, "debug_voxels.png")

    # Run actual initializer
    print("\n" + "="*60)
    print("  RUNNING ACTUAL INITIALIZER")
    print("="*60)

    print(f"\n--- DL3DV: {first_dl3dv_scene} ---")
    run_initializer_verbose(dl3dv_voxel, dl3dv_rgb)

    if Path(spp_test_split).exists():
        print(f"\n--- ScanNetPP: {first_spp_scene} ---")
        run_initializer_verbose(spp_voxel, spp_rgb)


if __name__ == "__main__":
    main()

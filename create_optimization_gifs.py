#!/usr/bin/env python3
"""
Create MP4 videos from optimization iteration images.
Given a run path, scans for optimization step images and creates videos for each scene.
Can be used standalone or imported into the inference pipeline.
"""

import argparse
from pathlib import Path
from collections import defaultdict
import re
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy import ImageSequenceClip


def extract_step_number(filename):
    """Extract step number from filename like scene_0000_opt_5.jpg or scene_0000_init.jpg"""
    if "init" in filename:
        return 0

    # Match opt_N pattern
    match = re.search(r'opt_(\d+)', filename)
    if match:
        return int(match.group(1))

    # Match ft pattern
    if "ft" in filename:
        return 999

    return -1


def collect_image_sequences(run_path, scene_id=None):
    """
    Collect all optimization image sequences for each scene.
    Returns dict: {scene_id: {frame_idx: [(step_num, image_path)]}}

    If scene_id is given, only collects for that scene.
    """
    outputs_dir = Path(run_path) / "outputs"

    if not outputs_dir.exists():
        print(f"Error: outputs directory not found at {outputs_dir}")
        return {}

    # Structure: {scene_id: {frame_idx: [(step_num, image_path)]}}
    sequences = defaultdict(lambda: defaultdict(list))

    # Scan scene directories
    if scene_id is not None:
        scene_dirs = [outputs_dir / scene_id]
    else:
        scene_dirs = [d for d in outputs_dir.iterdir() if d.is_dir()]

    for scene_dir in scene_dirs:
        if not scene_dir.is_dir():
            continue

        sid = scene_dir.name

        # Find all RGB images (not depth)
        for img_path in scene_dir.glob("*.jpg"):
            filename = img_path.name

            # Skip depth images
            if "depth" in filename:
                continue

            # Extract frame number (should be 4 digits like 0000, 0001, etc.)
            frame_idx = None
            parts = filename.replace(".jpg", "").split("_")
            for part in parts:
                if part.isdigit() and len(part) == 4:
                    frame_idx = int(part)
                    break

            if frame_idx is None:
                continue

            # Extract step number
            step_num = extract_step_number(filename)
            if step_num >= 0:
                sequences[sid][frame_idx].append((step_num, img_path))

    # Sort images by step number for each sequence
    for sid in sequences:
        for frame_idx in sequences[sid]:
            sequences[sid][frame_idx].sort(key=lambda x: x[0])

    return sequences


def add_text_to_image(image, text, position="top-left", font_size=40):
    """Add text overlay to an image."""
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except Exception:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    padding = 20
    if position == "top-left":
        x, y = padding, padding
    elif position == "top-right":
        x = img_copy.width - text_width - padding
        y = padding
    else:
        x, y = padding, padding

    bg_padding = 10
    draw.rectangle(
        [x - bg_padding, y - bg_padding,
         x + text_width + bg_padding, y + text_height + bg_padding],
        fill=(0, 0, 0, 180)
    )
    draw.text((x, y), text, fill=(255, 255, 255), font=font)

    return img_copy


def _step_label(step_num):
    if step_num == 0:
        return "init"
    elif step_num == 999:
        return "ft"
    else:
        return f"opt_{step_num}"


def create_video(image_step_pairs, output_path, fps=2):
    """
    Create an MP4 video from a list of (image_path, step_number) pairs using moviepy.
    """
    if not image_step_pairs:
        print(f"No images to create video at {output_path}")
        return

    frames = []
    for img_path, step_num in image_step_pairs:
        try:
            img = Image.open(img_path).convert("RGB")
            label = _step_label(step_num)
            img_with_text = add_text_to_image(img, label, position="top-left")
            frames.append(np.array(img_with_text))
        except Exception as e:
            print(f"Error loading {img_path}: {e}")

    if not frames:
        print(f"Failed to load any images for {output_path}")
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(
        str(output_path),
        codec="libx264",
        logger=None,
    )
    print(f"Created video: {output_path} ({len(frames)} frames @ {fps} fps)")


def create_scene_videos(run_path, scene_id, output_dir=None, fps=2, include_ft=True):
    """
    Create optimization videos for a single scene.
    Called from the inference pipeline after each scene completes.
    """
    run_path = Path(run_path)
    if output_dir is None:
        output_dir = run_path / "videos"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sequences = collect_image_sequences(run_path, scene_id=scene_id)
    if not sequences or scene_id not in sequences:
        print(f"No image sequences found for scene {scene_id}")
        return

    scene_sequences = sequences[scene_id]
    total = 0
    for frame_idx in sorted(scene_sequences.keys()):
        step_images = scene_sequences[frame_idx]

        if not include_ft:
            step_images = [(s, p) for s, p in step_images if s != 999]

        if not step_images:
            continue

        image_step_pairs = [(img_path, step_num) for step_num, img_path in step_images]

        video_filename = f"{scene_id}_{frame_idx:04d}_optimization.mp4"
        video_path = output_dir / video_filename

        create_video(image_step_pairs, video_path, fps=fps)
        total += 1

    print(f"Created {total} videos for scene {scene_id} in {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Create MP4 videos from optimization iteration images"
    )
    parser.add_argument(
        "run_path",
        type=str,
        help="Path to the inference run directory (e.g., inference_outputs/inference/phase2_eval_long)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=2,
        help="Frames per second for the video (default: 2)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for videos (default: <run_path>/videos)"
    )
    parser.add_argument(
        "--include-ft",
        action="store_true",
        help="Include the finetune (ft) step in the video"
    )

    args = parser.parse_args()

    run_path = Path(args.run_path)

    if not run_path.exists():
        print(f"Error: Run path does not exist: {run_path}")
        return

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = run_path / "videos"

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning: {run_path}")
    print(f"Output directory: {output_dir}")
    print(f"FPS: {args.fps}")
    print()

    sequences = collect_image_sequences(run_path)

    if not sequences:
        print("No image sequences found!")
        return

    print(f"Found {len(sequences)} scenes")

    total_videos = 0
    for scene_id in sorted(sequences.keys()):
        scene_sequences = sequences[scene_id]

        for frame_idx in sorted(scene_sequences.keys()):
            step_images = scene_sequences[frame_idx]

            if not args.include_ft:
                step_images = [(s, p) for s, p in step_images if s != 999]

            if not step_images:
                continue

            image_step_pairs = [(img_path, step_num) for step_num, img_path in step_images]

            video_filename = f"{scene_id}_{frame_idx:04d}_optimization.mp4"
            video_path = output_dir / video_filename

            create_video(image_step_pairs, video_path, fps=args.fps)
            total_videos += 1

    print()
    print(f"Done! Created {total_videos} videos in {output_dir}")


if __name__ == "__main__":
    main()

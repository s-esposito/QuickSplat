#!/usr/bin/env python3
"""
Create MP4 videos from optimization iteration images.
Given a run path, scans for optimization step images and creates videos for each scene.
"""

import argparse
from pathlib import Path
from collections import defaultdict
import re
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont


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


def collect_image_sequences(run_path):
    """
    Collect all optimization image sequences for each scene.
    Returns dict: {scene_id: {frame_idx: [(step_num, image_path)]}}
    """
    outputs_dir = Path(run_path) / "outputs"
    
    if not outputs_dir.exists():
        print(f"Error: outputs directory not found at {outputs_dir}")
        return {}
    
    # Structure: {scene_id: {frame_idx: [(step_num, image_path)]}}
    sequences = defaultdict(lambda: defaultdict(list))
    
    # Scan all scene directories
    for scene_dir in outputs_dir.iterdir():
        if not scene_dir.is_dir():
            continue
        
        scene_id = scene_dir.name
        
        # Find all RGB images (not depth)
        for img_path in scene_dir.glob("*.jpg"):
            filename = img_path.name
            
            # Skip depth images
            if "depth" in filename:
                continue
            
            # Parse filename: scene_XXXX_step.jpg
            # Match pattern: scene_id_framenum_step.jpg
            parts = filename.replace(".jpg", "").split("_")
            
            if len(parts) < 3:
                continue
            
            # Extract frame number (should be 4 digits like 0000, 0001, etc.)
            frame_idx = None
            for i, part in enumerate(parts):
                if part.isdigit() and len(part) == 4:
                    frame_idx = int(part)
                    break
            
            if frame_idx is None:
                continue
            
            # Extract step number
            step_num = extract_step_number(filename)
            if step_num >= 0:
                sequences[scene_id][frame_idx].append((step_num, img_path))
    
    # Sort images by step number for each sequence
    for scene_id in sequences:
        for frame_idx in sequences[scene_id]:
            sequences[scene_id][frame_idx].sort(key=lambda x: x[0])
    
    return sequences


def add_text_to_image(image, text, position="top-left", font_size=40):
    """
    Add text overlay to an image.
    
    Args:
        image: PIL Image
        text: Text to add
        position: Position of text ("top-left", "top-right", etc.)
        font_size: Font size
    
    Returns:
        PIL Image with text overlay
    """
    # Create a copy to avoid modifying original
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    
    # Try to use a nice font, fall back to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", font_size)
        except:
            # Use default font
            font = ImageFont.load_default()
    
    # Get text bounding box
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Calculate position
    padding = 20
    if position == "top-left":
        x, y = padding, padding
    elif position == "top-right":
        x = img_copy.width - text_width - padding
        y = padding
    else:
        x, y = padding, padding
    
    # Draw background rectangle for better visibility
    bg_padding = 10
    draw.rectangle(
        [x - bg_padding, y - bg_padding, 
         x + text_width + bg_padding, y + text_height + bg_padding],
        fill=(0, 0, 0, 180)
    )
    
    # Draw text in white
    draw.text((x, y), text, fill=(255, 255, 255), font=font)
    
    return img_copy


def create_video(image_step_pairs, output_path, fps=5):
    """
    Create an MP4 video from a list of (image_path, step_info) pairs.
    
    Args:
        image_step_pairs: List of (image_path, step_number) tuples
        output_path: Path to save the MP4 video
        fps: Frames per second
    """
    if not image_step_pairs:
        print(f"No images to create video at {output_path}")
        return
    
    # Load all images and add iteration labels
    frames = []
    for img_path, step_num in image_step_pairs:
        try:
            img = Image.open(img_path)
            
            # Determine label text
            if step_num == 0:
                label = "init"
            elif step_num == 999:
                label = "ft"
            else:
                label = f"opt_{step_num}"
            
            # Add text overlay
            img_with_text = add_text_to_image(img, label, position="top-left")
            
            # Convert PIL Image to numpy array (RGB -> BGR for OpenCV)
            frame = np.array(img_with_text)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frames.append(frame)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
    
    if not frames:
        print(f"Failed to load any images for {output_path}")
        return
    
    # Get dimensions from first frame
    height, width = frames[0].shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    if not video_writer.isOpened():
        print(f"Error: Could not open video writer for {output_path}")
        return
    
    # Write frames
    for frame in frames:
        video_writer.write(frame)
    
    video_writer.release()
    print(f"Created video: {output_path} ({len(frames)} frames @ {fps} fps)")


# Keep old function name for compatibility but redirect to create_video
def create_gif(image_step_pairs, output_path, duration=200, loop=0):
    """Legacy function - now creates MP4 video instead of GIF."""
    # Convert duration (ms) to fps
    fps = 1000 / duration if duration > 0 else 5
    fps = max(1, min(30, fps))  # Clamp between 1-30 fps
    create_video(image_step_pairs, output_path, fps=fps)


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
        default=5,
        help="Frames per second for the video (default: 5)"
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
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = run_path / "videos"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Scanning: {run_path}")
    print(f"Output directory: {output_dir}")
    print(f"FPS: {args.fps}")
    print()
    
    # Collect image sequences
    sequences = collect_image_sequences(run_path)
    
    if not sequences:
        print("No image sequences found!")
        return
    
    print(f"Found {len(sequences)} scenes")
    
    # Create GIFs for each scene and frame
    # Create videos for each scene and frame
    total_videos = 0
    for scene_id in sorted(sequences.keys()):
        scene_sequences = sequences[scene_id]
        
        for frame_idx in sorted(scene_sequences.keys()):
            # Get sorted list of images
            step_images = scene_sequences[frame_idx]
            
            # Filter out ft step if not requested
            if not args.include_ft:
                step_images = [(s, p) for s, p in step_images if s != 999]
            
            if not step_images:
                continue
            
            # Keep as list of (step_num, img_path) tuples
            image_step_pairs = [(img_path, step_num) for step_num, img_path in step_images]
            
            # Create video filename
            video_filename = f"{scene_id}_{frame_idx:04d}_optimization.mp4"
            video_path = output_dir / video_filename
            
            create_video(image_step_pairs, video_path, fps=args.fps)
            total_videos += 1
    
    print()
    print(f"Done! Created {total_videos} videos in {output_dir}")


if __name__ == "__main__":
    main()

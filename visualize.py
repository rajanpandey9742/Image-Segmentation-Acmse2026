"""
visualize.py — Debug/inspection helper for segmentation results.
Not needed for submission; useful for demos and the report.

Usage:
    python visualize.py <image_path> <mask_path> [output_path]
"""

import numpy as np
from PIL import Image
import sys
import os


def colorize_labels(labels: np.ndarray) -> np.ndarray:
    """
    Map integer segment IDs to a random RGB colormap for visualization.
    """
    rng = np.random.default_rng(seed=42)
    unique_ids = np.unique(labels)
    color_map = {sid: rng.integers(0, 255, 3).tolist() for sid in unique_ids}
    rgb = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for sid, color in color_map.items():
        mask = labels == sid
        rgb[mask] = color
    return rgb


def visualize(image_path: str, mask_path: str, output_path: str = None):
    """
    Create a side-by-side comparison: original | colorized segments.
    """
    img  = np.array(Image.open(image_path).convert("L"))
    mask = np.array(Image.open(mask_path)).astype(np.int32)

    num_segments = len(np.unique(mask))
    print(f"Image: {image_path}")
    print(f"Size : {img.shape[1]}x{img.shape[0]}")
    print(f"Segments: {num_segments}")

    # Convert grayscale to RGB for side-by-side display
    img_rgb = np.stack([img, img, img], axis=-1)
    colored = colorize_labels(mask)

    # Separator line
    sep = np.full((img.shape[0], 4, 3), 200, dtype=np.uint8)
    combined = np.concatenate([img_rgb, sep, colored], axis=1)
    out = Image.fromarray(combined)

    if output_path:
        out.save(output_path)
        print(f"Saved to: {output_path}")
    else:
        out.show()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python visualize.py <image_path> <mask_path> [output_path]")
        sys.exit(1)

    image_path  = sys.argv[1]
    mask_path   = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else None
    visualize(image_path, mask_path, output_path)

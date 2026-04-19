# visualize.py — Basic version
import numpy as np
from PIL import Image

def colorize_labels(mask):
    """Map segment IDs to random colors."""
    rng = np.random.default_rng(seed=42)
    colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for sid in np.unique(mask):
        colored[mask == sid] = rng.integers(0, 255, 3).tolist()
    return colored

def visualize(image_path, mask_path, output_path=None):
    img  = np.array(Image.open(image_path).convert("L"))
    mask = np.array(Image.open(mask_path))

    print(f"Segments found: {len(np.unique(mask))}")

    # Side-by-side with a thin separator
    img_rgb = np.stack([img, img, img], axis=-1)
    colored = colorize_labels(mask)
    sep     = np.full((img.shape[0], 4, 3), 200, dtype=np.uint8)
    combined = np.concatenate([img_rgb, sep, colored], axis=1)

    out = Image.fromarray(combined)
    if output_path:
        out.save(output_path)
        print(f"Saved to: {output_path}")
    else:
        out.show()
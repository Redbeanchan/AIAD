"""
This is a boilerplate pipeline 'model_evaluation'
generated using Kedro 0.19.14
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from PIL import Image


def _load_one_image(path: str, img_size: int) -> tf.Tensor:
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, (img_size, img_size))
    img = tf.cast(img, tf.float32) / 255.0
    return img


def generate_qualitative_preview(
    model: tf.keras.Model,
    sample_manifest: pd.DataFrame,
    img_size: int,
    n_samples: int,
    random_state: int,
    output_path: str,
    threshold: float = 0.5,
) -> str:
    """
    Saves a grid PNG:
      [Image | Pred Mask | BG Removed] x n_samples
    Returns the saved PNG path string.
    """
    if "images" not in sample_manifest.columns:
        raise ValueError("sample_manifest must contain an 'images' column")

    sample_paths = (
        sample_manifest.sample(min(n_samples, len(sample_manifest)), random_state=random_state)["images"]
        .astype(str)
        .tolist()
    )

    fig, axes = plt.subplots(len(sample_paths), 3, figsize=(12, 4 * len(sample_paths)))
    if len(sample_paths) == 1:
        axes = np.array([axes])  # keep indexing consistent

    for i, path in enumerate(sample_paths):
        img = _load_one_image(path, img_size)
        pred = model.predict(img[None, ...], verbose=0)[0, ..., 0]
        mask = (pred > threshold).astype(np.uint8)

        axes[i, 0].imshow(img.numpy())
        axes[i, 0].set_title("Image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(mask * 255, cmap="Blues_r")
        axes[i, 1].set_title("Pred Mask")
        axes[i, 1].axis("off")

        orig = np.array(Image.open(path).resize((img_size, img_size)))
        cut = orig * np.stack([mask] * 3, axis=-1)

        axes[i, 2].imshow(cut.astype(np.uint8))
        axes[i, 2].set_title("BG Removed")
        axes[i, 2].axis("off")

    plt.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return str(out)
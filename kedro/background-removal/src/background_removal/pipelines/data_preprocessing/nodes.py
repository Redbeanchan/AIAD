"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.19.14
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def build_manifest(
    supervisely_csv_path: str,
    supervisely_root: str,
    personseg_images_dir: str,
    personseg_masks_dir: str,
) -> pd.DataFrame:
    """
    Build a unified manifest DataFrame:
      columns: images, masks, source
    Using:
      - Supervisely df.csv (paths are relative, joined with supervisely_root)
      - Person Segmentation folders (paired by filename stem)
    """
    # --- Supervisely ---
    df_sup = pd.read_csv(supervisely_csv_path)[["images", "masks"]].copy()
    sup_root = Path(supervisely_root)

    df_sup["images"] = df_sup["images"].apply(lambda x: str(sup_root / x))
    df_sup["masks"] = df_sup["masks"].apply(lambda x: str(sup_root / x))
    df_sup["source"] = "supervisely"

    # --- Person Segmentation ---
    img_dir = Path(personseg_images_dir)
    msk_dir = Path(personseg_masks_dir)

    ps_imgs = [p for p in img_dir.iterdir() if p.is_file()]
    ps_msks = [p for p in msk_dir.iterdir() if p.is_file()]

    img_map = {p.stem: str(p) for p in ps_imgs}
    msk_map = {p.stem: str(p) for p in ps_msks}
    common = sorted(set(img_map) & set(msk_map))

    df_ps = pd.DataFrame(
        {"images": [img_map[k] for k in common], "masks": [msk_map[k] for k in common]}
    )
    df_ps["source"] = "person_segmentation"

    final_df = pd.concat([df_sup, df_ps], ignore_index=True)

    # sanity check
    missing_img = ~final_df["images"].apply(lambda p: Path(p).exists())
    missing_msk = ~final_df["masks"].apply(lambda p: Path(p).exists())
    if missing_img.any() or missing_msk.any():
        bad = final_df[missing_img | missing_msk].head(20)
        raise FileNotFoundError(
            "Missing files in manifest. Fix your supervisely_root / dataset paths.\n"
            f"Example rows:\n{bad.to_string(index=False)}"
        )

    return final_df


def split_manifest(
    manifest: pd.DataFrame,
    val_split: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df, val_df = train_test_split(
        manifest,
        test_size=val_split,
        random_state=random_state,
        shuffle=True,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)
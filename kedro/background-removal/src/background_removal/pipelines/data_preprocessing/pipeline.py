"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.19.14
"""
from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import build_manifest, split_manifest


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=build_manifest,
                inputs=dict(
                    supervisely_csv_path="params:supervisely_csv_path",
                    supervisely_root="params:supervisely_root",
                    personseg_images_dir="params:personseg_images_dir",
                    personseg_masks_dir="params:personseg_masks_dir",
                ),
                outputs="manifest",
                name="build_manifest",
            ),
            node(
                func=split_manifest,
                inputs=dict(
                    manifest="manifest",
                    val_split="params:val_split",
                    random_state="params:random_state",
                ),
                outputs=["train_manifest", "val_manifest"],
                name="split_manifest",
            ),
        ]
    )
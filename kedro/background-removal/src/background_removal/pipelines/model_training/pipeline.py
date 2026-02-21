"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 0.19.14
"""

from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import train_segmentation_model, save_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=train_segmentation_model,
                inputs=dict(
                    train_manifest="train_manifest",
                    val_manifest="val_manifest",
                    img_size="params:img_size",
                    batch_size="params:batch_size",
                    shuffle_buffer="params:shuffle_buffer",
                    base_channels="params:base_channels",
                    lr="params:lr",
                    epochs="params:epochs",
                ),
                outputs=["seg_model", "training_history"],
                name="train_segmentation_model",
            ),
            node(
                func=save_model,
                inputs=dict(
                    model="seg_model",
                    model_output_path="params:model_output_path",
                ),
                outputs="saved_model_path",
                name="save_model",
            ),
        ]
    )
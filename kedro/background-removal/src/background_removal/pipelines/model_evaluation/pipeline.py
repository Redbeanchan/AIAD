"""
This is a boilerplate pipeline 'model_evaluation'
generated using Kedro 0.19.14
"""
from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import generate_qualitative_preview


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=generate_qualitative_preview,
                inputs=dict(
                    model="seg_model",
                    sample_manifest="train_manifest",  # change to val_manifest if you want
                    img_size="params:img_size",
                    n_samples="params:eval_n_samples",
                    random_state="params:eval_random_state",
                    output_path="params:eval_output_path",
                    threshold="params:eval_threshold",
                ),
                outputs="eval_preview_path",
                name="generate_qualitative_preview",
            )
        ]
    )
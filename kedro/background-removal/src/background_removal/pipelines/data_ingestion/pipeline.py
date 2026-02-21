"""
This is a boilerplate pipeline 'data_ingestion'
generated using Kedro 0.19.14
"""
from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import download_and_extract_many_from_gdrive


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=download_and_extract_many_from_gdrive,
                inputs=dict(
                    files="params:gdrive_files",
                    output_root="params:raw_output_root",
                    delete_zip="params:delete_zip",
                ),
                outputs="extracted_dirs",
                name="download_extract_gdrive",
            )
        ]
    )
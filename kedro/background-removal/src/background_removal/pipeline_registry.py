"""Project pipelines."""
from __future__ import annotations

from kedro.pipeline import Pipeline

from background_removal.pipelines.data_ingestion import (
    create_pipeline as ingestion_pipeline,
)
from background_removal.pipelines.data_preprocessing import (
    create_pipeline as preprocessing_pipeline,
)
from background_removal.pipelines.model_training import (
    create_pipeline as training_pipeline,
)
from background_removal.pipelines.model_evaluation import (
    create_pipeline as evaluation_pipeline,
)


def register_pipelines() -> dict[str, Pipeline]:
    data_ingestion = ingestion_pipeline()
    data_preprocessing = preprocessing_pipeline()
    model_training = training_pipeline()
    model_evaluation = evaluation_pipeline()

    full = data_ingestion + data_preprocessing + model_training + model_evaluation

    return {
        "__default__": full,
        "full": full,
        "data_ingestion": data_ingestion,
        "data_preprocessing": data_preprocessing,
        "model_training": model_training,
        "model_evaluation": model_evaluation,
    }
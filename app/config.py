import os


class Config:

    MODEL_PATH = (
        "models/production_model.joblib"
    )

    FEATURE_PATH = (
        "models/production_feature_names.joblib"
    )

    METADATA_PATH = (
        "models/model_metadata.json"
    )

    DATABASE_PATH = (
        "data/predictions.db"
    )

    LOG_DIRECTORY = "logs"

    DEBUG = True
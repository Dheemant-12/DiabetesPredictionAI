import json
import os

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier


# ==========================================
# Paths
# ==========================================

COMPARISON_PATH = (
    "data/processed/model_comparison.csv"
)

TRAIN_PATH = (
    "data/processed/train.csv"
)

MODEL_DIRECTORY = "models"

METADATA_PATH = (
    "models/model_metadata.json"
)


# ==========================================
# Load Comparison Results
# ==========================================

results_df = pd.read_csv(
    COMPARISON_PATH
)


# ==========================================
# Select Best Model
# ==========================================

# Primary metric:
# ROC AUC
#
# Secondary metric:
# Recall

results_df = results_df.sort_values(
    by=[
        "ROC AUC",
        "Recall"
    ],
    ascending=False
)


best_model_name = (
    results_df.iloc[0]["Model"]
)


print("=" * 60)
print("PRODUCTION MODEL SELECTION")
print("=" * 60)

print(
    f"Selected Model: {best_model_name}"
)


# ==========================================
# Model Definitions
# ==========================================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=2000
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            max_depth=5,
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42
        ),

    "SVM":
        SVC(
            probability=True,
            random_state=42
        ),

    "XGBoost":
        XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            random_state=42,
            eval_metric="logloss"
        )
}


# ==========================================
# Load Training Data
# ==========================================

train_df = pd.read_csv(
    TRAIN_PATH
)


X_train = train_df.drop(
    "Outcome",
    axis=1
)

y_train = train_df["Outcome"]


# ==========================================
# Get Selected Model
# ==========================================

model = models[
    best_model_name
]


# ==========================================
# Train Production Model
# ==========================================

print(
    "\nTraining production model..."
)

model.fit(
    X_train,
    y_train
)


# ==========================================
# Create Models Directory
# ==========================================

os.makedirs(
    MODEL_DIRECTORY,
    exist_ok=True
)


# ==========================================
# Save Production Model
# ==========================================

import joblib


MODEL_PATH = (
    "models/production_model.joblib"
)


joblib.dump(
    model,
    MODEL_PATH
)


# ==========================================
# Save Feature Names
# ==========================================

feature_names = (
    X_train.columns.tolist()
)


joblib.dump(
    feature_names,
    "models/production_feature_names.joblib"
)


# ==========================================
# Get Selected Metrics
# ==========================================

selected_row = results_df[
    results_df["Model"] ==
    best_model_name
].iloc[0]


metadata = {

    "model_name":
        best_model_name,

    "selection_metric":
        "ROC AUC",

    "secondary_metric":
        "Recall",

    "accuracy":
        float(selected_row["Accuracy"]),

    "precision":
        float(selected_row["Precision"]),

    "recall":
        float(selected_row["Recall"]),

    "f1_score":
        float(selected_row["F1 Score"]),

    "roc_auc":
        float(selected_row["ROC AUC"]),

    "model_file":
        MODEL_PATH,

    "feature_file":
        "models/production_feature_names.joblib"

}


# ==========================================
# Save Metadata
# ==========================================

with open(
    METADATA_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metadata,
        file,
        indent=4
    )


# ==========================================
# Display Results
# ==========================================

print("\n" + "=" * 60)
print("PRODUCTION MODEL SAVED")
print("=" * 60)

print(
    f"Model      : {best_model_name}"
)

print(
    f"Accuracy   : "
    f"{metadata['accuracy']:.4f}"
)

print(
    f"Precision  : "
    f"{metadata['precision']:.4f}"
)

print(
    f"Recall     : "
    f"{metadata['recall']:.4f}"
)

print(
    f"F1 Score   : "
    f"{metadata['f1_score']:.4f}"
)

print(
    f"ROC AUC    : "
    f"{metadata['roc_auc']:.4f}"
)

print(
    f"\nSaved model:"
)

print(
    MODEL_PATH
)

print(
    "\nSaved metadata:"
)

print(
    METADATA_PATH
)
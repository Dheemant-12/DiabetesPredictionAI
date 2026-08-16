import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier


# ==========================================
# Load Dataset
# ==========================================

train_df = pd.read_csv(
    "data/processed/train.csv"
)

test_df = pd.read_csv(
    "data/processed/test.csv"
)


X_train = train_df.drop(
    "Outcome",
    axis=1
)

y_train = train_df["Outcome"]


X_test = test_df.drop(
    "Outcome",
    axis=1
)

y_test = test_df["Outcome"]


# ==========================================
# Define Models
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
# Store Results
# ==========================================

results = []


# ==========================================
# Train and Evaluate
# ==========================================

for name, model in models.items():

    print(
        f"\nTraining {name}..."
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]


    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )


    results.append({

        "Model": name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1 Score": f1,

        "ROC AUC": roc_auc

    })


# ==========================================
# Create Results DataFrame
# ==========================================

results_df = pd.DataFrame(
    results
)


results_df = results_df.sort_values(
    by="ROC AUC",
    ascending=False
)


# ==========================================
# Display Results
# ==========================================

print("\n")
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}"
    )
)


# ==========================================
# Best Model
# ==========================================

best_model = results_df.iloc[0]

print("\n")
print("=" * 70)
print("BEST MODEL BY ROC AUC")
print("=" * 70)

print(
    f"Model     : {best_model['Model']}"
)

print(
    f"Accuracy  : {best_model['Accuracy']:.4f}"
)

print(
    f"Precision : {best_model['Precision']:.4f}"
)

print(
    f"Recall    : {best_model['Recall']:.4f}"
)

print(
    f"F1 Score  : {best_model['F1 Score']:.4f}"
)

print(
    f"ROC AUC   : {best_model['ROC AUC']:.4f}"
)


# ==========================================
# Save Results
# ==========================================

results_df.to_csv(
    "data/processed/model_comparison.csv",
    index=False
)


print("\nModel comparison saved to:")

print(
    "data/processed/model_comparison.csv"
)
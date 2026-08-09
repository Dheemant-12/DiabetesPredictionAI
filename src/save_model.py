import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier


# Create models folder
os.makedirs("models", exist_ok=True)


# Load training data
train_df = pd.read_csv(
    "data/processed/train.csv"
)

X_train = train_df.drop(
    "Outcome",
    axis=1
)

y_train = train_df["Outcome"]


# Final model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=2,
    random_state=42
)


# Train
model.fit(
    X_train,
    y_train
)


# Save model
joblib.dump(
    model,
    "models/diabetes_model.joblib"
)


# Save feature names
feature_names = X_train.columns.tolist()

joblib.dump(
    feature_names,
    "models/feature_names.joblib"
)


print("Model saved successfully!")

print(
    "Feature names saved successfully!"
)


# Verify model
loaded_model = joblib.load(
    "models/diabetes_model.joblib"
)

print(
    "Loaded Model Type:",
    type(loaded_model)
)
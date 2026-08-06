import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

# Create models folder if it doesn't exist
os.makedirs("models", exist_ok=True)

# Load training data
train_df = pd.read_csv("data/processed/train.csv")

X_train = train_df.drop("Outcome", axis=1)
y_train = train_df["Outcome"]

# Train model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=2,
    random_state=42
)

model.fit(X_train, y_train)

# Save model
joblib.dump(
    model,
    "models/diabetes_model.joblib"
)

print("✅ Model saved successfully!")

# Verify saved model
loaded_model = joblib.load("models/diabetes_model.joblib")

print("Loaded Model Type:", type(loaded_model))
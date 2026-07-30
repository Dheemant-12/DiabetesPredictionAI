import pandas as pd
from sklearn.model_selection import train_test_split
df=pd.read_csv("data/processed/featured_diabetes.csv")
X = df.drop("Outcome",axis=1)
y=df["Outcome"]
X=pd.get_dummies(
    X,
    columns=["BMI_Category","Age_Group"],  
    drop_first=True
)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Merge features and target
train_df = X_train.copy()
train_df["Outcome"] = y_train

test_df = X_test.copy()
test_df["Outcome"] = y_test

# Save datasets
train_df.to_csv(
    "data/processed/train.csv",
    index=False
)

test_df.to_csv(
    "data/processed/test.csv",
    index=False
)

print("Training Dataset Shape:", train_df.shape)
print("Testing Dataset Shape:", test_df.shape)

print("\nDatasets saved successfully!")
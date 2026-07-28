import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("data/raw/diabetes.csv")

columns_with_zero_missing = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

# Replace invalid zeros with NaN
df[columns_with_zero_missing] = df[columns_with_zero_missing].replace(0, np.nan)

print("Missing values before cleaning:")
print(df.isnull().sum())

# Fill missing values using median
for column in columns_with_zero_missing:
    median_value = df[column].median()
    df[column].fillna(median_value, inplace=True)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# Save cleaned dataset
df.to_csv("data/processed/cleaned_diabetes.csv", index=False)

print("\nCleaned dataset saved successfully!")
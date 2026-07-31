import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("data/raw/diabetes.csv")

# Columns where 0 means missing
columns_with_zero_missing = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

# Replace impossible 0 values with NaN
df[columns_with_zero_missing] = df[columns_with_zero_missing].replace(0, np.nan)

print("Missing values before cleaning:")
print(df.isnull().sum())

# Fill missing values with the median
for column in columns_with_zero_missing:
    median_value = df[column].median()
    df[column] = df[column].fillna(median_value)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# Save cleaned dataset
df.to_csv("data/processed/cleaned_diabetes.csv", index=False)

print("\nCleaned dataset saved successfully!")
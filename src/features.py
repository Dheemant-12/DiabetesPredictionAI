import pandas as pd
import numpy as np

# Load cleaned dataset
df = pd.read_csv("data/processed/cleaned_diabetes.csv")

# BMI Categories
def bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"

df["BMI_Category"] = df["BMI"].apply(bmi_category)

# Age Groups
def age_group(age):
    if age < 30:
        return "Young"
    elif age < 50:
        return "Adult"
    else:
        return "Senior"

df["Age_Group"] = df["Age"].apply(age_group)

# Glucose to Insulin Ratio
df["Glucose_Insulin_Ratio"] = (
    df["Glucose"] / (df["Insulin"] + 1)
)

# Preview
print(df[[
    "BMI",
    "BMI_Category",
    "Age",
    "Age_Group",
    "Glucose",
    "Insulin",
    "Glucose_Insulin_Ratio"
]].head())

# Save
df.to_csv(
    "data/processed/featured_diabetes.csv",
    index=False
)

print("\nFeature engineered dataset saved successfully!")
import pandas as pd

# Load the dataset
df = pd.read_csv("data/raw/diabetes.csv")

# Display dataset information
print("Dataset Shape:", df.shape)

print("\nFirst Five Rows:")
print(df.head())
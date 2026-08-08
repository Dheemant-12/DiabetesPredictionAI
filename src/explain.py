import os
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt


# Load trained model
model = joblib.load(
    "models/diabetes_model.joblib"
)

# Load test data
test_df = pd.read_csv(
    "data/processed/test.csv"
)

X_test = test_df.drop(
    "Outcome",
    axis=1
)

# Create SHAP explainer
explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_test)

# Handle different SHAP versions
if isinstance(shap_values, list):
    values_for_positive_class = shap_values[1]

else:
    if shap_values.ndim == 3:
        values_for_positive_class = shap_values[:, :, 1]
    else:
        values_for_positive_class = shap_values

# Create output directory
os.makedirs(
    "notebooks/figures",
    exist_ok=True
)

# SHAP summary plot
shap.summary_plot(
    values_for_positive_class,
    X_test,
    show=False
)

plt.tight_layout()

plt.savefig(
    "notebooks/figures/shap_summary.png",
    bbox_inches="tight"
)

plt.close()

# SHAP feature importance plot
shap.summary_plot(
    values_for_positive_class,
    X_test,
    plot_type="bar",
    show=False
)

plt.tight_layout()

plt.savefig(
    "notebooks/figures/shap_feature_importance.png",
    bbox_inches="tight"
)

plt.close()

print("SHAP analysis completed successfully!")

print("\nGenerated files:")

print(
    "notebooks/figures/shap_summary.png"
)

print(
    "notebooks/figures/shap_feature_importance.png"
)
import joblib
import pandas as pd


MODEL_PATH = "models/diabetes_model.joblib"
FEATURE_PATH = "models/feature_names.joblib"


# Load model
model = joblib.load(MODEL_PATH)

# Load training feature names
feature_names = joblib.load(FEATURE_PATH)


def bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"


def age_group(age):
    if age < 30:
        return "Young"
    elif age < 50:
        return "Adult"
    else:
        return "Senior"


def predict_diabetes(patient_data):

    # Convert dictionary to DataFrame
    df = pd.DataFrame([patient_data])

    # Feature engineering
    df["BMI_Category"] = df["BMI"].apply(
        bmi_category
    )

    df["Age_Group"] = df["Age"].apply(
        age_group
    )

    df["Glucose_Insulin_Ratio"] = (
        df["Glucose"] /
        (df["Insulin"] + 1)
    )

    # One-hot encoding
    df = pd.get_dummies(
        df,
        columns=[
            "BMI_Category",
            "Age_Group"
        ],
        drop_first=True
    )

    # Make sure all training columns exist
    df = df.reindex(
        columns=feature_names,
        fill_value=0
    )

    # Prediction
    prediction = model.predict(df)[0]

    probability = model.predict_proba(
        df
    )[0][1]

    return {
        "prediction": int(prediction),
        "probability": float(probability)
    }


if __name__ == "__main__":

    sample_patient = {
        "Pregnancies": 2,
        "Glucose": 140,
        "BloodPressure": 80,
        "SkinThickness": 25,
        "Insulin": 100,
        "BMI": 31,
        "DiabetesPedigreeFunction": 0.5,
        "Age": 45
    }

    result = predict_diabetes(
        sample_patient
    )

    print("\nPrediction Result:")
    print(result)
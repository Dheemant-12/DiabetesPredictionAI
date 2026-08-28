import joblib
import pandas as pd


MODEL_PATH = (
    "models/diabetes_model.joblib"
)

FEATURE_PATH = (
    "models/feature_names.joblib"
)


# Load production model
model = joblib.load(
    MODEL_PATH
)

# Load production feature names
feature_names = joblib.load(
    FEATURE_PATH
)


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


def get_confidence(probability):

    if probability >= 0.75 or probability <= 0.25:
        return "High"

    elif probability >= 0.60 or probability <= 0.40:
        return "Medium"

    else:
        return "Low"


def predict_diabetes(patient_data):

    # Convert input to DataFrame
    df = pd.DataFrame(
        [patient_data]
    )


    # Feature engineering
    df["BMI_Category"] = df[
        "BMI"
    ].apply(
        bmi_category
    )


    df["Age_Group"] = df[
        "Age"
    ].apply(
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


    # Match training features exactly
    df = df.reindex(
        columns=feature_names,
        fill_value=0
    )


    # Make prediction
    prediction = model.predict(
        df
    )[0]


    probability = model.predict_proba(
        df
    )[0][1]


    probability = float(
        probability
    )


    # Human-readable result
    if prediction == 1:

        prediction_label = (
            "Diabetes Risk"
        )

        message = (
            "The model predicts a "
            "higher risk of diabetes."
        )

    else:

        prediction_label = (
            "Lower Diabetes Risk"
        )

        message = (
            "The model predicts a "
            "lower risk of diabetes."
        )


    confidence = get_confidence(
        probability
    )


    return {

        "prediction":
            int(prediction),

        "prediction_label":
            prediction_label,

        "probability":
            round(
                probability,
                4
            ),

        "confidence":
            confidence,

        "message":
            message

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


    print(
        "\nProduction Model Prediction:"
    )


    for key, value in result.items():

        print(
            f"{key}: {value}"
        )

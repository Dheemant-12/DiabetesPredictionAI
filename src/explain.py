import joblib
import pandas as pd


MODEL_PATH = (
    "models/production_model.joblib"
)

FEATURE_PATH = (
    "models/production_feature_names.joblib"
)


model = joblib.load(
    MODEL_PATH
)

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


def prepare_features(patient_data):

    df = pd.DataFrame(
        [patient_data]
    )


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


    df = pd.get_dummies(
        df,
        columns=[
            "BMI_Category",
            "Age_Group"
        ],
        drop_first=True
    )


    df = df.reindex(
        columns=feature_names,
        fill_value=0
    )


    return df


def explain_prediction(patient_data):

    df = prepare_features(
        patient_data
    )


    prediction = model.predict(
        df
    )[0]


    probability = model.predict_proba(
        df
    )[0][1]


    explanations = []


    # --------------------------------------
    # Glucose
    # --------------------------------------

    glucose = patient_data["Glucose"]

    if glucose >= 140:

        explanations.append({
            "feature": "Glucose",
            "value": glucose,
            "impact": "High",
            "message":
                "The glucose value is relatively high."
        })

    elif glucose >= 100:

        explanations.append({
            "feature": "Glucose",
            "value": glucose,
            "impact": "Moderate",
            "message":
                "The glucose value is moderately elevated."
        })

    else:

        explanations.append({
            "feature": "Glucose",
            "value": glucose,
            "impact": "Lower",
            "message":
                "The glucose value is relatively low."
        })


    # --------------------------------------
    # BMI
    # --------------------------------------

    bmi = patient_data["BMI"]

    if bmi >= 30:

        explanations.append({
            "feature": "BMI",
            "value": bmi,
            "impact": "High",
            "message":
                "The BMI falls in the obese range."
        })

    elif bmi >= 25:

        explanations.append({
            "feature": "BMI",
            "value": bmi,
            "impact": "Moderate",
            "message":
                "The BMI falls in the overweight range."
        })

    else:

        explanations.append({
            "feature": "BMI",
            "value": bmi,
            "impact": "Lower",
            "message":
                "The BMI is below the overweight range."
        })


    # --------------------------------------
    # Age
    # --------------------------------------

    age = patient_data["Age"]

    if age >= 50:

        explanations.append({
            "feature": "Age",
            "value": age,
            "impact": "High",
            "message":
                "Age is relatively high."
        })

    elif age >= 30:

        explanations.append({
            "feature": "Age",
            "value": age,
            "impact": "Moderate",
            "message":
                "Age falls within the adult range."
        })

    else:

        explanations.append({
            "feature": "Age",
            "value": age,
            "impact": "Lower",
            "message":
                "Age is relatively young."
        })


    # --------------------------------------
    # Blood Pressure
    # --------------------------------------

    blood_pressure = (
        patient_data["BloodPressure"]
    )

    if blood_pressure >= 90:

        explanations.append({
            "feature": "Blood Pressure",
            "value": blood_pressure,
            "impact": "High",
            "message":
                "Blood pressure is relatively high."
        })

    elif blood_pressure >= 80:

        explanations.append({
            "feature": "Blood Pressure",
            "value": blood_pressure,
            "impact": "Moderate",
            "message":
                "Blood pressure is moderately elevated."
        })

    else:

        explanations.append({
            "feature": "Blood Pressure",
            "value": blood_pressure,
            "impact": "Lower",
            "message":
                "Blood pressure is relatively lower."
        })


    # --------------------------------------
    # Insulin
    # --------------------------------------

    insulin = patient_data["Insulin"]

    if insulin >= 200:

        explanations.append({
            "feature": "Insulin",
            "value": insulin,
            "impact": "High",
            "message":
                "The insulin value is relatively high."
        })

    elif insulin >= 100:

        explanations.append({
            "feature": "Insulin",
            "value": insulin,
            "impact": "Moderate",
            "message":
                "The insulin value is moderately elevated."
        })

    else:

        explanations.append({
            "feature": "Insulin",
            "value": insulin,
            "impact": "Lower",
            "message":
                "The insulin value is relatively lower."
        })


    # --------------------------------------
    # Sort explanations
    # --------------------------------------

    impact_order = {
        "High": 3,
        "Moderate": 2,
        "Lower": 1
    }


    explanations.sort(
        key=lambda item:
            impact_order[item["impact"]],
        reverse=True
    )


    return {

        "prediction":
            int(prediction),

        "probability":
            round(
                float(probability),
                4
            ),

        "explanations":
            explanations

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


    result = explain_prediction(
        sample_patient
    )


    print(
        "\nPrediction Explanation:"
    )


    for explanation in result[
        "explanations"
    ]:

        print(
            f"{explanation['feature']}: "
            f"{explanation['impact']} impact"
        )

        print(
            f"  {explanation['message']}"
        )

import os
import sqlite3


DATABASE_PATH = "data/predictions.db"


def get_connection():
    os.makedirs("data", exist_ok=True)

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    connection.row_factory = sqlite3.Row

    return connection


def initialize_database():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pregnancies INTEGER NOT NULL,
            glucose REAL NOT NULL,
            blood_pressure REAL NOT NULL,
            skin_thickness REAL NOT NULL,
            insulin REAL NOT NULL,
            bmi REAL NOT NULL,
            diabetes_pedigree REAL NOT NULL,
            age INTEGER NOT NULL,
            prediction INTEGER NOT NULL,
            probability REAL NOT NULL,
            confidence TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    connection.commit()

    connection.close()


def save_prediction(
    patient_data,
    result
):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO predictions (
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            diabetes_pedigree,
            age,
            prediction,
            probability,
            confidence
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            patient_data["Pregnancies"],
            patient_data["Glucose"],
            patient_data["BloodPressure"],
            patient_data["SkinThickness"],
            patient_data["Insulin"],
            patient_data["BMI"],
            patient_data[
                "DiabetesPedigreeFunction"
            ],
            patient_data["Age"],
            result["prediction"],
            result["probability"],
            result["confidence"]
        )
    )

    connection.commit()

    connection.close()


def get_predictions():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT *
        FROM predictions
        ORDER BY id DESC
        """
    )

    rows = cursor.fetchall()

    connection.close()

    return [
        dict(row)
        for row in rows
    ]


def clear_predictions():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        "DELETE FROM predictions"
    )

    connection.commit()

    connection.close()
def get_prediction_analytics():

    predictions = get_predictions()

    total_predictions = len(
        predictions
    )

    if total_predictions == 0:

        return {
            "total_predictions": 0,
            "higher_risk": 0,
            "lower_risk": 0,
            "higher_risk_percentage": 0,
            "lower_risk_percentage": 0,
            "average_probability": 0
        }


    higher_risk = sum(
        1
        for prediction in predictions
        if prediction["prediction"] == 1
    )


    lower_risk = (
        total_predictions -
        higher_risk
    )


    average_probability = sum(
        float(
            prediction["probability"]
        )
        for prediction in predictions
    ) / total_predictions


    return {

        "total_predictions":
            total_predictions,

        "higher_risk":
            higher_risk,

        "lower_risk":
            lower_risk,

        "higher_risk_percentage":
            round(
                (
                    higher_risk /
                    total_predictions
                ) * 100,
                2
            ),

        "lower_risk_percentage":
            round(
                (
                    lower_risk /
                    total_predictions
                ) * 100,
                2
            ),

        "average_probability":
            round(
                average_probability * 100,
                2
            )

    } 
def get_recent_predictions(limit=10):

    predictions = get_predictions()

    return predictions[:limit]
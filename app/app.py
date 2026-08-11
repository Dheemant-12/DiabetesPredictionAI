from flask import Flask, request, jsonify

from src.prediction_pipeline import predict_diabetes


app = Flask(__name__)


REQUIRED_FIELDS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]


FIELD_RANGES = {
    "Pregnancies": (0, 20),
    "Glucose": (1, 300),
    "BloodPressure": (1, 200),
    "SkinThickness": (0, 100),
    "Insulin": (0, 1000),
    "BMI": (1, 80),
    "DiabetesPedigreeFunction": (0, 3),
    "Age": (1, 120)
}


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Diabetes Prediction API is running!"
    })


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    if not data:
        return jsonify({
            "error": "No input data provided."
        }), 400

    # Check required fields
    missing_fields = [
        field
        for field in REQUIRED_FIELDS
        if field not in data
    ]

    if missing_fields:
        return jsonify({
            "error": "Missing required fields.",
            "missing_fields": missing_fields
        }), 400

    # Validate values
    for field in REQUIRED_FIELDS:

        value = data[field]

        if isinstance(value, bool) or not isinstance(
            value,
            (int, float)
        ):
            return jsonify({
                "error": f"{field} must be a number."
            }), 400

        minimum, maximum = FIELD_RANGES[field]

        if value < minimum or value > maximum:
            return jsonify({
                "error": (
                    f"{field} must be between "
                    f"{minimum} and {maximum}."
                )
            }), 400

    try:
        result = predict_diabetes(data)

        return jsonify(result)

    except Exception as error:
        return jsonify({
            "error": "Prediction failed.",
            "details": str(error)
        }), 500


if __name__ == "__main__":
    app.run(
        debug=True
    )
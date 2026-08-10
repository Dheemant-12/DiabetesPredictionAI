from flask import Flask, request, jsonify

from src.prediction_pipeline import predict_diabetes


app = Flask(__name__)


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
            "error": "No input data provided"
        }), 400

    try:
        result = predict_diabetes(data)

        return jsonify(result)

    except Exception as error:
        return jsonify({
            "error": str(error)
        }), 400


if __name__ == "__main__":
    app.run(
        debug=True
    )
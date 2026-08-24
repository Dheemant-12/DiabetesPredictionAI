import json
from flask import (
    Flask,
    request,
    jsonify,
    render_template
)

from src.prediction_pipeline import (
    predict_diabetes
)

from src.explain import (
    explain_prediction
)

from app.database import (
    initialize_database,
    save_prediction,
    get_predictions,
    clear_predictions,
    get_prediction_analytics
)

from app.logger import (
    setup_logger
)


app = Flask(__name__)

logger = setup_logger()


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

    return render_template(
        "index.html"
    )


@app.route(
    "/health",
    methods=["GET"]
)
def health():

    return jsonify({
        "status": "healthy",
        "service": "Diabetes Prediction API"
    })


@app.route(
    "/model-status",
    methods=["GET"]
)
def model_status():

    try:

        with open(
            "models/model_metadata.json",
            "r",
            encoding="utf-8"
        ) as file:

            metadata = json.load(
                file
            )


        return jsonify({

            "status":
                "loaded",

            "model":
                metadata["model_name"],

            "selection_metric":
                metadata["selection_metric"],

            "secondary_metric":
                metadata["secondary_metric"],

            "accuracy":
                metadata["accuracy"],

            "precision":
                metadata["precision"],

            "recall":
                metadata["recall"],

            "f1_score":
                metadata["f1_score"],

            "roc_auc":
                metadata["roc_auc"]

        })


    except Exception as error:

        logger.exception(
            "Model status check failed"
        )

        return jsonify({

            "status":
                "error",

            "message":
                str(error)

        }), 500


@app.route(
    "/database-status",
    methods=["GET"]
)
def database_status():

    try:

        initialize_database()

        return jsonify({
            "status": "connected",
            "database": "SQLite"
        })

    except Exception as error:

        logger.error(
            "Database status check failed: %s",
            error
        )

        return jsonify({
            "status": "error",
            "message": str(error)
        }), 500


@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    data = request.get_json()

    logger.info(
        "Prediction request received"
    )

    if not data:

        logger.warning(
            "Prediction request contained no data"
        )

        return jsonify({
            "error":
                "No input data provided."
        }), 400


    missing_fields = [
        field
        for field in REQUIRED_FIELDS
        if field not in data
    ]


    if missing_fields:

        logger.warning(
            "Missing fields: %s",
            missing_fields
        )

        return jsonify({
            "error":
                "Missing required fields.",
            "missing_fields":
                missing_fields
        }), 400


    for field in REQUIRED_FIELDS:

        value = data[field]


        if (
            isinstance(value, bool)
            or not isinstance(
                value,
                (int, float)
            )
        ):

            logger.warning(
                "Invalid value for %s",
                field
            )

            return jsonify({
                "error":
                    f"{field} must be a number."
            }), 400


        minimum, maximum = (
            FIELD_RANGES[field]
        )


        if (
            value < minimum
            or value > maximum
        ):

            logger.warning(
                "%s outside valid range",
                field
            )

            return jsonify({
                "error": (
                    f"{field} must be between "
                    f"{minimum} and {maximum}."
                )
            }), 400


    try:

        result = predict_diabetes(
            data
        )


        save_prediction(
            data,
            result
        )


        logger.info(
            "Prediction completed successfully: %s",
            result["prediction"]
        )


        return jsonify(result)


    except Exception as error:

        logger.exception(
            "Prediction failed"
        )

        return jsonify({
            "error":
                "Prediction failed.",
            "details":
                str(error)
        }), 500


@app.route(
    "/history",
    methods=["GET"]
)
def history():

    try:

        predictions = get_predictions()


        return jsonify({
            "predictions": predictions
        })


    except Exception as error:

        logger.exception(
            "Failed to retrieve prediction history"
        )


        return jsonify({
            "error":
                "Failed to retrieve prediction history.",
            "details":
                str(error)
        }), 500


@app.route(
    "/history/clear",
    methods=["DELETE"]
)
def clear_history():

    try:

        clear_predictions()


        logger.info(
            "Prediction history cleared"
        )


        return jsonify({
            "message":
                "Prediction history cleared."
        })


    except Exception as error:

        logger.exception(
            "Failed to clear prediction history"
        )


        return jsonify({
            "error":
                "Failed to clear prediction history.",
            "details":
                str(error)
        }), 500

@app.route(
    "/explain",
    methods=["POST"]
)
def explain():

    data = request.get_json()


    logger.info(
        "Explanation request received"
    )


    if not data:

        return jsonify({
            "error":
                "No input data provided."
        }), 400


    missing_fields = [
        field
        for field in REQUIRED_FIELDS
        if field not in data
    ]


    if missing_fields:

        return jsonify({

            "error":
                "Missing required fields.",

            "missing_fields":
                missing_fields

        }), 400


    for field in REQUIRED_FIELDS:

        value = data[field]


        if (
            isinstance(value, bool)
            or not isinstance(
                value,
                (int, float)
            )
        ):

            return jsonify({

                "error":
                    f"{field} must be a number."

            }), 400


        minimum, maximum = (
            FIELD_RANGES[field]
        )


        if (
            value < minimum
            or value > maximum
        ):

            return jsonify({

                "error": (
                    f"{field} must be between "
                    f"{minimum} and {maximum}."
                )

            }), 400


    try:

        result = explain_prediction(
            data
        )


        return jsonify(
            result
        )


    except Exception as error:

        logger.exception(
            "Explanation failed"
        )


        return jsonify({

            "error":
                "Failed to generate explanation.",

            "details":
                str(error)

        }), 500
@app.route(
    "/analytics",
    methods=["GET"]
)
def analytics():

    try:

        analytics_data = (
            get_prediction_analytics()
        )


        return jsonify(
            analytics_data
        )


    except Exception as error:

        logger.exception(
            "Analytics request failed"
        )


        return jsonify({

            "error":
                "Failed to retrieve analytics.",

            "details":
                str(error)

        }), 500
if __name__ == "__main__":

    initialize_database()


    logger.info(
        "Starting Diabetes Prediction API"
    )


    app.run(
        debug=True
    )
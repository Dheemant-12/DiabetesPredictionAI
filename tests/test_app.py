import pytest

from app.app import app


@pytest.fixture
def client():

    app.config["TESTING"] = True

    with app.test_client() as client:

        yield client


def valid_patient():

    return {

        "Pregnancies": 2,

        "Glucose": 140,

        "BloodPressure": 80,

        "SkinThickness": 25,

        "Insulin": 100,

        "BMI": 31,

        "DiabetesPedigreeFunction": 0.5,

        "Age": 45

    }


# ==========================================
# HEALTH TEST
# ==========================================

def test_health(client):

    response = client.get(
        "/health"
    )

    assert response.status_code == 200

    data = response.get_json()

    assert data["status"] == "healthy"


# ==========================================
# DATABASE STATUS TEST
# ==========================================

def test_database_status(client):

    response = client.get(
        "/database-status"
    )

    assert response.status_code == 200

    data = response.get_json()

    assert data["status"] == "connected"


# ==========================================
# MODEL STATUS TEST
# ==========================================

def test_model_status(client):

    response = client.get(
        "/model-status"
    )

    assert response.status_code == 200


# ==========================================
# HISTORY TEST
# ==========================================

def test_history(client):

    response = client.get(
        "/history"
    )

    assert response.status_code == 200

    data = response.get_json()

    assert "predictions" in data


# ==========================================
# ANALYTICS TEST
# ==========================================

def test_analytics(client):

    response = client.get(
        "/analytics"
    )

    assert response.status_code == 200

    data = response.get_json()

    assert "total_predictions" in data

    assert "higher_risk" in data

    assert "lower_risk" in data

    assert "average_probability" in data


# ==========================================
# RECENT PREDICTIONS TEST
# ==========================================

def test_recent_predictions(client):

    response = client.get(
        "/analytics/recent"
    )

    assert response.status_code == 200

    data = response.get_json()

    assert "predictions" in data


# ==========================================
# VALID PREDICTION TEST
# ==========================================

def test_prediction(client):

    response = client.post(
        "/predict",
        json=valid_patient()
    )

    assert response.status_code == 200

    data = response.get_json()

    assert "prediction" in data

    assert "probability" in data

    assert "confidence" in data


# ==========================================
# EXPLANATION TEST
# ==========================================

def test_explanation(client):

    response = client.post(
        "/explain",
        json=valid_patient()
    )

    assert response.status_code == 200

    data = response.get_json()

    assert "prediction" in data

    assert "probability" in data

    assert "explanations" in data


# ==========================================
# MISSING FIELD TEST
# ==========================================

def test_missing_fields(client):

    data = {

        "Pregnancies": 2,

        "Glucose": 140,

        "Age": 45

    }


    response = client.post(
        "/predict",
        json=data
    )


    assert response.status_code == 400

    result = response.get_json()

    assert (
        "missing_fields"
        in result
    )


# ==========================================
# INVALID GLUCOSE TEST
# ==========================================

def test_invalid_glucose(client):

    data = valid_patient()

    data["Glucose"] = 900


    response = client.post(
        "/predict",
        json=data
    )


    assert response.status_code == 400

    result = response.get_json()

    assert (
        "Glucose must be between 1 and 300."
        == result["error"]
    )


# ==========================================
# INVALID AGE TEST
# ==========================================

def test_invalid_age(client):

    data = valid_patient()

    data["Age"] = 150


    response = client.post(
        "/predict",
        json=data
    )


    assert response.status_code == 400

    result = response.get_json()

    assert (
        "Age must be between 1 and 120."
        == result["error"]
    )
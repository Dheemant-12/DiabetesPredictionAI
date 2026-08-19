import requests


BASE_URL = (
    "http://127.0.0.1:5000"
)


def test_health():

    response = requests.get(
        f"{BASE_URL}/health"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"


def test_model_status():

    response = requests.get(
        f"{BASE_URL}/model-status"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "loaded"


def test_database_status():

    response = requests.get(
        f"{BASE_URL}/database-status"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "connected"


def test_prediction():

    patient = {

        "Pregnancies": 2,

        "Glucose": 140,

        "BloodPressure": 80,

        "SkinThickness": 25,

        "Insulin": 100,

        "BMI": 31,

        "DiabetesPedigreeFunction": 0.5,

        "Age": 45

    }


    response = requests.post(

        f"{BASE_URL}/predict",

        json=patient

    )


    assert response.status_code == 200


    data = response.json()


    assert "prediction" in data

    assert "probability" in data

    assert "confidence" in data

    assert "prediction_label" in data


if __name__ == "__main__":

    test_health()

    test_model_status()

    test_database_status()

    test_prediction()

    print(
        "All backend API tests passed!"
    )
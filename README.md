# Diabetes Prediction AI

A Machine Learning web application that predicts the likelihood of diabetes using the Pima Indians Diabetes Dataset.

## Features

- Diabetes prediction using Machine Learning
- Multiple model comparison
- Saved trained model
- Prediction probability and confidence
- Input validation
- Prediction history
- SQLite database
- Prediction analytics
- Recent prediction trends
- Prediction explanations
- REST API
- Automated API testing
- Production-ready Flask configuration

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Flask
- SQLite
- Joblib
- Pytest
- HTML
- CSS
- JavaScript

## Project Structure

```text
DiabetesPredictionAI/
│
├── app/
│   ├── app.py
│   ├── database.py
│   ├── logger.py
│   │
│   ├── templates/
│   │   └── index.html
│   │
│   └── static/
│       └── style.css
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── diabetes_model.joblib
│   └── model_metadata.json
│
├── src/
│   ├── clean_data.py
│   ├── prediction_pipeline.py
│   ├── explain.py
│   ├── save_model.py
│   └── models/
│       ├── logistic.py
│       ├── tree.py
│       └── xgboost_model.py
│
├── tests/
│   ├── test_app.py
│   └── test_api.py
│
├── notebooks/
│
├── requirements.txt
├── pytest.ini
├── .gitignore
└── README.md
# House Price Prediction API

A machine learning project that predicts house price per unit area from property details. It includes a Flask web app with a simple UI and a FastAPI REST API.

## Features
- Compares Linear Regression, Random Forest and Gradient Boosting using 5-fold cross-validation
- Tunes the best model with GridSearchCV
- Flask web app with a landing page and a prediction form
- FastAPI endpoint `POST /predict` with automatic docs at `/docs`

## Model Performance
| Model | Cross-validation R2 |
|---|---|
| Linear Regression | 0.54 |
| Random Forest (selected) | 0.67 |
| Gradient Boosting | 0.66 |

Final model on the test set: R2 = 0.82, RMSE = 5.52

## Dataset
Real Estate Valuation dataset (414 records). Input features: transaction date, house age, distance to the nearest MRT station, number of convenience stores, latitude and longitude. Target: house price per unit area.

## Tech Stack
Python, scikit-learn, pandas, NumPy, Flask, FastAPI, HTML, CSS

## Project Structure
    app.py                 Flask web app
    api.py                 FastAPI REST API
    requirements.txt       Dependencies
    Dataset/               Real estate CSV
    Model/                 Training script and saved model
    templates/             HTML pages

## How to Run

1. Install dependencies

        pip install -r requirements.txt

2. Run the Flask web app

        python app.py

   Open http://127.0.0.1:5000

3. Run the FastAPI API

        python -m uvicorn api:app --reload

   Open http://127.0.0.1:8000/docs and try `POST /predict`

## Sample API Request

    {
      "transaction_date": 2013.5,
      "house_age": 10,
      "distance_to_mrt": 500,
      "convenience_stores": 5,
      "latitude": 24.982,
      "longitude": 121.54
    }

## Author
Tanvi Choukade
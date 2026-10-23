from flask import Flask, render_template, request
import os
import pickle
import pandas as pd

app = Flask(__name__)

# Load model (path relative to this file, so it works on any computer)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "Model", "real_estate_model_v2.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Same column names and order as used during training
FEATURES = [
    "X1 transaction date",
    "X2 house age",
    "X3 distance to the nearest MRT station",
    "X4 number of convenience stores",
    "X5 latitude",
    "X6 longitude",
]


@app.route('/')
def landing():
    return render_template("landing.html")


@app.route('/home')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    try:
        values = [
            float(request.form['transaction_date']),
            float(request.form['house_age']),
            float(request.form['distance']),
            float(request.form['stores']),
            float(request.form['latitude']),
            float(request.form['longitude']),
        ]
    except (ValueError, KeyError):
        return render_template("index.html",
                               prediction_text="Please enter valid numbers in all fields.")

    features = pd.DataFrame([values], columns=FEATURES)
    prediction = model.predict(features)
    result = round(float(prediction[0]), 2)

    return render_template("index.html",
                           prediction_text="ok",
                           prediction_value=result)


if __name__ == "__main__":
    app.run(debug=True)





from flask import Flask, jsonify
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load model, scaler, and training columns
model = joblib.load("ml_api/logistic_regression_model.pkl")
scaler = joblib.load("ml_api/scaler.pkl")  # Load your fitted scaler
training_columns = joblib.load("training_columns.pkl")  # Load the training feature columns

@app.route("/api/predict", methods=["GET"])
def predict():
    # Example input data
    data = {
        "age": 30,
        "job": "technician",
        "marital": "single",
        "education": "university.degree",
        "default": "no",
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "month": "may",
        "day_of_week": "mon",
        "duration": 200,
        "campaign": 1,
        "pdays": 999,
        "previous": 0,
        "poutcome": "nonexistent",
        "emp_var_rate": 1.1,
        "cons_price_idx": 93.994,
        "cons_conf_idx": -36.4,
        "euribor3m": 4.860,
        "nr_employed": 5191.0
    }

    # Create a DataFrame from the input data
    input_data = pd.DataFrame([data])

    # One-hot encode categorical variables
    input_data_encoded = pd.get_dummies(input_data, drop_first=True)

    # Reindex the encoded input data to match the training columns
    input_data_encoded = input_data_encoded.reindex(columns=training_columns, fill_value=0)

    # Scale the input data
    input_data_scaled = scaler.transform(input_data_encoded)

    # Make predictions
    predictions = model.predict(input_data_scaled)

    # Convert predictions to probabilities
    probabilities = model.predict_proba(input_data_scaled)[:, 1]

    # Return the results as JSON
    return jsonify({
        "predictions": predictions.tolist(),
        "probabilities": probabilities.tolist()
    })

if __name__ == "__main__":
    app.run(debug=True)

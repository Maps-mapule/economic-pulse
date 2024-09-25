from flask import Flask
import joblib

app = Flask(__name__)
model = joblib.load("logistic_regression_model.pkl")

@app.route("/api/predict",methods=["POST"])
def predict():
    return "<p>Hello, World!</p>"
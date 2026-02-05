from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os

# FLASK SETUP
app = Flask(__name__)
CORS(app)

price_model = None
model_features = None
encoder = None

# LOAD MODEL ARTIFACTS
def load_resources():
    global price_model, model_features, encoder

    if price_model is None:
        price_model = joblib.load("price_model.pkl")
        model_features = joblib.load("model_features.pkl")
        encoder = joblib.load("ordinal_encoder.pkl")

# PREDICT API
@app.route("/predict", methods=["POST"])
def predict_price():
    try:
        load_resources()

        data = request.get_json()
        df = pd.DataFrame([data])

        # Drop non-inference columns
        if "Property ID" in df.columns:
            df = df.drop(columns=["Property ID"])

        # Date feature engineering
        if "Date Sold" in df.columns:
            df["Date Sold"] = pd.to_datetime(df["Date Sold"])

            df["Sold_Year"] = df["Date Sold"].dt.year
            df["Sold_Month"] = df["Date Sold"].dt.month
            df["Sold_Quarter"] = df["Date Sold"].dt.quarter

            if "Year Built" in df.columns:
                df["Property_Age_At_Sale"] = df["Sold_Year"] - df["Year Built"]

            df = df.drop(columns=["Date Sold"])

        # Encode categorical features
        cat_cols = ["Location", "Condition", "Type"]
        df[cat_cols] = encoder.transform(df[cat_cols])

        # Align with training features
        df = df.reindex(columns=model_features)

        prediction = price_model.predict(df)[0]

        return jsonify({
            "predicted_price": round(float(prediction), 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

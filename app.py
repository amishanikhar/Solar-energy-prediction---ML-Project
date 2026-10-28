from pathlib import Path
import json
import joblib
import pandas as pd

from flask import Flask, render_template, request

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent

MODEL_FILE = BASE_DIR / "best_solar_model.pkl"
SCALER_FILE = BASE_DIR / "scaler.pkl"
THRESHOLD_FILE = BASE_DIR / "generation_thresholds.json"

# Load files created by train_model.py
model = joblib.load(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)

with open(THRESHOLD_FILE, "r", encoding="utf-8") as file:
    thresholds = json.load(file)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Read values entered through the website
        amb_temp = float(request.form["amb_temp"])
        mod_temp = float(request.form["mod_temp"])
        irradiation = float(request.form["irradiation"])

        # Basic input validation
        if not all(map(
            lambda value: value == value and abs(value) != float("inf"),
            [amb_temp, mod_temp, irradiation]
        )):
            raise ValueError("Please enter valid numeric values.")

        if not (0 <= amb_temp <= 50):
            raise ValueError("Ambient temperature must be between 0 and 50.")

        if not (0 <= mod_temp <= 80):
            raise ValueError("Module temperature must be between 0 and 80.")

        if not (0 <= irradiation <= 1.25):
            raise ValueError("Irradiation must be between 0 and 1.25.")

        # Arrange input in the same feature order used for training
        input_data = pd.DataFrame(
            [[amb_temp, mod_temp, irradiation]],
            columns=[
                "AMBIENT_TEMPERATURE",
                "MODULE_TEMPERATURE",
                "IRRADIATION"
            ]
        )

        # Apply the already-fitted scaler and predict
        input_scaled = scaler.transform(input_data)
        prediction_value = float(model.predict(input_scaled)[0])

        # Power generation cannot be negative
        prediction_value = max(0.0, prediction_value)

        # Status thresholds come from the training data
        if prediction_value < thresholds["low"]:
            generation_status = "Low Generation"
        elif prediction_value < thresholds["high"]:
            generation_status = "Normal Generation"
        else:
            generation_status = "High Generation"

        return render_template(
            "index.html",
            prediction=f"{prediction_value:.2f} kW",
            low_threshold=round(thresholds["low"], 1),
            high_threshold=round(thresholds["high"], 1),
            generation_status=generation_status,
            amb_temp=amb_temp,
            mod_temp=mod_temp,
            irradiation=irradiation
        )

    except (ValueError, KeyError) as error:
        return render_template("index.html", error=str(error)), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

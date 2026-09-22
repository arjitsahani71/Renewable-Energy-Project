import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from sklearn.ensemble import RandomForestRegressor


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "models"
FRONTEND_DIR = BASE_DIR / "frontend"

MODEL_PATH = MODEL_DIR / "solar_power_model.pkl"


# =========================================================
# FLASK APP
# =========================================================

app = Flask(__name__)
CORS(app)


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

model = joblib.load(MODEL_PATH)


# =========================================================
# LOAD DATASET
# =========================================================

generation = pd.read_csv(
    DATASET_DIR / "Plant_1_Generation_Data.csv"
)

weather = pd.read_csv(
    DATASET_DIR / "Plant_1_Weather_Sensor_Data.csv"
)


# =========================================================
# PREPARE DATA
# =========================================================

generation["DATE_TIME"] = pd.to_datetime(generation["DATE_TIME"])
weather["DATE_TIME"] = pd.to_datetime(weather["DATE_TIME"])

data = pd.merge(
    generation,
    weather,
    on=["DATE_TIME", "PLANT_ID"],
    how="inner"
)

data = data.drop(
    columns=["SOURCE_KEY_x", "SOURCE_KEY_y"],
    errors="ignore"
)

data = data.dropna().copy()

# Time features
data["hour"] = data["DATE_TIME"].dt.hour
data["minute"] = data["DATE_TIME"].dt.minute
data["day"] = data["DATE_TIME"].dt.day
data["month"] = data["DATE_TIME"].dt.month
data["day_of_week"] = data["DATE_TIME"].dt.dayofweek


# =========================================================
# MODEL FEATURES
# =========================================================

MODEL_FEATURES = [
    "DC_POWER",
    "DAILY_YIELD",
    "AMBIENT_TEMPERATURE",
    "MODULE_TEMPERATURE",
    "IRRADIATION",
    "hour",
    "minute",
    "day",
    "month",
    "day_of_week"
]


# =========================================================
# FUTURE FORECAST MODEL
# =========================================================

FORECAST_FEATURES = [
    "IRRADIATION",
    "AMBIENT_TEMPERATURE",
    "MODULE_TEMPERATURE",
    "hour",
    "minute",
    "day",
    "month",
    "day_of_week"
]

forecast_model = RandomForestRegressor(
    n_estimators=120,
    random_state=42,
    n_jobs=-1
)

forecast_training_data = data[
    FORECAST_FEATURES + ["AC_POWER"]
].copy()

forecast_model.fit(
    forecast_training_data[FORECAST_FEATURES],
    forecast_training_data["AC_POWER"]
)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def generate_recommendation(
    actual_power,
    predicted_power,
    irradiation,
    efficiency,
    alert_count
):
    if alert_count > 0:

        if irradiation > 0.4 and actual_power < predicted_power * 0.8:
            return (
                "High irradiation but low power generation detected. "
                "Check panel cleanliness, shading, wiring, or inverter performance."
            )

        if irradiation < 0.1:
            return (
                "Low solar irradiation detected. "
                "Power generation is expected to be low."
            )

        return (
            "Performance deviation detected. "
            "Inspect the solar system for possible abnormal conditions."
        )

    if efficiency >= 90:
        return (
            "Solar system is operating efficiently. "
            "Continue regular monitoring and maintenance."
        )

    if efficiency >= 70:
        return (
            "System performance is acceptable. "
            "Regular cleaning and preventive maintenance are recommended."
        )

    return (
        "System efficiency is relatively low. "
        "Check panel cleanliness, shading, temperature, and inverter performance."
    )


def estimate_future_weather(selected_date):
    """
    Estimate irradiation and temperature using historical
    month/hour patterns from the dataset.
    """

    month = selected_date.month
    day_of_week = selected_date.weekday()

    historical = data[
        (data["month"] == month)
    ].copy()

    if historical.empty:
        historical = data.copy()

    hourly = (
        historical
        .groupby("hour")
        .agg({
            "IRRADIATION": "mean",
            "AMBIENT_TEMPERATURE": "mean",
            "MODULE_TEMPERATURE": "mean"
        })
        .reset_index()
    )

    return hourly


def get_historical_date_data(selected_date):
    day_data = data[
        data["DATE_TIME"].dt.date == selected_date
    ].copy()

    return day_data


# =========================================================
# FRONTEND ROUTES
# =========================================================

@app.route("/")
def serve_index():
    return send_from_directory(
        FRONTEND_DIR,
        "index.html"
    )


@app.route("/style.css")
def serve_css():
    return send_from_directory(
        FRONTEND_DIR,
        "style.css"
    )


@app.route("/script.js")
def serve_script():
    return send_from_directory(
        FRONTEND_DIR,
        "script.js"
    )


# =========================================================
# BASIC API
# =========================================================

@app.route("/api/test")
def test_api():
    return jsonify({
        "status": "success",
        "message": "Solar Energy Monitoring API is working"
    })


@app.route("/api/model")
def model_info():
    return jsonify({
        "status": "success",
        "model": "Random Forest Regressor",
        "features": MODEL_FEATURES
    })


# =========================================================
# DATE INFORMATION
# =========================================================

@app.route("/api/date-info")
def date_info():

    min_date = data["DATE_TIME"].min().date()
    max_date = data["DATE_TIME"].max().date()

    return jsonify({
        "status": "success",
        "min_date": str(min_date),
        "max_date": str(max_date)
    })


# =========================================================
# DASHBOARD API
# =========================================================

@app.route("/api/dashboard")
def dashboard():

    selected_date_string = request.args.get("date")

    if selected_date_string:
        try:
            selected_date = pd.to_datetime(
                selected_date_string
            ).date()
        except Exception:
            return jsonify({
                "status": "error",
                "message": "Invalid date format"
            }), 400
    else:
        selected_date = data["DATE_TIME"].max().date()


    # =====================================================
    # HISTORICAL MODE
    # =====================================================

    day_data = get_historical_date_data(selected_date)

    if not day_data.empty:

        # Select a generating timestamp rather than
        # selecting a nighttime zero-power row.
        generating_data = day_data[
            day_data["AC_POWER"] > 1
        ]

        if not generating_data.empty:
            row = generating_data.iloc[-1]
        else:
            row = day_data.iloc[-1]


        X = pd.DataFrame([{
            feature: row[feature]
            for feature in MODEL_FEATURES
        }])

        predicted_power = float(
            model.predict(X)[0]
        )

        actual_power = float(
            row["AC_POWER"]
        )

        dc_power = float(
            row["DC_POWER"]
        )

        irradiation = float(
            row["IRRADIATION"]
        )

        ambient_temperature = float(
            row["AMBIENT_TEMPERATURE"]
        )

        module_temperature = float(
            row["MODULE_TEMPERATURE"]
        )

        daily_yield = float(
            row["DAILY_YIELD"]
        )

        # Prediction error
        if actual_power != 0:
            error_percentage = abs(
                predicted_power - actual_power
            ) / abs(actual_power) * 100
        else:
            error_percentage = 0

        # Efficiency
        if dc_power > 0:
            efficiency = (
                actual_power / dc_power
            ) * 100
        else:
            efficiency = 0

        # Limit efficiency to sensible display range
        efficiency = max(
            0,
            min(efficiency, 100)
        )

        # Alert
        alert_count = 0

        if (
            error_percentage > 20
            and irradiation > 0.1
        ):
            alert_count = 1

        if (
            irradiation > 0.4
            and actual_power < predicted_power * 0.8
        ):
            alert_count = 1

        if alert_count > 0:
            system_status = "Warning"
        else:
            system_status = "Normal"

        recommendation = generate_recommendation(
            actual_power,
            predicted_power,
            irradiation,
            efficiency,
            alert_count
        )

        return jsonify({
            "status": "success",
            "data": {
                "mode": "historical",
                "selected_date": str(selected_date),

                "date_time": str(
                    row["DATE_TIME"]
                ),

                "actual_power": round(
                    actual_power, 2
                ),

                "predicted_power": round(
                    predicted_power, 2
                ),

                "dc_power": round(
                    dc_power, 2
                ),

                "irradiation": round(
                    irradiation, 3
                ),

                "ambient_temperature": round(
                    ambient_temperature, 2
                ),

                "module_temperature": round(
                    module_temperature, 2
                ),

                "daily_yield": round(
                    daily_yield, 2
                ),

                "efficiency": round(
                    efficiency, 2
                ),

                "error_percentage": round(
                    error_percentage, 2
                ),

                "alert_count": alert_count,

                "system_status": system_status,

                "recommendation": recommendation
            }
        })


    # =====================================================
    # FUTURE FORECAST MODE
    # =====================================================

    weather_pattern = estimate_future_weather(
        selected_date
    )

    predictions = []

    for _, weather_row in weather_pattern.iterrows():

        hour = int(
            weather_row["hour"]
        )

        # Skip nighttime
        if hour < 5 or hour > 19:
            continue

        # Use a standard 15-minute interval
        for minute in [0, 15, 30, 45]:

            features = pd.DataFrame([{
                "IRRADIATION": float(
                    weather_row["IRRADIATION"]
                ),

                "AMBIENT_TEMPERATURE": float(
                    weather_row["AMBIENT_TEMPERATURE"]
                ),

                "MODULE_TEMPERATURE": float(
                    weather_row["MODULE_TEMPERATURE"]
                ),

                "hour": hour,
                "minute": minute,
                "day": selected_date.day,
                "month": selected_date.month,
                "day_of_week": selected_date.weekday()
            }])

            prediction = float(
                forecast_model.predict(
                    features
                )[0]
            )

            prediction = max(
                prediction,
                0
            )

            predictions.append({
                "hour": hour,
                "minute": minute,
                "power": prediction
            })


    if predictions:

        predicted_powers = [
            item["power"]
            for item in predictions
        ]

        peak_power = max(
            predicted_powers
        )

        average_power = np.mean(
            predicted_powers
        )

        # Approximate daily energy from
        # 15-minute intervals.
        daily_energy = sum(
            predicted_powers
        ) * 0.25

        max_prediction = max(
            predicted_powers
        )

        peak_index = predicted_powers.index(
            max_prediction
        )

        peak_item = predictions[
            peak_index
        ]

        peak_time = (
            f"{peak_item['hour']:02d}:"
            f"{peak_item['minute']:02d}"
        )

    else:

        peak_power = 0
        average_power = 0
        daily_energy = 0
        peak_time = "N/A"


    recommendation = (
        "This is an AI-based forecast using historical "
        "solar generation and weather patterns. "
        "Actual future generation can vary with real-time "
        "weather, cloud cover, shading, and system conditions."
    )

    return jsonify({
        "status": "success",
        "data": {
            "mode": "forecast",
            "selected_date": str(selected_date),

            "date_time": (
                f"{selected_date} "
                f"{peak_time}"
            ),

            "actual_power": None,

            "predicted_power": round(
                peak_power, 2
            ),

            "dc_power": None,

            "irradiation": round(
                float(
                    weather_pattern["IRRADIATION"].mean()
                ),
                3
            ),

            "ambient_temperature": round(
                float(
                    weather_pattern[
                        "AMBIENT_TEMPERATURE"
                    ].mean()
                ),
                2
            ),

            "module_temperature": round(
                float(
                    weather_pattern[
                        "MODULE_TEMPERATURE"
                    ].mean()
                ),
                2
            ),

            "daily_yield": round(
                daily_energy,
                2
            ),

            "efficiency": None,

            "error_percentage": None,

            "alert_count": 0,

            "system_status": "AI Forecast",

            "recommendation": recommendation
        }
    })


# =========================================================
# CHART API
# =========================================================

@app.route("/api/chart")
def chart():

    selected_date_string = request.args.get("date")

    if selected_date_string:
        try:
            selected_date = pd.to_datetime(
                selected_date_string
            ).date()
        except Exception:
            return jsonify({
                "status": "error",
                "message": "Invalid date"
            }), 400
    else:
        selected_date = data["DATE_TIME"].max().date()


    # =====================================================
    # HISTORICAL CHART
    # =====================================================

    day_data = get_historical_date_data(
        selected_date
    )

    if not day_data.empty:

        chart_data = []

        for _, row in day_data.iterrows():

            X = pd.DataFrame([{
                feature: row[feature]
                for feature in MODEL_FEATURES
            }])

            prediction = float(
                model.predict(X)[0]
            )

            chart_data.append({
                "time": row["DATE_TIME"].strftime(
                    "%H:%M"
                ),

                "actual": round(
                    float(row["AC_POWER"]),
                    2
                ),

                "predicted": round(
                    prediction,
                    2
                )
            })

        return jsonify({
            "status": "success",
            "mode": "historical",
            "data": chart_data
        })


    # =====================================================
    # FUTURE FORECAST CHART
    # =====================================================

    weather_pattern = estimate_future_weather(
        selected_date
    )

    chart_data = []

    for _, weather_row in weather_pattern.iterrows():

        hour = int(
            weather_row["hour"]
        )

        if hour < 5 or hour > 19:
            continue

        for minute in [0, 15, 30, 45]:

            features = pd.DataFrame([{
                "IRRADIATION": float(
                    weather_row["IRRADIATION"]
                ),

                "AMBIENT_TEMPERATURE": float(
                    weather_row[
                        "AMBIENT_TEMPERATURE"
                    ]
                ),

                "MODULE_TEMPERATURE": float(
                    weather_row[
                        "MODULE_TEMPERATURE"
                    ]
                ),

                "hour": hour,
                "minute": minute,
                "day": selected_date.day,
                "month": selected_date.month,
                "day_of_week": selected_date.weekday()
            }])

            prediction = float(
                forecast_model.predict(
                    features
                )[0]
            )

            chart_data.append({
                "time": f"{hour:02d}:{minute:02d}",

                "actual": None,

                "predicted": round(
                    max(prediction, 0),
                    2
                )
            })

    return jsonify({
        "status": "success",
        "mode": "forecast",
        "data": chart_data
    })


# =========================================================
# MANUAL PREDICTION API
# =========================================================

@app.route("/api/predict", methods=["POST"])
def predict():

    try:

        input_data = request.get_json()

        X = pd.DataFrame([{
            feature: input_data[feature]
            for feature in MODEL_FEATURES
        }])

        prediction = float(
            model.predict(X)[0]
        )

        return jsonify({
            "status": "success",
            "predicted_power": round(
                prediction,
                2
            )
        })

    except Exception as e:

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=True
    )
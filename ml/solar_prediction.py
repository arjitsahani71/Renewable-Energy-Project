# ============================================================
# RENEWABLE ENERGY MONITORING AND OPTIMIZATION SYSTEM
# Machine Learning - Solar Power Prediction
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import joblib


# ============================================================
# 2. PROJECT PATHS
# ============================================================

# Get the main project folder
# Current file:
# Renewable-Energy-Project/ml/solar_prediction.py
#
# parent      -> ml
# parent.parent -> Renewable-Energy-Project

BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset folder
DATASET_DIR = BASE_DIR / "dataset"

# Models folder
MODEL_DIR = BASE_DIR / "models"

# Results folder
RESULTS_DIR = BASE_DIR / "results"

# Create folders if they don't exist
MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


print("============================================")
print("RENEWABLE ENERGY ML SYSTEM")
print("============================================")

print("\nProject directory:")
print(BASE_DIR)

print("\nDataset directory:")
print(DATASET_DIR)

print("\nModel directory:")
print(MODEL_DIR)


# ============================================================
# 3. LOAD DATASET
# ============================================================

print("\n============================================")
print("1. LOADING DATA")
print("============================================")


generation_file = DATASET_DIR / "Plant_1_Generation_Data.csv"

weather_file = DATASET_DIR / "Plant_1_Weather_Sensor_Data.csv"


# Check whether files exist

if not generation_file.exists():

    raise FileNotFoundError(
        f"\nGeneration dataset not found:\n{generation_file}"
    )


if not weather_file.exists():

    raise FileNotFoundError(
        f"\nWeather dataset not found:\n{weather_file}"
    )


# Read CSV files

generation = pd.read_csv(
    generation_file
)

weather = pd.read_csv(
    weather_file
)


print("\nGeneration data loaded successfully.")

print(
    "Generation data shape:",
    generation.shape
)


print("\nWeather data loaded successfully.")

print(
    "Weather data shape:",
    weather.shape
)


# ============================================================
# 4. DISPLAY BASIC INFORMATION
# ============================================================

print("\n============================================")
print("2. DATA INFORMATION")
print("============================================")


print("\nGeneration columns:")

print(
    generation.columns.tolist()
)


print("\nWeather columns:")

print(
    weather.columns.tolist()
)


# ============================================================
# 5. CONVERT DATE/TIME
# ============================================================

print("\n============================================")
print("3. CONVERTING DATE/TIME")
print("============================================")


# Generation dataset uses DD-MM-YYYY format

generation["DATE_TIME"] = pd.to_datetime(
    generation["DATE_TIME"],
    dayfirst=True
)


# Weather dataset uses YYYY-MM-DD format

weather["DATE_TIME"] = pd.to_datetime(
    weather["DATE_TIME"]
)


print("Date conversion completed.")


# ============================================================
# 6. CHECK MISSING VALUES
# ============================================================

print("\n============================================")
print("4. CHECKING MISSING VALUES")
print("============================================")


print("\nGeneration missing values:")

print(
    generation.isnull().sum()
)


print("\nWeather missing values:")

print(
    weather.isnull().sum()
)


# ============================================================
# 7. MERGE GENERATION AND WEATHER DATA
# ============================================================

print("\n============================================")
print("5. MERGING DATA")
print("============================================")


data = pd.merge(
    generation,
    weather,
    on=["DATE_TIME", "PLANT_ID"],
    how="inner"
)


print(
    "\nMerged data shape:",
    data.shape
)


# ============================================================
# 8. REMOVE UNNECESSARY COLUMNS
# ============================================================

print("\n============================================")
print("6. REMOVING UNNECESSARY COLUMNS")
print("============================================")


# After merging, SOURCE_KEY becomes:
#
# SOURCE_KEY_x
# SOURCE_KEY_y
#
# We don't need these identifiers for this model.

columns_to_remove = [
    "PLANT_ID",
    "SOURCE_KEY_x",
    "SOURCE_KEY_y"
]


data = data.drop(
    columns=columns_to_remove,
    errors="ignore"
)


print(
    "Remaining columns:"
)

print(
    data.columns.tolist()
)


# ============================================================
# 9. REMOVE MISSING VALUES
# ============================================================

print("\n============================================")
print("7. CLEANING DATA")
print("============================================")


before_cleaning = len(data)


data = data.dropna()


after_cleaning = len(data)


print(
    "Rows before cleaning:",
    before_cleaning
)

print(
    "Rows after cleaning:",
    after_cleaning
)

print(
    "Rows removed:",
    before_cleaning - after_cleaning
)


# ============================================================
# 10. FEATURE ENGINEERING
# ============================================================

print("\n============================================")
print("8. FEATURE ENGINEERING")
print("============================================")


# Extract time-related information

data["hour"] = data["DATE_TIME"].dt.hour

data["minute"] = data["DATE_TIME"].dt.minute

data["day"] = data["DATE_TIME"].dt.day

data["month"] = data["DATE_TIME"].dt.month

data["day_of_week"] = (
    data["DATE_TIME"].dt.dayofweek
)


print("\nTime features created:")

print(
    [
        "hour",
        "minute",
        "day",
        "month",
        "day_of_week"
    ]
)


# ============================================================
# 11. SELECT FEATURES
# ============================================================

print("\n============================================")
print("9. SELECTING FEATURES")
print("============================================")


features = [

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


# Target variable

target = "AC_POWER"


X = data[features]

y = data[target]


print("\nInput features:")

for feature in features:

    print(
        " -",
        feature
    )


print("\nTarget:")

print(
    " -",
    target
)


print("\nX shape:")

print(
    X.shape
)


print("\ny shape:")

print(
    y.shape
)


# ============================================================
# 12. TRAIN / TEST SPLIT
# ============================================================

print("\n============================================")
print("10. SPLITTING DATA")
print("============================================")


X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42

)


print(
    "\nTraining samples:",
    len(X_train)
)


print(
    "Testing samples:",
    len(X_test)
)


# ============================================================
# 13. CREATE RANDOM FOREST MODEL
# ============================================================

print("\n============================================")
print("11. CREATING RANDOM FOREST MODEL")
print("============================================")


model = RandomForestRegressor(

    n_estimators=200,

    random_state=42,

    n_jobs=-1

)


print(
    "Random Forest created."
)


# ============================================================
# 14. TRAIN MODEL
# ============================================================

print("\n============================================")
print("12. TRAINING MODEL")
print("============================================")


print(
    "\nTraining started..."
)


model.fit(

    X_train,

    y_train

)


print(
    "Training completed successfully."
)


# ============================================================
# 15. MAKE PREDICTIONS
# ============================================================

print("\n============================================")
print("13. MAKING PREDICTIONS")
print("============================================")


y_pred = model.predict(
    X_test
)


print(
    "Predictions generated."
)


# ============================================================
# 16. MODEL EVALUATION
# ============================================================

print("\n============================================")
print("14. MODEL EVALUATION")
print("============================================")


mae = mean_absolute_error(

    y_test,

    y_pred

)


rmse = np.sqrt(

    mean_squared_error(

        y_test,

        y_pred

    )

)


r2 = r2_score(

    y_test,

    y_pred

)


print("\nModel performance:")

print(
    "MAE  :",
    mae
)

print(
    "RMSE :",
    rmse
)

print(
    "R2   :",
    r2
)


# ============================================================
# 17. FEATURE IMPORTANCE
# ============================================================

print("\n============================================")
print("15. FEATURE IMPORTANCE")
print("============================================")


feature_importance = pd.DataFrame({

    "Feature": features,

    "Importance": model.feature_importances_

})


feature_importance = (
    feature_importance
    .sort_values(
        by="Importance",
        ascending=False
    )
)


print(
    "\nFeature importance:"
)

print(
    feature_importance
)


# Save feature importance

feature_importance.to_csv(

    RESULTS_DIR / "feature_importance.csv",

    index=False

)


# ============================================================
# 18. ACTUAL VS PREDICTED GRAPH
# ============================================================

print("\n============================================")
print("16. CREATING PREDICTION GRAPH")
print("============================================")


plt.figure(
    figsize=(10, 6)
)


plt.scatter(

    y_test,

    y_pred,

    alpha=0.3

)


plt.xlabel(
    "Actual AC Power"
)


plt.ylabel(
    "Predicted AC Power"
)


plt.title(
    "Actual vs Predicted Solar Power"
)


plt.tight_layout()


# Save graph

plt.savefig(

    RESULTS_DIR / "actual_vs_predicted.png"

)


plt.show()


# ============================================================
# 19. CREATE RESULTS DATAFRAME
# ============================================================

print("\n============================================")
print("17. CREATING PREDICTION RESULTS")
print("============================================")


results = X_test.copy()


results["actual_power"] = (
    y_test.values
)


results["predicted_power"] = (
    y_pred
)


# Difference between actual and predicted

results["difference"] = (

    results["actual_power"]

    -

    results["predicted_power"]

)


# ============================================================
# 20. CALCULATE ERROR PERCENTAGE
# ============================================================


results["error_percentage"] = (

    abs(
        results["difference"]
    )

    /

    (
        abs(
            results["predicted_power"]
        )

        + 1
    )

) * 100


# ============================================================
# 21. ANOMALY DETECTION
# ============================================================

print("\n============================================")
print("18. ANOMALY DETECTION")
print("============================================")


# Threshold used for this project

ANOMALY_THRESHOLD = 20


results["anomaly"] = (

    results["error_percentage"]

    >

    ANOMALY_THRESHOLD

)


# Convert True/False into readable status

results["status"] = np.where(

    results["anomaly"],

    "ANOMALY",

    "NORMAL"

)


# ============================================================
# 22. RECOMMENDATION SYSTEM
# ============================================================

print("\n============================================")
print("19. GENERATING RECOMMENDATIONS")
print("============================================")


def recommendation(row):

    # If system is normal

    if not row["anomaly"]:

        return (
            "System operating normally"
        )


    # If anomaly and irradiation is high

    if row["IRRADIATION"] > 0.5:

        return (
            "Check panel cleanliness, "
            "inverter performance, "
            "or possible equipment fault"
        )


    # If anomaly and irradiation is low

    return (
        "Low solar irradiation; "
        "low generation may be expected"
    )


results["recommendation"] = (

    results.apply(

        recommendation,

        axis=1

    )

)


# ============================================================
# 23. DISPLAY RESULTS
# ============================================================

print("\n============================================")
print("20. SAMPLE RESULTS")
print("============================================")


display_columns = [

    "actual_power",

    "predicted_power",

    "difference",

    "error_percentage",

    "status",

    "recommendation"

]


print(

    results[
        display_columns
    ].head(20)

)


# ============================================================
# 24. ANOMALY SUMMARY
# ============================================================

print("\n============================================")
print("21. ANOMALY SUMMARY")
print("============================================")


total_records = len(results)


anomaly_count = (

    results["anomaly"]

    .sum()

)


normal_count = (

    total_records

    -

    anomaly_count

)


print(
    "Total test records:",
    total_records
)


print(
    "Normal records:",
    normal_count
)


print(
    "Anomalies detected:",
    anomaly_count
)


print(
    "Anomaly percentage:",
    round(
        (anomaly_count / total_records) * 100,
        2
    ),
    "%"
)


# ============================================================
# 25. SAVE PREDICTION RESULTS
# ============================================================

print("\n============================================")
print("22. SAVING RESULTS")
print("============================================")


results.to_csv(

    RESULTS_DIR / "solar_prediction_results.csv",

    index=False

)


print(
    "Prediction results saved."
)


# ============================================================
# 26. SAVE TRAINED MODEL
# ============================================================

print("\n============================================")
print("23. SAVING ML MODEL")
print("============================================")


model_path = (

    MODEL_DIR

    /

    "solar_power_model.pkl"

)


joblib.dump(

    model,

    model_path

)


print(
    "\nModel saved successfully!"
)


print(
    "Model location:"
)


print(
    model_path
)


# ============================================================
# 27. FINISHED
# ============================================================

print("\n============================================")
print("ML PIPELINE COMPLETED")
print("============================================")


print(
    "\nYour ML system is ready."
)


print(
    "\nGenerated files:"
)


print(
    "1. models/solar_power_model.pkl"
)


print(
    "2. results/solar_prediction_results.csv"
)


print(
    "3. results/feature_importance.csv"
)


print(
    "4. results/actual_vs_predicted.png"
)


print(
    "\nNext step: Flask backend + frontend dashboard."
)
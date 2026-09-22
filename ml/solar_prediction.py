from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

generation_path = DATASET_DIR / "Plant_1_Generation_Data.csv"
weather_path = DATASET_DIR / "Plant_1_Weather_Sensor_Data.csv"

print("Loading generation data...")
generation = pd.read_csv(generation_path)

print("Loading weather data...")
weather = pd.read_csv(weather_path)

print("Generation data shape:", generation.shape)
print("Weather data shape:", weather.shape)


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

generation.columns = generation.columns.str.strip()
weather.columns = weather.columns.str.strip()


# ============================================================
# CONVERT DATE_TIME
# ============================================================

generation["DATE_TIME"] = pd.to_datetime(
    generation["DATE_TIME"],
    errors="coerce"
)

weather["DATE_TIME"] = pd.to_datetime(
    weather["DATE_TIME"],
    errors="coerce"
)


# ============================================================
# MERGE GENERATION + WEATHER DATA
# ============================================================

print("Merging generation and weather data...")

df = pd.merge(
    generation,
    weather,
    on=["DATE_TIME", "PLANT_ID"],
    how="inner"
)

print("Merged data shape:", df.shape)


# ============================================================
# REMOVE UNNECESSARY COLUMNS
# ============================================================

columns_to_drop = [
    "PLANT_ID",
    "SOURCE_KEY_x",
    "SOURCE_KEY_y"
]

for column in columns_to_drop:
    if column in df.columns:
        df.drop(column, axis=1, inplace=True)


# ============================================================
# TIME-BASED FEATURES
# ============================================================

df["hour"] = df["DATE_TIME"].dt.hour
df["minute"] = df["DATE_TIME"].dt.minute
df["day"] = df["DATE_TIME"].dt.day
df["month"] = df["DATE_TIME"].dt.month
df["day_of_week"] = df["DATE_TIME"].dt.dayofweek


# ============================================================
# REMOVE MISSING VALUES
# ============================================================

print("Missing values before cleaning:")
print(df.isnull().sum())

df.dropna(inplace=True)

print("Data shape after removing missing values:", df.shape)


# ============================================================
# FEATURES AND TARGET
# ============================================================

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

target = "AC_POWER"


X = df[features]
y = df[target]


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# RANDOM FOREST MODEL
# ============================================================
#
# Reduced from 200 trees to 20 trees so the model can fit
# within the free deployment memory constraints.
#

model = RandomForestRegressor(
    n_estimators=20,
    random_state=42,
    n_jobs=-1
)


print("\nTraining Random Forest model...")

model.fit(X_train, y_train)

print("Training completed.")


# ============================================================
# PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# MODEL EVALUATION
# ============================================================

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)


print("\n========================================")
print("MODEL PERFORMANCE")
print("========================================")

print(f"MAE  : {mae:.6f}")
print(f"RMSE : {rmse:.6f}")
print(f"R2   : {r2:.6f}")


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n========================================")
print("FEATURE IMPORTANCE")
print("========================================")

print(importance)


# Save feature importance
importance.to_csv(
    RESULTS_DIR / "feature_importance.csv",
    index=False
)


# ============================================================
# SAVE MODEL
# ============================================================

model_path = MODEL_DIR / "solar_power_model.pkl"

print("\nSaving model...")

joblib.dump(
    model,
    model_path
)

print("Model saved to:")
print(model_path)


# ============================================================
# MODEL SIZE
# ============================================================

model_size_mb = model_path.stat().st_size / (1024 * 1024)

print(f"\nModel size: {model_size_mb:.2f} MB")


# ============================================================
# SAVE METRICS
# ============================================================

metrics = pd.DataFrame({
    "Metric": [
        "MAE",
        "RMSE",
        "R2",
        "Number of Trees",
        "Model Size (MB)"
    ],
    "Value": [
        mae,
        rmse,
        r2,
        20,
        model_size_mb
    ]
})

metrics.to_csv(
    RESULTS_DIR / "model_metrics.csv",
    index=False
)


print("\n========================================")
print("ML PIPELINE COMPLETED")
print("========================================")

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# ==========================================
# 1. PROJECT FILE PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

GENERATION_FILE = BASE_DIR / "Plant_1_Generation_Data.csv"
WEATHER_FILE = BASE_DIR / "Plant_1_Weather_Sensor_Data.csv"

MODEL_FILE = BASE_DIR / "best_solar_model.pkl"
SCALER_FILE = BASE_DIR / "scaler.pkl"
THRESHOLD_FILE = BASE_DIR / "generation_thresholds.json"

FEATURES = [
    "AMBIENT_TEMPERATURE",
    "MODULE_TEMPERATURE",
    "IRRADIATION"
]

TARGET = "AC_POWER"
JOIN_KEYS = ["DATE_TIME", "PLANT_ID"]


def save_figures(results_df, y_true, y_pred, best_name):
    """Save report figures (skipped quietly if matplotlib is missing)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping figures.")
        return

    out = BASE_DIR / "figures"
    out.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, metric in zip(axes, ["MAE", "RMSE", "R2"]):
        ax.bar(results_df["Model"], results_df[metric], color="#176b4d")
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=25)
    fig.suptitle("Model comparison on unseen test days")
    fig.tight_layout()
    fig.savefig(out / "model_comparison.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(y_true, y_pred, s=3, alpha=0.3, color="#176b4d")
    lim = max(y_true.max(), y_pred.max())
    ax.plot([0, lim], [0, lim], "r--", linewidth=1)
    ax.set_xlabel("Actual AC power (kW)")
    ax.set_ylabel("Predicted AC power (kW)")
    ax.set_title(f"Actual vs predicted - {best_name}")
    fig.tight_layout()
    fig.savefig(out / "actual_vs_predicted.png", dpi=200)
    plt.close(fig)
    print("Figures saved in:", out)


def main():

    print("\n" + "=" * 60)
    print("       SOLAR POWER FORECASTING - MODEL TRAINING")
    print("=" * 60)

    # ==========================================
    # 2. CHECK AND LOAD CSV FILES
    # ==========================================

    if not GENERATION_FILE.exists() or not WEATHER_FILE.exists():
        raise FileNotFoundError(
            "Keep both CSV files in the same folder as train_model.py."
        )

    generation = pd.read_csv(GENERATION_FILE)
    weather = pd.read_csv(WEATHER_FILE)

    print("\n--- ORIGINAL DATASET SIZES ---")
    print("Generation dataset:", generation.shape)
    print("Weather dataset:   ", weather.shape)

    # ==========================================
    # 3. CHECK MISSING VALUES AND DUPLICATES
    # ==========================================

    print("\n--- BEFORE CLEANING ---")

    print(
        "Generation missing values:",
        generation.isnull().sum().sum()
    )
    print(
        "Weather missing values:   ",
        weather.isnull().sum().sum()
    )
    print(
        "Generation duplicate rows:",
        generation.duplicated().sum()
    )
    print(
        "Weather duplicate rows:   ",
        weather.duplicated().sum()
    )

    # Remove exact duplicate rows
    generation = generation.drop_duplicates().copy()
    weather = weather.drop_duplicates().copy()

    # ==========================================
    # 4. CLEAN DATE AND JOIN KEY COLUMNS
    # ==========================================

    # The generation file mixes "15-05-2020 00:00" and "1/6/2020 0:00"
    # (both day-first). A single fixed format would silently drop ~37% rows.
    generation["DATE_TIME"] = pd.to_datetime(
        generation["DATE_TIME"],
        format="mixed",
        dayfirst=True,
        errors="coerce"
    )

    # Weather file uses month-first dates, e.g. "5/15/2020 0:00"
    weather["DATE_TIME"] = pd.to_datetime(
        weather["DATE_TIME"],
        format="%m/%d/%Y %H:%M",
        errors="coerce"
    )

    # Remove invalid dates and missing join keys
    generation = generation.dropna(subset=JOIN_KEYS).copy()
    weather = weather.dropna(subset=JOIN_KEYS).copy()

    # Keep only columns required for this project
    # SOURCE_KEY (inverter id) is kept so that different inverters with the same
    # output (e.g. 0 kW at night) are NOT mistaken for duplicate rows.
    generation = generation[JOIN_KEYS + ["SOURCE_KEY", TARGET]].copy()
    weather = weather[JOIN_KEYS + FEATURES].copy()

    # Convert model columns to numeric values
    for column in [TARGET]:
        generation[column] = pd.to_numeric(
            generation[column], errors="coerce"
        )

    for column in FEATURES:
        weather[column] = pd.to_numeric(
            weather[column], errors="coerce"
        )

    # Remove rows with missing model values
    generation = generation.dropna(subset=[TARGET]).copy()
    weather = weather.dropna(subset=FEATURES).copy()

    # ==========================================
    # 5. HANDLE DUPLICATE WEATHER JOIN KEYS
    # ==========================================

    duplicate_keys = weather.duplicated(subset=JOIN_KEYS).sum()

    print("\nDuplicate weather join keys:", duplicate_keys)

    if duplicate_keys > 0:
        print("Keeping the first weather record per date and plant.")
        weather = weather.drop_duplicates(
            subset=JOIN_KEYS,
            keep="first"
        )

    # ==========================================
    # 6. MERGE BOTH DATASETS
    # ==========================================

    data = pd.merge(
        generation,
        weather,
        on=JOIN_KEYS,
        how="inner",
        validate="many_to_one"
    )

    print("\n--- AFTER MERGING ---")
    print("Merged dataset size:", data.shape)
    print("Generation rows not matched:", len(generation) - len(data))

    required_columns = FEATURES + [TARGET]

    print("\nMissing values in model columns:")
    print(data[required_columns].isnull().sum())

    print(
        "Duplicate merged rows:",
        data.duplicated().sum()
    )

    # Remove incomplete rows and exact duplicates
    data = data.dropna(subset=required_columns)
    data = data.drop_duplicates().copy()

    print("\n--- FINAL CLEANED DATASET ---")
    print("Final dataset size:", data.shape)
    print("Total usable rows:", len(data))

    if len(data) < 10:
        raise ValueError(
            "Too few usable rows. Check CSV files and date formats."
        )

    # ==========================================
    # 7. SELECT FEATURES AND TARGET
    # ==========================================

    # Sort by time so the split can be chronological
    data = data.sort_values("DATE_TIME").reset_index(drop=True)

    X = data[FEATURES]
    y = data[TARGET]

    print("\nInput features (X):", FEATURES)
    print("Target variable (y):", TARGET)
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Date range:", data["DATE_TIME"].min(), "->", data["DATE_TIME"].max())

    # ==========================================
    # 8. TRAIN-TEST SPLIT (CHRONOLOGICAL, 80/20 BY DAY)
    # ==========================================
    # Readings are 15-minute time series and 22 inverters share the same
    # weather at each timestamp. A random split would put near-identical rows
    # in both train and test (data leakage -> inflated scores), so we train on
    # the earlier days and test on the later, unseen days.

    days = sorted(data["DATE_TIME"].dt.date.unique())
    split_index = int(len(days) * 0.80)
    train_days = set(days[:split_index])

    is_train = data["DATE_TIME"].dt.date.isin(train_days)

    X_train, y_train = X[is_train], y[is_train]
    X_test, y_test = X[~is_train], y[~is_train]

    print("\n--- TRAIN-TEST SPLIT (chronological by day) ---")
    print("Training days:", split_index, f"({days[0]} to {days[split_index - 1]})")
    print("Testing days: ", len(days) - split_index, f"({days[split_index]} to {days[-1]})")
    print("Training rows:", len(X_train))
    print("Testing rows: ", len(X_test))
    print(
        "Training percentage:",
        round(len(X_train) / len(X) * 100, 2), "%"
    )
    print(
        "Testing percentage: ",
        round(len(X_test) / len(X) * 100, 2), "%"
    )

    # Fit scaler ONLY on training data
    scaler = MinMaxScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ==========================================
    # 9. DEFINE FOUR REGRESSION MODELS
    # ==========================================

    models = {
        "Decision Tree": DecisionTreeRegressor(
            random_state=42
        ),

        "SVR": SVR(
            kernel="rbf",
            C=100,
            epsilon=0.1
        ),

        "KNN": KNeighborsRegressor(
            n_neighbors=5
        ),

        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
    }

    # ==========================================
    # 10. TRAIN AND EVALUATE ALL MODELS
    # ==========================================

    results = []
    trained_models = {}

    print("\n" + "=" * 60)
    print("         TRAINING AND EVALUATING MODELS")
    print("=" * 60)

    for name, model in models.items():

        print(f"\nTraining {name}...")

        model.fit(X_train_scaled, y_train)

        # Power cannot be negative (same rule the website applies)
        predictions = np.maximum(model.predict(X_test_scaled), 0)

        mae = mean_absolute_error(y_test, predictions)
        rmse = mean_squared_error(y_test, predictions) ** 0.5
        r2 = r2_score(y_test, predictions)

        results.append({
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })

        trained_models[name] = model

        print(f"{name} training completed.")

    # ==========================================
    # 11. DISPLAY COMPARISON TABLE
    # ==========================================

    results_df = pd.DataFrame(results)

    # Lowest RMSE is our primary selection criterion
    results_df = results_df.sort_values(
        by="RMSE",
        ascending=True
    ).reset_index(drop=True)

    print("\n" + "=" * 76)
    print("                  MODEL COMPARISON TABLE")
    print("=" * 76)

    print(
        results_df.to_string(
            index=False,
            formatters={
                "MAE": "{:.4f}".format,
                "RMSE": "{:.4f}".format,
                "R2": "{:.4f}".format
            }
        )
    )

    print("\nMAE: lower is better")
    print("RMSE: lower is better")
    print("R2: higher is generally better")

    # ==========================================
    # 12. SELECT AND SAVE THE BEST MODEL
    # ==========================================

    best_model_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]

    # Compare actual and predicted values on unseen test data
    best_predictions = np.maximum(best_model.predict(X_test_scaled), 0)

    comparison = pd.DataFrame({
        "Actual Power": y_test.values[:10],
        "Predicted Power": best_predictions[:10]
    })

    comparison["Absolute Error"] = (
        comparison["Actual Power"] -
        comparison["Predicted Power"]
    ).abs()

    print("\n--- ACTUAL VS PREDICTED POWER (10 TEST SAMPLES) ---")
    print(comparison.round(3).to_string(index=False))

    results_df.to_csv(BASE_DIR / "model_comparison.csv", index=False)
    save_figures(results_df, y_test.values, best_predictions, best_model_name)

    joblib.dump(best_model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)

    # Thresholds used by the website to show generation status
    # Night-time readings are all 0 kW, so quantiles of ALL rows would give
    # low = 0. Thresholds are therefore taken from daytime (AC_POWER > 0) rows.
    daytime_power = y_train[y_train > 0]
    thresholds = {
        "low": float(daytime_power.quantile(1 / 3)),
        "high": float(daytime_power.quantile(2 / 3))
    }

    with open(THRESHOLD_FILE, "w", encoding="utf-8") as file:
        json.dump(thresholds, file, indent=4)

    print("\n" + "=" * 60)
    print("                 BEST MODEL SELECTED")
    print("=" * 60)

    print("Best model:", best_model_name)
    print(
        "Best RMSE:",
        f"{results_df.iloc[0]['RMSE']:.4f}"
    )
    print(
        "Corresponding MAE:",
        f"{results_df.iloc[0]['MAE']:.4f}"
    )
    print(
        "Corresponding R2:",
        f"{results_df.iloc[0]['R2']:.4f}"
    )

    print("\n--- FILES SAVED ---")
    print("Model:     ", MODEL_FILE)
    print("Scaler:    ", SCALER_FILE)
    print("Thresholds:", THRESHOLD_FILE)

    print("\nAll four models evaluated successfully!")
    print("Training completed successfully!")


if __name__ == "__main__":
    main()

# Step 3: train and tune a CatBoost regressor on ionization potential (IP).
# Reuses the chemical-property-predictor training pipeline: VarianceThreshold +
# StandardScaler, Optuna-tuned CatBoost, 70/15/15 train/val/test split
# (test_size=0.3, then 0.15 of the remainder held out as validation, same
# convention as chemical_property_predictor.ipynb cell 12).
import time

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from sklearn import set_config
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

optuna.logging.set_verbosity(optuna.logging.WARNING)
set_config(transform_output="pandas")

IN_PATH = "data/descriptors.csv"
TARGET = "ionization_potential_eV"
N_TRIALS = 15


def main():
    t0 = time.time()
    properties = pd.read_csv(IN_PATH, low_memory=False)

    subset = properties[properties[TARGET].notna()].copy()
    Q1, Q3 = subset[TARGET].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    subset = subset[
        (subset[TARGET] >= Q1 - 1.5 * IQR) & (subset[TARGET] <= Q3 + 1.5 * IQR)
    ]
    print(f"{TARGET}: {len(subset)} rows after IQR outlier removal (from {len(properties)})")

    X = subset.drop(columns=["qm9_id", "smiles", TARGET])
    X = X.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all").dropna()
    y = subset[TARGET][X.index]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)
    print(f"Split sizes -- train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

    preprocessor = Pipeline(steps=[
        ("variance", VarianceThreshold(threshold=0.0)),
        ("scaler", StandardScaler()),
    ])
    preprocessor.fit(X_train)
    X_train_p = preprocessor.transform(X_train)
    X_val_p = preprocessor.transform(X_val)
    X_test_p = preprocessor.transform(X_test)

    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 500, 3000),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "depth": trial.suggest_int("depth", 3, 6),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 100.0, log=True),
            "rsm": trial.suggest_float("rsm", 0.1, 1.0),
            "loss_function": "RMSE",
            "early_stopping_rounds": 100,
            "verbose": False,
        }
        m = CatBoostRegressor(**params)
        m.fit(X_train_p, y_train, eval_set=(X_val_p, y_val), use_best_model=True)
        return np.sqrt(mean_squared_error(y_val, m.predict(X_val_p)))

    tune_t0 = time.time()
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_TRIALS)
    tune_time = time.time() - tune_t0
    print(f"Best trial RMSE (val): {study.best_value:.4f}")
    print(f"Optuna tuning time ({N_TRIALS} trials): {tune_time:.1f}s")

    best_params = study.best_params
    best_params.update({"loss_function": "RMSE", "early_stopping_rounds": 100, "verbose": 200})
    model = CatBoostRegressor(**best_params)
    fit_t0 = time.time()
    model.fit(X_train_p, y_train, eval_set=(X_val_p, y_val), use_best_model=True)
    fit_time = time.time() - fit_t0

    predictions = model.predict(X_test_p)
    mask = y_test.abs() > 1e-8
    mape = float(np.mean(np.abs((y_test[mask] - predictions[mask]) / y_test[mask])))
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    total_time = time.time() - t0

    print()
    print(f"=== Test set results: {TARGET} ===")
    print(f"n_test = {len(y_test)}")
    print(f"R2   = {r2:.4f}")
    print(f"MAE  = {mae:.4f} eV")
    print(f"RMSE = {rmse:.4f} eV")
    print(f"MAPE = {mape:.4f}")
    print(f"Best params: {study.best_params}")
    print(f"Total training script time: {total_time:.1f}s")

    importances = model.get_feature_importance()
    feature_names = X_train_p.columns
    importance_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    importance_df = importance_df.sort_values("importance", ascending=False)
    print()
    print("Top 15 features:")
    print(importance_df.head(15).to_string(index=False))

    # Save a small summary for RESULTS.md consumption
    with open("data/results_summary.txt", "w") as f:
        f.write(f"n_start_molecules=7000\n")
        f.write(f"n_after_extraction=5699\n")
        f.write(f"n_after_featurization={len(properties)}\n")
        f.write(f"n_after_iqr_and_split={len(subset)}\n")
        f.write(f"n_train={len(X_train)}\n")
        f.write(f"n_val={len(X_val)}\n")
        f.write(f"n_test={len(X_test)}\n")
        f.write(f"r2={r2:.4f}\n")
        f.write(f"mae_eV={mae:.4f}\n")
        f.write(f"rmse_eV={rmse:.4f}\n")
        f.write(f"mape={mape:.4f}\n")
        f.write(f"optuna_trials={N_TRIALS}\n")
        f.write(f"tune_time_s={tune_time:.1f}\n")
        f.write(f"final_fit_time_s={fit_time:.1f}\n")
        f.write(f"total_script_time_s={total_time:.1f}\n")
        f.write(f"best_params={study.best_params}\n")


if __name__ == "__main__":
    main()

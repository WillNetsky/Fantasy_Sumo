# train_model_stacking.py
import pandas as pd
import joblib
import argparse
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sumo_utils import preprocess_data, _parse_rank

# --- Constants ---
FEATURES = [
    'division_numeric', 'rank_in_division',
    'prev_division_numeric', 'prev_rank_in_division',
    'prev_w', 'prev_l',
    'age', 'bmi', 'heya_strength', 'has_uni_sumo', 'win_consistency',
    'rank_gap',
    'was_kyujo_last_basho',
    'kachi_koshi_streak',
    'division_strength',
    'oshi_ratio',
    'avg_opponent_oshi_ratio',
    'avg_h2h_win_pct',
    # Injury risk features
    'prev_absences',
    'absences_last_3',
    'completion_rate_last_3',
    'age_x_prev_absences',
    'basho_since_full'
]
TARGET = 'w'

# Best params from previous tuning for LightGBM
LGBM_PARAMS = {
    'n_estimators': 1000,
    'learning_rate': 0.01,
    'num_leaves': 20,
    'max_depth': 7,
    'reg_alpha': 0.1,
    'reg_lambda': 0.5,
    'colsample_bytree': 0.7,
    'subsample': 1.0,
    'random_state': 42,
    'n_jobs': 1,  # Single thread for stacking base models
    'verbosity': -1
}

# Reasonable defaults for XGBoost
XGB_PARAMS = {
    'n_estimators': 1000,
    'learning_rate': 0.01,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': 1, # Single thread for stacking base models
    'verbosity': 0
}

# Reasonable defaults for Random Forest
RF_PARAMS = {
    'n_estimators': 200,
    'max_depth': 10,
    'min_samples_leaf': 2,
    'random_state': 42,
    'n_jobs': 1 # Single thread for stacking base models
}


def is_top_division(rank_str: str) -> bool:
    division, _, _ = _parse_rank(rank_str)
    return division in ['Y', 'O', 'S', 'K', 'M', 'J']


def train_and_save_stacking_model(file_path: str, model_output_path: str):
    print(f"Loading historical data from '{file_path}'...")
    try:
        df_historical = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return

    print("Loading match history...")
    try:
        match_history_df = pd.read_csv("match_history_with_kimarite.csv")
    except FileNotFoundError:
        print("Warning: Match history not found. Advanced features disabled.")
        match_history_df = None

    # Filter for top divisions
    mask_top_division = df_historical['rank'].apply(is_top_division)
    df_top_divisions = df_historical[mask_top_division].copy()
    print(f"Filtered records: {len(df_top_divisions)}")

    print("Preprocessing data...")
    df_processed = preprocess_data(df_top_divisions, match_history_df=match_history_df)
    df_clean = df_processed[FEATURES + [TARGET]].dropna()
    print(f"Cleaned records for training: {len(df_clean)}")

    if df_clean.empty:
        print("Error: No data available for training.")
        return

    X = df_clean[FEATURES]
    y = df_clean[TARGET]

    # --- Define Base Models ---
    # We use n_jobs=1 for base models to avoid contention, as StackingRegressor
    # can parallelize the cross-validation of these models.
    estimators = [
        ('lgbm', lgb.LGBMRegressor(**LGBM_PARAMS)),
        ('xgb', xgb.XGBRegressor(**XGB_PARAMS)),
        ('rf', RandomForestRegressor(**RF_PARAMS))
    ]

    # --- Define Meta-Learner ---
    # RidgeCV is a good default meta-learner (regularized linear regression).
    # It learns how to best weight the predictions of the base models.
    final_estimator = RidgeCV()

    print("\nInitializing Stacking Regressor...")
    # n_jobs=-1 here allows the base models to be trained in parallel on different folds
    stacking_model = StackingRegressor(
        estimators=estimators,
        final_estimator=final_estimator,
        n_jobs=-1,
        passthrough=False # False means meta-learner only sees base model predictions, not original features
    )

    print("Performing validation check (Train/Test Split)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Stacking Model on split (this may take a moment)...")
    stacking_model.fit(X_train, y_train)
    predictions = stacking_model.predict(X_test)

    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    print("--- Stacking Model Performance ---")
    print(f"  R-squared (R²): {r2:.4f}")
    print(f"  Mean Absolute Error (MAE): {mae:.4f} wins")
    print("-" * 35)

    print("\nRetraining Stacking Model on ALL data...")
    stacking_model.fit(X, y)

    model_and_features = {
        'model': stacking_model,
        'features': FEATURES,
        'mae': mae  # Save MAE for use in prediction/fantasy scoring
    }
    print(f"Saving stacking model to '{model_output_path}'...")
    joblib.dump(model_and_features, model_output_path)
    print("Model saved successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Stacking Ensemble model.")
    parser.add_argument('--data-file', type=str, default="banzuke_detailed.csv")
    parser.add_argument('--model-file', type=str, default="sumo_win_predictor_model.joblib")
    args = parser.parse_args()

    train_and_save_stacking_model(args.data_file, args.model_file)

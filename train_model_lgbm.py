# train_model_lgbm.py
import pandas as pd
import joblib
import argparse
import lightgbm as lgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error
from sumo_utils import preprocess_data, _parse_rank

# --- Constants for better maintainability ---
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
    'basho_since_full',
    # Elo features
    'elo',
    'elo_change_last_2_bouts',
    'elo_change_last_5_bouts',
    'elo_change_last_15_bouts',
    'elo_change_last_1_basho',
    'elo_change_last_2_basho',
]
TARGET = 'w'

# --- MODIFICATION: Update with the new champion parameters ---
# These are the best parameters found when training with all H2H and style features.
DEFAULT_BEST_PARAMS = {
    'n_estimators': 1000,
    'learning_rate': 0.01,
    'num_leaves': 20,
    'max_depth': 10,
    'reg_alpha': 0.0,
    'reg_lambda': 0.0,
    'colsample_bytree': 0.8,
    'subsample': 0.8,
    'random_state': 42,
    'n_jobs': -1
}
# --- END MODIFICATION ---

# The hyperparameter grid remains the same for further tuning.
PARAM_GRID = {
    'n_estimators': [1000, 1500, 2000],
    'learning_rate': [0.01, 0.02],
    'num_leaves': [20, 31, 40],
    'max_depth': [7, 10, -1],
    'reg_alpha': [0.0, 0.1, 0.5],
    'reg_lambda': [0.0, 0.1, 0.5],
    'colsample_bytree': [0.7, 0.8, 1.0],
    'subsample': [0.7, 0.8, 1.0]
}


def is_top_division(rank_str: str) -> bool:
    """
    Checks if a rank string belongs to the top two divisions (Makuuchi or Juryo).
    """
    division, _, _ = _parse_rank(rank_str)
    return division in ['Y', 'O', 'S', 'K', 'M', 'J']


def train_and_save_model(file_path: str, model_output_path: str, tune_hyperparameters: bool):
    """
    Loads historical data, preprocesses it, trains the final model on all available data,
    and saves it to a file. Optionally performs hyperparameter tuning.
    """
    print(f"Loading historical data from '{file_path}' for training...")
    try:
        df_historical = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return

    print("Loading match history for style feature engineering...")
    try:
        match_history_df = pd.read_csv("data/match_history_with_kimarite.csv")
    except FileNotFoundError:
        print("Warning: 'data/match_history_with_kimarite.csv' not found. Style features will not be used.")
        match_history_df = None

    print(f"Original records loaded: {len(df_historical)}")

    # --- REFACTORED: Use centralized parsing logic for filtering ---
    mask_top_division = df_historical['rank'].apply(is_top_division)
    df_top_divisions = df_historical[mask_top_division].copy()
    # --- END REFACTOR ---

    print(f"Filtered for Makuuchi/Juryo divisions. Records remaining: {len(df_top_divisions)}")

    print("\nPreprocessing filtered data and engineering features...")
    df_processed = preprocess_data(df_top_divisions, match_history_df=match_history_df)

    if df_processed.empty:
        print("Error: No data left after preprocessing. Halting.")
        return

    df_clean = df_processed[FEATURES + [TARGET]].dropna()
    print(f"Data cleaned. Using {len(df_clean)} records for final model.")

    if df_clean.empty:
        print("\nError: No records remaining after dropping rows with missing values.")
        print("This can happen if the initial dataset has many new rikishi or incomplete records.")
        return

    X = df_clean[FEATURES]
    y = df_clean[TARGET]

    best_params = {}
    if tune_hyperparameters:
        print("\nPerforming hyperparameter tuning with RandomizedSearchCV...")
        # FIX: Re-enable parallel processing with n_jobs=-1 and use a less frequent verbose level.
        # Set n_jobs=1 in the estimator to prevent thread contention.
        random_search = RandomizedSearchCV(
            estimator=lgb.LGBMRegressor(random_state=42, n_jobs=1, verbosity=-1),
            param_distributions=PARAM_GRID,
            n_iter=100,
            cv=3,
            scoring='r2',
            verbose=10,
            n_jobs=-1,
            random_state=42
        )
        random_search.fit(X, y)

        best_params = DEFAULT_BEST_PARAMS.copy()
        best_params.update(random_search.best_params_)

        print("\nBest parameters found by RandomizedSearchCV:")
        print(best_params)
    else:
        print("\nSkipping hyperparameter tuning. Using default best parameters.")
        best_params = DEFAULT_BEST_PARAMS

    print("\nPerforming a validation check on a test split...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    validation_model = lgb.LGBMRegressor(**best_params)
    validation_model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        eval_names=['train', 'val'],
        callbacks=[lgb.log_evaluation(period=100)],
    )
    predictions = validation_model.predict(X_test)

    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    print("--- Validation Model Performance ---")
    print(f"  R-squared (R²): {r2:.4f}")
    print(f"  Mean Absolute Error (MAE): {mae:.4f} wins")
    print("-" * 35)

    print(f"\nTraining final model on all {len(X)} records with parameters: {best_params}")
    final_model = lgb.LGBMRegressor(**best_params)
    final_model.fit(
        X, y,
        callbacks=[lgb.log_evaluation(period=100)],
    )

    model_and_features = {
        'model': final_model,
        'features': FEATURES,
        'mae': mae  # Save MAE for use in prediction/fantasy scoring
    }
    print(f"Saving trained model and feature list to '{model_output_path}'...")
    joblib.dump(model_and_features, model_output_path)
    print("Model saved successfully.")

    print("\n--- Feature Importance (from Final Model) ---")
    importances = final_model.feature_importances_
    total_importance = sum(importances)
    if total_importance > 0:
        normalized_importances = [i / total_importance for i in importances]
    else:
        normalized_importances = [0.0] * len(importances)

    feature_importance_df = pd.DataFrame({'Feature': FEATURES, 'Importance': normalized_importances}).sort_values(
        by='Importance',
        ascending=False
    )
    print(feature_importance_df.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and save a sumo win prediction model.")
    parser.add_argument(
        '--data-file',
        type=str,
        default="data/banzuke_detailed.csv",
        help="Path to the historical data CSV file."
    )
    parser.add_argument(
        '--model-file',
        type=str,
        default="sumo_win_predictor_model.joblib",
        help="Path to save the output model file."
    )
    parser.add_argument(
        '--no-tune',
        action='store_true',
        help="Skip hyperparameter tuning and use default parameters for faster retraining."
    )
    args = parser.parse_args()

    train_and_save_model(
        file_path=args.data_file,
        model_output_path=args.model_file,
        tune_hyperparameters=(not args.no_tune)
    )

# train_model.py
import pandas as pd
import joblib
import argparse
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
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
    'division_strength'
]
TARGET = 'w'

DEFAULT_BEST_PARAMS = {
    'max_depth': 10,
    'min_samples_leaf': 2,
    'min_samples_split': 2,
    'n_estimators': 200,
    'random_state': 42,
    'n_jobs': -1
}

PARAM_GRID = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
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

    print(f"Original records loaded: {len(df_historical)}")

    # --- REFACTORED: Use centralized parsing logic for filtering ---
    # This ensures consistency with the rest of the pipeline.
    mask_top_division = df_historical['rank'].apply(is_top_division)
    df_top_divisions = df_historical[mask_top_division].copy()
    # --- END REFACTOR ---

    print(f"Filtered for Makuuchi/Juryo divisions. Records remaining: {len(df_top_divisions)}")

    print("\nPreprocessing filtered data and engineering features...")
    df_processed = preprocess_data(df_top_divisions)

    if df_processed.empty:
        print("Error: No data left after preprocessing. Halting.")
        return

    df_clean = df_processed[FEATURES + [TARGET]].dropna()
    print(f"Data cleaned. Using {len(df_clean)} records for final model.")

    if df_clean.empty:
        print("\nError: No records remaining after dropping rows with missing values.")
        return

    X = df_clean[FEATURES]
    y = df_clean[TARGET]

    best_params = {}
    if tune_hyperparameters:
        print("\nPerforming hyperparameter tuning with GridSearchCV... (This may take a few minutes)")
        grid_search = GridSearchCV(
            estimator=RandomForestRegressor(random_state=42, n_jobs=-1),
            param_grid=PARAM_GRID,
            cv=3,
            scoring='r2',
            verbose=1,
            n_jobs=-1
        )
        grid_search.fit(X, y)

        best_params = DEFAULT_BEST_PARAMS.copy()
        best_params.update(grid_search.best_params_)

        print("\nBest parameters found by GridSearchCV:")
        print(best_params)
    else:
        print("\nSkipping hyperparameter tuning. Using default best parameters.")
        best_params = DEFAULT_BEST_PARAMS

    print("\nPerforming a validation check on a test split...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    validation_model = RandomForestRegressor(**best_params)
    validation_model.fit(X_train, y_train)
    predictions = validation_model.predict(X_test)

    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    print("--- Validation Model Performance ---")
    print(f"  R-squared (R²): {r2:.4f}")
    print(f"  Mean Absolute Error (MAE): {mae:.4f} wins")
    print("-" * 35)

    print(f"\nTraining final model on all {len(X)} records with parameters: {best_params}")
    final_model = RandomForestRegressor(**best_params)
    final_model.fit(X, y)

    model_and_features = {
        'model': final_model,
        'features': FEATURES
    }
    print(f"Saving trained model and feature list to '{model_output_path}'...")
    joblib.dump(model_and_features, model_output_path)
    print("Model saved successfully.")

    print("\n--- Feature Importance (from Final Model) ---")
    importances = final_model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': FEATURES, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    print(feature_importance_df.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and save a sumo win prediction model.")
    parser.add_argument('--data-file', type=str, default="banzuke_detailed.csv", help="Path to the historical data CSV file.")
    parser.add_argument('--model-file', type=str, default="sumo_win_predictor_model.joblib", help="Path to save the output model file.")
    parser.add_argument('--no-tune', action='store_true', help="Skip hyperparameter tuning and use default parameters.")
    args = parser.parse_args()

    train_and_save_model(
        file_path=args.data_file,
        model_output_path=args.model_file,
        tune_hyperparameters=(not args.no_tune)
    )

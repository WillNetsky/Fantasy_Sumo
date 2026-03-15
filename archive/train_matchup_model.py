# train_matchup_model.py
"""
Matchup-based prediction model for sumo wrestling.
Predicts P(wrestler A beats wrestler B) for individual bouts.
"""

import pandas as pd
import numpy as np
import joblib
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
import lightgbm as lgb
from sumo_utils import get_absolute_rank_score, _parse_rank

# Features for matchup prediction (differences and interactions)
MATCHUP_FEATURES = [
    # Rank-based
    'rank_diff',              # A's rank score - B's rank score
    'rank_in_div_diff',       # Position within division difference

    # Physical attributes
    'bmi_diff',
    'height_diff',
    'weight_diff',
    'age_diff',

    # Style matchup
    'oshi_ratio_A',
    'oshi_ratio_B',
    'oshi_ratio_diff',        # Style contrast

    # Head-to-head history
    'h2h_win_pct_A',          # A's historical win % vs B
    'h2h_total_matches',      # Number of previous meetings

    # Recent form
    'win_pct_last_3_A',
    'win_pct_last_3_B',
    'win_pct_last_3_diff',

    # Current tournament (for in-tournament prediction)
    'current_wins_A',
    'current_wins_B',
    'current_wins_diff',
]

LGBM_PARAMS = {
    'n_estimators': 500,
    'learning_rate': 0.02,
    'num_leaves': 31,
    'max_depth': 6,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'colsample_bytree': 0.8,
    'subsample': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': -1
}


def load_and_prepare_data(banzuke_path: str, match_history_path: str):
    """Load banzuke and match history data."""
    print("Loading data...")
    banzuke = pd.read_csv(banzuke_path)
    matches = pd.read_csv(match_history_path)

    # Convert IDs to numeric
    banzuke['wrestler_id'] = pd.to_numeric(banzuke['wrestler_id'], errors='coerce')
    banzuke['tournament_id'] = pd.to_numeric(banzuke['tournament_id'], errors='coerce')
    matches['winner_id'] = pd.to_numeric(matches['winner_id'], errors='coerce')
    matches['loser_id'] = pd.to_numeric(matches['loser_id'], errors='coerce')
    matches['tournament_id'] = pd.to_numeric(matches['tournament_id'], errors='coerce')
    matches['day'] = pd.to_numeric(matches['day'], errors='coerce')

    return banzuke, matches


def compute_wrestler_features(banzuke: pd.DataFrame) -> pd.DataFrame:
    """Compute per-wrestler features for each tournament."""
    print("Computing wrestler features...")

    df = banzuke.copy()
    df = df.sort_values(['wrestler_id', 'tournament_id']).reset_index(drop=True)

    # Rank score
    df['rank_score'] = df['rank'].apply(get_absolute_rank_score)

    # Rank in division
    def get_rank_in_div(rank_str):
        div, num, side = _parse_rank(rank_str)
        if not div:
            return np.nan
        sanyaku_map = {'Y': 22, 'O': 20, 'S': 19, 'K': 18}
        if div in sanyaku_map:
            score = sanyaku_map[div]
        elif div == 'M':
            score = 18 - num
        elif div == 'J':
            score = 15 - num
        else:
            score = 0
        if side == 'e':
            score += 0.5
        return score

    df['rank_in_div'] = df['rank'].apply(get_rank_in_div)

    # Physical attributes
    hw_data = df['height_weight'].str.extract(r'(\d+\.?\d*)\s*cm\s*(\d+\.?\d*)\s*kg', expand=True)
    df['height'] = pd.to_numeric(hw_data[0], errors='coerce')
    df['weight'] = pd.to_numeric(hw_data[1], errors='coerce')
    df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)

    # Age
    df['birth_date'] = pd.to_datetime(df['birth_date'], format='%d.%m.%Y', errors='coerce')
    df['tournament_date'] = pd.to_datetime(df['tournament_id'], format='%Y%m')
    df['age'] = (df['tournament_date'] - df['birth_date']).dt.days / 365.25

    # Wins/losses
    df['w'] = pd.to_numeric(df['w'], errors='coerce')
    df['l'] = pd.to_numeric(df['l'], errors='coerce')

    # Win percentage in last 3 basho
    df['total_bouts'] = df['w'] + df['l']
    df['win_pct'] = df['w'] / df['total_bouts'].replace(0, np.nan)
    df['win_pct_last_3'] = df.groupby('wrestler_id')['win_pct'].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
    ).fillna(0.5)

    # Fill missing physical attributes with division medians
    for col in ['height', 'weight', 'bmi', 'age']:
        df[col] = df.groupby('tournament_id')[col].transform(
            lambda x: x.fillna(x.median())
        )

    return df


def compute_oshi_ratio(matches: pd.DataFrame) -> pd.DataFrame:
    """Compute cumulative oshi ratio for each wrestler over time."""
    print("Computing fighting style (oshi ratio)...")

    oshi_techniques = ['oshidashi', 'tsukidashi', 'tsukiotoshi', 'hatakikomi', 'oshitaoshi', 'tsukitaoshi']
    yotsu_techniques = ['yorikiri', 'uwatenage', 'shitatenage', 'yoritaoshi', 'kotenage', 'sukuinage', 'utchari']

    matches = matches.copy()
    matches['is_oshi'] = matches['kimarite'].isin(oshi_techniques).astype(int)
    matches['is_yotsu'] = matches['kimarite'].isin(yotsu_techniques).astype(int)

    # Sort by time
    matches = matches.sort_values(['winner_id', 'tournament_id', 'day'])

    # Cumulative counts per winner
    matches['cum_oshi'] = matches.groupby('winner_id')['is_oshi'].cumsum()
    matches['cum_yotsu'] = matches.groupby('winner_id')['is_yotsu'].cumsum()
    matches['cum_style_wins'] = matches['cum_oshi'] + matches['cum_yotsu']

    # Get last cumulative values per tournament
    style_by_tourney = matches.groupby(['winner_id', 'tournament_id']).agg({
        'cum_oshi': 'last',
        'cum_yotsu': 'last',
        'cum_style_wins': 'last'
    }).reset_index()

    style_by_tourney.rename(columns={'winner_id': 'wrestler_id'}, inplace=True)

    # Shift to get "before this tournament" values
    style_by_tourney = style_by_tourney.sort_values(['wrestler_id', 'tournament_id'])
    style_by_tourney['prev_cum_oshi'] = style_by_tourney.groupby('wrestler_id')['cum_oshi'].shift(1).fillna(0)
    style_by_tourney['prev_cum_style'] = style_by_tourney.groupby('wrestler_id')['cum_style_wins'].shift(1).fillna(0)

    style_by_tourney['oshi_ratio'] = np.where(
        style_by_tourney['prev_cum_style'] > 0,
        style_by_tourney['prev_cum_oshi'] / style_by_tourney['prev_cum_style'],
        0.5
    )

    return style_by_tourney[['wrestler_id', 'tournament_id', 'oshi_ratio']]


def compute_h2h_history(matches: pd.DataFrame) -> pd.DataFrame:
    """Compute head-to-head history for all wrestler pairs."""
    print("Computing head-to-head history...")

    matches = matches.copy()

    # Create canonical pair ordering (lower ID first)
    matches['w1'] = np.minimum(matches['winner_id'], matches['loser_id'])
    matches['w2'] = np.maximum(matches['winner_id'], matches['loser_id'])
    matches['w1_won'] = (matches['w1'] == matches['winner_id']).astype(int)

    # Sort by time
    matches = matches.sort_values(['w1', 'w2', 'tournament_id', 'day'])

    # Cumulative wins for w1 in each pair
    matches['w1_cum_wins'] = matches.groupby(['w1', 'w2'])['w1_won'].cumsum()
    matches['match_num'] = matches.groupby(['w1', 'w2']).cumcount() + 1
    matches['w2_cum_wins'] = matches['match_num'] - matches['w1_cum_wins']

    # Shift to get "before this match" values
    matches['w1_prev_wins'] = matches.groupby(['w1', 'w2'])['w1_cum_wins'].shift(1).fillna(0)
    matches['w2_prev_wins'] = matches.groupby(['w1', 'w2'])['w2_cum_wins'].shift(1).fillna(0)
    matches['prev_total'] = matches['w1_prev_wins'] + matches['w2_prev_wins']

    return matches[['tournament_id', 'day', 'winner_id', 'loser_id', 'w1', 'w2',
                    'w1_prev_wins', 'w2_prev_wins', 'prev_total']]


def build_matchup_dataset(banzuke: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Build training dataset with matchup features."""
    print("Building matchup dataset...")

    # Get wrestler features
    wrestler_features = compute_wrestler_features(banzuke)

    # Get oshi ratios
    oshi_ratios = compute_oshi_ratio(matches)

    # Merge oshi ratio into wrestler features
    wrestler_features = wrestler_features.merge(
        oshi_ratios, on=['wrestler_id', 'tournament_id'], how='left'
    )
    wrestler_features['oshi_ratio'] = wrestler_features['oshi_ratio'].fillna(0.5)

    # Get H2H history
    h2h_data = compute_h2h_history(matches)

    # Build matchup records
    print("Merging features for each match...")

    # For each match, we need features for both wrestlers
    # Winner is "A", Loser is "B" - then we'll also create B vs A with flipped label

    matchups = h2h_data.copy()

    # Merge wrestler A (winner) features
    matchups = matchups.merge(
        wrestler_features[['wrestler_id', 'tournament_id', 'rank_score', 'rank_in_div',
                          'bmi', 'height', 'weight', 'age', 'win_pct_last_3', 'oshi_ratio']],
        left_on=['winner_id', 'tournament_id'],
        right_on=['wrestler_id', 'tournament_id'],
        how='left',
        suffixes=('', '_A')
    ).drop(columns=['wrestler_id'])

    # Rename A columns
    for col in ['rank_score', 'rank_in_div', 'bmi', 'height', 'weight', 'age', 'win_pct_last_3', 'oshi_ratio']:
        matchups.rename(columns={col: f'{col}_A'}, inplace=True)

    # Merge wrestler B (loser) features
    matchups = matchups.merge(
        wrestler_features[['wrestler_id', 'tournament_id', 'rank_score', 'rank_in_div',
                          'bmi', 'height', 'weight', 'age', 'win_pct_last_3', 'oshi_ratio']],
        left_on=['loser_id', 'tournament_id'],
        right_on=['wrestler_id', 'tournament_id'],
        how='left',
        suffixes=('', '_B')
    ).drop(columns=['wrestler_id'])

    # Rename B columns
    for col in ['rank_score', 'rank_in_div', 'bmi', 'height', 'weight', 'age', 'win_pct_last_3', 'oshi_ratio']:
        if col in matchups.columns:
            matchups.rename(columns={col: f'{col}_B'}, inplace=True)

    # Compute H2H win percentage for A (winner)
    # Need to figure out if winner is w1 or w2
    matchups['winner_is_w1'] = (matchups['winner_id'] == matchups['w1']).astype(int)
    matchups['h2h_wins_A'] = np.where(
        matchups['winner_is_w1'] == 1,
        matchups['w1_prev_wins'],
        matchups['w2_prev_wins']
    )
    matchups['h2h_wins_B'] = np.where(
        matchups['winner_is_w1'] == 1,
        matchups['w2_prev_wins'],
        matchups['w1_prev_wins']
    )
    matchups['h2h_total_matches'] = matchups['prev_total']
    matchups['h2h_win_pct_A'] = np.where(
        matchups['h2h_total_matches'] > 0,
        matchups['h2h_wins_A'] / matchups['h2h_total_matches'],
        0.5  # No prior history
    )

    # Compute difference features
    matchups['rank_diff'] = matchups['rank_score_A'] - matchups['rank_score_B']
    matchups['rank_in_div_diff'] = matchups['rank_in_div_A'] - matchups['rank_in_div_B']
    matchups['bmi_diff'] = matchups['bmi_A'] - matchups['bmi_B']
    matchups['height_diff'] = matchups['height_A'] - matchups['height_B']
    matchups['weight_diff'] = matchups['weight_A'] - matchups['weight_B']
    matchups['age_diff'] = matchups['age_A'] - matchups['age_B']
    matchups['oshi_ratio_diff'] = matchups['oshi_ratio_A'] - matchups['oshi_ratio_B']
    matchups['win_pct_last_3_diff'] = matchups['win_pct_last_3_A'] - matchups['win_pct_last_3_B']

    # Current tournament wins (set to 0 for training - would be filled during simulation)
    matchups['current_wins_A'] = 0
    matchups['current_wins_B'] = 0
    matchups['current_wins_diff'] = 0

    # Target: A wins (always 1 in this setup since A is the winner)
    matchups['A_wins'] = 1

    # Create symmetric examples (flip A and B, label becomes 0)
    matchups_flipped = matchups.copy()

    # Swap A and B columns
    for col in ['rank_score', 'rank_in_div', 'bmi', 'height', 'weight', 'age', 'win_pct_last_3', 'oshi_ratio']:
        matchups_flipped[f'{col}_A'], matchups_flipped[f'{col}_B'] = \
            matchups[f'{col}_B'].copy(), matchups[f'{col}_A'].copy()

    matchups_flipped['h2h_win_pct_A'] = 1 - matchups['h2h_win_pct_A']
    matchups_flipped['rank_diff'] = -matchups['rank_diff']
    matchups_flipped['rank_in_div_diff'] = -matchups['rank_in_div_diff']
    matchups_flipped['bmi_diff'] = -matchups['bmi_diff']
    matchups_flipped['height_diff'] = -matchups['height_diff']
    matchups_flipped['weight_diff'] = -matchups['weight_diff']
    matchups_flipped['age_diff'] = -matchups['age_diff']
    matchups_flipped['oshi_ratio_diff'] = -matchups['oshi_ratio_diff']
    matchups_flipped['win_pct_last_3_diff'] = -matchups['win_pct_last_3_diff']
    matchups_flipped['A_wins'] = 0

    # Combine
    all_matchups = pd.concat([matchups, matchups_flipped], ignore_index=True)

    print(f"Total matchup examples: {len(all_matchups):,}")

    return all_matchups


def train_matchup_model(matchups: pd.DataFrame, model_output_path: str):
    """Train the matchup prediction model."""

    features = [
        'rank_diff', 'rank_in_div_diff',
        'bmi_diff', 'height_diff', 'weight_diff', 'age_diff',
        'oshi_ratio_A', 'oshi_ratio_B', 'oshi_ratio_diff',
        'h2h_win_pct_A', 'h2h_total_matches',
        'win_pct_last_3_A', 'win_pct_last_3_B', 'win_pct_last_3_diff',
    ]

    # Clean data
    df = matchups.dropna(subset=features + ['A_wins'])
    print(f"Training samples after dropping NaN: {len(df):,}")

    X = df[features]
    y = df['A_wins']

    # Split by tournament to avoid leakage
    tournaments = df['tournament_id'].unique()
    train_tournaments = tournaments[:-10]  # Hold out last 10 tournaments
    test_tournaments = tournaments[-10:]

    train_mask = df['tournament_id'].isin(train_tournaments)
    test_mask = df['tournament_id'].isin(test_tournaments)

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    print(f"\nTraining set: {len(X_train):,} examples")
    print(f"Test set: {len(X_test):,} examples (last 10 tournaments)")

    # Train model
    print("\nTraining LightGBM classifier...")
    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    model.fit(X_train, y_train)

    # Evaluate
    train_pred_proba = model.predict_proba(X_train)[:, 1]
    test_pred_proba = model.predict_proba(X_test)[:, 1]

    train_pred = (train_pred_proba > 0.5).astype(int)
    test_pred = (test_pred_proba > 0.5).astype(int)

    print("\n--- Model Performance ---")
    print(f"Training Accuracy: {accuracy_score(y_train, train_pred):.4f}")
    print(f"Training AUC: {roc_auc_score(y_train, train_pred_proba):.4f}")
    print(f"Training Log Loss: {log_loss(y_train, train_pred_proba):.4f}")
    print()
    print(f"Test Accuracy: {accuracy_score(y_test, test_pred):.4f}")
    print(f"Test AUC: {roc_auc_score(y_test, test_pred_proba):.4f}")
    print(f"Test Log Loss: {log_loss(y_test, test_pred_proba):.4f}")
    print("-" * 30)

    # Baseline comparison
    print(f"\nBaseline (always predict higher-ranked): ~52.6%")
    print(f"Model improvement: +{(accuracy_score(y_test, test_pred) - 0.526) * 100:.1f}%")

    # Feature importance
    print("\n--- Feature Importance ---")
    importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    print(importance.to_string(index=False))

    # Save model
    model_bundle = {
        'model': model,
        'features': features,
        'accuracy': accuracy_score(y_test, test_pred),
        'auc': roc_auc_score(y_test, test_pred_proba)
    }

    print(f"\nSaving model to '{model_output_path}'...")
    joblib.dump(model_bundle, model_output_path)
    print("Done!")

    return model_bundle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train matchup prediction model.")
    parser.add_argument('--banzuke-file', type=str, default="banzuke_detailed.csv")
    parser.add_argument('--match-file', type=str, default="match_history_with_kimarite.csv")
    parser.add_argument('--model-file', type=str, default="matchup_model.joblib")
    args = parser.parse_args()

    banzuke, matches = load_and_prepare_data(args.banzuke_file, args.match_file)
    matchups = build_matchup_dataset(banzuke, matches)
    train_matchup_model(matchups, args.model_file)

# train_matchup_model_v3.py
"""
Improved matchup prediction model v3 targeting 65%+ accuracy.
Key improvements:
1. ELO rating system for wrestlers
2. Momentum/form features
3. Surface match difficulty adjustment
4. Better feature engineering for matchup-specific predictions
5. Calibrated probability outputs
"""

import pandas as pd
import numpy as np
import joblib
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb
from sumo_utils import get_absolute_rank_score, _parse_rank

# === TECHNIQUE CATEGORIES ===
TECHNIQUE_CATEGORIES = {
    'oshi': ['oshidashi', 'tsukidashi', 'oshitaoshi', 'tsukitaoshi'],
    'yotsu': ['yorikiri', 'uwatenage', 'shitatenage', 'yoritaoshi', 'kotenage',
              'sukuinage', 'utchari', 'uwatedashinage', 'shitatedashinage',
              'shitatehineri', 'uwatehineri', 'tsuridashi', 'tsuriotoshi',
              'kimedashi', 'kimetaoshi', 'abisetaoshi'],
    'pull': ['hatakikomi', 'hikiotoshi', 'tsukiotoshi', 'hikkake', 'katasukashi',
             'sokubiotoshi', 'kubinage'],
    'leg': ['ketaguri', 'kekaeshi', 'kawazugake', 'sototogake', 'uchigake',
            'sotogake', 'ashitori', 'chongake', 'kirikaeshi', 'komatasukui',
            'watashikomi', 'nimaigeri', 'kozumatori', 'susoharai', 'susotori',
            'kakenage', 'tsumatori'],
    'okuri': ['okuridashi', 'okuritaoshi', 'okurinage', 'okurigake', 'okuritsuriotoshi'],
}

TECHNIQUE_TO_CATEGORY = {}
for cat, techs in TECHNIQUE_CATEGORIES.items():
    for tech in techs:
        TECHNIQUE_TO_CATEGORY[tech] = cat

# Hyperparameters - tuned for higher accuracy
LGBM_PARAMS = {
    'n_estimators': 1200,
    'learning_rate': 0.012,
    'num_leaves': 50,
    'max_depth': 8,
    'reg_alpha': 0.02,
    'reg_lambda': 0.05,
    'colsample_bytree': 0.85,
    'subsample': 0.85,
    'min_child_samples': 40,
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': -1
}


def compute_elo_ratings(matches: pd.DataFrame, k_factor: float = 32, initial_elo: float = 1500) -> dict:
    """
    Compute ELO ratings for all wrestlers based on match history.
    Returns a dict: {(wrestler_id, tournament_id): elo_rating}
    """
    print("Computing ELO ratings...")
    matches = matches.sort_values(['tournament_id', 'day']).reset_index(drop=True)

    # Current ELO for each wrestler
    current_elo = {}
    # Historical ELO at each tournament
    elo_history = {}

    current_tournament = None

    for idx, row in matches.iterrows():
        tournament = row['tournament_id']
        winner = row['winner_id']
        loser = row['loser_id']

        # At start of new tournament, record ELO for all wrestlers
        if tournament != current_tournament:
            current_tournament = tournament
            for w_id, elo in current_elo.items():
                elo_history[(w_id, tournament)] = elo

        # Get current ELO for both wrestlers
        elo_winner = current_elo.get(winner, initial_elo)
        elo_loser = current_elo.get(loser, initial_elo)

        # Store pre-match ELO for this tournament
        if (winner, tournament) not in elo_history:
            elo_history[(winner, tournament)] = elo_winner
        if (loser, tournament) not in elo_history:
            elo_history[(loser, tournament)] = elo_loser

        # Calculate expected scores
        exp_winner = 1 / (1 + 10 ** ((elo_loser - elo_winner) / 400))
        exp_loser = 1 - exp_winner

        # Update ELO
        current_elo[winner] = elo_winner + k_factor * (1 - exp_winner)
        current_elo[loser] = elo_loser + k_factor * (0 - exp_loser)

    return elo_history


def compute_momentum(matches: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Compute recent momentum for each wrestler entering each tournament.
    Returns DataFrame with (wrestler_id, tournament_id, momentum_score, win_streak)
    """
    print("Computing momentum features...")
    matches = matches.sort_values(['tournament_id', 'day'])

    # Create match records for each wrestler
    results = []
    for w_id in set(matches['winner_id'].tolist() + matches['loser_id'].tolist()):
        w_matches = matches[(matches['winner_id'] == w_id) | (matches['loser_id'] == w_id)].copy()
        w_matches = w_matches.sort_values(['tournament_id', 'day'])
        w_matches['won'] = (w_matches['winner_id'] == w_id).astype(int)

        # Group by tournament and compute stats
        for tournament in w_matches['tournament_id'].unique():
            t_matches = w_matches[w_matches['tournament_id'] < tournament].tail(window)

            if len(t_matches) == 0:
                momentum = 0.5
                win_streak = 0
                recent_win_pct = 0.5
            else:
                wins = t_matches['won'].sum()
                total = len(t_matches)
                momentum = wins / total
                recent_win_pct = wins / total

                # Calculate win streak (consecutive wins at end of period)
                streak = 0
                for val in reversed(t_matches['won'].values):
                    if val == 1:
                        streak += 1
                    else:
                        break
                win_streak = streak

            results.append({
                'wrestler_id': w_id,
                'tournament_id': tournament,
                'momentum': momentum,
                'win_streak': win_streak,
                'recent_win_pct': recent_win_pct
            })

    return pd.DataFrame(results)


def load_data(banzuke_path: str, match_history_path: str):
    """Load and prepare data."""
    print("Loading data...")
    banzuke = pd.read_csv(banzuke_path)
    matches = pd.read_csv(match_history_path)

    for df in [banzuke, matches]:
        for col in ['wrestler_id', 'winner_id', 'loser_id', 'tournament_id', 'day']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

    return banzuke, matches


def compute_wrestler_profiles(banzuke: pd.DataFrame, matches: pd.DataFrame,
                               elo_history: dict, momentum_df: pd.DataFrame) -> pd.DataFrame:
    """Compute comprehensive wrestler profiles including ELO and momentum."""
    print("Computing wrestler profiles...")

    df = banzuke.copy()
    df = df.sort_values(['wrestler_id', 'tournament_id']).reset_index(drop=True)

    # Basic features
    df['rank_score'] = df['rank'].apply(get_absolute_rank_score)

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

    # Division
    def get_division(rank_str):
        div, _, _ = _parse_rank(rank_str)
        if div in ['Y', 'O', 'S', 'K', 'M']:
            return 'makuuchi'
        elif div == 'J':
            return 'juryo'
        return 'other'

    df['division'] = df['rank'].apply(get_division)

    # Physical attributes
    hw = df['height_weight'].str.extract(r'(\d+\.?\d*)\s*cm\s*(\d+\.?\d*)\s*kg', expand=True)
    df['height'] = pd.to_numeric(hw[0], errors='coerce')
    df['weight'] = pd.to_numeric(hw[1], errors='coerce')
    df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)

    # Age
    df['birth_date'] = pd.to_datetime(df['birth_date'], format='%d.%m.%Y', errors='coerce')
    df['tournament_date'] = pd.to_datetime(df['tournament_id'], format='%Y%m')
    df['age'] = (df['tournament_date'] - df['birth_date']).dt.days / 365.25

    # Win percentage
    df['w'] = pd.to_numeric(df['w'], errors='coerce')
    df['l'] = pd.to_numeric(df['l'], errors='coerce')
    df['total_bouts'] = df['w'] + df['l']
    df['win_pct'] = df['w'] / df['total_bouts'].replace(0, np.nan)

    # Rolling win percentages (lagged)
    df['win_pct_last_3'] = df.groupby('wrestler_id')['win_pct'].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
    ).fillna(0.5)

    df['win_pct_last_6'] = df.groupby('wrestler_id')['win_pct'].transform(
        lambda x: x.shift(1).rolling(window=6, min_periods=1).mean()
    ).fillna(0.5)

    # ELO rating
    df['elo'] = df.apply(
        lambda row: elo_history.get((row['wrestler_id'], row['tournament_id']), 1500),
        axis=1
    )

    # Momentum features
    df = df.merge(momentum_df, on=['wrestler_id', 'tournament_id'], how='left')
    df['momentum'] = df['momentum'].fillna(0.5)
    df['win_streak'] = df['win_streak'].fillna(0)
    df['recent_win_pct'] = df['recent_win_pct'].fillna(0.5)

    # Fill missing values
    for col in ['height', 'weight', 'bmi', 'age']:
        df[col] = df.groupby('tournament_id')[col].transform(lambda x: x.fillna(x.median()))

    return df


def compute_style_profiles(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Compute detailed style profiles for each wrestler at each tournament.
    """
    print("Computing detailed style profiles...")

    matches = matches.copy()
    matches['category'] = matches['kimarite'].map(TECHNIQUE_TO_CATEGORY).fillna('other')

    for cat in TECHNIQUE_CATEGORIES.keys():
        matches[f'is_{cat}'] = (matches['category'] == cat).astype(int)

    matches = matches.sort_values(['winner_id', 'tournament_id', 'day'])

    for cat in TECHNIQUE_CATEGORIES.keys():
        matches[f'cum_{cat}'] = matches.groupby('winner_id')[f'is_{cat}'].cumsum()

    matches['cum_wins'] = matches.groupby('winner_id').cumcount() + 1

    style_cols = [f'cum_{cat}' for cat in TECHNIQUE_CATEGORIES.keys()] + ['cum_wins']
    style_history = matches.groupby(['winner_id', 'tournament_id'])[style_cols].last().reset_index()
    style_history.rename(columns={'winner_id': 'wrestler_id'}, inplace=True)

    style_history = style_history.sort_values(['wrestler_id', 'tournament_id'])
    for col in style_cols:
        style_history[f'prev_{col}'] = style_history.groupby('wrestler_id')[col].shift(1).fillna(0)

    total_style_wins = style_history['prev_cum_wins'].replace(0, 1)
    for cat in TECHNIQUE_CATEGORIES.keys():
        style_history[f'{cat}_ratio'] = style_history[f'prev_cum_{cat}'] / total_style_wins

    ratio_cols = ['wrestler_id', 'tournament_id'] + [f'{cat}_ratio' for cat in TECHNIQUE_CATEGORIES.keys()]
    return style_history[ratio_cols]


def compute_vulnerability_profiles(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Compute how often each wrestler LOSES to each technique category.
    """
    print("Computing vulnerability profiles...")

    matches = matches.copy()
    matches['category'] = matches['kimarite'].map(TECHNIQUE_TO_CATEGORY).fillna('other')

    for cat in TECHNIQUE_CATEGORIES.keys():
        matches[f'lost_to_{cat}'] = (matches['category'] == cat).astype(int)

    matches = matches.sort_values(['loser_id', 'tournament_id', 'day'])

    for cat in TECHNIQUE_CATEGORIES.keys():
        matches[f'cum_lost_{cat}'] = matches.groupby('loser_id')[f'lost_to_{cat}'].cumsum()

    matches['cum_losses'] = matches.groupby('loser_id').cumcount() + 1

    vuln_cols = [f'cum_lost_{cat}' for cat in TECHNIQUE_CATEGORIES.keys()] + ['cum_losses']
    vuln_history = matches.groupby(['loser_id', 'tournament_id'])[vuln_cols].last().reset_index()
    vuln_history.rename(columns={'loser_id': 'wrestler_id'}, inplace=True)

    vuln_history = vuln_history.sort_values(['wrestler_id', 'tournament_id'])
    for col in vuln_cols:
        vuln_history[f'prev_{col}'] = vuln_history.groupby('wrestler_id')[col].shift(1).fillna(0)

    total_losses = vuln_history['prev_cum_losses'].replace(0, 1)
    for cat in TECHNIQUE_CATEGORIES.keys():
        vuln_history[f'vuln_{cat}'] = vuln_history[f'prev_cum_lost_{cat}'] / total_losses

    vuln_cols = ['wrestler_id', 'tournament_id'] + [f'vuln_{cat}' for cat in TECHNIQUE_CATEGORIES.keys()]
    return vuln_history[vuln_cols]


def compute_h2h_history(matches: pd.DataFrame) -> pd.DataFrame:
    """Compute head-to-head history."""
    print("Computing H2H history...")

    matches = matches.copy()
    matches['w1'] = np.minimum(matches['winner_id'], matches['loser_id'])
    matches['w2'] = np.maximum(matches['winner_id'], matches['loser_id'])
    matches['w1_won'] = (matches['w1'] == matches['winner_id']).astype(int)

    matches = matches.sort_values(['w1', 'w2', 'tournament_id', 'day'])
    matches['w1_cum_wins'] = matches.groupby(['w1', 'w2'])['w1_won'].cumsum()
    matches['match_num'] = matches.groupby(['w1', 'w2']).cumcount() + 1
    matches['w2_cum_wins'] = matches['match_num'] - matches['w1_cum_wins']

    matches['w1_prev_wins'] = matches.groupby(['w1', 'w2'])['w1_cum_wins'].shift(1).fillna(0)
    matches['w2_prev_wins'] = matches.groupby(['w1', 'w2'])['w2_cum_wins'].shift(1).fillna(0)
    matches['prev_total'] = matches['w1_prev_wins'] + matches['w2_prev_wins']

    return matches[['tournament_id', 'day', 'winner_id', 'loser_id', 'w1', 'w2',
                    'w1_prev_wins', 'w2_prev_wins', 'prev_total', 'kimarite']]


def build_training_data(banzuke: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Build comprehensive training dataset."""
    print("\nBuilding training dataset...")

    # Compute ELO and momentum
    elo_history = compute_elo_ratings(matches)
    momentum_df = compute_momentum(matches)

    # Get all profile data
    wrestler_profiles = compute_wrestler_profiles(banzuke, matches, elo_history, momentum_df)
    style_profiles = compute_style_profiles(matches)
    vuln_profiles = compute_vulnerability_profiles(matches)
    h2h_data = compute_h2h_history(matches)

    # Merge style and vulnerability into wrestler profiles
    wrestler_profiles = wrestler_profiles.merge(
        style_profiles, on=['wrestler_id', 'tournament_id'], how='left'
    )
    wrestler_profiles = wrestler_profiles.merge(
        vuln_profiles, on=['wrestler_id', 'tournament_id'], how='left'
    )

    # Fill missing style/vuln ratios with neutral values
    for cat in TECHNIQUE_CATEGORIES.keys():
        wrestler_profiles[f'{cat}_ratio'] = wrestler_profiles[f'{cat}_ratio'].fillna(0.2)
        wrestler_profiles[f'vuln_{cat}'] = wrestler_profiles[f'vuln_{cat}'].fillna(0.2)

    # Build matchup features
    print("Merging features for each match...")

    matchups = h2h_data.copy()

    # Features we need from wrestler profiles
    profile_cols = ['wrestler_id', 'tournament_id', 'rank_score', 'rank_in_div',
                    'bmi', 'height', 'weight', 'age', 'elo', 'momentum', 'win_streak',
                    'win_pct_last_3', 'win_pct_last_6', 'recent_win_pct']
    profile_cols += [f'{cat}_ratio' for cat in TECHNIQUE_CATEGORIES.keys()]
    profile_cols += [f'vuln_{cat}' for cat in TECHNIQUE_CATEGORIES.keys()]

    # Merge winner (A) features
    matchups = matchups.merge(
        wrestler_profiles[profile_cols],
        left_on=['winner_id', 'tournament_id'],
        right_on=['wrestler_id', 'tournament_id'],
        how='left'
    ).drop(columns=['wrestler_id'])

    rename_cols = {c: f'{c}_A' for c in profile_cols if c not in ['wrestler_id', 'tournament_id']}
    matchups.rename(columns=rename_cols, inplace=True)

    # Merge loser (B) features
    matchups = matchups.merge(
        wrestler_profiles[profile_cols],
        left_on=['loser_id', 'tournament_id'],
        right_on=['wrestler_id', 'tournament_id'],
        how='left'
    ).drop(columns=['wrestler_id'])

    rename_cols = {c: f'{c}_B' for c in profile_cols if c not in ['wrestler_id', 'tournament_id']}
    matchups.rename(columns=rename_cols, inplace=True)

    # Compute H2H win percentage for A
    matchups['winner_is_w1'] = (matchups['winner_id'] == matchups['w1']).astype(int)
    matchups['h2h_wins_A'] = np.where(
        matchups['winner_is_w1'] == 1,
        matchups['w1_prev_wins'],
        matchups['w2_prev_wins']
    )
    matchups['h2h_total_matches'] = matchups['prev_total']
    matchups['h2h_win_pct_A'] = np.where(
        matchups['h2h_total_matches'] > 0,
        matchups['h2h_wins_A'] / matchups['h2h_total_matches'],
        0.5
    )

    # === DIFFERENCE FEATURES ===
    matchups['rank_diff'] = matchups['rank_score_A'] - matchups['rank_score_B']
    matchups['rank_in_div_diff'] = matchups['rank_in_div_A'] - matchups['rank_in_div_B']
    matchups['bmi_diff'] = matchups['bmi_A'] - matchups['bmi_B']
    matchups['height_diff'] = matchups['height_A'] - matchups['height_B']
    matchups['weight_diff'] = matchups['weight_A'] - matchups['weight_B']
    matchups['age_diff'] = matchups['age_A'] - matchups['age_B']
    matchups['elo_diff'] = matchups['elo_A'] - matchups['elo_B']
    matchups['momentum_diff'] = matchups['momentum_A'] - matchups['momentum_B']
    matchups['win_streak_diff'] = matchups['win_streak_A'] - matchups['win_streak_B']
    matchups['win_pct_last_3_diff'] = matchups['win_pct_last_3_A'] - matchups['win_pct_last_3_B']
    matchups['win_pct_last_6_diff'] = matchups['win_pct_last_6_A'] - matchups['win_pct_last_6_B']

    # Expected win probability from ELO
    matchups['elo_expected_A'] = 1 / (1 + 10 ** ((matchups['elo_B'] - matchups['elo_A']) / 400))

    # === STYLE MATCHUP FEATURES ===
    for cat in TECHNIQUE_CATEGORIES.keys():
        matchups[f'{cat}_advantage_A'] = matchups[f'{cat}_ratio_A'] * matchups[f'vuln_{cat}_B']
        matchups[f'{cat}_advantage_B'] = matchups[f'{cat}_ratio_B'] * matchups[f'vuln_{cat}_A']
        matchups[f'{cat}_advantage_diff'] = matchups[f'{cat}_advantage_A'] - matchups[f'{cat}_advantage_B']

    adv_cols_A = [f'{cat}_advantage_A' for cat in TECHNIQUE_CATEGORIES.keys()]
    adv_cols_B = [f'{cat}_advantage_B' for cat in TECHNIQUE_CATEGORIES.keys()]
    matchups['total_style_advantage_A'] = matchups[adv_cols_A].sum(axis=1)
    matchups['total_style_advantage_B'] = matchups[adv_cols_B].sum(axis=1)
    matchups['style_advantage_diff'] = matchups['total_style_advantage_A'] - matchups['total_style_advantage_B']

    # === TARGET ===
    matchups['A_wins'] = 1

    # === CREATE SYMMETRIC EXAMPLES ===
    print("Creating symmetric training examples...")
    matchups_flipped = matchups.copy()

    a_cols = [c for c in matchups.columns if c.endswith('_A')]
    b_cols = [c for c in matchups.columns if c.endswith('_B')]

    for a_col, b_col in zip(sorted(a_cols), sorted([c.replace('_A', '_B') for c in a_cols])):
        if b_col in matchups.columns:
            matchups_flipped[a_col], matchups_flipped[b_col] = matchups[b_col].copy(), matchups[a_col].copy()

    diff_cols = [c for c in matchups.columns if c.endswith('_diff')]
    for col in diff_cols:
        matchups_flipped[col] = -matchups[col]

    matchups_flipped['h2h_win_pct_A'] = 1 - matchups['h2h_win_pct_A']
    matchups_flipped['elo_expected_A'] = 1 - matchups['elo_expected_A']
    matchups_flipped['A_wins'] = 0

    all_matchups = pd.concat([matchups, matchups_flipped], ignore_index=True)
    print(f"Total training examples: {len(all_matchups):,}")

    return all_matchups


def train_model(matchups: pd.DataFrame, model_output_path: str, calibrate: bool = True):
    """Train the improved matchup model."""

    # Define features
    features = [
        # Basic differences
        'rank_diff', 'rank_in_div_diff',
        'bmi_diff', 'height_diff', 'weight_diff', 'age_diff',

        # ELO and expected probability
        'elo_diff', 'elo_expected_A',

        # Momentum features
        'momentum_diff', 'win_streak_diff',

        # H2H
        'h2h_win_pct_A', 'h2h_total_matches',

        # Form
        'win_pct_last_3_A', 'win_pct_last_3_B', 'win_pct_last_3_diff',
        'win_pct_last_6_A', 'win_pct_last_6_B', 'win_pct_last_6_diff',

        # Momentum raw values
        'momentum_A', 'momentum_B',
        'win_streak_A', 'win_streak_B',
    ]

    # Style ratios for both wrestlers
    for cat in TECHNIQUE_CATEGORIES.keys():
        features.extend([f'{cat}_ratio_A', f'{cat}_ratio_B'])

    # Vulnerability ratios
    for cat in TECHNIQUE_CATEGORIES.keys():
        features.extend([f'vuln_{cat}_A', f'vuln_{cat}_B'])

    # Style advantage features
    for cat in TECHNIQUE_CATEGORIES.keys():
        features.append(f'{cat}_advantage_diff')

    features.append('style_advantage_diff')

    print(f"\nTotal features: {len(features)}")

    # Clean data
    df = matchups.dropna(subset=features + ['A_wins'])
    print(f"Training samples after dropping NaN: {len(df):,}")

    X = df[features]
    y = df['A_wins']

    # Split by tournament (temporal split)
    tournaments = sorted(df['tournament_id'].unique())
    train_tournaments = tournaments[:-10]
    test_tournaments = tournaments[-10:]

    train_mask = df['tournament_id'].isin(train_tournaments)
    test_mask = df['tournament_id'].isin(test_tournaments)

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    print(f"\nTraining: {len(X_train):,} | Test: {len(X_test):,}")

    # Train
    print("\nTraining LightGBM...")
    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    model.fit(X_train, y_train)

    # Optionally calibrate
    if calibrate:
        print("Calibrating probability outputs...")
        calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
        calibrated_model.fit(X_train, y_train)
        final_model = calibrated_model
    else:
        final_model = model

    # Evaluate
    train_pred_proba = final_model.predict_proba(X_train)[:, 1]
    test_pred_proba = final_model.predict_proba(X_test)[:, 1]
    train_pred = (train_pred_proba > 0.5).astype(int)
    test_pred = (test_pred_proba > 0.5).astype(int)

    print("\n" + "=" * 50)
    print("MODEL PERFORMANCE")
    print("=" * 50)
    print(f"\nTraining:")
    print(f"  Accuracy: {accuracy_score(y_train, train_pred):.4f}")
    print(f"  AUC:      {roc_auc_score(y_train, train_pred_proba):.4f}")
    print(f"  Brier:    {brier_score_loss(y_train, train_pred_proba):.4f}")

    test_accuracy = accuracy_score(y_test, test_pred)
    test_auc = roc_auc_score(y_test, test_pred_proba)
    test_brier = brier_score_loss(y_test, test_pred_proba)

    print(f"\nTest (last 10 tournaments):")
    print(f"  Accuracy: {test_accuracy:.4f}")
    print(f"  AUC:      {test_auc:.4f}")
    print(f"  Brier:    {test_brier:.4f}")

    print(f"\nBaseline (higher rank wins): ~52.6%")
    print(f"Previous v2 model: ~59.6%")
    print(f"Improvement over v2: +{(test_accuracy - 0.596) * 100:.1f}%")

    # Feature importance (from base model)
    print("\n" + "=" * 50)
    print("TOP 20 FEATURE IMPORTANCE")
    print("=" * 50)
    importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    print(importance.head(20).to_string(index=False))

    # Save
    model_bundle = {
        'model': final_model,
        'base_model': model,
        'features': features,
        'technique_categories': TECHNIQUE_CATEGORIES,
        'accuracy': test_accuracy,
        'auc': test_auc,
        'brier': test_brier,
        'calibrated': calibrate,
        'version': 3
    }

    print(f"\nSaving model to '{model_output_path}'...")
    joblib.dump(model_bundle, model_output_path)
    print("Done!")

    return model_bundle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train improved matchup model v3.")
    parser.add_argument('--banzuke-file', type=str, default="banzuke_detailed.csv")
    parser.add_argument('--match-file', type=str, default="match_history_with_kimarite.csv")
    parser.add_argument('--model-file', type=str, default="matchup_model_v3.joblib")
    parser.add_argument('--no-calibrate', action='store_true', help="Skip probability calibration")
    args = parser.parse_args()

    banzuke, matches = load_data(args.banzuke_file, args.match_file)
    matchups = build_training_data(banzuke, matches)
    train_model(matchups, args.model_file, calibrate=not args.no_calibrate)

# train_matchup_model_v5.py
"""
Matchup prediction model v5 - Push for 65%+ accuracy.
Key changes:
1. Feature engineering focused on highest predictive power
2. More aggressive hyperparameter tuning
3. Glicko-2 style ratings (accounts for uncertainty)
4. Better handling of H2H significance
5. Recent form decay weighting
"""

import pandas as pd
import numpy as np
import joblib
import argparse
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
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


def compute_glicko_ratings(matches: pd.DataFrame, initial_rating: float = 1500,
                           initial_rd: float = 350, c: float = 15) -> dict:
    """
    Compute Glicko-style ratings with rating deviation (uncertainty).
    Returns dict: {(wrestler_id, tournament_id): {'rating': float, 'rd': float}}
    """
    print("Computing Glicko-style ratings...")
    matches = matches.sort_values(['tournament_id', 'day']).reset_index(drop=True)

    # Current ratings for each wrestler
    current_ratings = {}  # wrestler_id -> {'rating': float, 'rd': float, 'last_active': int}
    # Historical ratings
    rating_history = {}

    q = np.log(10) / 400

    def g(rd):
        return 1 / np.sqrt(1 + 3 * q**2 * rd**2 / np.pi**2)

    def expected_score(rating_a, rating_b, rd_b):
        return 1 / (1 + 10**(-g(rd_b) * (rating_a - rating_b) / 400))

    current_tournament = None

    for idx, row in matches.iterrows():
        tournament = row['tournament_id']
        winner = row['winner_id']
        loser = row['loser_id']

        # At start of new tournament, record ratings and update RD for inactivity
        if tournament != current_tournament:
            current_tournament = tournament
            for w_id, data in current_ratings.items():
                rating_history[(w_id, tournament)] = {'rating': data['rating'], 'rd': data['rd']}

        # Get/initialize ratings
        if winner not in current_ratings:
            current_ratings[winner] = {'rating': initial_rating, 'rd': initial_rd, 'last_active': tournament}
        if loser not in current_ratings:
            current_ratings[loser] = {'rating': initial_rating, 'rd': initial_rd, 'last_active': tournament}

        # Store pre-match ratings
        if (winner, tournament) not in rating_history:
            rating_history[(winner, tournament)] = {
                'rating': current_ratings[winner]['rating'],
                'rd': current_ratings[winner]['rd']
            }
        if (loser, tournament) not in rating_history:
            rating_history[(loser, tournament)] = {
                'rating': current_ratings[loser]['rating'],
                'rd': current_ratings[loser]['rd']
            }

        # Get current values
        r_w, rd_w = current_ratings[winner]['rating'], current_ratings[winner]['rd']
        r_l, rd_l = current_ratings[loser]['rating'], current_ratings[loser]['rd']

        # Calculate expected scores
        g_w, g_l = g(rd_l), g(rd_w)
        e_w = expected_score(r_w, r_l, rd_l)
        e_l = expected_score(r_l, r_w, rd_w)

        # Update ratings (simplified Glicko)
        d2_w = 1 / (q**2 * g_w**2 * e_w * (1 - e_w))
        d2_l = 1 / (q**2 * g_l**2 * e_l * (1 - e_l))

        new_rd_w = np.sqrt(1 / (1/rd_w**2 + 1/d2_w))
        new_rd_l = np.sqrt(1 / (1/rd_l**2 + 1/d2_l))

        # Winner won (s=1), loser lost (s=0)
        current_ratings[winner]['rating'] = r_w + q * new_rd_w**2 * g_w * (1 - e_w)
        current_ratings[loser]['rating'] = r_l + q * new_rd_l**2 * g_l * (0 - e_l)

        # Update RDs (clamped)
        current_ratings[winner]['rd'] = max(30, min(350, new_rd_w))
        current_ratings[loser]['rd'] = max(30, min(350, new_rd_l))

        current_ratings[winner]['last_active'] = tournament
        current_ratings[loser]['last_active'] = tournament

    return rating_history


def compute_weighted_form(matches: pd.DataFrame, decay: float = 0.9) -> pd.DataFrame:
    """
    Compute exponentially weighted recent form.
    More recent matches count more than older ones.
    """
    print("Computing weighted form...")

    matches = matches.sort_values(['tournament_id', 'day'])
    results = []

    all_wrestlers = set(matches['winner_id'].tolist() + matches['loser_id'].tolist())

    for w_id in all_wrestlers:
        w_matches = matches[(matches['winner_id'] == w_id) | (matches['loser_id'] == w_id)].copy()
        w_matches = w_matches.sort_values(['tournament_id', 'day'])
        w_matches['won'] = (w_matches['winner_id'] == w_id).astype(int)

        tournaments = sorted(w_matches['tournament_id'].unique())

        for tournament in tournaments:
            # Get all matches before this tournament
            prev_matches = w_matches[w_matches['tournament_id'] < tournament].tail(30)

            if len(prev_matches) == 0:
                weighted_form = 0.5
                recent_streak = 0
                volatility = 0.15
            else:
                # Apply exponential decay weights
                n = len(prev_matches)
                weights = np.array([decay ** (n - 1 - i) for i in range(n)])
                weights = weights / weights.sum()

                weighted_form = np.sum(prev_matches['won'].values * weights)

                # Recent streak
                streak = 0
                for val in reversed(prev_matches['won'].values):
                    if val == 1:
                        streak += 1
                    else:
                        break
                recent_streak = streak

                # Volatility (weighted std)
                mean_form = weighted_form
                volatility = np.sqrt(np.sum(weights * (prev_matches['won'].values - mean_form)**2))

            results.append({
                'wrestler_id': w_id,
                'tournament_id': tournament,
                'weighted_form': weighted_form,
                'recent_streak': recent_streak,
                'form_volatility': volatility
            })

    return pd.DataFrame(results)


def compute_h2h_significance(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Compute H2H history with significance weighting.
    More recent H2H matches and larger sample sizes are more significant.
    """
    print("Computing H2H significance...")

    matches = matches.copy()
    matches['w1'] = np.minimum(matches['winner_id'], matches['loser_id'])
    matches['w2'] = np.maximum(matches['winner_id'], matches['loser_id'])
    matches['w1_won'] = (matches['w1'] == matches['winner_id']).astype(int)

    matches = matches.sort_values(['w1', 'w2', 'tournament_id', 'day'])

    results = []
    for (w1, w2), group in matches.groupby(['w1', 'w2']):
        group = group.sort_values(['tournament_id', 'day'])

        running_w1_wins = 0
        running_total = 0

        for idx, row in group.iterrows():
            # Store pre-match stats
            h2h_win_pct_w1 = running_w1_wins / running_total if running_total > 0 else 0.5

            # Significance based on sample size (logistic curve)
            significance = 1 - 1 / (1 + running_total / 3)  # Approaches 1 as matches increase

            results.append({
                'tournament_id': row['tournament_id'],
                'day': row['day'],
                'w1': w1,
                'w2': w2,
                'winner_id': row['winner_id'],
                'loser_id': row['loser_id'],
                'h2h_win_pct_w1': h2h_win_pct_w1,
                'h2h_total': running_total,
                'h2h_significance': significance,
                'kimarite': row['kimarite']
            })

            # Update stats
            running_w1_wins += row['w1_won']
            running_total += 1

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
                               glicko_history: dict, form_df: pd.DataFrame) -> pd.DataFrame:
    """Compute comprehensive wrestler profiles."""
    print("Computing wrestler profiles...")

    df = banzuke.copy()
    df = df.sort_values(['wrestler_id', 'tournament_id']).reset_index(drop=True)

    # Basic rank features
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

    # Physical attributes
    hw = df['height_weight'].str.extract(r'(\d+\.?\d*)\s*cm\s*(\d+\.?\d*)\s*kg', expand=True)
    df['height'] = pd.to_numeric(hw[0], errors='coerce')
    df['weight'] = pd.to_numeric(hw[1], errors='coerce')
    df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)

    # Age
    df['birth_date'] = pd.to_datetime(df['birth_date'], format='%d.%m.%Y', errors='coerce')
    df['tournament_date'] = pd.to_datetime(df['tournament_id'], format='%Y%m')
    df['age'] = (df['tournament_date'] - df['birth_date']).dt.days / 365.25

    # Glicko rating
    df['glicko_rating'] = df.apply(
        lambda row: glicko_history.get((row['wrestler_id'], row['tournament_id']),
                                        {'rating': 1500, 'rd': 350})['rating'],
        axis=1
    )
    df['glicko_rd'] = df.apply(
        lambda row: glicko_history.get((row['wrestler_id'], row['tournament_id']),
                                        {'rating': 1500, 'rd': 350})['rd'],
        axis=1
    )

    # Weighted form
    df = df.merge(form_df, on=['wrestler_id', 'tournament_id'], how='left')
    df['weighted_form'] = df['weighted_form'].fillna(0.5)
    df['recent_streak'] = df['recent_streak'].fillna(0)
    df['form_volatility'] = df['form_volatility'].fillna(0.15)

    # Win percentage rolling
    df['w'] = pd.to_numeric(df['w'], errors='coerce')
    df['l'] = pd.to_numeric(df['l'], errors='coerce')
    df['total_bouts'] = df['w'] + df['l']
    df['win_pct'] = df['w'] / df['total_bouts'].replace(0, np.nan)

    df['win_pct_last_3'] = df.groupby('wrestler_id')['win_pct'].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
    ).fillna(0.5)

    df['win_pct_last_6'] = df.groupby('wrestler_id')['win_pct'].transform(
        lambda x: x.shift(1).rolling(window=6, min_periods=1).mean()
    ).fillna(0.5)

    # Fill missing values
    for col in ['height', 'weight', 'bmi', 'age']:
        df[col] = df.groupby('tournament_id')[col].transform(lambda x: x.fillna(x.median()))

    return df


def compute_style_profiles(matches: pd.DataFrame) -> pd.DataFrame:
    """Compute style profiles."""
    print("Computing style profiles...")

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
    """Compute vulnerability profiles."""
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

    vuln_cols_out = ['wrestler_id', 'tournament_id'] + [f'vuln_{cat}' for cat in TECHNIQUE_CATEGORIES.keys()]
    return vuln_history[vuln_cols_out]


def build_training_data(banzuke: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Build training dataset."""
    print("\nBuilding training dataset...")

    # Compute helper data
    glicko_history = compute_glicko_ratings(matches)
    form_df = compute_weighted_form(matches)
    h2h_data = compute_h2h_significance(matches)

    # Get profiles
    wrestler_profiles = compute_wrestler_profiles(banzuke, matches, glicko_history, form_df)
    style_profiles = compute_style_profiles(matches)
    vuln_profiles = compute_vulnerability_profiles(matches)

    # Merge profiles
    wrestler_profiles = wrestler_profiles.merge(
        style_profiles, on=['wrestler_id', 'tournament_id'], how='left'
    )
    wrestler_profiles = wrestler_profiles.merge(
        vuln_profiles, on=['wrestler_id', 'tournament_id'], how='left'
    )

    # Fill missing
    for cat in TECHNIQUE_CATEGORIES.keys():
        wrestler_profiles[f'{cat}_ratio'] = wrestler_profiles[f'{cat}_ratio'].fillna(0.2)
        wrestler_profiles[f'vuln_{cat}'] = wrestler_profiles[f'vuln_{cat}'].fillna(0.2)

    # Build matchup features
    print("Merging features for each match...")

    matchups = h2h_data.copy()

    # Features from profiles
    profile_cols = ['wrestler_id', 'tournament_id', 'rank_score', 'rank_in_div',
                    'bmi', 'height', 'weight', 'age',
                    'glicko_rating', 'glicko_rd',
                    'weighted_form', 'recent_streak', 'form_volatility',
                    'win_pct_last_3', 'win_pct_last_6']
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

    # === COMPUTE DERIVED FEATURES ===

    # H2H features
    matchups['winner_is_w1'] = (matchups['winner_id'] == matchups['w1']).astype(int)
    matchups['h2h_win_pct_A'] = np.where(
        matchups['winner_is_w1'] == 1,
        matchups['h2h_win_pct_w1'],
        1 - matchups['h2h_win_pct_w1']
    )
    matchups['h2h_total_matches'] = matchups['h2h_total']

    # Weighted H2H (significance-adjusted)
    matchups['h2h_weighted_A'] = (matchups['h2h_win_pct_A'] - 0.5) * matchups['h2h_significance'] + 0.5

    # === DIFFERENCE FEATURES ===
    matchups['rank_diff'] = matchups['rank_score_A'] - matchups['rank_score_B']
    matchups['rank_in_div_diff'] = matchups['rank_in_div_A'] - matchups['rank_in_div_B']
    matchups['bmi_diff'] = matchups['bmi_A'] - matchups['bmi_B']
    matchups['height_diff'] = matchups['height_A'] - matchups['height_B']
    matchups['weight_diff'] = matchups['weight_A'] - matchups['weight_B']
    matchups['age_diff'] = matchups['age_A'] - matchups['age_B']

    # Glicko features
    matchups['glicko_diff'] = matchups['glicko_rating_A'] - matchups['glicko_rating_B']
    matchups['glicko_combined_rd'] = np.sqrt(matchups['glicko_rd_A']**2 + matchups['glicko_rd_B']**2)

    # Glicko expected score
    q = np.log(10) / 400
    g_val = 1 / np.sqrt(1 + 3 * q**2 * matchups['glicko_combined_rd']**2 / np.pi**2)
    matchups['glicko_expected_A'] = 1 / (1 + 10**(-g_val * matchups['glicko_diff'] / 400))

    # Form features
    matchups['form_diff'] = matchups['weighted_form_A'] - matchups['weighted_form_B']
    matchups['streak_diff'] = matchups['recent_streak_A'] - matchups['recent_streak_B']
    matchups['volatility_diff'] = matchups['form_volatility_A'] - matchups['form_volatility_B']

    matchups['win_pct_last_3_diff'] = matchups['win_pct_last_3_A'] - matchups['win_pct_last_3_B']
    matchups['win_pct_last_6_diff'] = matchups['win_pct_last_6_A'] - matchups['win_pct_last_6_B']

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

    for a_col in sorted(a_cols):
        b_col = a_col.replace('_A', '_B')
        if b_col in matchups.columns:
            matchups_flipped[a_col], matchups_flipped[b_col] = matchups[b_col].copy(), matchups[a_col].copy()

    diff_cols = [c for c in matchups.columns if c.endswith('_diff')]
    for col in diff_cols:
        matchups_flipped[col] = -matchups[col]

    matchups_flipped['h2h_win_pct_A'] = 1 - matchups['h2h_win_pct_A']
    matchups_flipped['h2h_weighted_A'] = 1 - matchups['h2h_weighted_A']
    matchups_flipped['glicko_expected_A'] = 1 - matchups['glicko_expected_A']
    matchups_flipped['A_wins'] = 0

    all_matchups = pd.concat([matchups, matchups_flipped], ignore_index=True)
    print(f"Total training examples: {len(all_matchups):,}")

    return all_matchups


def train_model(matchups: pd.DataFrame, model_output_path: str):
    """Train the model."""

    # Define features - focused on highest predictive power
    features = [
        # Core rating/rank features
        'rank_diff', 'rank_in_div_diff',
        'glicko_diff', 'glicko_expected_A', 'glicko_combined_rd',

        # Physical
        'bmi_diff', 'height_diff', 'weight_diff', 'age_diff',

        # H2H
        'h2h_win_pct_A', 'h2h_weighted_A', 'h2h_total_matches', 'h2h_significance',

        # Form
        'form_diff', 'streak_diff', 'volatility_diff',
        'weighted_form_A', 'weighted_form_B',
        'recent_streak_A', 'recent_streak_B',
        'win_pct_last_3_diff', 'win_pct_last_6_diff',
        'win_pct_last_3_A', 'win_pct_last_3_B',
        'win_pct_last_6_A', 'win_pct_last_6_B',
    ]

    # Style ratios
    for cat in TECHNIQUE_CATEGORIES.keys():
        features.extend([f'{cat}_ratio_A', f'{cat}_ratio_B'])

    # Vulnerability ratios
    for cat in TECHNIQUE_CATEGORIES.keys():
        features.extend([f'vuln_{cat}_A', f'vuln_{cat}_B'])

    # Style advantage
    for cat in TECHNIQUE_CATEGORIES.keys():
        features.append(f'{cat}_advantage_diff')

    features.append('style_advantage_diff')

    print(f"\nTotal features: {len(features)}")

    # Clean data
    df = matchups.dropna(subset=features + ['A_wins'])
    print(f"Training samples after dropping NaN: {len(df):,}")

    X = df[features]
    y = df['A_wins']

    # Split by tournament
    tournaments = sorted(df['tournament_id'].unique())
    train_tournaments = tournaments[:-10]
    test_tournaments = tournaments[-10:]

    train_mask = df['tournament_id'].isin(train_tournaments)
    test_mask = df['tournament_id'].isin(test_tournaments)

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    print(f"\nTraining: {len(X_train):,} | Test: {len(X_test):,}")

    # More aggressive hyperparameters
    print("\nTraining LightGBM with optimized hyperparameters...")

    model = lgb.LGBMClassifier(
        n_estimators=1500,
        learning_rate=0.008,
        num_leaves=60,
        max_depth=10,
        reg_alpha=0.01,
        reg_lambda=0.03,
        colsample_bytree=0.9,
        subsample=0.9,
        min_child_samples=30,
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )

    model.fit(X_train, y_train)

    # Evaluate
    train_pred_proba = model.predict_proba(X_train)[:, 1]
    test_pred_proba = model.predict_proba(X_test)[:, 1]
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
    print(f"Previous v4 model: ~63.0%")

    if test_accuracy >= 0.65:
        print("\n*** TARGET ACHIEVED: 65%+ ACCURACY! ***")
    else:
        print(f"\n*** Gap to 65%: {(0.65 - test_accuracy) * 100:.1f}% ***")

    # Feature importance
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
        'model': model,
        'features': features,
        'technique_categories': TECHNIQUE_CATEGORIES,
        'accuracy': test_accuracy,
        'auc': test_auc,
        'brier': test_brier,
        'version': 5
    }

    print(f"\nSaving model to '{model_output_path}'...")
    joblib.dump(model_bundle, model_output_path)
    print("Done!")

    return model_bundle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train matchup model v5.")
    parser.add_argument('--banzuke-file', type=str, default="banzuke_detailed.csv")
    parser.add_argument('--match-file', type=str, default="match_history_with_kimarite.csv")
    parser.add_argument('--model-file', type=str, default="matchup_model_v5.joblib")
    args = parser.parse_args()

    banzuke, matches = load_data(args.banzuke_file, args.match_file)
    matchups = build_training_data(banzuke, matches)
    train_model(matchups, args.model_file)

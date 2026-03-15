# train_matchup_model_v4.py
"""
Hybrid matchup prediction model v4 targeting 65%+ accuracy.
Key improvements over v3:
1. Stacking ensemble (LightGBM + XGBoost + Logistic Regression meta-learner)
2. Day-of-tournament features (performance tracking within basho)
3. Career experience features
4. Improved feature selection based on importance
5. Better handling of matchup dynamics
"""

import pandas as pd
import numpy as np
import joblib
import argparse
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
from sklearn.ensemble import StackingClassifier
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


def compute_elo_ratings(matches: pd.DataFrame, k_factor: float = 32, initial_elo: float = 1500) -> dict:
    """Compute ELO ratings for all wrestlers based on match history."""
    print("Computing ELO ratings...")
    matches = matches.sort_values(['tournament_id', 'day']).reset_index(drop=True)

    current_elo = {}
    elo_history = {}
    current_tournament = None

    for idx, row in matches.iterrows():
        tournament = row['tournament_id']
        winner = row['winner_id']
        loser = row['loser_id']

        if tournament != current_tournament:
            current_tournament = tournament
            for w_id, elo in current_elo.items():
                elo_history[(w_id, tournament)] = elo

        elo_winner = current_elo.get(winner, initial_elo)
        elo_loser = current_elo.get(loser, initial_elo)

        if (winner, tournament) not in elo_history:
            elo_history[(winner, tournament)] = elo_winner
        if (loser, tournament) not in elo_history:
            elo_history[(loser, tournament)] = elo_loser

        exp_winner = 1 / (1 + 10 ** ((elo_loser - elo_winner) / 400))
        exp_loser = 1 - exp_winner

        current_elo[winner] = elo_winner + k_factor * (1 - exp_winner)
        current_elo[loser] = elo_loser + k_factor * (0 - exp_loser)

    return elo_history


def compute_career_stats(banzuke: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Compute career statistics for each wrestler at each tournament."""
    print("Computing career stats...")

    results = []
    matches = matches.sort_values(['tournament_id', 'day'])

    for w_id in banzuke['wrestler_id'].unique():
        if pd.isna(w_id):
            continue
        w_matches = matches[(matches['winner_id'] == w_id) | (matches['loser_id'] == w_id)]
        w_matches = w_matches.sort_values(['tournament_id', 'day'])
        tournaments = w_matches['tournament_id'].unique()

        # Track career stats
        career_wins = 0
        career_losses = 0
        career_tournaments = 0

        for i, tourn in enumerate(tournaments):
            # Stats BEFORE this tournament
            results.append({
                'wrestler_id': w_id,
                'tournament_id': tourn,
                'career_wins': career_wins,
                'career_losses': career_losses,
                'career_tournaments': career_tournaments,
                'career_win_pct': career_wins / max(1, career_wins + career_losses),
                'career_total_bouts': career_wins + career_losses
            })

            # Update for next tournament
            t_matches = w_matches[w_matches['tournament_id'] == tourn]
            career_wins += (t_matches['winner_id'] == w_id).sum()
            career_losses += (t_matches['loser_id'] == w_id).sum()
            career_tournaments += 1

    return pd.DataFrame(results)


def compute_within_tournament_stats(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Compute running stats within each tournament for each wrestler.
    This gives us day-of-match context.
    """
    print("Computing within-tournament stats...")

    matches = matches.sort_values(['tournament_id', 'day']).reset_index(drop=True)

    # For each match, we need the stats of both wrestlers BEFORE that match
    results = []

    # Track cumulative within-tournament stats
    tournament_stats = {}  # (wrestler_id, tournament_id) -> {'wins': int, 'losses': int}

    current_tournament = None

    for idx, row in matches.iterrows():
        tournament = row['tournament_id']
        day = row['day']
        winner = row['winner_id']
        loser = row['loser_id']

        # Reset stats at tournament start
        if tournament != current_tournament:
            current_tournament = tournament
            tournament_stats = {}

        # Get current stats for both wrestlers (before this match)
        w_stats = tournament_stats.get((winner, tournament), {'wins': 0, 'losses': 0})
        l_stats = tournament_stats.get((loser, tournament), {'wins': 0, 'losses': 0})

        results.append({
            'tournament_id': tournament,
            'day': day,
            'winner_id': winner,
            'loser_id': loser,
            'winner_basho_wins': w_stats['wins'],
            'winner_basho_losses': w_stats['losses'],
            'loser_basho_wins': l_stats['wins'],
            'loser_basho_losses': l_stats['losses']
        })

        # Update stats after match
        if (winner, tournament) not in tournament_stats:
            tournament_stats[(winner, tournament)] = {'wins': 0, 'losses': 0}
        if (loser, tournament) not in tournament_stats:
            tournament_stats[(loser, tournament)] = {'wins': 0, 'losses': 0}

        tournament_stats[(winner, tournament)]['wins'] += 1
        tournament_stats[(loser, tournament)]['losses'] += 1

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
                               elo_history: dict, career_stats: pd.DataFrame) -> pd.DataFrame:
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

    # Is sanyaku (top ranks)
    def is_sanyaku(rank_str):
        div, _, _ = _parse_rank(rank_str)
        return 1 if div in ['Y', 'O', 'S', 'K'] else 0

    df['is_sanyaku'] = df['rank'].apply(is_sanyaku)

    # Physical attributes
    hw = df['height_weight'].str.extract(r'(\d+\.?\d*)\s*cm\s*(\d+\.?\d*)\s*kg', expand=True)
    df['height'] = pd.to_numeric(hw[0], errors='coerce')
    df['weight'] = pd.to_numeric(hw[1], errors='coerce')
    df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)

    # Reach advantage proxy (height - typical arm span relationship)
    df['reach'] = df['height'] * 1.02  # Approximation

    # Age
    df['birth_date'] = pd.to_datetime(df['birth_date'], format='%d.%m.%Y', errors='coerce')
    df['tournament_date'] = pd.to_datetime(df['tournament_id'], format='%Y%m')
    df['age'] = (df['tournament_date'] - df['birth_date']).dt.days / 365.25

    # Age categories (peak performance age)
    df['is_peak_age'] = ((df['age'] >= 24) & (df['age'] <= 30)).astype(int)
    df['is_young'] = (df['age'] < 24).astype(int)
    df['is_veteran'] = (df['age'] > 32).astype(int)

    # Win percentage
    df['w'] = pd.to_numeric(df['w'], errors='coerce')
    df['l'] = pd.to_numeric(df['l'], errors='coerce')
    df['total_bouts'] = df['w'] + df['l']
    df['win_pct'] = df['w'] / df['total_bouts'].replace(0, np.nan)

    # Rolling win percentages
    df['win_pct_last_3'] = df.groupby('wrestler_id')['win_pct'].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
    ).fillna(0.5)

    df['win_pct_last_6'] = df.groupby('wrestler_id')['win_pct'].transform(
        lambda x: x.shift(1).rolling(window=6, min_periods=1).mean()
    ).fillna(0.5)

    # Variance in performance (consistency)
    df['win_pct_std_6'] = df.groupby('wrestler_id')['win_pct'].transform(
        lambda x: x.shift(1).rolling(window=6, min_periods=3).std()
    ).fillna(0.15)

    # ELO rating
    df['elo'] = df.apply(
        lambda row: elo_history.get((row['wrestler_id'], row['tournament_id']), 1500),
        axis=1
    )

    # Career stats
    df = df.merge(career_stats, on=['wrestler_id', 'tournament_id'], how='left')
    df['career_win_pct'] = df['career_win_pct'].fillna(0.5)
    df['career_tournaments'] = df['career_tournaments'].fillna(0)
    df['career_total_bouts'] = df['career_total_bouts'].fillna(0)

    # Experience tier
    df['is_experienced'] = (df['career_tournaments'] >= 20).astype(int)
    df['is_newcomer'] = (df['career_tournaments'] <= 3).astype(int)

    # Fill missing values
    for col in ['height', 'weight', 'bmi', 'age', 'reach']:
        df[col] = df.groupby('tournament_id')[col].transform(lambda x: x.fillna(x.median()))

    return df


def compute_style_profiles(matches: pd.DataFrame) -> pd.DataFrame:
    """Compute detailed style profiles."""
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

    # Also compute dominant style
    style_history['dominant_style'] = style_history[[f'{cat}_ratio' for cat in TECHNIQUE_CATEGORIES.keys()]].idxmax(axis=1)
    style_history['dominant_style'] = style_history['dominant_style'].str.replace('_ratio', '')

    ratio_cols = ['wrestler_id', 'tournament_id'] + [f'{cat}_ratio' for cat in TECHNIQUE_CATEGORIES.keys()] + ['dominant_style']
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

    # Compute all helper data
    elo_history = compute_elo_ratings(matches)
    career_stats = compute_career_stats(banzuke, matches)
    within_tournament_stats = compute_within_tournament_stats(matches)

    # Get all profile data
    wrestler_profiles = compute_wrestler_profiles(banzuke, matches, elo_history, career_stats)
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

    # Fill missing style/vuln ratios
    for cat in TECHNIQUE_CATEGORIES.keys():
        wrestler_profiles[f'{cat}_ratio'] = wrestler_profiles[f'{cat}_ratio'].fillna(0.2)
        wrestler_profiles[f'vuln_{cat}'] = wrestler_profiles[f'vuln_{cat}'].fillna(0.2)

    # Build matchup features
    print("Merging features for each match...")

    matchups = h2h_data.copy()

    # Merge within-tournament stats
    matchups = matchups.merge(
        within_tournament_stats,
        on=['tournament_id', 'day', 'winner_id', 'loser_id'],
        how='left'
    )

    # Features from wrestler profiles
    profile_cols = ['wrestler_id', 'tournament_id', 'rank_score', 'rank_in_div', 'is_sanyaku',
                    'bmi', 'height', 'weight', 'reach', 'age', 'is_peak_age', 'is_veteran',
                    'elo', 'win_pct_last_3', 'win_pct_last_6', 'win_pct_std_6',
                    'career_win_pct', 'career_tournaments', 'is_experienced', 'is_newcomer']
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

    # H2H win percentage for A
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
    matchups['has_h2h_history'] = (matchups['h2h_total_matches'] > 0).astype(int)

    # Within-tournament performance (winning-ness entering match)
    matchups['basho_win_pct_A'] = matchups['winner_basho_wins'] / (
        matchups['winner_basho_wins'] + matchups['winner_basho_losses'] + 1
    )
    matchups['basho_win_pct_B'] = matchups['loser_basho_wins'] / (
        matchups['loser_basho_wins'] + matchups['loser_basho_losses'] + 1
    )
    matchups['basho_momentum_diff'] = matchups['basho_win_pct_A'] - matchups['basho_win_pct_B']

    # Day of tournament (early vs late)
    matchups['day_normalized'] = matchups['day'] / 15.0
    matchups['is_late_basho'] = (matchups['day'] >= 10).astype(int)

    # === DIFFERENCE FEATURES ===
    matchups['rank_diff'] = matchups['rank_score_A'] - matchups['rank_score_B']
    matchups['rank_in_div_diff'] = matchups['rank_in_div_A'] - matchups['rank_in_div_B']
    matchups['bmi_diff'] = matchups['bmi_A'] - matchups['bmi_B']
    matchups['height_diff'] = matchups['height_A'] - matchups['height_B']
    matchups['weight_diff'] = matchups['weight_A'] - matchups['weight_B']
    matchups['reach_diff'] = matchups['reach_A'] - matchups['reach_B']
    matchups['age_diff'] = matchups['age_A'] - matchups['age_B']
    matchups['elo_diff'] = matchups['elo_A'] - matchups['elo_B']
    matchups['win_pct_last_3_diff'] = matchups['win_pct_last_3_A'] - matchups['win_pct_last_3_B']
    matchups['win_pct_last_6_diff'] = matchups['win_pct_last_6_A'] - matchups['win_pct_last_6_B']
    matchups['career_win_pct_diff'] = matchups['career_win_pct_A'] - matchups['career_win_pct_B']
    matchups['experience_diff'] = matchups['career_tournaments_A'] - matchups['career_tournaments_B']

    # ELO expected probability
    matchups['elo_expected_A'] = 1 / (1 + 10 ** ((matchups['elo_B'] - matchups['elo_A']) / 400))

    # Experience matchup (newcomer vs veteran)
    matchups['newcomer_vs_veteran'] = matchups['is_newcomer_A'] * matchups['is_veteran_B'] - matchups['is_newcomer_B'] * matchups['is_veteran_A']

    # Sanyaku matchup
    matchups['sanyaku_diff'] = matchups['is_sanyaku_A'] - matchups['is_sanyaku_B']

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

    # Swap A and B features
    a_cols = [c for c in matchups.columns if c.endswith('_A') and not c.startswith('basho_win_pct')]
    b_cols = [c for c in matchups.columns if c.endswith('_B') and not c.startswith('basho_win_pct')]

    for a_col in sorted(a_cols):
        b_col = a_col.replace('_A', '_B')
        if b_col in matchups.columns:
            matchups_flipped[a_col], matchups_flipped[b_col] = matchups[b_col].copy(), matchups[a_col].copy()

    # Swap basho stats
    matchups_flipped['basho_win_pct_A'], matchups_flipped['basho_win_pct_B'] = matchups['basho_win_pct_B'].copy(), matchups['basho_win_pct_A'].copy()

    # Flip difference columns
    diff_cols = [c for c in matchups.columns if c.endswith('_diff')]
    for col in diff_cols:
        matchups_flipped[col] = -matchups[col]

    matchups_flipped['h2h_win_pct_A'] = 1 - matchups['h2h_win_pct_A']
    matchups_flipped['elo_expected_A'] = 1 - matchups['elo_expected_A']
    matchups_flipped['A_wins'] = 0

    all_matchups = pd.concat([matchups, matchups_flipped], ignore_index=True)
    print(f"Total training examples: {len(all_matchups):,}")

    return all_matchups


def train_model(matchups: pd.DataFrame, model_output_path: str, use_stacking: bool = True):
    """Train the hybrid matchup model with optional stacking."""

    # Define features - carefully selected based on importance analysis
    features = [
        # Core difference features
        'rank_diff', 'rank_in_div_diff', 'elo_diff', 'elo_expected_A',

        # Physical differences
        'bmi_diff', 'height_diff', 'weight_diff', 'age_diff',

        # Form and performance
        'win_pct_last_3_diff', 'win_pct_last_6_diff', 'career_win_pct_diff',
        'win_pct_last_3_A', 'win_pct_last_3_B',
        'win_pct_last_6_A', 'win_pct_last_6_B',

        # H2H
        'h2h_win_pct_A', 'h2h_total_matches', 'has_h2h_history',

        # Within-tournament performance
        'basho_win_pct_A', 'basho_win_pct_B', 'basho_momentum_diff',
        'day_normalized', 'is_late_basho',

        # Experience
        'experience_diff', 'newcomer_vs_veteran',

        # Sanyaku status
        'sanyaku_diff', 'is_sanyaku_A', 'is_sanyaku_B',

        # Peak age
        'is_peak_age_A', 'is_peak_age_B', 'is_veteran_A', 'is_veteran_B',
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

    # Split by tournament (temporal split)
    tournaments = sorted(df['tournament_id'].unique())
    train_tournaments = tournaments[:-10]
    test_tournaments = tournaments[-10:]

    train_mask = df['tournament_id'].isin(train_tournaments)
    test_mask = df['tournament_id'].isin(test_tournaments)

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    print(f"\nTraining: {len(X_train):,} | Test: {len(X_test):,}")

    if use_stacking:
        print("\nTraining stacking ensemble...")

        # Define base estimators
        lgbm_model = lgb.LGBMClassifier(
            n_estimators=800,
            learning_rate=0.015,
            num_leaves=40,
            max_depth=7,
            reg_alpha=0.02,
            reg_lambda=0.05,
            colsample_bytree=0.85,
            subsample=0.85,
            min_child_samples=40,
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )

        lgbm_model_2 = lgb.LGBMClassifier(
            n_estimators=600,
            learning_rate=0.02,
            num_leaves=30,
            max_depth=6,
            reg_alpha=0.1,
            reg_lambda=0.1,
            colsample_bytree=0.8,
            subsample=0.8,
            min_child_samples=50,
            random_state=123,
            n_jobs=-1,
            verbosity=-1
        )

        # Stacking classifier
        model = StackingClassifier(
            estimators=[
                ('lgbm1', lgbm_model),
                ('lgbm2', lgbm_model_2),
            ],
            final_estimator=LogisticRegression(max_iter=1000),
            cv=3,
            n_jobs=-1,
            passthrough=True
        )

        model.fit(X_train, y_train)
        base_model = lgbm_model  # For feature importance

    else:
        print("\nTraining single LightGBM model...")

        model = lgb.LGBMClassifier(
            n_estimators=1200,
            learning_rate=0.012,
            num_leaves=50,
            max_depth=8,
            reg_alpha=0.02,
            reg_lambda=0.05,
            colsample_bytree=0.85,
            subsample=0.85,
            min_child_samples=40,
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )

        model.fit(X_train, y_train)
        base_model = model

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
    print(f"Previous v3 model: ~62.4%")
    print(f"{'Improvement' if test_accuracy > 0.624 else 'Change'} over v3: {(test_accuracy - 0.624) * 100:+.1f}%")

    if test_accuracy >= 0.65:
        print("\n*** TARGET ACHIEVED: 65%+ ACCURACY! ***")
    else:
        print(f"\n*** Still {(0.65 - test_accuracy) * 100:.1f}% short of 65% target ***")

    # Feature importance (from base model)
    if hasattr(base_model, 'feature_importances_'):
        print("\n" + "=" * 50)
        print("TOP 20 FEATURE IMPORTANCE")
        print("=" * 50)
        importance = pd.DataFrame({
            'Feature': features,
            'Importance': base_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        print(importance.head(20).to_string(index=False))

    # Save
    model_bundle = {
        'model': model,
        'base_model': base_model,
        'features': features,
        'technique_categories': TECHNIQUE_CATEGORIES,
        'accuracy': test_accuracy,
        'auc': test_auc,
        'brier': test_brier,
        'version': 4
    }

    print(f"\nSaving model to '{model_output_path}'...")
    joblib.dump(model_bundle, model_output_path)
    print("Done!")

    return model_bundle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train hybrid matchup model v4.")
    parser.add_argument('--banzuke-file', type=str, default="banzuke_detailed.csv")
    parser.add_argument('--match-file', type=str, default="match_history_with_kimarite.csv")
    parser.add_argument('--model-file', type=str, default="matchup_model_v4.joblib")
    parser.add_argument('--no-stacking', action='store_true', help="Skip stacking ensemble")
    args = parser.parse_args()

    banzuke, matches = load_data(args.banzuke_file, args.match_file)
    matchups = build_training_data(banzuke, matches)
    train_model(matchups, args.model_file, use_stacking=not args.no_stacking)

# train_matchup_model_v7.py
"""
Matchup model v7 - New features with ablation study.
Key changes:
1. In-tournament record features (wins/losses before each match, day)
2. Kachi-koshi/Make-koshi pressure features (critical match situations)
3. Physical × style interaction features (height/weight/age × fighting style)
4. Ablation study framework to measure each feature group's contribution
"""

import pandas as pd
import numpy as np
import joblib
import argparse
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
import lightgbm as lgb
from sumo_utils import get_absolute_rank_score, _parse_rank

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

# Feature groups for ablation study
FEATURE_GROUPS = {
    'in_tournament': [
        'wins_before_A', 'wins_before_B', 'losses_before_A', 'losses_before_B',
        'record_diff', 'day'
    ],
    'pressure': [
        'is_kk_match_A', 'is_kk_match_B', 'is_mk_match_A', 'is_mk_match_B',
        'kk_pressure_diff', 'mk_pressure_diff', 'pressure_diff'
    ],
    'height_style': ['height_x_oshi_diff', 'height_x_yotsu_diff'],
    'weight_style': ['weight_x_yotsu_diff', 'bmi_x_pull_diff'],
    'age_style': ['age_x_oshi_diff', 'age_x_yotsu_diff']
}


def compute_elo_ratings(matches: pd.DataFrame, k_factor: float = 32, initial_elo: float = 1500) -> dict:
    """Compute ELO ratings. Optimized with numpy arrays."""
    print("Computing ELO ratings...", flush=True)
    matches = matches.sort_values(['tournament_id', 'day']).reset_index(drop=True)

    # Convert to numpy arrays for speed
    tournaments = matches['tournament_id'].values
    winners = matches['winner_id'].values
    losers = matches['loser_id'].values

    current_elo = {}
    elo_history = {}
    current_tournament = None
    n_matches = len(matches)

    for i in range(n_matches):
        tournament = tournaments[i]
        winner = winners[i]
        loser = losers[i]

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

        current_elo[winner] = elo_winner + k_factor * (1 - exp_winner)
        current_elo[loser] = elo_loser - k_factor * exp_winner

        if i % 50000 == 0 and i > 0:
            print(f"  ELO progress: {i:,}/{n_matches:,} matches processed", flush=True)

    print(f"  ELO complete: {n_matches:,} matches processed", flush=True)
    return elo_history


def compute_in_tournament_records(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Compute running W-L record for each wrestler before each match.
    Optimized using vectorized operations.
    """
    print("Computing in-tournament records...", flush=True)
    matches = matches.sort_values(['tournament_id', 'day']).reset_index(drop=True)

    # Create a long-form dataframe for wrestlers' participation in each match
    # Each match generates two rows: one for winner (result=1), one for loser (result=0)
    winner_rows = matches[['tournament_id', 'day', 'winner_id']].copy()
    winner_rows['wrestler_id'] = winner_rows['winner_id']
    winner_rows['is_win'] = 1
    winner_rows['match_idx'] = matches.index

    loser_rows = matches[['tournament_id', 'day', 'loser_id']].copy()
    loser_rows['wrestler_id'] = loser_rows['loser_id']
    loser_rows['is_win'] = 0
    loser_rows['match_idx'] = matches.index

    all_participations = pd.concat([
        winner_rows[['tournament_id', 'day', 'wrestler_id', 'is_win', 'match_idx']],
        loser_rows[['tournament_id', 'day', 'wrestler_id', 'is_win', 'match_idx']]
    ]).sort_values(['tournament_id', 'wrestler_id', 'day', 'match_idx'])

    # Compute cumulative wins and losses BEFORE each match
    all_participations['cum_wins'] = all_participations.groupby(
        ['tournament_id', 'wrestler_id']
    )['is_win'].cumsum()
    all_participations['match_in_tourn'] = all_participations.groupby(
        ['tournament_id', 'wrestler_id']
    ).cumcount() + 1

    # Shift to get records BEFORE the current match
    all_participations['wins_before'] = all_participations['cum_wins'] - all_participations['is_win']
    all_participations['losses_before'] = (
        all_participations['match_in_tourn'] - 1 - all_participations['wins_before']
    )

    # Pivot back to match-level data
    winner_records = all_participations[all_participations['is_win'] == 1][
        ['match_idx', 'wins_before', 'losses_before']
    ].rename(columns={
        'wins_before': 'wins_before_winner',
        'losses_before': 'losses_before_winner'
    })

    loser_records = all_participations[all_participations['is_win'] == 0][
        ['match_idx', 'wins_before', 'losses_before']
    ].rename(columns={
        'wins_before': 'wins_before_loser',
        'losses_before': 'losses_before_loser'
    })

    result = matches[['day']].copy()
    result['idx'] = matches.index
    result = result.merge(winner_records, left_on='idx', right_on='match_idx', how='left')
    result = result.merge(loser_records, left_on='idx', right_on='match_idx', how='left')
    result = result.drop(columns=['match_idx_x', 'match_idx_y'], errors='ignore')

    return result.set_index('idx')


def compute_pressure_features(wins: int, losses: int) -> dict:
    """Compute KK/MK pressure indicators for a single wrestler."""
    return {
        'is_kk_match': 1 if wins == 7 and losses < 8 else 0,
        'is_mk_match': 1 if losses == 7 and wins < 8 else 0,
    }


def compute_physical_style_interactions(matchups: pd.DataFrame) -> pd.DataFrame:
    """Add physical × style interaction features."""
    # Height × oshi (tall + pushing style)
    matchups['height_x_oshi_A'] = matchups['height_A'] * matchups['oshi_ratio_A']
    matchups['height_x_oshi_B'] = matchups['height_B'] * matchups['oshi_ratio_B']
    matchups['height_x_oshi_diff'] = matchups['height_x_oshi_A'] - matchups['height_x_oshi_B']

    # Height × yotsu (tall + belt work)
    matchups['height_x_yotsu_A'] = matchups['height_A'] * matchups['yotsu_ratio_A']
    matchups['height_x_yotsu_B'] = matchups['height_B'] * matchups['yotsu_ratio_B']
    matchups['height_x_yotsu_diff'] = matchups['height_x_yotsu_A'] - matchups['height_x_yotsu_B']

    # Weight × yotsu (heavy + belt grappling)
    matchups['weight_x_yotsu_A'] = matchups['weight_A'] * matchups['yotsu_ratio_A']
    matchups['weight_x_yotsu_B'] = matchups['weight_B'] * matchups['yotsu_ratio_B']
    matchups['weight_x_yotsu_diff'] = matchups['weight_x_yotsu_A'] - matchups['weight_x_yotsu_B']

    # BMI × pull (body proportion + pulling techniques)
    matchups['bmi_x_pull_A'] = matchups['bmi_A'] * matchups['pull_ratio_A']
    matchups['bmi_x_pull_B'] = matchups['bmi_B'] * matchups['pull_ratio_B']
    matchups['bmi_x_pull_diff'] = matchups['bmi_x_pull_A'] - matchups['bmi_x_pull_B']

    # Age × oshi (experience + pushing style)
    matchups['age_x_oshi_A'] = matchups['age_A'] * matchups['oshi_ratio_A']
    matchups['age_x_oshi_B'] = matchups['age_B'] * matchups['oshi_ratio_B']
    matchups['age_x_oshi_diff'] = matchups['age_x_oshi_A'] - matchups['age_x_oshi_B']

    # Age × yotsu (experience + belt technique)
    matchups['age_x_yotsu_A'] = matchups['age_A'] * matchups['yotsu_ratio_A']
    matchups['age_x_yotsu_B'] = matchups['age_B'] * matchups['yotsu_ratio_B']
    matchups['age_x_yotsu_diff'] = matchups['age_x_yotsu_A'] - matchups['age_x_yotsu_B']

    return matchups


def compute_career_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Compute career-level features for each wrestler at each tournament.
    Optimized using vectorized operations.
    """
    print("Computing career features...", flush=True)
    matches = matches.sort_values(['tournament_id', 'day'])

    # Create long-form data: each match generates two rows (winner and loser)
    winner_rows = matches[['tournament_id', 'winner_id']].copy()
    winner_rows['wrestler_id'] = winner_rows['winner_id']
    winner_rows['is_win'] = 1

    loser_rows = matches[['tournament_id', 'loser_id']].copy()
    loser_rows['wrestler_id'] = loser_rows['loser_id']
    loser_rows['is_win'] = 0

    all_matches = pd.concat([
        winner_rows[['tournament_id', 'wrestler_id', 'is_win']],
        loser_rows[['tournament_id', 'wrestler_id', 'is_win']]
    ]).sort_values(['wrestler_id', 'tournament_id'])

    # Aggregate by tournament: wins and losses per wrestler per tournament
    tourn_stats = all_matches.groupby(['wrestler_id', 'tournament_id']).agg(
        wins_in_tourn=('is_win', 'sum'),
        matches_in_tourn=('is_win', 'count')
    ).reset_index()
    tourn_stats['losses_in_tourn'] = tourn_stats['matches_in_tourn'] - tourn_stats['wins_in_tourn']

    # Sort and compute cumulative stats
    tourn_stats = tourn_stats.sort_values(['wrestler_id', 'tournament_id'])

    # Cumulative career stats BEFORE each tournament (shift by 1)
    tourn_stats['career_wins'] = tourn_stats.groupby('wrestler_id')['wins_in_tourn'].cumsum().shift(1)
    tourn_stats['career_losses'] = tourn_stats.groupby('wrestler_id')['losses_in_tourn'].cumsum().shift(1)
    tourn_stats['career_tournaments'] = tourn_stats.groupby('wrestler_id').cumcount()

    # Fill NaN for first tournaments
    tourn_stats['career_wins'] = tourn_stats['career_wins'].fillna(0)
    tourn_stats['career_losses'] = tourn_stats['career_losses'].fillna(0)

    tourn_stats['career_total_bouts'] = tourn_stats['career_wins'] + tourn_stats['career_losses']
    tourn_stats['career_win_pct'] = tourn_stats['career_wins'] / tourn_stats['career_total_bouts'].replace(0, 1)

    # For weighted_recent_10, use a simpler rolling window approach on tournament-level win %
    tourn_stats['tourn_win_pct'] = tourn_stats['wins_in_tourn'] / tourn_stats['matches_in_tourn'].replace(0, 1)
    # Use exponential weighted average of last few tournaments as proxy for recent form
    tourn_stats['weighted_recent_10'] = tourn_stats.groupby('wrestler_id')['tourn_win_pct'].transform(
        lambda x: x.shift(1).ewm(span=3, min_periods=1).mean()
    ).fillna(0.5)

    return tourn_stats[['wrestler_id', 'tournament_id', 'career_wins', 'career_losses',
                        'career_tournaments', 'career_win_pct', 'career_total_bouts',
                        'weighted_recent_10']]


def load_data(banzuke_path: str, match_history_path: str):
    """Load data."""
    print("Loading data...", flush=True)
    banzuke = pd.read_csv(banzuke_path)
    matches = pd.read_csv(match_history_path)

    for df in [banzuke, matches]:
        for col in ['wrestler_id', 'winner_id', 'loser_id', 'tournament_id', 'day']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

    return banzuke, matches


def compute_wrestler_profiles(banzuke: pd.DataFrame, matches: pd.DataFrame,
                               elo_history: dict, career_df: pd.DataFrame) -> pd.DataFrame:
    """Compute wrestler profiles with career features."""
    print("Computing wrestler profiles...", flush=True)

    df = banzuke.copy()
    df = df.sort_values(['wrestler_id', 'tournament_id']).reset_index(drop=True)

    # Rank features
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

    def get_is_sanyaku(rank_str):
        div, _, _ = _parse_rank(rank_str)
        return 1 if div in ['Y', 'O', 'S', 'K'] else 0

    df['is_sanyaku'] = df['rank'].apply(get_is_sanyaku)

    # Physical
    hw = df['height_weight'].str.extract(r'(\d+\.?\d*)\s*cm\s*(\d+\.?\d*)\s*kg', expand=True)
    df['height'] = pd.to_numeric(hw[0], errors='coerce')
    df['weight'] = pd.to_numeric(hw[1], errors='coerce')
    df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)

    # Age
    df['birth_date'] = pd.to_datetime(df['birth_date'], format='%d.%m.%Y', errors='coerce')
    df['tournament_date'] = pd.to_datetime(df['tournament_id'], format='%Y%m')
    df['age'] = (df['tournament_date'] - df['birth_date']).dt.days / 365.25

    # Win pct rolling
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

    # Rank trajectory features
    df['highest_rank_score'] = df['high'].apply(get_absolute_rank_score)
    df['rank_gap'] = (df['highest_rank_score'] - df['rank_score']).clip(lower=0)

    # Rank momentum: mean rank score change over last 3 basho (positive = improving)
    rank_delta = df.groupby('wrestler_id')['rank_score'].diff()
    df['rank_momentum'] = (
        rank_delta.groupby(df['wrestler_id'])
        .transform(lambda x: x.rolling(window=3, min_periods=1).mean().shift(1))
        .fillna(0)
    )

    # Basho since peak: tournaments since rank_gap was last ~0
    def _basho_since_peak(rank_gaps):
        result, count = [], 0
        for gap in rank_gaps:
            result.append(count)
            count = 0 if gap <= 0.6 else count + 1
        return result

    df['basho_since_peak'] = (
        df.groupby('wrestler_id')['rank_gap']
        .transform(lambda x: pd.Series(_basho_since_peak(x.values), index=x.index))
        .shift(1)
        .fillna(0)
    )

    # ELO
    df['elo'] = df.apply(
        lambda row: elo_history.get((row['wrestler_id'], row['tournament_id']), 1500),
        axis=1
    )

    # Career features
    df = df.merge(career_df, on=['wrestler_id', 'tournament_id'], how='left')
    df['career_win_pct'] = df['career_win_pct'].fillna(0.5)
    df['career_tournaments'] = df['career_tournaments'].fillna(0)
    df['weighted_recent_10'] = df['weighted_recent_10'].fillna(0.5)

    # Fill missing
    for col in ['height', 'weight', 'bmi', 'age']:
        df[col] = df.groupby('tournament_id')[col].transform(lambda x: x.fillna(x.median()))

    return df


def compute_style_profiles(matches: pd.DataFrame) -> pd.DataFrame:
    """Compute style profiles."""
    print("Computing style profiles...", flush=True)

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
    print("Computing vulnerability profiles...", flush=True)

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


def compute_h2h_history(matches: pd.DataFrame) -> pd.DataFrame:
    """Compute H2H history."""
    print("Computing H2H history...", flush=True)

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
    """Build training dataset with v7 features."""
    print("\nBuilding training dataset...", flush=True)

    # Compute helper data
    elo_history = compute_elo_ratings(matches)
    career_df = compute_career_features(matches)
    in_tournament_records = compute_in_tournament_records(matches)

    # Get profiles
    wrestler_profiles = compute_wrestler_profiles(banzuke, matches, elo_history, career_df)
    style_profiles = compute_style_profiles(matches)
    vuln_profiles = compute_vulnerability_profiles(matches)
    h2h_data = compute_h2h_history(matches)

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
    print("Merging features for each match...", flush=True)

    matchups = h2h_data.copy()

    # Features from profiles
    profile_cols = ['wrestler_id', 'tournament_id', 'rank_score', 'rank_in_div', 'is_sanyaku',
                    'bmi', 'height', 'weight', 'age', 'elo',
                    'win_pct_last_3', 'win_pct_last_6',
                    'career_win_pct', 'career_tournaments', 'weighted_recent_10',
                    'rank_momentum', 'basho_since_peak', 'rank_gap']
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

    # === MERGE IN-TOURNAMENT RECORDS ===
    matchups = matchups.reset_index(drop=True)
    # Drop 'day' from matchups since in_tournament_records also has it
    matchups = matchups.drop(columns=['day'], errors='ignore')
    matchups = matchups.join(in_tournament_records)

    # Map winner/loser records to A/B (A = winner for original examples)
    matchups['wins_before_A'] = matchups['wins_before_winner']
    matchups['losses_before_A'] = matchups['losses_before_winner']
    matchups['wins_before_B'] = matchups['wins_before_loser']
    matchups['losses_before_B'] = matchups['losses_before_loser']

    # Compute record differential
    matchups['record_diff'] = (
        (matchups['wins_before_A'] - matchups['losses_before_A']) -
        (matchups['wins_before_B'] - matchups['losses_before_B'])
    )

    # === COMPUTE PRESSURE FEATURES ===
    print("Computing pressure features...", flush=True)
    matchups['is_kk_match_A'] = ((matchups['wins_before_A'] == 7) &
                                  (matchups['losses_before_A'] < 8)).astype(int)
    matchups['is_mk_match_A'] = ((matchups['losses_before_A'] == 7) &
                                  (matchups['wins_before_A'] < 8)).astype(int)
    matchups['is_kk_match_B'] = ((matchups['wins_before_B'] == 7) &
                                  (matchups['losses_before_B'] < 8)).astype(int)
    matchups['is_mk_match_B'] = ((matchups['losses_before_B'] == 7) &
                                  (matchups['wins_before_B'] < 8)).astype(int)

    # Pressure differentials
    matchups['kk_pressure_diff'] = matchups['is_kk_match_A'] - matchups['is_kk_match_B']
    matchups['mk_pressure_diff'] = matchups['is_mk_match_A'] - matchups['is_mk_match_B']
    matchups['pressure_diff'] = matchups['kk_pressure_diff'] - matchups['mk_pressure_diff']

    # === COMPUTE DERIVED FEATURES (v6 features) ===

    # H2H
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

    # === DIFFERENCE FEATURES ===
    matchups['rank_diff'] = matchups['rank_score_A'] - matchups['rank_score_B']
    matchups['rank_in_div_diff'] = matchups['rank_in_div_A'] - matchups['rank_in_div_B']
    matchups['bmi_diff'] = matchups['bmi_A'] - matchups['bmi_B']
    matchups['height_diff'] = matchups['height_A'] - matchups['height_B']
    matchups['weight_diff'] = matchups['weight_A'] - matchups['weight_B']
    matchups['age_diff'] = matchups['age_A'] - matchups['age_B']
    matchups['elo_diff'] = matchups['elo_A'] - matchups['elo_B']
    matchups['win_pct_last_3_diff'] = matchups['win_pct_last_3_A'] - matchups['win_pct_last_3_B']
    matchups['win_pct_last_6_diff'] = matchups['win_pct_last_6_A'] - matchups['win_pct_last_6_B']
    matchups['career_win_pct_diff'] = matchups['career_win_pct_A'] - matchups['career_win_pct_B']
    matchups['experience_diff'] = matchups['career_tournaments_A'] - matchups['career_tournaments_B']
    matchups['weighted_recent_diff'] = matchups['weighted_recent_10_A'] - matchups['weighted_recent_10_B']

    # ELO expected
    matchups['elo_expected_A'] = 1 / (1 + 10 ** ((matchups['elo_B'] - matchups['elo_A']) / 400))

    # === INTERACTION FEATURES ===
    matchups['rank_x_form_A'] = matchups['rank_score_A'] * matchups['win_pct_last_3_A']
    matchups['rank_x_form_B'] = matchups['rank_score_B'] * matchups['win_pct_last_3_B']
    matchups['rank_x_form_diff'] = matchups['rank_x_form_A'] - matchups['rank_x_form_B']

    matchups['elo_x_age_A'] = matchups['elo_A'] * (1 / (1 + np.exp((matchups['age_A'] - 30) / 3)))
    matchups['elo_x_age_B'] = matchups['elo_B'] * (1 / (1 + np.exp((matchups['age_B'] - 30) / 3)))
    matchups['elo_x_age_diff'] = matchups['elo_x_age_A'] - matchups['elo_x_age_B']

    matchups['sanyaku_diff'] = matchups['is_sanyaku_A'] - matchups['is_sanyaku_B']

    # === RANK TRAJECTORY FEATURES ===
    matchups['rank_momentum_diff'] = matchups['rank_momentum_A'] - matchups['rank_momentum_B']
    matchups['basho_since_peak_diff'] = matchups['basho_since_peak_A'] - matchups['basho_since_peak_B']
    matchups['rank_gap_diff'] = matchups['rank_gap_A'] - matchups['rank_gap_B']

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

    # === PHYSICAL × STYLE INTERACTION FEATURES (v7 new) ===
    matchups = compute_physical_style_interactions(matchups)

    # === TARGET ===
    matchups['A_wins'] = 1

    # === CREATE SYMMETRIC EXAMPLES ===
    print("Creating symmetric training examples...", flush=True)
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
    matchups_flipped['elo_expected_A'] = 1 - matchups['elo_expected_A']
    matchups_flipped['A_wins'] = 0

    # Also swap in-tournament records and pressure features
    matchups_flipped['day'] = matchups['day']  # day stays same

    all_matchups = pd.concat([matchups, matchups_flipped], ignore_index=True)
    print(f"Total training examples: {len(all_matchups):,}")

    return all_matchups


def get_base_features():
    """Get the baseline (v6) feature list."""
    features = [
        # Core ranking
        'rank_diff', 'rank_in_div_diff', 'sanyaku_diff',

        # ELO
        'elo_diff', 'elo_expected_A',

        # Physical
        'bmi_diff', 'height_diff', 'weight_diff', 'age_diff',

        # Form
        'win_pct_last_3_diff', 'win_pct_last_6_diff',
        'win_pct_last_3_A', 'win_pct_last_3_B',
        'win_pct_last_6_A', 'win_pct_last_6_B',

        # Career
        'career_win_pct_diff', 'experience_diff', 'weighted_recent_diff',
        'career_win_pct_A', 'career_win_pct_B',
        'weighted_recent_10_A', 'weighted_recent_10_B',

        # H2H
        'h2h_win_pct_A', 'h2h_total_matches', 'has_h2h_history',

        # Interactions
        'rank_x_form_diff', 'elo_x_age_diff',

        # Rank trajectory
        'rank_momentum_diff', 'rank_momentum_A', 'rank_momentum_B',
        'basho_since_peak_diff', 'basho_since_peak_A', 'basho_since_peak_B',
        'rank_gap_diff',
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

    return features


def train_and_evaluate(matchups: pd.DataFrame, features: list, verbose: bool = False):
    """Train model with given features and return test accuracy."""
    # Clean data
    df = matchups.dropna(subset=features + ['A_wins'])

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

    # Train model
    model = lgb.LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.006,
        num_leaves=70,
        max_depth=12,
        reg_alpha=0.005,
        reg_lambda=0.02,
        colsample_bytree=0.95,
        subsample=0.95,
        min_child_samples=25,
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )

    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])

    # Evaluate
    test_pred_proba = model.predict_proba(X_test)[:, 1]
    test_pred = (test_pred_proba > 0.5).astype(int)
    test_accuracy = accuracy_score(y_test, test_pred)

    if verbose:
        test_auc = roc_auc_score(y_test, test_pred_proba)
        test_brier = brier_score_loss(y_test, test_pred_proba)
        print(f"  Accuracy: {test_accuracy:.4f}")
        print(f"  AUC:      {test_auc:.4f}")
        print(f"  Brier:    {test_brier:.4f}")

    return test_accuracy, model


def run_ablation_study(matchups: pd.DataFrame, base_features: list, feature_groups: dict):
    """Test each feature group's contribution to model accuracy."""
    print("\n" + "=" * 60)
    print("ABLATION STUDY", flush=True)
    print("=" * 60)

    results = []

    # Baseline (v6 features)
    print("\nTraining baseline model (v6 features)...", flush=True)
    baseline_acc, _ = train_and_evaluate(matchups, base_features, verbose=True)
    print(f"\nBaseline accuracy: {baseline_acc:.4f}")

    # Test each group added to baseline
    print("\nTesting each feature group added to baseline...", flush=True)
    for group_name, group_features in feature_groups.items():
        # Filter to features that exist in the dataframe
        valid_features = [f for f in group_features if f in matchups.columns]
        if not valid_features:
            print(f"\n{group_name}: No valid features found")
            continue

        features = base_features + valid_features
        print(f"\n{group_name} ({len(valid_features)} features):")
        acc, _ = train_and_evaluate(matchups, features, verbose=False)
        delta = acc - baseline_acc
        results.append({
            'group': group_name,
            'features_added': len(valid_features),
            'accuracy': acc,
            'delta': delta
        })
        print(f"  Accuracy: {acc:.4f} (delta: {delta:+.4f})")

    # Test all new features together
    print("\nTraining with all new features...", flush=True)
    all_new = []
    for group_features in feature_groups.values():
        valid_features = [f for f in group_features if f in matchups.columns]
        all_new.extend(valid_features)
    all_new = list(set(all_new))  # Remove duplicates

    all_features = base_features + all_new
    all_acc, _ = train_and_evaluate(matchups, all_features, verbose=True)
    all_delta = all_acc - baseline_acc
    print(f"\nAll new features accuracy: {all_acc:.4f} (delta: {all_delta:+.4f})")

    return pd.DataFrame(results), baseline_acc, all_acc


def train_model(matchups: pd.DataFrame, model_output_path: str, run_ablation: bool = False):
    """Train the model with optimized hyperparameters."""

    base_features = get_base_features()

    # Run ablation study if requested
    if run_ablation:
        ablation_results, baseline_acc, all_acc = run_ablation_study(
            matchups, base_features, FEATURE_GROUPS
        )

        print("\n" + "=" * 60)
        print("ABLATION STUDY SUMMARY", flush=True)
        print("=" * 60)
        print(f"\nBaseline (v6 features): {baseline_acc:.1%}")
        print("\nFeature Group          | Features | Accuracy | Delta", flush=True)
        print("-" * 55)
        for _, row in ablation_results.iterrows():
            print(f"{row['group']:<22} | {row['features_added']:>8} | {row['accuracy']:.1%}    | {row['delta']:+.1%}")
        print("-" * 55)
        print(f"All new features: {all_acc:.1%} (delta: {all_acc - baseline_acc:+.1%})")

    # Build final feature list (include features with positive delta)
    # For now, include all new features
    all_new_features = []
    for group_features in FEATURE_GROUPS.values():
        valid_features = [f for f in group_features if f in matchups.columns]
        all_new_features.extend(valid_features)
    all_new_features = list(set(all_new_features))

    features = base_features + all_new_features
    print(f"\nTotal features for final model: {len(features)}")

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

    # Optimized hyperparameters
    print("\nTraining LightGBM with optimized hyperparameters...", flush=True)

    model = lgb.LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.006,
        num_leaves=70,
        max_depth=12,
        reg_alpha=0.005,
        reg_lambda=0.02,
        colsample_bytree=0.95,
        subsample=0.95,
        min_child_samples=25,
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )

    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])

    # Evaluate
    train_pred_proba = model.predict_proba(X_train)[:, 1]
    test_pred_proba = model.predict_proba(X_test)[:, 1]
    train_pred = (train_pred_proba > 0.5).astype(int)
    test_pred = (test_pred_proba > 0.5).astype(int)

    print("\n" + "=" * 50)
    print("MODEL PERFORMANCE", flush=True)
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
    print(f"Previous best (v6): ~63.0%")

    if test_accuracy >= 0.65:
        print("\n*** TARGET ACHIEVED: 65%+ ACCURACY! ***", flush=True)
    else:
        print(f"\n*** Gap to 65%: {(0.65 - test_accuracy) * 100:.1f}% ***")

    # Feature importance
    print("\n" + "=" * 50)
    print("TOP 30 FEATURE IMPORTANCE", flush=True)
    print("=" * 50)
    importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    print(importance.head(30).to_string(index=False))

    # Show new features specifically
    print("\n" + "=" * 50)
    print("NEW FEATURE IMPORTANCE (v7)", flush=True)
    print("=" * 50)
    new_importance = importance[importance['Feature'].isin(all_new_features)]
    print(new_importance.to_string(index=False))

    # Save
    model_bundle = {
        'model': model,
        'features': features,
        'base_features': base_features,
        'new_features': all_new_features,
        'feature_groups': FEATURE_GROUPS,
        'technique_categories': TECHNIQUE_CATEGORIES,
        'accuracy': test_accuracy,
        'auc': test_auc,
        'brier': test_brier,
        'version': 7
    }

    print(f"\nSaving model to '{model_output_path}'...")
    joblib.dump(model_bundle, model_output_path)
    print("Done!", flush=True)

    return model_bundle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train matchup model v7.")
    parser.add_argument('--banzuke-file', type=str, default="banzuke_detailed.csv")
    parser.add_argument('--match-file', type=str, default="match_history_with_kimarite.csv")
    parser.add_argument('--model-file', type=str, default="matchup_model_v7.joblib")
    parser.add_argument('--ablation', action='store_true', help="Run ablation study")
    args = parser.parse_args()

    banzuke, matches = load_data(args.banzuke_file, args.match_file)
    matchups = build_training_data(banzuke, matches)
    train_model(matchups, args.model_file, run_ablation=args.ablation)

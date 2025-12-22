# sumo_utils.py
import pandas as pd
import numpy as np
import re

# --- NEW: Centralized Rank Parsing Utility ---
def _parse_rank(rank_str: str) -> tuple:
    """
    Parses a sumo rank string into its core components.
    Returns a tuple: (division_prefix, rank_number, side_suffix)
    Example: 'M1e' -> ('M', 1, 'e')
             'Y1w' -> ('Y', 1, 'w')
             'J13' -> ('J', 13, None)
    """
    if not isinstance(rank_str, str):
        return (None, None, None)

    # More robust regex to handle all known rank formats
    match = re.match(r'(Ms|Sd|Jd|Jk|Bg|[YOSKMJ])(\d+)?([ew])?', rank_str)

    if not match:
        return (None, None, None)

    division, number_str, side = match.groups()
    number = int(number_str) if number_str else 1 # Default to 1 if no number (e.g., Y, O)

    return (division, number, side)


# --- REFACTORED: Simplified using _parse_rank ---
def get_absolute_rank_score(rank_str: str) -> float:
    """
    Converts a sumo rank string into a single, continuous numerical score
    representing absolute standing across all divisions. Higher is better.
    """
    division, number, side = _parse_rank(rank_str)
    if not division:
        # Handle special cases or return NaN for unparseable ranks
        if isinstance(rank_str, str) and ('td' in rank_str or 'sd' in rank_str):
            return -15.0
        return np.nan

    # Division base scores
    rank_map = {
        'Y': 60, 'O': 50, 'S': 45, 'K': 42, 'M': 41,
        'J': 21, 'Ms': 1, 'Sd': -79, 'Jd': -199, 'Jk': -319, 'Bg': -400
    }
    score = rank_map.get(division, -400.0)

    # Adjust score based on number within the division
    if division in ['M', 'J', 'Ms', 'Sd', 'Jd', 'Jk']:
        score -= number

    # Apply side bonus/penalty
    if side == 'e':
        score += 0.5
    # Yokozuna/Ozeki are ranked; a west Ozeki is lower than an east Ozeki
    elif side == 'w' and division in ['Y', 'O']:
        score -= 0.5

    return float(score)


# --- REFACTORED: Simplified using _parse_rank ---
def get_divisional_rank_features(rank_str: str) -> pd.Series:
    """
    Converts a sumo rank string into two features:
    1. division_numeric: A score for the division (Makuuchi > Juryo).
    2. rank_in_division: A score for the rank within that division (higher is better).
    """
    division, number, side = _parse_rank(rank_str)
    if not division:
        return pd.Series({'division_numeric': np.nan, 'rank_in_division': np.nan})

    # Makuuchi (2), Juryo (1), others (0)
    division_map = {'Y': 2, 'O': 2, 'S': 2, 'K': 2, 'M': 2, 'J': 1}
    division_score = division_map.get(division, 0)

    # Sanyaku ranks have fixed scores, Maegashira/Juryo are calculated
    sanyaku_rank_map = {'Y': 22, 'O': 20, 'S': 19, 'K': 18}
    if division in sanyaku_rank_map:
        rank_in_division_score = sanyaku_rank_map[division]
    elif division == 'M':
        rank_in_division_score = 18 - number
    elif division == 'J':
        rank_in_division_score = 15 - number
    else:
        rank_in_division_score = 0.0 # No intra-division rank score for lower divisions

    # Apply side bonus/penalty
    if side == 'e':
        rank_in_division_score += 0.5
    elif side == 'w' and division in ['Y', 'O']:
        rank_in_division_score -= 0.5

    return pd.Series({
        'division_numeric': float(division_score),
        'rank_in_division': float(rank_in_division_score)
    })


def preprocess_data(df: pd.DataFrame, match_history_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Applies all feature engineering and cleaning steps to the DataFrame.
    Now optionally accepts a DataFrame of match history to create style and H2H features.
    """
    df = df.sort_values(by=['wrestler_id', 'tournament_id']).reset_index(drop=True)

    # --- 1. Basic Wrestler Attributes ---
    static_cols = ['birth_date', 'height_weight', 'university', 'high', 'hoshitori']
    for col in static_cols:
        if col not in df.columns:
            df[col] = np.nan
        # FIX: Use direct groupby().ffill() to avoid downcasting warning
        filled_col = df.groupby('wrestler_id')[col].ffill().bfill()
        df[col] = filled_col.infer_objects(copy=False)

    df['tournament_date'] = pd.to_datetime(df['tournament_id'], format='%Y%m')
    df['birth_date'] = pd.to_datetime(df['birth_date'], format='%d.%m.%Y', errors='coerce')
    df['age'] = (df['tournament_date'] - df['birth_date']).dt.days / 365.25

    hw_data = df['height_weight'].str.extract(r'(\d+\.?\d*)\s*cm\s*(\d+\.?\d*)\s*kg', expand=True)
    df['height_m'] = pd.to_numeric(hw_data[0], errors='coerce') / 100
    df['weight_kg'] = pd.to_numeric(hw_data[1], errors='coerce')
    df['bmi'] = df['weight_kg'] / (df['height_m'] ** 2)
    df['has_uni_sumo'] = (~df['university'].isnull()).astype(int)

    # --- 2. Rank-Based Features (CRITICAL - MUST BE EARLY) ---
    print("Generating core rank-based features...")
    df['absolute_rank_score'] = df['rank'].apply(get_absolute_rank_score)
    df['highest_rank_score'] = df['high'].apply(get_absolute_rank_score)
    df['rank_gap'] = df['highest_rank_score'] - df['absolute_rank_score']

    rank_features = df['rank'].apply(get_divisional_rank_features)
    df = df.join(rank_features) # More efficient than assigning columns one by one

    df['prev'] = df.groupby('wrestler_id')['rank'].shift(1)
    prev_rank_features = df['prev'].apply(get_divisional_rank_features)
    df = df.join(prev_rank_features.add_prefix('prev_'))

    # --- 3. Performance & Momentum Features ---
    df['w'] = pd.to_numeric(df.get('w'), errors='coerce')
    df['l'] = pd.to_numeric(df.get('l'), errors='coerce')
    df['prev_w'] = df.groupby('wrestler_id')['w'].shift(1)
    df['prev_l'] = df.groupby('wrestler_id')['l'].shift(1)

    if 'hoshitori' in df.columns and not df['hoshitori'].isnull().all():
        def count_absences(hoshitori_str: str) -> int:
            if not isinstance(hoshitori_str, str): return 0
            return hoshitori_str.count('-')
        df['absences'] = df['hoshitori'].apply(count_absences)
        df['prev_absences'] = df.groupby('wrestler_id')['absences'].shift(1).fillna(0)
        df['was_kyujo_last_basho'] = (df['prev_absences'] > 0).astype(int)
        df = df.drop(columns=['absences', 'prev_absences'])
    else:
        df['was_kyujo_last_basho'] = 0

    df['kachi_koshi_flag'] = (df['w'] > df['l']).astype(int)
    streak_groups = (df.groupby('wrestler_id')['kachi_koshi_flag'].diff() != 0).cumsum()
    cumulative_streak = df.groupby(['wrestler_id', streak_groups])['kachi_koshi_flag'].cumsum()
    current_streak = cumulative_streak * df['kachi_koshi_flag']
    df['kachi_koshi_streak'] = current_streak.groupby(df['wrestler_id']).shift(1).fillna(0)
    df = df.drop(columns=['kachi_koshi_flag'])

    df['win_consistency'] = df.groupby('wrestler_id')['w'].transform(lambda x: x.rolling(window=6, min_periods=3).std().shift(1))

    # --- 4. Contextual Features (Division & Heya Strength) ---
    print("Generating contextual strength features...")
    df['heya_strength'] = df.groupby(['tournament_id', 'heya'])['absolute_rank_score'].transform('mean')

    div_stats = df.groupby(['tournament_id', 'division_numeric'])['absolute_rank_score'].agg(
        sum_strength='sum',
        count_strength='count'
    ).reset_index()
    df = pd.merge(df, div_stats, on=['tournament_id', 'division_numeric'], how='left')
    # Handle division with only one wrestler
    df['count_strength'] = df['count_strength'].replace(1, 0)
    df['division_strength'] = (df['sum_strength'] - df['absolute_rank_score']) / (df['count_strength'] - 1)
    df['division_strength'] = df['division_strength'].fillna(0)
    df = df.drop(columns=['sum_strength', 'count_strength'])

    # --- 5. Advanced Features from Match History ---
    if match_history_df is not None and not match_history_df.empty:
        # --- Fighting Style Profile (Oshi Ratio) ---
        print("Generating 'oshi_ratio' style feature...")
        oshi_techniques = ['oshidashi', 'tsukidashi', 'tsukiotoshi', 'hatakikomi', 'oshitaoshi', 'tsukitaoshi']
        match_history_df['is_oshi'] = match_history_df['kimarite'].isin(oshi_techniques).astype(int)
        yotsu_techniques = ['yorikiri', 'uwatenage', 'shitatenage', 'yoritaoshi', 'kotenage', 'sukuinage', 'utchari']
        match_history_df['is_yotsu'] = match_history_df['kimarite'].isin(yotsu_techniques).astype(int)

        match_history_df = match_history_df.sort_values(by=['winner_id', 'tournament_id', 'day'])
        match_history_df['cumulative_oshi'] = match_history_df.groupby('winner_id')['is_oshi'].cumsum()
        match_history_df['cumulative_yotsu'] = match_history_df.groupby('winner_id')['is_yotsu'].cumsum()

        style_history = match_history_df.groupby(['winner_id', 'tournament_id'])[['cumulative_oshi', 'cumulative_yotsu']].last().reset_index()
        style_history.rename(columns={'winner_id': 'wrestler_id'}, inplace=True)
        style_history['prev_cumulative_oshi'] = style_history.groupby('wrestler_id')['cumulative_oshi'].shift(1)
        style_history['prev_cumulative_yotsu'] = style_history.groupby('wrestler_id')['cumulative_yotsu'].shift(1)

        df = pd.merge(df, style_history[['tournament_id', 'wrestler_id', 'prev_cumulative_oshi', 'prev_cumulative_yotsu']], on=['tournament_id', 'wrestler_id'], how='left')
        df['prev_cumulative_oshi'] = df.groupby('wrestler_id')['prev_cumulative_oshi'].ffill().fillna(0)
        df['prev_cumulative_yotsu'] = df.groupby('wrestler_id')['prev_cumulative_yotsu'].ffill().fillna(0)

        total_style_wins = df['prev_cumulative_oshi'] + df['prev_cumulative_yotsu']
        df['oshi_ratio'] = np.where(total_style_wins > 0, df['prev_cumulative_oshi'] / total_style_wins, 0.5)
        df = df.drop(columns=['prev_cumulative_oshi', 'prev_cumulative_yotsu'])

        # --- Opponent Style Profile ---
        print("Generating 'avg_opponent_oshi_ratio' feature...")
        style_stats = df.groupby(['tournament_id', 'division_numeric'])['oshi_ratio'].agg(
            sum_style='sum',
            count_style='count'
        ).reset_index()
        df = pd.merge(df, style_stats, on=['tournament_id', 'division_numeric'], how='left')
        # Handle division with only one wrestler
        df['count_style'] = df['count_style'].replace(1, 0)
        df['avg_opponent_oshi_ratio'] = (df['sum_style'] - df['oshi_ratio']) / (df['count_style'] - 1)
        df['avg_opponent_oshi_ratio'] = df['avg_opponent_oshi_ratio'].fillna(0.5)
        df = df.drop(columns=['sum_style', 'count_style'])

        # --- Average Head-to-Head Win Percentage ---
        print("Generating 'avg_h2h_win_pct' feature...")
        matches = match_history_df.copy()
        matches['w1'] = np.minimum(matches['winner_id'], matches['loser_id'])
        matches['w2'] = np.maximum(matches['winner_id'], matches['loser_id'])
        matches['w1_won'] = (matches['w1'] == matches['winner_id']).astype(int)

        matches = matches.sort_values(by=['w1', 'w2', 'tournament_id', 'day'])
        matches['w1_cum_wins'] = matches.groupby(['w1', 'w2'])['w1_won'].cumsum()
        matches['total_matches'] = matches.groupby(['w1', 'w2']).cumcount() + 1
        matches['w2_cum_wins'] = matches['total_matches'] - matches['w1_cum_wins']

        h2h_history = matches.groupby(['w1', 'w2', 'tournament_id'])[['w1_cum_wins', 'w2_cum_wins']].last().reset_index()
        h2h_history['prev_w1_wins'] = h2h_history.groupby(['w1', 'w2'])['w1_cum_wins'].shift(1).fillna(0)
        h2h_history['prev_w2_wins'] = h2h_history.groupby(['w1', 'w2'])['w2_cum_wins'].shift(1).fillna(0)

        rosters = df[['tournament_id', 'division_numeric', 'wrestler_id']].dropna()
        matchups = pd.merge(rosters, rosters, on=['tournament_id', 'division_numeric'], suffixes=('_A', '_B'))
        matchups = matchups[matchups['wrestler_id_A'] != matchups['wrestler_id_B']]
        matchups['w1'] = np.minimum(matchups['wrestler_id_A'], matchups['wrestler_id_B'])
        matchups['w2'] = np.maximum(matchups['wrestler_id_A'], matchups['wrestler_id_B'])

        matchups = pd.merge(matchups, h2h_history, on=['w1', 'w2', 'tournament_id'], how='left')
        matchups['prev_w1_wins'] = matchups.groupby(['w1', 'w2'])['prev_w1_wins'].ffill().fillna(0)
        matchups['prev_w2_wins'] = matchups.groupby(['w1', 'w2'])['prev_w2_wins'].ffill().fillna(0)

        w1_is_A = (matchups['w1'] == matchups['wrestler_id_A']).astype(int)
        wins_A = w1_is_A * matchups['prev_w1_wins'] + (1 - w1_is_A) * matchups['prev_w2_wins']
        losses_A = (1 - w1_is_A) * matchups['prev_w1_wins'] + w1_is_A * matchups['prev_w2_wins']
        total_h2h_matches = wins_A + losses_A
        matchups['h2h_win_pct_A'] = np.where(total_h2h_matches > 0, wins_A / total_h2h_matches, 0.5)

        avg_h2h = matchups.groupby(['tournament_id', 'wrestler_id_A'])['h2h_win_pct_A'].mean().reset_index()
        avg_h2h.rename(columns={'wrestler_id_A': 'wrestler_id', 'h2h_win_pct_A': 'avg_h2h_win_pct'}, inplace=True)

        df = pd.merge(df, avg_h2h, on=['tournament_id', 'wrestler_id'], how='left')
        # FIX: Avoid inplace modification on a potential copy to fix SettingWithCopyWarning.
        df['avg_h2h_win_pct'] = df['avg_h2h_win_pct'].fillna(0.5)
    else:
        print("Warning: Match history data not provided. Style and H2H features will be set to defaults.")
        df['oshi_ratio'] = 0.5
        df['avg_opponent_oshi_ratio'] = 0.5
        df['avg_h2h_win_pct'] = 0.5

    # --- 6. Final Cleanup ---
    df = df.drop(columns=['absolute_rank_score', 'highest_rank_score', 'prev'], errors='ignore')

    return df

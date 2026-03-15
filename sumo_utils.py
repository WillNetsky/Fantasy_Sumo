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


def compute_elo_features(match_history_df: pd.DataFrame, k_factor: float = 32, initial_elo: float = 1500) -> pd.DataFrame:
    """
    Computes Elo-based features for each (wrestler_id, tournament_id):
      - elo: Elo rating going INTO this tournament
      - elo_change_prev_basho: Elo delta during the previous tournament
      - elo_change_last_2_bouts: Elo delta over the last 2 bouts before this tournament
    """
    mh = match_history_df.sort_values(['tournament_id', 'day']).reset_index(drop=True)

    current_elo = {}
    # wrestler_id -> list of (tournament_id, delta) in chronological order
    bout_deltas = {}

    for row in mh.itertuples(index=False):
        winner, loser, tournament = row.winner_id, row.loser_id, row.tournament_id

        elo_w = current_elo.get(winner, initial_elo)
        elo_l = current_elo.get(loser, initial_elo)

        exp_w = 1 / (1 + 10 ** ((elo_l - elo_w) / 400))
        delta_w = k_factor * (1 - exp_w)
        delta_l = -k_factor * exp_w

        current_elo[winner] = elo_w + delta_w
        current_elo[loser] = elo_l + delta_l

        bout_deltas.setdefault(winner, []).append((tournament, delta_w))
        bout_deltas.setdefault(loser, []).append((tournament, delta_l))

    BOUT_WINDOWS = [2, 5, 15]
    BASHO_WINDOWS = [1, 2]

    records = []
    for wrestler_id, deltas in bout_deltas.items():
        # Group deltas by tournament (preserving order)
        by_tournament = {}
        for tid, delta in deltas:
            by_tournament.setdefault(tid, []).append(delta)

        running_elo = initial_elo
        all_deltas_so_far = []   # flat list of all bout deltas in order
        basho_changes = []       # per-tournament net Elo change, in order

        for tid in sorted(by_tournament):
            t_deltas = by_tournament[tid]
            t_change = sum(t_deltas)
            running_elo += t_change
            all_deltas_so_far.extend(t_deltas)
            basho_changes.append(t_change)

            record = {
                'wrestler_id': wrestler_id,
                'tournament_id': tid,
                'elo_end': running_elo,
            }
            for n in BOUT_WINDOWS:
                record[f'elo_change_last_{n}_bouts'] = sum(all_deltas_so_far[-n:])
            for m in BASHO_WINDOWS:
                record[f'elo_change_last_{m}_basho'] = sum(basho_changes[-m:])

            records.append(record)

    result = pd.DataFrame(records).sort_values(['wrestler_id', 'tournament_id'])
    g = result.groupby('wrestler_id')
    result['elo'] = g['elo_end'].shift(1).fillna(initial_elo)

    bout_cols = [f'elo_change_last_{n}_bouts' for n in BOUT_WINDOWS]
    basho_cols = [f'elo_change_last_{m}_basho' for m in BASHO_WINDOWS]
    for col in bout_cols + basho_cols:
        result[col] = g[col].shift(1).fillna(0.0)

    return result[['tournament_id', 'wrestler_id', 'elo'] + bout_cols + basho_cols]


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
        # FIX: Use transform to keep ffill and bfill within each wrestler's group
        filled_col = df.groupby('wrestler_id')[col].transform(lambda x: x.ffill().bfill())
        df[col] = filled_col

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

    rank_features = df['rank'].apply(get_divisional_rank_features).reset_index(drop=True)
    df = df.reset_index(drop=True)
    df = pd.concat([df, rank_features], axis=1)

    df['prev'] = df.groupby('wrestler_id')['rank'].shift(1)
    prev_rank_features = df['prev'].apply(get_divisional_rank_features).reset_index(drop=True)
    prev_rank_features = prev_rank_features.add_prefix('prev_')
    df = pd.concat([df, prev_rank_features], axis=1)

    # --- 3. Performance & Momentum Features ---
    df['w'] = pd.to_numeric(df.get('w'), errors='coerce')
    df['l'] = pd.to_numeric(df.get('l'), errors='coerce')
    df['prev_w'] = df.groupby('wrestler_id')['w'].shift(1)
    df['prev_l'] = df.groupby('wrestler_id')['l'].shift(1)

    # --- Injury Risk Features ---
    print("Generating injury risk features...")
    if 'hoshitori' in df.columns and not df['hoshitori'].isnull().all():
        def count_absences(hoshitori_str: str) -> int:
            if not isinstance(hoshitori_str, str): return 0
            return hoshitori_str.count('-')

        def count_scheduled_bouts(hoshitori_str: str) -> int:
            """Count total scheduled bouts (all characters except spaces)"""
            if not isinstance(hoshitori_str, str): return 15  # Default to 15 for top division
            return len([c for c in hoshitori_str if c in 'OoSsXx*-#'])

        df['absences'] = df['hoshitori'].apply(count_absences)
        df['scheduled_bouts'] = df['hoshitori'].apply(count_scheduled_bouts)
        df['bouts_completed'] = df['scheduled_bouts'] - df['absences']

        # Previous basho absences
        df['prev_absences'] = df.groupby('wrestler_id')['absences'].shift(1).fillna(0)
        df['was_kyujo_last_basho'] = (df['prev_absences'] > 0).astype(int)

        # Absences in last 3 basho (rolling sum)
        df['absences_last_3'] = df.groupby('wrestler_id')['absences'].transform(
            lambda x: x.shift(1).rolling(window=3, min_periods=1).sum()
        ).fillna(0)

        # Completion rate in last 3 basho
        df['completion_rate'] = df['bouts_completed'] / df['scheduled_bouts']
        df['completion_rate_last_3'] = df.groupby('wrestler_id')['completion_rate'].transform(
            lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
        ).fillna(1.0)  # Default to 100% if no history

        # Age × previous absences interaction (older + injured = higher risk)
        df['age_x_prev_absences'] = df['age'] * df['prev_absences']

        # Basho since last full completion (no absences)
        def calc_basho_since_full(group):
            result = []
            basho_count = 0
            for absences in group:
                result.append(basho_count)
                if absences == 0:
                    basho_count = 0
                else:
                    basho_count += 1
            return result

        df['basho_since_full'] = df.groupby('wrestler_id')['absences'].transform(
            lambda x: pd.Series(calc_basho_since_full(x.values), index=x.index)
        ).shift(1).fillna(0)

        # Clean up intermediate columns
        df = df.drop(columns=['absences', 'scheduled_bouts', 'bouts_completed', 'completion_rate'])
    else:
        df['was_kyujo_last_basho'] = 0
        df['prev_absences'] = 0
        df['absences_last_3'] = 0
        df['completion_rate_last_3'] = 1.0
        df['age_x_prev_absences'] = 0
        df['basho_since_full'] = 0

    df['kachi_koshi_flag'] = (df['w'] > df['l']).astype(int)
    streak_groups = (df.groupby('wrestler_id')['kachi_koshi_flag'].diff() != 0).cumsum()
    cumulative_streak = df.groupby(['wrestler_id', streak_groups])['kachi_koshi_flag'].cumsum()
    # Use .values to avoid index misalignment between MultiIndex result and DataFrame
    current_streak = cumulative_streak.values * df['kachi_koshi_flag'].values
    df['kachi_koshi_streak'] = pd.Series(current_streak, index=df.index).groupby(df['wrestler_id']).shift(1).fillna(0)
    df = df.drop(columns=['kachi_koshi_flag'])

    df['win_consistency'] = df.groupby('wrestler_id')['w'].transform(lambda x: x.rolling(window=6, min_periods=3).std().shift(1))

    # --- Rank Trajectory Features ---
    print("Generating rank trajectory features...")

    # Rank momentum: mean rank score change over last 3 basho (positive = improving)
    rank_delta = df.groupby('wrestler_id')['absolute_rank_score'].diff()
    df['rank_momentum'] = (
        rank_delta.groupby(df['wrestler_id'])
        .transform(lambda x: x.rolling(window=3, min_periods=1).mean().shift(1))
        .fillna(0)
    )

    # Basho since peak: how many tournaments since rank_gap was last ~0
    # (i.e. since they were at their career-best rank)
    def _basho_since_peak(rank_gaps):
        result = []
        count = 0
        for gap in rank_gaps:
            result.append(count)
            if gap <= 0.6:  # at or within east/west of career best
                count = 0
            else:
                count += 1
        return result

    df['basho_since_peak'] = (
        df.groupby('wrestler_id')['rank_gap']
        .transform(lambda x: pd.Series(
            _basho_since_peak(x.values), index=x.index
        ))
        .shift(1)
        .fillna(0)
    )

    # --- 4. Contextual Features (Division & Heya Strength) ---
    print("Generating contextual strength features...")
    df['heya_strength'] = df.groupby(['tournament_id', 'heya'])['absolute_rank_score'].transform('mean')

    div_stats = df.groupby(['tournament_id', 'division_numeric'])['absolute_rank_score'].agg(
        sum_strength='sum',
        count_strength='count'
    ).reset_index()
    df = pd.merge(df, div_stats, on=['tournament_id', 'division_numeric'], how='left')
    # Handle division with only one wrestler using np.where to avoid division by zero/negative
    df['division_strength'] = np.where(
        df['count_strength'] > 1,
        (df['sum_strength'] - df['absolute_rank_score']) / (df['count_strength'] - 1),
        0.0
    )
    df = df.drop(columns=['sum_strength', 'count_strength'])

    # --- 5. Advanced Features from Match History ---
    if match_history_df is not None and not match_history_df.empty:
        # --- Consecutive Upset Losses Feature ---
        print("Generating 'consec_upset_losses' feature...")
        mh_upset = match_history_df[['tournament_id', 'day', 'winner_id', 'winner_rank', 'loser_id', 'loser_rank']].copy()
        mh_upset['winner_rank_score'] = mh_upset['winner_rank'].apply(get_absolute_rank_score)
        mh_upset['loser_rank_score'] = mh_upset['loser_rank'].apply(get_absolute_rank_score)
        # Upset = winner was lower-ranked (smaller score) than loser
        mh_upset['is_upset'] = (mh_upset['winner_rank_score'] < mh_upset['loser_rank_score']).astype(int)

        # Build full bout timeline from both winner and loser perspectives
        winner_bouts = mh_upset[['tournament_id', 'day', 'winner_id']].copy()
        winner_bouts.rename(columns={'winner_id': 'wrestler_id'}, inplace=True)
        winner_bouts['is_upset_loss'] = 0

        loser_bouts = mh_upset[['tournament_id', 'day', 'loser_id', 'is_upset']].copy()
        loser_bouts.rename(columns={'loser_id': 'wrestler_id', 'is_upset': 'is_upset_loss'}, inplace=True)

        all_bouts = pd.concat([winner_bouts, loser_bouts], ignore_index=True)
        all_bouts = all_bouts.sort_values(['wrestler_id', 'tournament_id', 'day']).reset_index(drop=True)

        # Compute trailing consecutive upset-loss streak (resets on any non-upset-loss bout)
        def _consec_streak(vals):
            result = []
            streak = 0
            for v in vals:
                streak = streak + 1 if v else 0
                result.append(streak)
            return result

        all_bouts['consec_streak'] = all_bouts.groupby('wrestler_id')['is_upset_loss'].transform(
            lambda x: pd.Series(_consec_streak(x.values), index=x.index)
        )

        # Take the streak value at the end of each tournament, then shift forward
        # so the feature reflects the state *going into* the next basho
        eob_streak = all_bouts.groupby(['wrestler_id', 'tournament_id'])['consec_streak'].last().reset_index()
        eob_streak['consec_upset_losses'] = (
            eob_streak.groupby('wrestler_id')['consec_streak'].shift(1).fillna(0).astype(int)
        )
        eob_streak = eob_streak[['tournament_id', 'wrestler_id', 'consec_upset_losses']]

        df = pd.merge(df, eob_streak, on=['tournament_id', 'wrestler_id'], how='left')
        df['consec_upset_losses'] = df['consec_upset_losses'].fillna(0).astype(int)

        # --- Elo Features ---
        print("Generating Elo features...")
        elo_features = compute_elo_features(match_history_df)
        df = pd.merge(df, elo_features, on=['tournament_id', 'wrestler_id'], how='left')
        elo_cols = ['elo'] + [f'elo_change_last_{n}_bouts' for n in [2, 5, 15]] + \
                   [f'elo_change_last_{m}_basho' for m in [1, 2]]
        df['elo'] = df['elo'].fillna(1500.0)
        for col in elo_cols[1:]:
            df[col] = df[col].fillna(0.0)

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
        # Handle division with only one wrestler using np.where to avoid division by zero/negative
        df['avg_opponent_oshi_ratio'] = np.where(
            df['count_style'] > 1,
            (df['sum_style'] - df['oshi_ratio']) / (df['count_style'] - 1),
            0.5
        )
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
        df['consec_upset_losses'] = 0
        df['elo'] = 1500.0
        for col in [f'elo_change_last_{n}_bouts' for n in [2, 5, 15]] + \
                   [f'elo_change_last_{m}_basho' for m in [1, 2]]:
            df[col] = 0.0
        df['oshi_ratio'] = 0.5
        df['avg_opponent_oshi_ratio'] = 0.5
        df['avg_h2h_win_pct'] = 0.5

    # --- 6. Final Cleanup ---
    df = df.drop(columns=['absolute_rank_score', 'highest_rank_score', 'prev'], errors='ignore')

    return df

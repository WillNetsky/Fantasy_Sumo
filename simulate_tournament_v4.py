# simulate_tournament_v4.py
"""
Tournament simulator v4 with:
1. Multi-segment torikumi scheduling (5 day segments)
2. Banzuke prediction for future tournaments
3. Tournament chaining for multi-basho forecasts
4. Improved JSA-style scheduling rules
"""

import pandas as pd
import numpy as np
import joblib
import argparse
from typing import Dict, List, Tuple, Optional

from sumo_utils import get_absolute_rank_score, _parse_rank, compute_elo_features
from torikumi_scheduler import (
    WrestlerState, generate_torikumi, create_wrestler_states_from_df,
    get_segment, SchedulingSegment
)
from banzuke_predictor import predict_next_banzuke, RankPosition

# Import technique categories
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


def load_data(matchup_model_path: str, banzuke_path: str, match_history_path: str,
              direct_model_path: Optional[str] = None):
    """Load models and data."""
    print("Loading matchup model...")
    matchup_bundle = joblib.load(matchup_model_path)
    print(f"  Version: v{matchup_bundle.get('version', 'unknown')}")
    print(f"  Accuracy: {matchup_bundle.get('accuracy', 'N/A'):.4f}")

    direct_bundle = None
    if direct_model_path:
        print("Loading direct win model...")
        direct_bundle = joblib.load(direct_model_path)

    print("Loading historical data...")
    banzuke = pd.read_csv(banzuke_path)
    match_history = pd.read_csv(match_history_path)

    for df in [banzuke, match_history]:
        for col in ['wrestler_id', 'winner_id', 'loser_id', 'tournament_id']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

    return matchup_bundle, direct_bundle, banzuke, match_history


def prepare_wrestlers(banzuke: pd.DataFrame, match_history: pd.DataFrame,
                      tournament_id: int) -> pd.DataFrame:
    """Prepare all wrestler features for a tournament."""

    roster = banzuke[banzuke['tournament_id'] == tournament_id].copy()

    def is_top_div(r):
        d, _, _ = _parse_rank(r)
        return d in ['Y', 'O', 'S', 'K', 'M', 'J']

    roster = roster[roster['rank'].apply(is_top_div)]
    if roster.empty:
        return pd.DataFrame()

    hist_banzuke = banzuke[banzuke['tournament_id'] < tournament_id]
    hist_matches = match_history[match_history['tournament_id'] < tournament_id]

    wrestlers = []
    for _, row in roster.iterrows():
        w_id = row['wrestler_id']
        w_hist = hist_banzuke[hist_banzuke['wrestler_id'] == w_id]
        w_matches_won = hist_matches[hist_matches['winner_id'] == w_id]
        w_matches_lost = hist_matches[hist_matches['loser_id'] == w_id]

        # Rank
        rank_score = get_absolute_rank_score(row['rank'])
        div, num, side = _parse_rank(row['rank'])
        sanyaku_map = {'Y': 22, 'O': 20, 'S': 19, 'K': 18}
        if div in sanyaku_map:
            rank_in_div = sanyaku_map[div]
        elif div == 'M':
            rank_in_div = 18 - num
        elif div == 'J':
            rank_in_div = 15 - num
        else:
            rank_in_div = 0
        if side == 'e':
            rank_in_div += 0.5

        # Division
        if div in ['Y', 'O', 'S', 'K', 'M']:
            division = 'makuuchi'
        elif div == 'J':
            division = 'juryo'
        else:
            division = 'other'

        # Is sanyaku
        is_sanyaku = 1 if div in ['Y', 'O', 'S', 'K'] else 0

        # Physical
        hw = pd.Series([row.get('height_weight', '')]).str.extract(r'(\d+\.?\d*)\s*cm\s*(\d+\.?\d*)\s*kg')
        height = pd.to_numeric(hw[0].iloc[0], errors='coerce') if len(hw) > 0 else 175.0
        weight = pd.to_numeric(hw[1].iloc[0], errors='coerce') if len(hw) > 0 else 130.0
        bmi = weight / ((height / 100) ** 2) if height and weight else 40.0

        # Age
        bd = pd.to_datetime(row.get('birth_date'), format='%d.%m.%Y', errors='coerce')
        td = pd.to_datetime(str(tournament_id), format='%Y%m')
        age = (td - bd).days / 365.25 if pd.notna(bd) else 28.0

        # Recent form
        if not w_hist.empty:
            recent = w_hist.tail(3)
            rw = pd.to_numeric(recent['w'], errors='coerce').sum()
            rl = pd.to_numeric(recent['l'], errors='coerce').sum()
            win_pct_last_3 = rw / (rw + rl) if (rw + rl) > 0 else 0.5

            recent_6 = w_hist.tail(6)
            rw6 = pd.to_numeric(recent_6['w'], errors='coerce').sum()
            rl6 = pd.to_numeric(recent_6['l'], errors='coerce').sum()
            win_pct_last_6 = rw6 / (rw6 + rl6) if (rw6 + rl6) > 0 else 0.5
        else:
            win_pct_last_3 = 0.5
            win_pct_last_6 = 0.5

        # Style ratios (from wins)
        style_counts = {cat: 0 for cat in TECHNIQUE_CATEGORIES}
        total_style = 0
        for _, m in w_matches_won.iterrows():
            cat = TECHNIQUE_TO_CATEGORY.get(m['kimarite'])
            if cat:
                style_counts[cat] += 1
                total_style += 1

        style_ratios = {}
        for cat in TECHNIQUE_CATEGORIES:
            style_ratios[f'{cat}_ratio'] = style_counts[cat] / total_style if total_style > 0 else 0.2

        # Vulnerability ratios (from losses)
        vuln_counts = {cat: 0 for cat in TECHNIQUE_CATEGORIES}
        total_vuln = 0
        for _, m in w_matches_lost.iterrows():
            cat = TECHNIQUE_TO_CATEGORY.get(m['kimarite'])
            if cat:
                vuln_counts[cat] += 1
                total_vuln += 1

        vuln_ratios = {}
        for cat in TECHNIQUE_CATEGORIES:
            vuln_ratios[f'vuln_{cat}'] = vuln_counts[cat] / total_vuln if total_vuln > 0 else 0.2

        # Simple ELO approximation based on rank and recent form
        elo = 1500 + (rank_score - 40) * 10 + (win_pct_last_3 - 0.5) * 100

        wrestlers.append({
            'wrestler_id': w_id,
            'rikishi': row['rikishi'],
            'rank': row['rank'],
            'rank_score': rank_score,
            'rank_in_div': rank_in_div,
            'is_sanyaku': is_sanyaku,
            'division': division,
            'height': height,
            'weight': weight,
            'bmi': bmi,
            'age': age,
            'win_pct_last_3': win_pct_last_3,
            'win_pct_last_6': win_pct_last_6,
            'elo': elo,
            'heya': row.get('heya', ''),
            **style_ratios,
            **vuln_ratios
        })

    df = pd.DataFrame(wrestlers)

    for col in ['height', 'weight', 'bmi', 'age']:
        df[col] = df[col].fillna(df[col].median())

    return df


def compute_h2h(match_history: pd.DataFrame, tournament_id: int) -> Dict:
    """Get H2H records."""
    hist = match_history[match_history['tournament_id'] < tournament_id]
    h2h = {}
    for _, m in hist.iterrows():
        w1 = min(m['winner_id'], m['loser_id'])
        w2 = max(m['winner_id'], m['loser_id'])
        key = (w1, w2)
        if key not in h2h:
            h2h[key] = {'w1': 0, 'w2': 0}
        if m['winner_id'] == w1:
            h2h[key]['w1'] += 1
        else:
            h2h[key]['w2'] += 1
    return h2h


def get_h2h_stats(h2h: Dict, a_id: int, b_id: int) -> Tuple[float, int]:
    """Get A's win % vs B and total matches."""
    w1 = min(a_id, b_id)
    w2 = max(a_id, b_id)
    key = (w1, w2)
    if key not in h2h:
        return 0.5, 0
    rec = h2h[key]
    total = rec['w1'] + rec['w2']
    if a_id == w1:
        return rec['w1'] / total, total
    else:
        return rec['w2'] / total, total


def build_matchup_features(model_version: int, a: dict, b: dict,
                            h2h_pct_a: float, h2h_total: int,
                            basho_wins_a: int = 0, basho_losses_a: int = 0,
                            basho_wins_b: int = 0, basho_losses_b: int = 0,
                            day: int = 1) -> dict:
    """Build features dict for matchup prediction based on model version."""

    # Common features
    f = {
        'rank_diff': a['rank_score'] - b['rank_score'],
        'rank_in_div_diff': a['rank_in_div'] - b['rank_in_div'],
        'bmi_diff': a['bmi'] - b['bmi'],
        'height_diff': a['height'] - b['height'],
        'weight_diff': a['weight'] - b['weight'],
        'age_diff': a['age'] - b['age'],
        'h2h_win_pct_A': h2h_pct_a,
        'h2h_total_matches': h2h_total,
        'win_pct_last_3_A': a['win_pct_last_3'],
        'win_pct_last_3_B': b['win_pct_last_3'],
        'win_pct_last_3_diff': a['win_pct_last_3'] - b['win_pct_last_3'],
    }

    # V3+ features
    if model_version >= 3:
        f['elo_diff'] = a.get('elo', 1500) - b.get('elo', 1500)
        f['elo_expected_A'] = 1 / (1 + 10 ** ((b.get('elo', 1500) - a.get('elo', 1500)) / 400))
        f['momentum_diff'] = a.get('win_pct_last_3', 0.5) - b.get('win_pct_last_3', 0.5)
        f['win_streak_diff'] = 0
        f['momentum_A'] = a.get('win_pct_last_3', 0.5)
        f['momentum_B'] = b.get('win_pct_last_3', 0.5)
        f['win_streak_A'] = 0
        f['win_streak_B'] = 0
        f['win_pct_last_6_A'] = a.get('win_pct_last_6', 0.5)
        f['win_pct_last_6_B'] = b.get('win_pct_last_6', 0.5)
        f['win_pct_last_6_diff'] = a.get('win_pct_last_6', 0.5) - b.get('win_pct_last_6', 0.5)

    # V4+ features
    if model_version >= 4:
        f['has_h2h_history'] = 1 if h2h_total > 0 else 0
        f['basho_win_pct_A'] = basho_wins_a / (basho_wins_a + basho_losses_a + 1)
        f['basho_win_pct_B'] = basho_wins_b / (basho_wins_b + basho_losses_b + 1)
        f['basho_momentum_diff'] = f['basho_win_pct_A'] - f['basho_win_pct_B']
        f['day_normalized'] = day / 15.0
        f['is_late_basho'] = 1 if day >= 10 else 0
        f['experience_diff'] = 0
        f['newcomer_vs_veteran'] = 0
        f['sanyaku_diff'] = a.get('is_sanyaku', 0) - b.get('is_sanyaku', 0)
        f['is_sanyaku_A'] = a.get('is_sanyaku', 0)
        f['is_sanyaku_B'] = b.get('is_sanyaku', 0)
        f['is_peak_age_A'] = 1 if 24 <= a.get('age', 28) <= 30 else 0
        f['is_peak_age_B'] = 1 if 24 <= b.get('age', 28) <= 30 else 0
        f['is_veteran_A'] = 1 if a.get('age', 28) > 32 else 0
        f['is_veteran_B'] = 1 if b.get('age', 28) > 32 else 0
        f['career_win_pct_diff'] = 0

    # V5 features
    if model_version >= 5:
        f['glicko_diff'] = a.get('elo', 1500) - b.get('elo', 1500)
        f['glicko_expected_A'] = f.get('elo_expected_A', 0.5)
        f['glicko_combined_rd'] = 200
        f['form_diff'] = a.get('win_pct_last_3', 0.5) - b.get('win_pct_last_3', 0.5)
        f['streak_diff'] = 0
        f['volatility_diff'] = 0
        f['weighted_form_A'] = a.get('win_pct_last_3', 0.5)
        f['weighted_form_B'] = b.get('win_pct_last_3', 0.5)
        f['recent_streak_A'] = 0
        f['recent_streak_B'] = 0
        h2h_significance = 1 - 1 / (1 + h2h_total / 3)
        f['h2h_weighted_A'] = (h2h_pct_a - 0.5) * h2h_significance + 0.5
        f['h2h_significance'] = h2h_significance

    # Style ratios for both wrestlers
    for cat in TECHNIQUE_CATEGORIES:
        f[f'{cat}_ratio_A'] = a.get(f'{cat}_ratio', 0.2)
        f[f'{cat}_ratio_B'] = b.get(f'{cat}_ratio', 0.2)

    # Vulnerability ratios
    for cat in TECHNIQUE_CATEGORIES:
        f[f'vuln_{cat}_A'] = a.get(f'vuln_{cat}', 0.2)
        f[f'vuln_{cat}_B'] = b.get(f'vuln_{cat}', 0.2)

    # Style advantage features
    for cat in TECHNIQUE_CATEGORIES:
        adv_a = a.get(f'{cat}_ratio', 0.2) * b.get(f'vuln_{cat}', 0.2)
        adv_b = b.get(f'{cat}_ratio', 0.2) * a.get(f'vuln_{cat}', 0.2)
        f[f'{cat}_advantage_diff'] = adv_a - adv_b

    total_adv_a = sum(a.get(f'{c}_ratio', 0.2) * b.get(f'vuln_{c}', 0.2) for c in TECHNIQUE_CATEGORIES)
    total_adv_b = sum(b.get(f'{c}_ratio', 0.2) * a.get(f'vuln_{c}', 0.2) for c in TECHNIQUE_CATEGORIES)
    f['style_advantage_diff'] = total_adv_a - total_adv_b

    return f


def predict_match(model, features: List[str], f: dict) -> float:
    """Predict P(A wins)."""
    X = pd.DataFrame([f])[features]
    return model.predict_proba(X)[0][1]


def simulate_tournament(model_bundle: dict, wrestlers_df: pd.DataFrame,
                        h2h: Dict, n_sims: int = 500,
                        scheduling: str = 'jsa_v2',
                        verbose: bool = True,
                        h2h_recent: Dict = None) -> pd.DataFrame:
    """
    Run tournament simulation with improved multi-segment scheduling.

    Args:
        model_bundle: Trained matchup model
        wrestlers_df: Prepared wrestler DataFrame
        h2h: Head-to-head history dict
        n_sims: Number of simulations
        scheduling: 'jsa_v2' (5-segment), 'jsa' (2-segment), or 'simple'
        verbose: Print progress

    Returns:
        DataFrame with expected wins and probabilities
    """
    model = model_bundle['model']
    features = model_bundle['features']
    model_version = model_bundle.get('version', 2)

    w_ids = wrestlers_df['wrestler_id'].tolist()
    w_dict = wrestlers_df.set_index('wrestler_id').to_dict('index')

    total_wins = {w: 0 for w in w_ids}
    yusho_count = {w: 0 for w in w_ids}
    jun_yusho_count = {w: 0 for w in w_ids}
    kachi_koshi_count = {w: 0 for w in w_ids}

    if verbose:
        print(f"Running {n_sims} simulations with '{scheduling}' scheduling...")

    for sim in range(n_sims):
        if verbose and (sim + 1) % 100 == 0:
            print(f"  {sim + 1}/{n_sims}")

        # Initialize wrestler states for this simulation
        if scheduling == 'jsa_v2':
            wrestler_states = create_wrestler_states_from_df(wrestlers_df)
            w_state_dict = {w.wrestler_id: w for w in wrestler_states}

        records = {w: (0, 0) for w in w_ids}
        faced = {w: set() for w in w_ids}
        streaks = {w: 0 for w in w_ids}  # signed streak: +N wins / -N losses

        for day in range(1, 16):
            if scheduling == 'jsa_v2':
                # Use new multi-segment scheduler
                matchups = generate_torikumi(wrestler_states, day)
            elif scheduling == 'jsa':
                # Use old 2-segment scheduler (from v3)
                matchups = generate_jsa_torikumi(wrestlers_df, day, records, faced)
            else:
                # Simple matching
                matchups = generate_simple_torikumi(wrestlers_df, day, records, faced, w_dict)

            # Batch all of today's bouts into a single predict_proba call.
            feat_rows = []
            for a_id, b_id in matchups:
                h2h_pct, h2h_tot = get_h2h_stats(h2h, a_id, b_id)
                if model_version >= 9:
                    h2h_rp, h2h_rt = (get_h2h_recent_stats(h2h_recent, a_id, b_id)
                                      if h2h_recent is not None else (0.5, 0))
                    f = build_match_features_v9(
                        w_dict[a_id], w_dict[b_id], h2h_pct, h2h_tot,
                        wins_a=records[a_id][0], losses_a=records[a_id][1],
                        wins_b=records[b_id][0], losses_b=records[b_id][1],
                        day=day,
                        streak_a=streaks[a_id], streak_b=streaks[b_id],
                        h2h_recent_pct_a=h2h_rp, h2h_recent_total=h2h_rt,
                    )
                elif model_version >= 8:
                    f = build_match_features(
                        w_dict[a_id], w_dict[b_id], h2h_pct, h2h_tot,
                        wins_a=records[a_id][0], losses_a=records[a_id][1],
                        wins_b=records[b_id][0], losses_b=records[b_id][1],
                        day=day,
                    )
                else:
                    f = build_matchup_features(
                        model_version, w_dict[a_id], w_dict[b_id],
                        h2h_pct, h2h_tot,
                        records[a_id][0], records[a_id][1],
                        records[b_id][0], records[b_id][1],
                        day,
                    )
                feat_rows.append(f)

            if feat_rows:
                X_day = pd.DataFrame(feat_rows)[features]
                probs = model.predict_proba(X_day)[:, 1]
                rolls = np.random.random(len(probs))
            else:
                probs = np.array([])
                rolls = np.array([])

            for (a_id, b_id), prob, roll in zip(matchups, probs, rolls):
                if roll < prob:
                    winner, loser = a_id, b_id
                else:
                    winner, loser = b_id, a_id

                # Update records and signed streak
                w, l = records[winner]
                records[winner] = (w + 1, l)
                w, l = records[loser]
                records[loser] = (w, l + 1)
                streaks[winner] = max(1, streaks[winner] + 1)
                streaks[loser] = min(-1, streaks[loser] - 1)

                faced[a_id].add(b_id)
                faced[b_id].add(a_id)

                # Update wrestler states for v2 scheduling
                if scheduling == 'jsa_v2':
                    w_state_dict[winner].wins += 1
                    w_state_dict[loser].losses += 1
                    w_state_dict[a_id].faced.add(b_id)
                    w_state_dict[b_id].faced.add(a_id)
                    if w_state_dict[a_id].is_sanyaku and w_state_dict[b_id].is_sanyaku:
                        w_state_dict[a_id].sanyaku_faced.add(b_id)
                        w_state_dict[b_id].sanyaku_faced.add(a_id)

        # Track results
        for w_id, (wins, _) in records.items():
            total_wins[w_id] += wins
            if wins >= 8:
                kachi_koshi_count[w_id] += 1

        # Track yusho/jun-yusho
        max_wins = max(r[0] for r in records.values())
        winners = [w for w, r in records.items() if r[0] == max_wins]

        if len(winners) == 1:
            yusho_count[winners[0]] += 1
            second_wins = max(r[0] for w, r in records.items() if w not in winners)
            jun_winners = [w for w, r in records.items() if r[0] == second_wins]
            for w in jun_winners:
                jun_yusho_count[w] += 1 / len(jun_winners)
        else:
            for w in winners:
                yusho_count[w] += 1 / len(winners)
                jun_yusho_count[w] += (len(winners) - 1) / len(winners)

    results = wrestlers_df[['wrestler_id', 'rikishi', 'rank', 'division']].copy()
    results['expected_wins'] = results['wrestler_id'].map(lambda w: total_wins[w] / n_sims)
    results['yusho_prob'] = results['wrestler_id'].map(lambda w: yusho_count[w] / n_sims)
    results['jun_yusho_prob'] = results['wrestler_id'].map(lambda w: jun_yusho_count[w] / n_sims)
    results['kachi_koshi_prob'] = results['wrestler_id'].map(lambda w: kachi_koshi_count[w] / n_sims)

    return results.sort_values('expected_wins', ascending=False)


def generate_jsa_torikumi(wrestlers_df: pd.DataFrame, day: int,
                           records: Dict[int, Tuple[int, int]],
                           faced: Dict[int, set]) -> List[Tuple[int, int]]:
    """Original JSA-style scheduling (2-segment: days 1-7 vs 8-15)."""
    matchups = []

    for div in ['makuuchi', 'juryo']:
        div_wrestlers = wrestlers_df[wrestlers_df['division'] == div].copy()
        if div_wrestlers.empty:
            continue

        available = set(div_wrestlers['wrestler_id'].tolist())

        if day <= 7:
            div_wrestlers = div_wrestlers.sort_values('rank_score', ascending=False)
        else:
            div_wrestlers['record_diff'] = div_wrestlers['wrestler_id'].map(
                lambda w: records.get(w, (0, 0))[0] - records.get(w, (0, 0))[1]
            )
            div_wrestlers = div_wrestlers.sort_values(
                ['record_diff', 'rank_score'],
                ascending=[False, False]
            )

        w_list = div_wrestlers['wrestler_id'].tolist()
        w_heya = dict(zip(div_wrestlers['wrestler_id'], div_wrestlers['heya']))
        w_sanyaku = dict(zip(div_wrestlers['wrestler_id'], div_wrestlers.get('is_sanyaku', 0)))

        matched = set()

        for i, a in enumerate(w_list):
            if a in matched or a not in available:
                continue

            best_opponent = None
            best_score = -float('inf')

            for j, b in enumerate(w_list[i + 1:], start=i + 1):
                if b in matched or b not in available:
                    continue

                if w_heya.get(a) == w_heya.get(b):
                    continue

                if b in faced.get(a, set()):
                    continue

                score = 0

                if day > 7:
                    rec_a = records.get(a, (0, 0))[0] - records.get(a, (0, 0))[1]
                    rec_b = records.get(b, (0, 0))[0] - records.get(b, (0, 0))[1]
                    score -= abs(rec_a - rec_b) * 10

                rank_a = div_wrestlers[div_wrestlers['wrestler_id'] == a]['rank_score'].iloc[0]
                rank_b = div_wrestlers[div_wrestlers['wrestler_id'] == b]['rank_score'].iloc[0]
                score -= abs(rank_a - rank_b) * 0.5

                if day <= 14 and w_sanyaku.get(a, 0) and w_sanyaku.get(b, 0):
                    score += 5

                if score > best_score:
                    best_score = score
                    best_opponent = b

            if best_opponent is not None:
                matchups.append((a, best_opponent))
                matched.add(a)
                matched.add(best_opponent)
                available.discard(a)
                available.discard(best_opponent)

    return matchups


def generate_simple_torikumi(wrestlers_df: pd.DataFrame, day: int,
                              records: Dict[int, Tuple[int, int]],
                              faced: Dict[int, set],
                              w_dict: Dict) -> List[Tuple[int, int]]:
    """Simple greedy matching."""
    matchups = []

    for div in ['makuuchi', 'juryo']:
        div_ids = wrestlers_df[wrestlers_df['division'] == div]['wrestler_id'].tolist()
        available = [w for w in div_ids]

        if day <= 7:
            available.sort(key=lambda w: w_dict[w]['rank_score'], reverse=True)
        else:
            available.sort(key=lambda w: records[w][0] - records[w][1], reverse=True)

        while len(available) >= 2:
            a = available.pop(0)
            for i, b in enumerate(available):
                if w_dict[a]['heya'] != w_dict[b]['heya'] and b not in faced[a]:
                    available.pop(i)
                    matchups.append((a, b))
                    break
            else:
                if available:
                    b = available.pop(0)
                    matchups.append((a, b))

    return matchups


def _draft_category(rank: str) -> str:
    division, number, _ = _parse_rank(rank)
    if not division:
        return "Other"
    if division in ('Y', 'O'):
        return "Yokozuna/Ozeki"
    if division in ('S', 'K'):
        return "Sekiwake/Komusubi"
    if division == 'M':
        if 1 <= number <= 5:
            return "Maegashira 1-5"
        if 6 <= number <= 10:
            return "Maegashira 6-10"
        return "Maegashira 11+"
    if division == 'J':
        return "Juryo"
    return "Other"


def generate_html_report(results: pd.DataFrame, tournament_id: int,
                          model_version: int, n_sims: int, scheduling: str) -> str:
    """Render simulator results as a styled, sortable HTML table."""
    df = results.copy().sort_values('expected_wins', ascending=False)

    group_class_map = {
        "Yokozuna/Ozeki": "group-yo",
        "Sekiwake/Komusubi": "group-sk",
        "Maegashira 1-5": "group-m1-5",
        "Maegashira 6-10": "group-m6-10",
        "Maegashira 11+": "group-m11",
        "Juryo": "group-j",
    }

    rows_html = []
    for _, row in df.iterrows():
        cat = _draft_category(row['rank'])
        group_cls = group_class_map.get(cat, "")
        div_cls = f"division-{row['division']}"
        rows_html.append(
            f"<tr class='{div_cls} {group_cls}'>"
            f"<td>{row['rikishi']}</td>"
            f"<td>{row['rank']}</td>"
            f"<td><strong>{row['expected_wins']:.2f}</strong></td>"
            f"<td>{row['kachi_koshi_prob'] * 100:.1f}%</td>"
            f"<td>{row['yusho_prob'] * 100:.2f}%</td>"
            f"<td>{row['jun_yusho_prob'] * 100:.2f}%</td>"
            f"</tr>"
        )
    table_rows = "\n".join(rows_html)

    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>Sumo Simulation v{model_version} — {tournament_id}</title>
<style>
body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f4f4f9; color: #333; padding: 20px; }}
.container {{ max-width: 1200px; margin: auto; background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
h1, h2, .meta {{ text-align: center; color: #444; }}
.meta {{ color: #777; margin-top: -8px; font-size: 14px; }}
table {{ width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 13px; }}
th, td {{ padding: 8px 10px; border: 1px solid #ddd; text-align: left; }}
th {{ background-color: #333; color: #fff; cursor: pointer; user-select: none; position: sticky; top: 0; }}
.filters {{ text-align: center; margin-bottom: 20px; }}
.filters button {{ padding: 8px 15px; margin: 0 5px; border: 1px solid #ccc; background: #eee; cursor: pointer; border-radius: 4px; }}
.filters button.active {{ background: #333; color: #fff; border-color: #333; }}
.group-yo {{ background-color: #ffcccc; }} .group-sk {{ background-color: #ffedcc; }}
.group-m1-5 {{ background-color: #ffffcc; }} .group-m6-10 {{ background-color: #ccffcc; }}
.group-m11 {{ background-color: #ccffff; }} .group-j {{ background-color: #e6ccff; }}
</style>
<script>
function sortTable(n) {{
  var table = document.getElementById("reportTable");
  var rows, switching = true, i, x, y, shouldSwitch, dir = "asc", switchcount = 0;
  while (switching) {{
    switching = false; rows = table.rows;
    for (i = 1; i < rows.length - 1; i++) {{
      shouldSwitch = false;
      x = rows[i].getElementsByTagName("TD")[n];
      y = rows[i + 1].getElementsByTagName("TD")[n];
      var xc = isNaN(parseFloat(x.innerText)) ? x.innerText.toLowerCase() : parseFloat(x.innerText);
      var yc = isNaN(parseFloat(y.innerText)) ? y.innerText.toLowerCase() : parseFloat(y.innerText);
      if (dir === "asc" ? xc > yc : xc < yc) {{ shouldSwitch = true; break; }}
    }}
    if (shouldSwitch) {{ rows[i].parentNode.insertBefore(rows[i + 1], rows[i]); switching = true; switchcount++; }}
    else if (switchcount === 0 && dir === "asc") {{ dir = "desc"; switching = true; }}
  }}
}}
function filterDivision(division) {{
  var rows = document.getElementById("reportTable").getElementsByTagName("tr");
  for (var i = 1; i < rows.length; i++) {{
    rows[i].style.display = (division === 'all' || rows[i].classList.contains('division-' + division)) ? "" : "none";
  }}
  document.querySelectorAll('.filters button').forEach(b => b.classList.remove('active'));
  document.querySelector(".filters button[data-div='" + division + "']").classList.add('active');
}}
window.onload = () => {{ document.querySelector(".filters button[data-div='all']").classList.add('active'); }};
</script>
</head><body><div class="container">
<h1>Sumo Tournament Simulation — Matchup Model v{model_version}</h1>
<h2>Basho: {tournament_id}</h2>
<div class="meta">{n_sims} Monte Carlo simulations · {scheduling} scheduling</div>
<div class="filters">
  <button data-div="all" onclick="filterDivision('all')">All</button>
  <button data-div="makuuchi" onclick="filterDivision('makuuchi')">Makuuchi</button>
  <button data-div="juryo" onclick="filterDivision('juryo')">Juryo</button>
</div>
<table id="reportTable">
<thead><tr>
  <th onclick="sortTable(0)">Rikishi</th>
  <th onclick="sortTable(1)">Rank</th>
  <th onclick="sortTable(2)">Expected Wins</th>
  <th onclick="sortTable(3)">Kachi-koshi %</th>
  <th onclick="sortTable(4)">Yusho %</th>
  <th onclick="sortTable(5)">Jun-yusho %</th>
</tr></thead>
<tbody>{table_rows}</tbody>
</table>
</div></body></html>"""

    out_path = f"simulation_report_{tournament_id}.html"
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    return out_path


def main(model_path: str, banzuke_path: str, match_path: str,
         tournament_id: int, n_sims: int, scheduling: str = 'jsa_v2',
         predict_banzuke: bool = False, chain_tournaments: int = 0):
    """Main entry point."""

    matchup_bundle, _, banzuke, match_history = load_data(model_path, banzuke_path, match_path)
    model_version = matchup_bundle.get('version', 2)

    print(f"\nPreparing wrestlers for {tournament_id} (model v{model_version})...")
    if model_version >= 9:
        wrestlers = prepare_wrestlers_v9(banzuke, match_history, tournament_id)
    elif model_version >= 8:
        wrestlers = prepare_wrestlers_v8(banzuke, match_history, tournament_id)
    else:
        wrestlers = prepare_wrestlers(banzuke, match_history, tournament_id)

    if wrestlers.empty:
        print("No wrestlers found!")
        return

    # Add a 'division' column for v8/v9 prep paths (legacy prep already provides it).
    if 'division' not in wrestlers.columns:
        def _div_for(rank):
            d, _, _ = _parse_rank(rank)
            return 'makuuchi' if d in ('Y', 'O', 'S', 'K', 'M') else ('juryo' if d == 'J' else 'other')
        wrestlers['division'] = wrestlers['rank'].apply(_div_for)

    print(f"Found {len(wrestlers)} wrestlers")
    print(f"  Makuuchi: {len(wrestlers[wrestlers['division'] == 'makuuchi'])}")
    print(f"  Juryo: {len(wrestlers[wrestlers['division'] == 'juryo'])}")

    print("Computing H2H history...")
    h2h = compute_h2h(match_history, tournament_id)
    h2h_recent = compute_h2h_recent(match_history, tournament_id) if model_version >= 9 else None

    # Run simulation
    results = simulate_tournament(matchup_bundle, wrestlers, h2h, n_sims, scheduling,
                                   h2h_recent=h2h_recent)

    print(f"\n{'=' * 60}")
    print(f"PREDICTIONS FOR {tournament_id} ({n_sims} simulations, {scheduling} scheduling)")
    print(f"{'=' * 60}")

    print("\n--- MAKUUCHI ---")
    mak = results[results['division'] == 'makuuchi'][
        ['rikishi', 'rank', 'expected_wins', 'kachi_koshi_prob', 'yusho_prob']
    ].copy()
    mak['expected_wins'] = mak['expected_wins'].round(2)
    mak['kachi_koshi_prob'] = (mak['kachi_koshi_prob'] * 100).round(1).astype(str) + '%'
    mak['yusho_prob'] = (mak['yusho_prob'] * 100).round(1).astype(str) + '%'
    print(mak.to_string(index=False))

    print("\n--- JURYO ---")
    jur = results[results['division'] == 'juryo'][
        ['rikishi', 'rank', 'expected_wins', 'kachi_koshi_prob']
    ].copy()
    jur['expected_wins'] = jur['expected_wins'].round(2)
    jur['kachi_koshi_prob'] = (jur['kachi_koshi_prob'] * 100).round(1).astype(str) + '%'
    print(jur.to_string(index=False))

    # Save results
    output_file = f"outputs/simulation_v4_{tournament_id}.csv"
    results.to_csv(output_file, index=False)
    print(f"\nSaved to {output_file}")

    html_path = generate_html_report(results, tournament_id, model_version, n_sims, scheduling)
    print(f"Generated HTML report: '{html_path}'")

    # Predict next banzuke if requested
    if predict_banzuke:
        print("\n--- PREDICTING NEXT BANZUKE ---")

        # Convert results to tournament outcomes
        tournament_results = {}
        for _, row in results.iterrows():
            wins = int(round(row['expected_wins']))
            losses = 15 - wins
            tournament_results[row['wrestler_id']] = (wins, losses)

        # Get current banzuke for this tournament
        current_banzuke = banzuke[banzuke['tournament_id'] == tournament_id].copy()

        # Find yusho winner
        yusho_winner = results.loc[results['expected_wins'].idxmax()]['wrestler_id']

        # Predict next banzuke
        next_banzuke, _, _ = predict_next_banzuke(
            current_banzuke,
            tournament_results,
            yusho_winner_id=yusho_winner
        )

        print("\nPredicted Next Banzuke (Top 20):")
        print(next_banzuke[['rikishi', 'rank', 'prev_rank', 'prev_wins', 'prev_losses']].head(20).to_string(index=False))

        banzuke_output = f"predicted_banzuke_{tournament_id + 2}.csv"  # Next tournament
        next_banzuke.to_csv(banzuke_output, index=False)
        print(f"\nSaved predicted banzuke to {banzuke_output}")

    return results


# ---------------------------------------------------------------------------
# V8 feature functions — used by daily_matchup_app.py
# ---------------------------------------------------------------------------

def prepare_wrestlers_v8(banzuke: pd.DataFrame, match_history: pd.DataFrame,
                          tournament_id: int, all_divisions: bool = False) -> pd.DataFrame:
    """Prepare wrestler features for a tournament with full v8 model feature set.

    Computes proper Elo ratings (via compute_elo_features), Elo change windows,
    career stats, rank trajectory features, and style/vulnerability profiles.
    Set all_divisions=True to include wrestlers outside Makuuchi/Juryo.
    """
    roster = banzuke[banzuke['tournament_id'] == tournament_id].copy()

    def is_top_div(r):
        d, _, _ = _parse_rank(r)
        return d in ['Y', 'O', 'S', 'K', 'M', 'J']

    if not all_divisions:
        roster = roster[roster['rank'].apply(is_top_div)]
    if roster.empty:
        return pd.DataFrame()

    hist_banzuke = banzuke[banzuke['tournament_id'] < tournament_id]
    hist_matches = match_history[match_history['tournament_id'] < tournament_id]

    # Compute Elo ratings and elo_change_* features from full match history
    _elo_feats = compute_elo_features(match_history)
    _elo_for_tournament = _elo_feats[_elo_feats['tournament_id'] == tournament_id].set_index('wrestler_id')
    _elo_change_cols = [c for c in _elo_feats.columns if c.startswith('elo_change_')]

    wrestlers = []
    for _, row in roster.iterrows():
        w_id = row['wrestler_id']
        w_hist = hist_banzuke[hist_banzuke['wrestler_id'] == w_id]
        w_matches_won = hist_matches[hist_matches['winner_id'] == w_id]
        w_matches_lost = hist_matches[hist_matches['loser_id'] == w_id]

        # Rank
        rank_score = get_absolute_rank_score(row['rank'])
        div, num, side = _parse_rank(row['rank'])
        is_sanyaku = 1 if div in ['Y', 'O', 'S', 'K'] else 0
        sanyaku_map = {'Y': 22, 'O': 20, 'S': 19, 'K': 18}
        if div in sanyaku_map:
            rank_in_div = sanyaku_map[div]
        elif div == 'M':
            rank_in_div = 18 - num
        elif div == 'J':
            rank_in_div = 15 - num
        else:
            rank_in_div = 0
        if side == 'e':
            rank_in_div += 0.5

        # Physical
        hw = pd.Series([row.get('height_weight', '')]).str.extract(r'(\d+\.?\d*)\s*cm\s*(\d+\.?\d*)\s*kg')
        height = pd.to_numeric(hw[0].iloc[0], errors='coerce') if len(hw) > 0 else 175.0
        weight = pd.to_numeric(hw[1].iloc[0], errors='coerce') if len(hw) > 0 else 130.0
        bmi = weight / ((height / 100) ** 2) if height and weight else 40.0

        # Age
        bd = pd.to_datetime(row.get('birth_date'), format='%d.%m.%Y', errors='coerce')
        td = pd.to_datetime(str(tournament_id), format='%Y%m')
        age = (td - bd).days / 365.25 if pd.notna(bd) else 28.0

        # Elo from compute_elo_features
        elo = float(_elo_for_tournament.at[w_id, 'elo']) if w_id in _elo_for_tournament.index else 1500.0

        # Recent form
        if not w_hist.empty:
            recent = w_hist.tail(3)
            rw = pd.to_numeric(recent['w'], errors='coerce').sum()
            rl = pd.to_numeric(recent['l'], errors='coerce').sum()
            win_pct_last_3 = rw / (rw + rl) if (rw + rl) > 0 else 0.5

            recent6 = w_hist.tail(6)
            rw6 = pd.to_numeric(recent6['w'], errors='coerce').sum()
            rl6 = pd.to_numeric(recent6['l'], errors='coerce').sum()
            win_pct_last_6 = rw6 / (rw6 + rl6) if (rw6 + rl6) > 0 else 0.5

            # Career stats
            all_w = pd.to_numeric(w_hist['w'], errors='coerce').fillna(0)
            all_l = pd.to_numeric(w_hist['l'], errors='coerce').fillna(0)
            career_total = all_w.sum() + all_l.sum()
            career_win_pct = all_w.sum() / career_total if career_total > 0 else 0.5
            career_tournaments = len(w_hist)

            # Weighted recent form (EWM of per-tournament win pcts)
            tourn_totals = all_w + all_l
            tourn_win_pct = (all_w / tourn_totals.replace(0, 1)).values
            ewm_vals = pd.Series(tourn_win_pct).ewm(span=3, min_periods=1).mean().values
            weighted_recent_10 = float(ewm_vals[-1]) if len(ewm_vals) > 0 else 0.5
        else:
            win_pct_last_3 = 0.5
            win_pct_last_6 = 0.5
            career_win_pct = 0.5
            career_tournaments = 0
            weighted_recent_10 = 0.5

        # Rank trajectory
        high_rank = row.get('high', row['rank'])
        highest_rank_score = get_absolute_rank_score(high_rank) if high_rank else rank_score
        rank_gap = max(0, highest_rank_score - rank_score)

        if not w_hist.empty:
            hist_scores = w_hist['rank'].apply(get_absolute_rank_score).dropna().values
            if len(hist_scores) >= 2:
                deltas = [hist_scores[i] - hist_scores[i - 1] for i in range(1, len(hist_scores))]
                rank_momentum = sum(deltas[-3:]) / len(deltas[-3:])
            else:
                rank_momentum = 0.0
            hist_high = w_hist['high'].ffill().fillna(row['rank'])
            hist_gaps = (
                hist_high.apply(get_absolute_rank_score) -
                w_hist['rank'].apply(get_absolute_rank_score)
            ).clip(lower=0)
            basho_since_peak = 0
            for gap in reversed(hist_gaps.values):
                if gap <= 0.6:
                    break
                basho_since_peak += 1
        else:
            rank_momentum = 0.0
            basho_since_peak = 0

        # Style ratios (from wins)
        style_counts = {cat: 0 for cat in TECHNIQUE_CATEGORIES}
        total_style = 0
        for _, m in w_matches_won.iterrows():
            cat = TECHNIQUE_TO_CATEGORY.get(m['kimarite'])
            if cat:
                style_counts[cat] += 1
                total_style += 1
        style_ratios = {
            f'{cat}_ratio': style_counts[cat] / total_style if total_style > 0 else 0.2
            for cat in TECHNIQUE_CATEGORIES
        }

        # Vulnerability ratios (from losses)
        vuln_counts = {cat: 0 for cat in TECHNIQUE_CATEGORIES}
        total_vuln = 0
        for _, m in w_matches_lost.iterrows():
            cat = TECHNIQUE_TO_CATEGORY.get(m['kimarite'])
            if cat:
                vuln_counts[cat] += 1
                total_vuln += 1
        vuln_ratios = {
            f'vuln_{cat}': vuln_counts[cat] / total_vuln if total_vuln > 0 else 0.2
            for cat in TECHNIQUE_CATEGORIES
        }

        wrestlers.append({
            'wrestler_id': w_id,
            'rikishi': row['rikishi'],
            'rank': row['rank'],
            'rank_score': rank_score,
            'rank_in_div': rank_in_div,
            'is_sanyaku': is_sanyaku,
            'division': 'makuuchi' if div in ['Y', 'O', 'S', 'K', 'M'] else 'juryo',
            'height': height,
            'weight': weight,
            'bmi': bmi,
            'age': age,
            'elo': elo,
            'win_pct_last_3': win_pct_last_3,
            'win_pct_last_6': win_pct_last_6,
            'career_win_pct': career_win_pct,
            'career_tournaments': career_tournaments,
            'weighted_recent_10': weighted_recent_10,
            'rank_gap': rank_gap,
            'rank_momentum': rank_momentum,
            'basho_since_peak': basho_since_peak,
            'heya': row.get('heya', ''),
            **style_ratios,
            **vuln_ratios,
            **({col: _elo_for_tournament.at[w_id, col] for col in _elo_change_cols}
               if w_id in _elo_for_tournament.index else
               {col: 0.0 for col in _elo_change_cols}),
        })

    df = pd.DataFrame(wrestlers)
    for col in ['height', 'weight', 'bmi', 'age']:
        df[col] = df[col].fillna(df[col].median())
    return df


def build_match_features(a: dict, b: dict, h2h_pct_a: float, h2h_total: int,
                          wins_a: int = 0, losses_a: int = 0,
                          wins_b: int = 0, losses_b: int = 0,
                          day: int = 1) -> dict:
    """Build the full v8 feature dict for a matchup between wrestlers a and b."""
    elo_a = a.get('elo', 1500.0)
    elo_b = b.get('elo', 1500.0)
    age_a = a.get('age', 28.0)
    age_b = b.get('age', 28.0)
    win3_a = a.get('win_pct_last_3', 0.5)
    win3_b = b.get('win_pct_last_3', 0.5)
    win6_a = a.get('win_pct_last_6', win3_a)
    win6_b = b.get('win_pct_last_6', win3_b)
    career_a = a.get('career_win_pct', 0.5)
    career_b = b.get('career_win_pct', 0.5)
    exp_a = a.get('career_tournaments', 10)
    exp_b = b.get('career_tournaments', 10)
    wr10_a = a.get('weighted_recent_10', win3_a)
    wr10_b = b.get('weighted_recent_10', win3_b)
    h_a, h_b = a.get('height', 175.0), b.get('height', 175.0)
    w_a, w_b = a.get('weight', 130.0), b.get('weight', 130.0)
    bm_a, bm_b = a.get('bmi', 40.0), b.get('bmi', 40.0)
    rank_a, rank_b = a['rank_score'], b['rank_score']

    is_kk_a = 1 if wins_a == 7 and losses_a < 8 else 0
    is_mk_a = 1 if losses_a == 7 and wins_a < 8 else 0
    is_kk_b = 1 if wins_b == 7 and losses_b < 8 else 0
    is_mk_b = 1 if losses_b == 7 and wins_b < 8 else 0

    f = {
        'rank_diff': rank_a - rank_b,
        'rank_in_div_diff': a['rank_in_div'] - b['rank_in_div'],
        'sanyaku_diff': a.get('is_sanyaku', 0) - b.get('is_sanyaku', 0),
        'elo_diff': elo_a - elo_b,
        'elo_expected_A': 1 / (1 + 10 ** ((elo_b - elo_a) / 400)),
        'bmi_diff': bm_a - bm_b,
        'height_diff': h_a - h_b,
        'weight_diff': w_a - w_b,
        'age_diff': age_a - age_b,
        'win_pct_last_3_A': win3_a,
        'win_pct_last_3_B': win3_b,
        'win_pct_last_3_diff': win3_a - win3_b,
        'win_pct_last_6_A': win6_a,
        'win_pct_last_6_B': win6_b,
        'win_pct_last_6_diff': win6_a - win6_b,
        'career_win_pct_A': career_a,
        'career_win_pct_B': career_b,
        'career_win_pct_diff': career_a - career_b,
        'experience_diff': exp_a - exp_b,
        'weighted_recent_10_A': wr10_a,
        'weighted_recent_10_B': wr10_b,
        'weighted_recent_diff': wr10_a - wr10_b,
        'h2h_win_pct_A': h2h_pct_a,
        'h2h_total_matches': h2h_total,
        'has_h2h_history': 1 if h2h_total > 0 else 0,
        'rank_x_form_diff': rank_a * win3_a - rank_b * win3_b,
        'elo_x_age_diff': (
            elo_a / (1 + np.exp((age_a - 30) / 3)) -
            elo_b / (1 + np.exp((age_b - 30) / 3))
        ),
        'rank_momentum_A': a.get('rank_momentum', 0.0),
        'rank_momentum_B': b.get('rank_momentum', 0.0),
        'rank_momentum_diff': a.get('rank_momentum', 0.0) - b.get('rank_momentum', 0.0),
        'basho_since_peak_A': a.get('basho_since_peak', 0),
        'basho_since_peak_B': b.get('basho_since_peak', 0),
        'basho_since_peak_diff': a.get('basho_since_peak', 0) - b.get('basho_since_peak', 0),
        'rank_gap_diff': a.get('rank_gap', 0.0) - b.get('rank_gap', 0.0),
        'wins_before_A': wins_a,
        'losses_before_A': losses_a,
        'wins_before_B': wins_b,
        'losses_before_B': losses_b,
        'record_diff': (wins_a - losses_a) - (wins_b - losses_b),
        'day': day,
        'is_kk_match_A': is_kk_a,
        'is_kk_match_B': is_kk_b,
        'is_mk_match_A': is_mk_a,
        'is_mk_match_B': is_mk_b,
        'kk_pressure_diff': is_kk_a - is_kk_b,
        'mk_pressure_diff': is_mk_a - is_mk_b,
        'pressure_diff': (is_kk_a - is_kk_b) - (is_mk_a - is_mk_b),
    }

    for cat in TECHNIQUE_CATEGORIES:
        f[f'{cat}_ratio_A'] = a.get(f'{cat}_ratio', 0.2)
        f[f'{cat}_ratio_B'] = b.get(f'{cat}_ratio', 0.2)
        f[f'vuln_{cat}_A'] = a.get(f'vuln_{cat}', 0.2)
        f[f'vuln_{cat}_B'] = b.get(f'vuln_{cat}', 0.2)
        adv_a = a.get(f'{cat}_ratio', 0.2) * b.get(f'vuln_{cat}', 0.2)
        adv_b = b.get(f'{cat}_ratio', 0.2) * a.get(f'vuln_{cat}', 0.2)
        f[f'{cat}_advantage_diff'] = adv_a - adv_b

    total_adv_a = sum(a.get(f'{c}_ratio', 0.2) * b.get(f'vuln_{c}', 0.2) for c in TECHNIQUE_CATEGORIES)
    total_adv_b = sum(b.get(f'{c}_ratio', 0.2) * a.get(f'vuln_{c}', 0.2) for c in TECHNIQUE_CATEGORIES)
    f['style_advantage_diff'] = total_adv_a - total_adv_b

    f['height_x_oshi_diff'] = h_a * a.get('oshi_ratio', 0.2) - h_b * b.get('oshi_ratio', 0.2)
    f['height_x_yotsu_diff'] = h_a * a.get('yotsu_ratio', 0.2) - h_b * b.get('yotsu_ratio', 0.2)
    f['weight_x_yotsu_diff'] = w_a * a.get('yotsu_ratio', 0.2) - w_b * b.get('yotsu_ratio', 0.2)
    f['bmi_x_pull_diff'] = bm_a * a.get('pull_ratio', 0.2) - bm_b * b.get('pull_ratio', 0.2)
    f['age_x_oshi_diff'] = age_a * a.get('oshi_ratio', 0.2) - age_b * b.get('oshi_ratio', 0.2)
    f['age_x_yotsu_diff'] = age_a * a.get('yotsu_ratio', 0.2) - age_b * b.get('yotsu_ratio', 0.2)

    # Elo change features (v8 momentum)
    for key in [k for k in a if k.startswith('elo_change_')]:
        va = a.get(key, 0.0)
        vb = b.get(key, 0.0)
        f[f'{key}_A'] = va
        f[f'{key}_B'] = vb
        f[f'{key}_diff'] = va - vb

    return f


def predict_match_v8(model, features: List[str], a: dict, b: dict,
                     h2h_pct_a: float, h2h_total: int,
                     wins_a: int = 0, losses_a: int = 0,
                     wins_b: int = 0, losses_b: int = 0,
                     day: int = 1) -> float:
    """Predict P(A wins) using the v8 model feature set."""
    f = build_match_features(a, b, h2h_pct_a, h2h_total,
                             wins_a, losses_a, wins_b, losses_b, day)
    X = pd.DataFrame([f])[features]
    return model.predict_proba(X)[0][1]


# ---------------------------------------------------------------------------
# V9 feature functions — adds injury, streak, recent H2H, rank volatility
# ---------------------------------------------------------------------------

def compute_h2h_recent(match_history: pd.DataFrame, tournament_id: int, n: int = 10) -> Dict:
    """Get recent H2H records (last N meetings before tournament_id)."""
    hist = match_history[match_history['tournament_id'] < tournament_id].copy()
    hist = hist.sort_values(['tournament_id', 'day'])
    h2h: Dict = {}
    for _, m in hist.iterrows():
        w1 = min(m['winner_id'], m['loser_id'])
        w2 = max(m['winner_id'], m['loser_id'])
        key = (w1, w2)
        h2h.setdefault(key, []).append(1 if m['winner_id'] == w1 else 0)
    return {k: {'w1': sum(v[-n:]), 'total': len(v[-n:])} for k, v in h2h.items()}


def get_h2h_recent_stats(h2h_recent: Dict, a_id: int, b_id: int) -> Tuple[float, int]:
    """Get A's recent win % vs B and recent match count."""
    w1 = min(a_id, b_id)
    key = (w1, max(a_id, b_id))
    if key not in h2h_recent:
        return 0.5, 0
    rec = h2h_recent[key]
    total = rec['total']
    if total == 0:
        return 0.5, 0
    w1_pct = rec['w1'] / total
    return (w1_pct if a_id == w1 else 1 - w1_pct), total


def prepare_wrestlers_v9(banzuke: pd.DataFrame, match_history: pd.DataFrame,
                          tournament_id: int, all_divisions: bool = False) -> pd.DataFrame:
    """Prepare wrestler features for v9 model (extends v8 with injury + rank volatility)."""
    df = prepare_wrestlers_v8(banzuke, match_history, tournament_id, all_divisions)
    if df.empty:
        return df

    roster = banzuke[banzuke['tournament_id'] == tournament_id].copy()
    hist_banzuke = banzuke[banzuke['tournament_id'] < tournament_id]

    def _count_abs(s):
        return s.count('-') if isinstance(s, str) else 0

    def _count_sched(s):
        return len([c for c in str(s) if c in 'OoSsXx*-#']) if isinstance(s, str) else 15

    injury_rows = []
    for w_id in df['wrestler_id']:
        w_hist = hist_banzuke[hist_banzuke['wrestler_id'] == w_id]
        if not w_hist.empty and 'hoshitori' in w_hist.columns:
            abs_series = w_hist['hoshitori'].apply(_count_abs)
            sched_series = w_hist['hoshitori'].apply(_count_sched)
            comp_series = (sched_series - abs_series) / sched_series.replace(0, 1)
            prev_abs = float(abs_series.iloc[-1]) if len(abs_series) else 0
            was_kyujo = int(prev_abs > 0)
            absences_last_3 = float(abs_series.tail(3).sum())
            completion_rate_last_3 = float(comp_series.tail(3).mean())

            # Rank volatility: std of rank score changes over last 6 basho
            rank_scores = w_hist['rank'].apply(get_absolute_rank_score).dropna().values
            if len(rank_scores) >= 3:
                deltas = [rank_scores[i] - rank_scores[i-1] for i in range(1, len(rank_scores))]
                rank_volatility = float(np.std(deltas[-6:]))
            else:
                rank_volatility = 0.0
        else:
            was_kyujo = 0
            absences_last_3 = 0.0
            completion_rate_last_3 = 1.0
            rank_volatility = 0.0

        injury_rows.append({
            'wrestler_id': w_id,
            'was_kyujo_last_basho': was_kyujo,
            'absences_last_3': absences_last_3,
            'completion_rate_last_3': completion_rate_last_3,
            'rank_volatility': rank_volatility,
        })

    injury_df = pd.DataFrame(injury_rows)
    df = df.merge(injury_df, on='wrestler_id', how='left')
    df['was_kyujo_last_basho'] = df['was_kyujo_last_basho'].fillna(0)
    df['absences_last_3'] = df['absences_last_3'].fillna(0)
    df['completion_rate_last_3'] = df['completion_rate_last_3'].fillna(1.0)
    df['rank_volatility'] = df['rank_volatility'].fillna(0.0)
    return df


def build_match_features_v9(a: dict, b: dict, h2h_pct_a: float, h2h_total: int,
                              wins_a: int = 0, losses_a: int = 0,
                              wins_b: int = 0, losses_b: int = 0,
                              day: int = 1,
                              streak_a: int = 0, streak_b: int = 0,
                              h2h_recent_pct_a: float = 0.5,
                              h2h_recent_total: int = 0) -> dict:
    """Build full v9 feature dict (extends v8 with streak, recent H2H, injury, rank volatility)."""
    f = build_match_features(a, b, h2h_pct_a, h2h_total,
                             wins_a, losses_a, wins_b, losses_b, day)

    # In-tournament streak
    f['current_streak_A'] = streak_a
    f['current_streak_B'] = streak_b
    f['current_streak_diff'] = streak_a - streak_b

    # Recent H2H
    f['h2h_recent_win_pct_A'] = h2h_recent_pct_a
    f['has_recent_h2h'] = 1 if h2h_recent_total >= 3 else 0

    # Injury/kyujo
    f['was_kyujo_last_basho_A'] = a.get('was_kyujo_last_basho', 0)
    f['was_kyujo_last_basho_B'] = b.get('was_kyujo_last_basho', 0)
    f['was_kyujo_diff'] = f['was_kyujo_last_basho_A'] - f['was_kyujo_last_basho_B']
    f['absences_last_3_A'] = a.get('absences_last_3', 0.0)
    f['absences_last_3_B'] = b.get('absences_last_3', 0.0)
    f['absences_last_3_diff'] = f['absences_last_3_A'] - f['absences_last_3_B']
    f['completion_rate_last_3_A'] = a.get('completion_rate_last_3', 1.0)
    f['completion_rate_last_3_B'] = b.get('completion_rate_last_3', 1.0)
    f['completion_rate_diff'] = f['completion_rate_last_3_A'] - f['completion_rate_last_3_B']

    # Rank volatility
    f['rank_volatility_A'] = a.get('rank_volatility', 0.0)
    f['rank_volatility_B'] = b.get('rank_volatility', 0.0)
    f['rank_volatility_diff'] = f['rank_volatility_A'] - f['rank_volatility_B']

    return f


def predict_match_v9(model, features: List[str], a: dict, b: dict,
                     h2h_pct_a: float, h2h_total: int,
                     wins_a: int = 0, losses_a: int = 0,
                     wins_b: int = 0, losses_b: int = 0,
                     day: int = 1,
                     streak_a: int = 0, streak_b: int = 0,
                     h2h_recent_pct_a: float = 0.5,
                     h2h_recent_total: int = 0) -> float:
    """Predict P(A wins) using the v9 model feature set."""
    f = build_match_features_v9(a, b, h2h_pct_a, h2h_total,
                                 wins_a, losses_a, wins_b, losses_b, day,
                                 streak_a, streak_b,
                                 h2h_recent_pct_a, h2h_recent_total)
    X = pd.DataFrame([f])[features]
    return model.predict_proba(X)[0][1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sumo Tournament Simulator v4")
    parser.add_argument('--tournament', type=int, default=202601,
                        help="Tournament ID (YYYYMM format)")
    parser.add_argument('--simulations', type=int, default=500,
                        help="Number of Monte Carlo simulations")
    parser.add_argument('--model', type=str, default="matchup_model_v9.joblib",
                        help="Path to matchup model")
    parser.add_argument('--banzuke', type=str, default="data/banzuke_detailed.csv",
                        help="Path to banzuke data")
    parser.add_argument('--matches', type=str, default="data/match_history_with_kimarite.csv",
                        help="Path to match history")
    parser.add_argument('--scheduling', type=str, default='jsa_v2',
                        choices=['jsa_v2', 'jsa', 'simple'],
                        help="Scheduling algorithm: jsa_v2 (5-segment), jsa (2-segment), simple")
    parser.add_argument('--predict-banzuke', action='store_true',
                        help="Predict next tournament's banzuke")
    parser.add_argument('--chain', type=int, default=0,
                        help="Chain N future tournament simulations")

    args = parser.parse_args()

    main(
        args.model, args.banzuke, args.matches,
        args.tournament, args.simulations, args.scheduling,
        args.predict_banzuke, args.chain
    )

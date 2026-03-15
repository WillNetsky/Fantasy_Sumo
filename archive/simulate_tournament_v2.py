# simulate_tournament_v2.py
"""
Tournament simulator using improved matchup model v2.
"""

import pandas as pd
import numpy as np
import joblib
import argparse
from typing import Dict, List, Tuple
from sumo_utils import get_absolute_rank_score, _parse_rank, compute_elo_features

# Import technique categories from v2
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


def load_data(matchup_model_path: str, banzuke_path: str, match_history_path: str):
    """Load model and data."""
    print("Loading matchup model v2...")
    matchup_bundle = joblib.load(matchup_model_path)

    print("Loading historical data...")
    banzuke = pd.read_csv(banzuke_path)
    match_history = pd.read_csv(match_history_path)

    for df in [banzuke, match_history]:
        for col in ['wrestler_id', 'winner_id', 'loser_id', 'tournament_id']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

    return matchup_bundle, banzuke, match_history


def prepare_wrestlers(banzuke: pd.DataFrame, match_history: pd.DataFrame,
                      tournament_id: int, all_divisions: bool = False) -> pd.DataFrame:
    """Prepare all wrestler features for a tournament.

    Args:
        all_divisions: If True, include wrestlers from all divisions (not just
                       Makuuchi/Juryo). Useful for daily matchup predictions.
    """

    # Get tournament roster
    roster = banzuke[banzuke['tournament_id'] == tournament_id].copy()

    def is_top_div(r):
        d, _, _ = _parse_rank(r)
        return d in ['Y', 'O', 'S', 'K', 'M', 'J']

    if not all_divisions:
        roster = roster[roster['rank'].apply(is_top_div)]
    if roster.empty:
        return pd.DataFrame()

    # Historical data
    hist_banzuke = banzuke[banzuke['tournament_id'] < tournament_id]
    hist_matches = match_history[match_history['tournament_id'] < tournament_id]

    # Precompute ELO ratings and elo_change_* features from all historical matches
    _elo_feats = compute_elo_features(match_history)
    _elo_for_tournament = _elo_feats[_elo_feats['tournament_id'] == tournament_id].set_index('wrestler_id')
    _elo_change_cols = [c for c in _elo_feats.columns if c.startswith('elo_change_')]

    _elos: dict = {}
    _k = 32.0
    for _, _m in hist_matches.sort_values(['tournament_id', 'day']).iterrows():
        _w, _l = _m['winner_id'], _m['loser_id']
        _ew = _elos.get(_w, 1500.0)
        _el = _elos.get(_l, 1500.0)
        _exp = 1 / (1 + 10 ** ((_el - _ew) / 400))
        _elos[_w] = _ew + _k * (1 - _exp)
        _elos[_l] = _el - _k * _exp

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

        # ELO
        elo = _elos.get(w_id, 1500.0)

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

            # Weighted recent 10 (EWM of tournament win percentages)
            tourn_totals = all_w + all_l
            tourn_win_pct = (all_w / tourn_totals.replace(0, 1)).values
            if len(tourn_win_pct) > 0:
                ewm_vals = pd.Series(tourn_win_pct).ewm(span=3, min_periods=1).mean().values
                weighted_recent_10 = float(ewm_vals[-1])
            else:
                weighted_recent_10 = 0.5
        else:
            win_pct_last_3 = 0.5
            win_pct_last_6 = 0.5
            career_win_pct = 0.5
            career_tournaments = 0
            weighted_recent_10 = 0.5

        # Rank trajectory features
        high_rank = row.get('high', row['rank'])
        highest_rank_score = get_absolute_rank_score(high_rank) if high_rank else rank_score
        rank_gap = max(0, highest_rank_score - rank_score)

        if not w_hist.empty:
            hist_scores = w_hist['rank'].apply(get_absolute_rank_score).dropna().values
            if len(hist_scores) >= 2:
                deltas = [hist_scores[i] - hist_scores[i-1] for i in range(1, len(hist_scores))]
                rank_momentum = sum(deltas[-3:]) / len(deltas[-3:])
            else:
                rank_momentum = 0.0
            # Basho since peak: count back to last time rank_gap was ~0
            hist_high = w_hist['high'].ffill().fillna(row['rank'])
            hist_gaps = (hist_high.apply(get_absolute_rank_score) - w_hist['rank'].apply(get_absolute_rank_score)).clip(lower=0)
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
            **({col: _elo_for_tournament.at[w_id, col]
                for col in _elo_change_cols}
               if w_id in _elo_for_tournament.index else
               {col: 0.0 for col in _elo_change_cols}),
        })

    df = pd.DataFrame(wrestlers)

    # Fill missing
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


def build_match_features(a: dict, b: dict, h2h_pct_a: float, h2h_total: int,
                         wins_a: int = 0, losses_a: int = 0,
                         wins_b: int = 0, losses_b: int = 0,
                         day: int = 1) -> dict:
    """Build the full v7 feature dict for a matchup between wrestlers a and b."""
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

    # Pressure (KK = about to get winning record; MK = about to get losing record)
    is_kk_a = 1 if wins_a == 7 and losses_a < 8 else 0
    is_mk_a = 1 if losses_a == 7 and wins_a < 8 else 0
    is_kk_b = 1 if wins_b == 7 and losses_b < 8 else 0
    is_mk_b = 1 if losses_b == 7 and wins_b < 8 else 0

    f = {
        # Core ranking
        'rank_diff': rank_a - rank_b,
        'rank_in_div_diff': a['rank_in_div'] - b['rank_in_div'],
        'sanyaku_diff': a.get('is_sanyaku', 0) - b.get('is_sanyaku', 0),
        # ELO
        'elo_diff': elo_a - elo_b,
        'elo_expected_A': 1 / (1 + 10 ** ((elo_b - elo_a) / 400)),
        # Physical
        'bmi_diff': bm_a - bm_b,
        'height_diff': h_a - h_b,
        'weight_diff': w_a - w_b,
        'age_diff': age_a - age_b,
        # Form
        'win_pct_last_3_A': win3_a,
        'win_pct_last_3_B': win3_b,
        'win_pct_last_3_diff': win3_a - win3_b,
        'win_pct_last_6_A': win6_a,
        'win_pct_last_6_B': win6_b,
        'win_pct_last_6_diff': win6_a - win6_b,
        # Career
        'career_win_pct_A': career_a,
        'career_win_pct_B': career_b,
        'career_win_pct_diff': career_a - career_b,
        'experience_diff': exp_a - exp_b,
        'weighted_recent_10_A': wr10_a,
        'weighted_recent_10_B': wr10_b,
        'weighted_recent_diff': wr10_a - wr10_b,
        # H2H
        'h2h_win_pct_A': h2h_pct_a,
        'h2h_total_matches': h2h_total,
        'has_h2h_history': 1 if h2h_total > 0 else 0,
        # Interactions
        'rank_x_form_diff': rank_a * win3_a - rank_b * win3_b,
        'elo_x_age_diff': (elo_a / (1 + np.exp((age_a - 30) / 3))) - (elo_b / (1 + np.exp((age_b - 30) / 3))),
        # Rank trajectory
        'rank_momentum_A': a.get('rank_momentum', 0.0),
        'rank_momentum_B': b.get('rank_momentum', 0.0),
        'rank_momentum_diff': a.get('rank_momentum', 0.0) - b.get('rank_momentum', 0.0),
        'basho_since_peak_A': a.get('basho_since_peak', 0),
        'basho_since_peak_B': b.get('basho_since_peak', 0),
        'basho_since_peak_diff': a.get('basho_since_peak', 0) - b.get('basho_since_peak', 0),
        'rank_gap_diff': a.get('rank_gap', 0.0) - b.get('rank_gap', 0.0),
        # In-tournament record
        'wins_before_A': wins_a,
        'losses_before_A': losses_a,
        'wins_before_B': wins_b,
        'losses_before_B': losses_b,
        'record_diff': (wins_a - losses_a) - (wins_b - losses_b),
        'day': day,
        # Pressure
        'is_kk_match_A': is_kk_a,
        'is_kk_match_B': is_kk_b,
        'is_mk_match_A': is_mk_a,
        'is_mk_match_B': is_mk_b,
        'kk_pressure_diff': is_kk_a - is_kk_b,
        'mk_pressure_diff': is_mk_a - is_mk_b,
        'pressure_diff': (is_kk_a - is_kk_b) - (is_mk_a - is_mk_b),
    }

    # Style ratios and vulnerability
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

    # Physical × style interactions
    f['height_x_oshi_diff'] = h_a * a.get('oshi_ratio', 0.2) - h_b * b.get('oshi_ratio', 0.2)
    f['height_x_yotsu_diff'] = h_a * a.get('yotsu_ratio', 0.2) - h_b * b.get('yotsu_ratio', 0.2)
    f['weight_x_yotsu_diff'] = w_a * a.get('yotsu_ratio', 0.2) - w_b * b.get('yotsu_ratio', 0.2)
    f['bmi_x_pull_diff'] = bm_a * a.get('pull_ratio', 0.2) - bm_b * b.get('pull_ratio', 0.2)
    f['age_x_oshi_diff'] = age_a * a.get('oshi_ratio', 0.2) - age_b * b.get('oshi_ratio', 0.2)
    f['age_x_yotsu_diff'] = age_a * a.get('yotsu_ratio', 0.2) - age_b * b.get('yotsu_ratio', 0.2)

    # elo_change_* features (added in v8 model)
    for key in [k for k in a if k.startswith('elo_change_')]:
        va = a.get(key, 0.0)
        vb = b.get(key, 0.0)
        f[f'{key}_A'] = va
        f[f'{key}_B'] = vb
        f[f'{key}_diff'] = va - vb

    return f


def predict_match_v2(model, features: List[str], a: dict, b: dict,
                     h2h_pct_a: float, h2h_total: int,
                     wins_a: int = 0, losses_a: int = 0,
                     wins_b: int = 0, losses_b: int = 0,
                     day: int = 1) -> float:
    """Predict P(A wins) using v7 model."""
    f = build_match_features(a, b, h2h_pct_a, h2h_total,
                             wins_a, losses_a, wins_b, losses_b, day)
    X = pd.DataFrame([f])[features]
    return model.predict_proba(X)[0][1]


def generate_matchups(wrestlers_df: pd.DataFrame, day: int,
                      records: Dict[int, Tuple[int, int]],
                      faced: Dict[int, set]) -> List[Tuple[int, int]]:
    """Generate matchups for a day."""
    matchups = []
    available = set(wrestlers_df['wrestler_id'].tolist())

    for div in ['makuuchi', 'juryo']:
        div_ids = wrestlers_df[wrestlers_df['division'] == div]['wrestler_id'].tolist()
        div_avail = [w for w in div_ids if w in available]

        if day <= 7:
            div_avail = sorted(div_avail,
                              key=lambda w: wrestlers_df[wrestlers_df['wrestler_id'] == w]['rank_score'].iloc[0],
                              reverse=True)
        else:
            div_avail = sorted(div_avail,
                              key=lambda w: (records.get(w, (0, 0))[0], -records.get(w, (0, 0))[1]),
                              reverse=True)

        while len(div_avail) >= 2:
            a = div_avail.pop(0)
            heya_a = wrestlers_df[wrestlers_df['wrestler_id'] == a]['heya'].iloc[0]
            faced_a = faced.get(a, set())

            found = False
            for i, b in enumerate(div_avail):
                heya_b = wrestlers_df[wrestlers_df['wrestler_id'] == b]['heya'].iloc[0]
                if heya_a != heya_b and b not in faced_a:
                    div_avail.pop(i)
                    matchups.append((a, b))
                    available.discard(a)
                    available.discard(b)
                    found = True
                    break

            if not found and div_avail:
                b = div_avail.pop(0)
                matchups.append((a, b))
                available.discard(a)
                available.discard(b)

    return matchups


def get_draft_category(rank_str: str) -> str:
    """Returns the fantasy draft category for a rank string."""
    div, num, _ = _parse_rank(rank_str)
    if div in ['Y', 'O']:
        return '1. Yokozuna/Ozeki'
    if div in ['S', 'K']:
        return '2. Sekiwake/Komusubi'
    if div == 'M':
        if num <= 5:
            return '3. Maegashira 1-5'
        if num <= 10:
            return '4. Maegashira 6-10'
        return '5. Maegashira 11+'
    if div == 'J':
        return '6. Juryo'
    return '7. Other'


def simulate_tournament(model_bundle: dict, wrestlers_df: pd.DataFrame,
                        h2h: Dict, n_sims: int = 500) -> pd.DataFrame:
    """Run tournament simulation tracking wins and fantasy scoring."""

    model = model_bundle['model']
    features = model_bundle['features']

    w_ids = wrestlers_df['wrestler_id'].tolist()
    w_dict = wrestlers_df.set_index('wrestler_id').to_dict('index')

    yokozuna_ids = set(wrestlers_df[wrestlers_df['rank'].str.match(r'^Y', na=False)]['wrestler_id'].tolist())
    maegashira_ids = set(wrestlers_df[wrestlers_df['rank'].str.match(r'^M', na=False)]['wrestler_id'].tolist())
    makuuchi_ids = set(wrestlers_df[wrestlers_df['division'] == 'makuuchi']['wrestler_id'].tolist())

    total_wins = {w: 0 for w in w_ids}
    kk_count = {w: 0 for w in w_ids}
    yusho_count = {w: 0.0 for w in w_ids}
    jun_yusho_count = {w: 0.0 for w in w_ids}
    kinboshi_count = {w: 0 for w in w_ids}

    print(f"Running {n_sims} simulations...")
    for sim in range(n_sims):
        if (sim + 1) % 100 == 0:
            print(f"  {sim + 1}/{n_sims}")

        records = {w: (0, 0) for w in w_ids}
        faced = {w: set() for w in w_ids}

        for day in range(1, 16):
            matchups = generate_matchups(wrestlers_df, day, records, faced)

            for a, b in matchups:
                h2h_pct, h2h_tot = get_h2h_stats(h2h, a, b)
                wa, la = records.get(a, (0, 0))
                wb, lb = records.get(b, (0, 0))
                prob = predict_match_v2(model, features, w_dict[a], w_dict[b], h2h_pct, h2h_tot,
                                        wins_a=wa, losses_a=la, wins_b=wb, losses_b=lb, day=day)

                if np.random.random() < prob:
                    winner, loser = a, b
                else:
                    winner, loser = b, a

                w, l = records[winner]
                records[winner] = (w + 1, l)
                w, l = records[loser]
                records[loser] = (w, l + 1)

                faced[a].add(b)
                faced[b].add(a)

                if winner in maegashira_ids and loser in yokozuna_ids:
                    kinboshi_count[winner] += 1

        for w_id, (wins, _) in records.items():
            total_wins[w_id] += wins
            if wins >= 8:
                kk_count[w_id] += 1

        # Yusho / jun-yusho (Makuuchi only)
        mk_wins = {w: records[w][0] for w in makuuchi_ids}
        if mk_wins:
            max_wins = max(mk_wins.values())
            yusho_winners = [w for w, wins in mk_wins.items() if wins == max_wins]
            for w in yusho_winners:
                yusho_count[w] += 1

            n_tied = len(yusho_winners)
            if n_tied == 1:
                second_max = max((wins for w, wins in mk_wins.items() if w not in yusho_winners), default=-1)
                if second_max >= 0:
                    for w in [w for w, wins in mk_wins.items() if wins == second_max]:
                        jun_yusho_count[w] += 1
            else:
                for w in yusho_winners:
                    jun_yusho_count[w] += (n_tied - 1) / n_tied

    results = wrestlers_df[['wrestler_id', 'rikishi', 'rank', 'division']].copy()
    results['draft_category'] = results['rank'].apply(get_draft_category)

    kk_prob = results['wrestler_id'].map(lambda w: kk_count[w] / n_sims)
    mk_prob = 1 - kk_prob

    results['expected_wins'] = results['wrestler_id'].map(lambda w: round(total_wins[w] / n_sims, 1))
    results['kk_probability'] = kk_prob.round(3)
    results['wins_pts'] = results['expected_wins']
    results['kk_mk_pts'] = ((kk_prob * 1.0) - (mk_prob * 0.5)).round(3)
    results['yusho_pts'] = results['wrestler_id'].map(lambda w: round((yusho_count[w] / n_sims) * 5.0, 3))
    results['jun_yusho_pts'] = results['wrestler_id'].map(lambda w: round((jun_yusho_count[w] / n_sims) * 3.0, 3))
    results['kinboshi_pts'] = results['wrestler_id'].map(lambda w: round((kinboshi_count[w] / n_sims) * 2.0, 3))

    # Juryo has no Makuuchi yusho eligibility
    results.loc[results['division'] == 'juryo', ['yusho_pts', 'jun_yusho_pts']] = 0.0

    results['predicted_fantasy_score'] = results[['wins_pts', 'kk_mk_pts', 'yusho_pts', 'jun_yusho_pts', 'kinboshi_pts']].sum(axis=1).round(2)

    return results.sort_values(['draft_category', 'predicted_fantasy_score'], ascending=[True, False])


def main(model_path: str, banzuke_path: str, match_path: str,
         tournament_id: int, n_sims: int):
    """Main entry point."""

    model_bundle, banzuke, match_history = load_data(model_path, banzuke_path, match_path)

    print(f"\nPreparing wrestlers for {tournament_id}...")
    wrestlers = prepare_wrestlers(banzuke, match_history, tournament_id)

    if wrestlers.empty:
        print("No wrestlers found!")
        return

    print(f"Found {len(wrestlers)} wrestlers")

    print("Computing H2H history...")
    h2h = compute_h2h(match_history, tournament_id)

    results = simulate_tournament(model_bundle, wrestlers, h2h, n_sims)

    print(f"\n{'='*70}")
    print(f"PREDICTIONS FOR {tournament_id} ({n_sims} simulations)")
    print(f"{'='*70}")

    display_cols = ['rikishi', 'rank', 'expected_wins', 'kk_probability', 'kk_mk_pts', 'yusho_pts', 'jun_yusho_pts', 'kinboshi_pts', 'predicted_fantasy_score']
    for category in sorted(results['draft_category'].unique()):
        group = results[results['draft_category'] == category]
        print(f"\n--- {category.split('. ', 1)[1]} ---")
        print(group[display_cols].to_string(index=False))

    results.to_csv(f"simulation_v2_{tournament_id}.csv", index=False)
    print(f"\nSaved to simulation_v2_{tournament_id}.csv")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tournament', type=int, default=202511)
    parser.add_argument('--simulations', type=int, default=500)
    parser.add_argument('--model', type=str, default="matchup_model_v2.joblib")
    parser.add_argument('--banzuke', type=str, default="banzuke_detailed.csv")
    parser.add_argument('--matches', type=str, default="match_history_with_kimarite.csv")
    args = parser.parse_args()

    main(args.model, args.banzuke, args.matches, args.tournament, args.simulations)

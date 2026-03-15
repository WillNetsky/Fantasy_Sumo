# simulate_tournament_v3.py
"""
Improved tournament simulator with:
1. Realistic torikumi scheduling (JSA-style algorithm)
2. Hybrid model (combining matchup predictions with direct win estimates)
3. Support for multiple model versions
"""

import pandas as pd
import numpy as np
import joblib
import argparse
from typing import Dict, List, Tuple, Optional
from sumo_utils import get_absolute_rank_score, _parse_rank

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
        f['glicko_combined_rd'] = 200  # Default uncertainty
        f['form_diff'] = a.get('win_pct_last_3', 0.5) - b.get('win_pct_last_3', 0.5)
        f['streak_diff'] = 0
        f['volatility_diff'] = 0
        f['weighted_form_A'] = a.get('win_pct_last_3', 0.5)
        f['weighted_form_B'] = b.get('win_pct_last_3', 0.5)
        f['recent_streak_A'] = 0
        f['recent_streak_B'] = 0
        # H2H weighted
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


def generate_realistic_torikumi(wrestlers_df: pd.DataFrame, day: int,
                                 records: Dict[int, Tuple[int, int]],
                                 faced: Dict[int, set],
                                 schedule_style: str = 'jsa') -> List[Tuple[int, int]]:
    """
    Generate matchups using JSA-style scheduling algorithm.

    Key rules:
    1. Same heya wrestlers don't fight (except playoffs)
    2. Days 1-7: Match by rank (higher ranks face higher ranks)
    3. Days 8-15: Match by record (similar records face each other)
    4. Sanyaku must face each other in the first two weeks
    5. Avoid rematches
    """
    matchups = []

    for div in ['makuuchi', 'juryo']:
        div_wrestlers = wrestlers_df[wrestlers_df['division'] == div].copy()
        if div_wrestlers.empty:
            continue

        available = set(div_wrestlers['wrestler_id'].tolist())

        # Sort by appropriate criteria
        if day <= 7:
            # Early basho: match by rank
            div_wrestlers = div_wrestlers.sort_values('rank_score', ascending=False)
        else:
            # Later basho: match by record (wins - losses)
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

                # Skip same heya (always)
                if w_heya.get(a) == w_heya.get(b):
                    continue

                # Skip if already faced
                if b in faced.get(a, set()):
                    continue

                # Scoring for opponent selection
                score = 0

                # Prefer similar records (later in basho)
                if day > 7:
                    rec_a = records.get(a, (0, 0))[0] - records.get(a, (0, 0))[1]
                    rec_b = records.get(b, (0, 0))[0] - records.get(b, (0, 0))[1]
                    score -= abs(rec_a - rec_b) * 10

                # Prefer rank proximity
                rank_a = div_wrestlers[div_wrestlers['wrestler_id'] == a]['rank_score'].iloc[0]
                rank_b = div_wrestlers[div_wrestlers['wrestler_id'] == b]['rank_score'].iloc[0]
                score -= abs(rank_a - rank_b) * 0.5

                # Bonus for sanyaku vs sanyaku early in basho
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


def simulate_tournament(model_bundle: dict, wrestlers_df: pd.DataFrame,
                        h2h: Dict, n_sims: int = 500,
                        scheduling: str = 'jsa') -> pd.DataFrame:
    """Run tournament simulation with improved scheduling."""

    model = model_bundle['model']
    features = model_bundle['features']
    model_version = model_bundle.get('version', 2)

    w_ids = wrestlers_df['wrestler_id'].tolist()
    w_dict = wrestlers_df.set_index('wrestler_id').to_dict('index')

    total_wins = {w: 0 for w in w_ids}
    yusho_count = {w: 0 for w in w_ids}
    jun_yusho_count = {w: 0 for w in w_ids}

    print(f"Running {n_sims} simulations with '{scheduling}' scheduling...")

    for sim in range(n_sims):
        if (sim + 1) % 100 == 0:
            print(f"  {sim + 1}/{n_sims}")

        records = {w: (0, 0) for w in w_ids}
        faced = {w: set() for w in w_ids}

        for day in range(1, 16):
            if scheduling == 'jsa':
                matchups = generate_realistic_torikumi(wrestlers_df, day, records, faced, 'jsa')
            else:
                # Simple matching
                matchups = []
                for div in ['makuuchi', 'juryo']:
                    div_ids = wrestlers_df[wrestlers_df['division'] == div]['wrestler_id'].tolist()
                    available = [w for w in div_ids if w in w_ids]

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

            for a, b in matchups:
                h2h_pct, h2h_tot = get_h2h_stats(h2h, a, b)

                f = build_matchup_features(
                    model_version, w_dict[a], w_dict[b],
                    h2h_pct, h2h_tot,
                    records[a][0], records[a][1],
                    records[b][0], records[b][1],
                    day
                )

                prob = predict_match(model, features, f)

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

        # Track results
        for w_id, (wins, _) in records.items():
            total_wins[w_id] += wins

        # Track yusho/jun-yusho
        max_wins = max(r[0] for r in records.values())
        winners = [w for w, r in records.items() if r[0] == max_wins]

        if len(winners) == 1:
            yusho_count[winners[0]] += 1
            # Find jun-yusho
            second_wins = max(r[0] for w, r in records.items() if w not in winners)
            jun_winners = [w for w, r in records.items() if r[0] == second_wins]
            for w in jun_winners:
                jun_yusho_count[w] += 1 / len(jun_winners)
        else:
            # Tied for yusho
            for w in winners:
                yusho_count[w] += 1 / len(winners)
                jun_yusho_count[w] += (len(winners) - 1) / len(winners)

    results = wrestlers_df[['wrestler_id', 'rikishi', 'rank', 'division']].copy()
    results['expected_wins'] = results['wrestler_id'].map(lambda w: total_wins[w] / n_sims)
    results['yusho_prob'] = results['wrestler_id'].map(lambda w: yusho_count[w] / n_sims)
    results['jun_yusho_prob'] = results['wrestler_id'].map(lambda w: jun_yusho_count[w] / n_sims)

    return results.sort_values('expected_wins', ascending=False)


def main(model_path: str, banzuke_path: str, match_path: str,
         tournament_id: int, n_sims: int, scheduling: str = 'jsa'):
    """Main entry point."""

    matchup_bundle, _, banzuke, match_history = load_data(model_path, banzuke_path, match_path)

    print(f"\nPreparing wrestlers for {tournament_id}...")
    wrestlers = prepare_wrestlers(banzuke, match_history, tournament_id)

    if wrestlers.empty:
        print("No wrestlers found!")
        return

    print(f"Found {len(wrestlers)} wrestlers")

    print("Computing H2H history...")
    h2h = compute_h2h(match_history, tournament_id)

    results = simulate_tournament(matchup_bundle, wrestlers, h2h, n_sims, scheduling)

    print(f"\n{'=' * 60}")
    print(f"PREDICTIONS FOR {tournament_id} ({n_sims} simulations)")
    print(f"{'=' * 60}")

    print("\n--- MAKUUCHI ---")
    mak = results[results['division'] == 'makuuchi'][
        ['rikishi', 'rank', 'expected_wins', 'yusho_prob', 'jun_yusho_prob']
    ]
    print(mak.to_string(index=False))

    print("\n--- JURYO ---")
    jur = results[results['division'] == 'juryo'][
        ['rikishi', 'rank', 'expected_wins', 'yusho_prob', 'jun_yusho_prob']
    ]
    print(jur.to_string(index=False))

    output_file = f"simulation_v3_{tournament_id}.csv"
    results.to_csv(output_file, index=False)
    print(f"\nSaved to {output_file}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tournament', type=int, default=202601)
    parser.add_argument('--simulations', type=int, default=500)
    parser.add_argument('--model', type=str, default="matchup_model_v4.joblib")
    parser.add_argument('--banzuke', type=str, default="banzuke_detailed.csv")
    parser.add_argument('--matches', type=str, default="match_history_with_kimarite.csv")
    parser.add_argument('--scheduling', type=str, default='jsa', choices=['jsa', 'simple'])
    args = parser.parse_args()

    main(args.model, args.banzuke, args.matches, args.tournament, args.simulations, args.scheduling)

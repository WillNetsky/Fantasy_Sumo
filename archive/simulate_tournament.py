# simulate_tournament.py
"""
Tournament simulator using matchup prediction model.
Simulates a full 15-day tournament to predict total wins.
"""

import pandas as pd
import numpy as np
import joblib
import argparse
from typing import Dict, List, Tuple
from sumo_utils import get_absolute_rank_score, _parse_rank


def load_models_and_data(matchup_model_path: str, banzuke_path: str, match_history_path: str):
    """Load the matchup model and historical data."""
    print("Loading matchup model...")
    matchup_bundle = joblib.load(matchup_model_path)

    print("Loading historical data...")
    banzuke = pd.read_csv(banzuke_path)
    match_history = pd.read_csv(match_history_path)

    # Convert types
    banzuke['wrestler_id'] = pd.to_numeric(banzuke['wrestler_id'], errors='coerce')
    banzuke['tournament_id'] = pd.to_numeric(banzuke['tournament_id'], errors='coerce')
    match_history['winner_id'] = pd.to_numeric(match_history['winner_id'], errors='coerce')
    match_history['loser_id'] = pd.to_numeric(match_history['loser_id'], errors='coerce')
    match_history['tournament_id'] = pd.to_numeric(match_history['tournament_id'], errors='coerce')

    return matchup_bundle, banzuke, match_history


def prepare_wrestler_features(banzuke: pd.DataFrame, match_history: pd.DataFrame,
                              tournament_id: int) -> pd.DataFrame:
    """Prepare features for all wrestlers in a tournament."""

    # Get wrestlers for this tournament
    tourney_wrestlers = banzuke[banzuke['tournament_id'] == tournament_id].copy()

    # Filter for top divisions
    def is_top_division(rank_str):
        div, _, _ = _parse_rank(rank_str)
        return div in ['Y', 'O', 'S', 'K', 'M', 'J']

    tourney_wrestlers = tourney_wrestlers[tourney_wrestlers['rank'].apply(is_top_division)]

    if tourney_wrestlers.empty:
        return pd.DataFrame()

    # Historical data (before this tournament)
    historical = banzuke[banzuke['tournament_id'] < tournament_id].copy()

    # Compute features for each wrestler
    wrestlers = []
    for _, row in tourney_wrestlers.iterrows():
        wrestler_id = row['wrestler_id']
        wrestler_history = historical[historical['wrestler_id'] == wrestler_id]

        # Rank score
        rank_score = get_absolute_rank_score(row['rank'])

        # Rank in division
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

        # Physical attributes
        hw_match = pd.Series([row.get('height_weight', '')]).str.extract(
            r'(\d+\.?\d*)\s*cm\s*(\d+\.?\d*)\s*kg'
        )
        height = pd.to_numeric(hw_match[0].iloc[0], errors='coerce') if len(hw_match) > 0 else np.nan
        weight = pd.to_numeric(hw_match[1].iloc[0], errors='coerce') if len(hw_match) > 0 else np.nan
        bmi = weight / ((height / 100) ** 2) if height and weight else np.nan

        # Age
        birth_date = pd.to_datetime(row.get('birth_date'), format='%d.%m.%Y', errors='coerce')
        tourney_date = pd.to_datetime(str(tournament_id), format='%Y%m')
        age = (tourney_date - birth_date).days / 365.25 if pd.notna(birth_date) else np.nan

        # Recent form (win % in last 3 basho)
        if not wrestler_history.empty:
            recent = wrestler_history.tail(3)
            recent_w = pd.to_numeric(recent['w'], errors='coerce').sum()
            recent_l = pd.to_numeric(recent['l'], errors='coerce').sum()
            win_pct_last_3 = recent_w / (recent_w + recent_l) if (recent_w + recent_l) > 0 else 0.5
        else:
            win_pct_last_3 = 0.5

        # Oshi ratio (from match history)
        oshi_techniques = ['oshidashi', 'tsukidashi', 'tsukiotoshi', 'hatakikomi', 'oshitaoshi', 'tsukitaoshi']
        wrestler_wins = match_history[
            (match_history['winner_id'] == wrestler_id) &
            (match_history['tournament_id'] < tournament_id)
        ]
        if not wrestler_wins.empty:
            oshi_wins = wrestler_wins['kimarite'].isin(oshi_techniques).sum()
            yotsu_techniques = ['yorikiri', 'uwatenage', 'shitatenage', 'yoritaoshi', 'kotenage', 'sukuinage', 'utchari']
            yotsu_wins = wrestler_wins['kimarite'].isin(yotsu_techniques).sum()
            total_style = oshi_wins + yotsu_wins
            oshi_ratio = oshi_wins / total_style if total_style > 0 else 0.5
        else:
            oshi_ratio = 0.5

        wrestlers.append({
            'wrestler_id': wrestler_id,
            'rikishi': row['rikishi'],
            'rank': row['rank'],
            'rank_score': rank_score,
            'rank_in_div': rank_in_div,
            'division': 'makuuchi' if div in ['Y', 'O', 'S', 'K', 'M'] else 'juryo',
            'height': height,
            'weight': weight,
            'bmi': bmi,
            'age': age,
            'win_pct_last_3': win_pct_last_3,
            'oshi_ratio': oshi_ratio,
            'heya': row.get('heya', '')
        })

    df = pd.DataFrame(wrestlers)

    # Fill missing values with medians
    for col in ['height', 'weight', 'bmi', 'age']:
        df[col] = df[col].fillna(df[col].median())

    return df


def compute_h2h_history(match_history: pd.DataFrame, tournament_id: int) -> Dict:
    """Compute H2H history for all wrestler pairs before a tournament."""

    historical_matches = match_history[match_history['tournament_id'] < tournament_id].copy()

    h2h = {}
    for _, match in historical_matches.iterrows():
        w1 = min(match['winner_id'], match['loser_id'])
        w2 = max(match['winner_id'], match['loser_id'])
        winner = match['winner_id']

        key = (w1, w2)
        if key not in h2h:
            h2h[key] = {'w1_wins': 0, 'w2_wins': 0}

        if winner == w1:
            h2h[key]['w1_wins'] += 1
        else:
            h2h[key]['w2_wins'] += 1

    return h2h


def get_h2h_win_pct(h2h: Dict, wrestler_a: int, wrestler_b: int) -> Tuple[float, int]:
    """Get wrestler A's win percentage against wrestler B."""
    w1 = min(wrestler_a, wrestler_b)
    w2 = max(wrestler_a, wrestler_b)
    key = (w1, w2)

    if key not in h2h:
        return 0.5, 0  # No history

    record = h2h[key]
    total = record['w1_wins'] + record['w2_wins']

    if wrestler_a == w1:
        return record['w1_wins'] / total, total
    else:
        return record['w2_wins'] / total, total


def generate_matchups_day(wrestlers_df: pd.DataFrame, day: int,
                          current_records: Dict[int, Tuple[int, int]],
                          already_faced: Dict[int, set]) -> List[Tuple[int, int]]:
    """
    Generate matchups for a specific day.
    Days 1-7: Mostly by rank
    Days 8-15: By current record (winners vs winners)
    """

    matchups = []
    available = set(wrestlers_df['wrestler_id'].tolist())

    # Separate by division
    makuuchi = wrestlers_df[wrestlers_df['division'] == 'makuuchi']['wrestler_id'].tolist()
    juryo = wrestlers_df[wrestlers_df['division'] == 'juryo']['wrestler_id'].tolist()

    for division_wrestlers in [makuuchi, juryo]:
        div_available = [w for w in division_wrestlers if w in available]

        if day <= 7:
            # Early days: match by rank (with some randomness)
            div_available_sorted = sorted(div_available,
                                         key=lambda w: wrestlers_df[wrestlers_df['wrestler_id'] == w]['rank_score'].iloc[0],
                                         reverse=True)
        else:
            # Later days: match by current record
            div_available_sorted = sorted(div_available,
                                         key=lambda w: (current_records.get(w, (0, 0))[0],  # wins desc
                                                       -current_records.get(w, (0, 0))[1]),  # losses asc
                                         reverse=True)

        # Pair up wrestlers
        while len(div_available_sorted) >= 2:
            wrestler_a = div_available_sorted.pop(0)

            # Find opponent (hasn't faced yet, different heya)
            heya_a = wrestlers_df[wrestlers_df['wrestler_id'] == wrestler_a]['heya'].iloc[0]
            faced_a = already_faced.get(wrestler_a, set())

            opponent_found = False
            for i, wrestler_b in enumerate(div_available_sorted):
                heya_b = wrestlers_df[wrestlers_df['wrestler_id'] == wrestler_b]['heya'].iloc[0]

                # Can't face same heya, can't face someone already faced
                if heya_a != heya_b and wrestler_b not in faced_a:
                    div_available_sorted.pop(i)
                    matchups.append((wrestler_a, wrestler_b))
                    available.discard(wrestler_a)
                    available.discard(wrestler_b)
                    opponent_found = True
                    break

            if not opponent_found and div_available_sorted:
                # Fallback: just pick next available
                wrestler_b = div_available_sorted.pop(0)
                matchups.append((wrestler_a, wrestler_b))
                available.discard(wrestler_a)
                available.discard(wrestler_b)

    return matchups


def predict_match(model, features: List[str], wrestler_a: dict, wrestler_b: dict,
                  h2h_win_pct_a: float, h2h_total: int,
                  current_wins_a: int, current_wins_b: int) -> float:
    """Predict probability that wrestler A beats wrestler B."""

    feature_values = {
        'rank_diff': wrestler_a['rank_score'] - wrestler_b['rank_score'],
        'rank_in_div_diff': wrestler_a['rank_in_div'] - wrestler_b['rank_in_div'],
        'bmi_diff': wrestler_a['bmi'] - wrestler_b['bmi'],
        'height_diff': wrestler_a['height'] - wrestler_b['height'],
        'weight_diff': wrestler_a['weight'] - wrestler_b['weight'],
        'age_diff': wrestler_a['age'] - wrestler_b['age'],
        'oshi_ratio_A': wrestler_a['oshi_ratio'],
        'oshi_ratio_B': wrestler_b['oshi_ratio'],
        'oshi_ratio_diff': wrestler_a['oshi_ratio'] - wrestler_b['oshi_ratio'],
        'h2h_win_pct_A': h2h_win_pct_a,
        'h2h_total_matches': h2h_total,
        'win_pct_last_3_A': wrestler_a['win_pct_last_3'],
        'win_pct_last_3_B': wrestler_b['win_pct_last_3'],
        'win_pct_last_3_diff': wrestler_a['win_pct_last_3'] - wrestler_b['win_pct_last_3'],
    }

    X = pd.DataFrame([feature_values])[features]
    prob = model.predict_proba(X)[0][1]

    return prob


def simulate_tournament(matchup_bundle: dict, wrestlers_df: pd.DataFrame,
                        h2h: Dict, n_simulations: int = 1000) -> pd.DataFrame:
    """Simulate a tournament multiple times and return expected wins."""

    model = matchup_bundle['model']
    features = matchup_bundle['features']

    wrestler_ids = wrestlers_df['wrestler_id'].tolist()
    wrestler_dict = wrestlers_df.set_index('wrestler_id').to_dict('index')

    # Track wins across simulations
    total_wins = {w: 0 for w in wrestler_ids}

    print(f"Running {n_simulations} tournament simulations...")

    for sim in range(n_simulations):
        if (sim + 1) % 100 == 0:
            print(f"  Simulation {sim + 1}/{n_simulations}")

        # Track state for this simulation
        current_records = {w: (0, 0) for w in wrestler_ids}  # (wins, losses)
        already_faced = {w: set() for w in wrestler_ids}

        # Simulate each day
        for day in range(1, 16):
            matchups = generate_matchups_day(wrestlers_df, day, current_records, already_faced)

            for wrestler_a, wrestler_b in matchups:
                # Get H2H history
                h2h_win_pct_a, h2h_total = get_h2h_win_pct(h2h, wrestler_a, wrestler_b)

                # Get current wins
                wins_a, _ = current_records[wrestler_a]
                wins_b, _ = current_records[wrestler_b]

                # Predict match outcome
                prob_a_wins = predict_match(
                    model, features,
                    wrestler_dict[wrestler_a], wrestler_dict[wrestler_b],
                    h2h_win_pct_a, h2h_total,
                    wins_a, wins_b
                )

                # Simulate outcome
                if np.random.random() < prob_a_wins:
                    winner, loser = wrestler_a, wrestler_b
                else:
                    winner, loser = wrestler_b, wrestler_a

                # Update records
                w, l = current_records[winner]
                current_records[winner] = (w + 1, l)
                w, l = current_records[loser]
                current_records[loser] = (w, l + 1)

                # Track who faced whom
                already_faced[wrestler_a].add(wrestler_b)
                already_faced[wrestler_b].add(wrestler_a)

        # Accumulate wins for this simulation
        for w_id, (wins, _) in current_records.items():
            total_wins[w_id] += wins

    # Calculate expected wins
    results = wrestlers_df[['wrestler_id', 'rikishi', 'rank', 'division']].copy()
    results['expected_wins'] = results['wrestler_id'].map(
        lambda w: total_wins[w] / n_simulations
    )
    results = results.sort_values('expected_wins', ascending=False)

    return results


def main(matchup_model_path: str, banzuke_path: str, match_history_path: str,
         tournament_id: int, n_simulations: int):
    """Main function to run tournament simulation."""

    # Load data
    matchup_bundle, banzuke, match_history = load_models_and_data(
        matchup_model_path, banzuke_path, match_history_path
    )

    # Prepare wrestler features
    print(f"\nPreparing wrestler features for tournament {tournament_id}...")
    wrestlers_df = prepare_wrestler_features(banzuke, match_history, tournament_id)

    if wrestlers_df.empty:
        print("No wrestlers found for this tournament!")
        return

    print(f"Found {len(wrestlers_df)} wrestlers in top divisions")

    # Compute H2H history
    print("Computing head-to-head history...")
    h2h = compute_h2h_history(match_history, tournament_id)
    print(f"Found {len(h2h)} historical matchup pairs")

    # Run simulations
    results = simulate_tournament(matchup_bundle, wrestlers_df, h2h, n_simulations)

    # Display results
    print(f"\n{'='*60}")
    print(f"TOURNAMENT {tournament_id} - PREDICTED WINS (via {n_simulations} simulations)")
    print(f"{'='*60}")

    print("\n--- MAKUUCHI ---")
    mak = results[results['division'] == 'makuuchi']
    print(mak[['rikishi', 'rank', 'expected_wins']].to_string(index=False))

    print("\n--- JURYO ---")
    jur = results[results['division'] == 'juryo']
    print(jur[['rikishi', 'rank', 'expected_wins']].to_string(index=False))

    # Save results
    output_file = f"simulation_predictions_{tournament_id}.csv"
    results.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate tournament using matchup model.")
    parser.add_argument('--tournament', type=int, default=202511,
                        help="Tournament ID to simulate (e.g., 202511)")
    parser.add_argument('--simulations', type=int, default=1000,
                        help="Number of tournament simulations to run")
    parser.add_argument('--matchup-model', type=str, default="matchup_model.joblib")
    parser.add_argument('--banzuke-file', type=str, default="banzuke_detailed.csv")
    parser.add_argument('--match-file', type=str, default="match_history_with_kimarite.csv")
    args = parser.parse_args()

    main(args.matchup_model, args.banzuke_file, args.match_file,
         args.tournament, args.simulations)

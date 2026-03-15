# predict_wins_hybrid.py
"""
Hybrid prediction system combining:
1. Direct win prediction model (LightGBM regressor)
2. Matchup simulation model (classifier)
3. Weighted ensemble for optimal predictions

This maximizes predictive power by leveraging:
- Direct model: Better at predicting total wins (lower MAE)
- Matchup model: Captures opponent-specific dynamics (style, H2H)
"""

import pandas as pd
import numpy as np
import joblib
import argparse
import requests
import re
from bs4 import BeautifulSoup
from scipy.stats import norm
from concurrent.futures import ThreadPoolExecutor
from sumo_utils import preprocess_data, _parse_rank
from simulate_tournament_v3 import (
    prepare_wrestlers, compute_h2h, get_h2h_stats,
    build_matchup_features, generate_realistic_torikumi,
    TECHNIQUE_CATEGORIES
)

# Configuration
BASE_URL = "https://sumodb.sumogames.de/"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}
BANZUKE_PARAMS = "&heya=-1&shusshin=-1&h=on&w=on&bd=on&hd=on&su=on&hr=on&ho=on&ch=on&cr=on&spr=on&sps=on&c=on&snr=on&sns=on&simple=on"


def scrape_banzuke(basho_id: int) -> pd.DataFrame:
    """Scrape banzuke from SumoDB."""
    url = f"{BASE_URL}Banzuke.aspx?b={basho_id}{BANZUKE_PARAMS}"
    print(f"Scraping banzuke from: {url}")

    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL: {e}")
        return pd.DataFrame()

    soup = BeautifulSoup(response.content, 'lxml')
    banzuke_table = soup.find('table', class_='banzuke')
    if not banzuke_table:
        print("Could not find banzuke table.")
        return pd.DataFrame()

    header_row = None
    thead = banzuke_table.find('thead')
    if thead:
        header_row = thead.find('tr')
    if not header_row:
        first_row = banzuke_table.find('tr')
        if first_row and first_row.find('th'):
            header_row = first_row

    if not header_row:
        print("Could not find header row.")
        return pd.DataFrame()

    headers = [th.get_text(strip=True) for th in header_row.find_all('th')]
    clean_headers = [re.sub(r'[^a-z0-9_]+', '_', h.lower()).strip('_') for h in headers]

    try:
        shikona_col_index = clean_headers.index('rikishi')
    except ValueError:
        print("Could not find 'rikishi' column.")
        return pd.DataFrame()

    wrestlers_data = []
    all_rows = banzuke_table.find_all('tr')
    start_index = 1 if all_rows and all_rows[0] == header_row else 0

    for row in all_rows[start_index:]:
        if row.find('th'):
            continue
        cells = row.find_all('td')
        if len(cells) != len(clean_headers):
            continue
        data = {header: cell.get_text(strip=True) for header, cell in zip(clean_headers, cells)}
        shikona_anchor = cells[shikona_col_index].find('a')
        if shikona_anchor and 'href' in shikona_anchor.attrs:
            data['rikishi'] = shikona_anchor.get_text(strip=True)
            id_match = re.search(r'r=(\d+)', shikona_anchor['href'])
            data['wrestler_id'] = id_match.group(1) if id_match else None
        data['tournament_id'] = basho_id
        data['w'] = pd.to_numeric(data.get('w'), errors='coerce')
        data['l'] = pd.to_numeric(data.get('l'), errors='coerce')
        data['hoshitori'] = np.nan
        wrestlers_data.append(data)

    df = pd.DataFrame(wrestlers_data)
    print(f"Scraped {len(df)} wrestlers for basho {basho_id}")
    return df


def predict_direct_wins(model_bundle: dict, df_to_predict: pd.DataFrame) -> pd.DataFrame:
    """Predict wins using direct model."""
    model = model_bundle['model']
    features = model_bundle['features']

    X = df_to_predict[features].copy()
    valid_mask = X.notna().all(axis=1)
    X = X[valid_mask]

    if X.empty:
        return pd.DataFrame()

    predictions = model.predict(X)

    results = df_to_predict[valid_mask][['wrestler_id', 'rikishi']].copy()
    results['direct_wins'] = np.round(predictions, 1)

    return results


def predict_matchup_wins(matchup_bundle: dict, wrestlers_df: pd.DataFrame,
                          h2h: dict, n_sims: int = 300) -> pd.DataFrame:
    """Predict wins using matchup simulation."""

    model = matchup_bundle['model']
    features = matchup_bundle['features']
    model_version = matchup_bundle.get('version', 2)

    w_ids = wrestlers_df['wrestler_id'].tolist()
    w_dict = wrestlers_df.set_index('wrestler_id').to_dict('index')

    total_wins = {w: 0 for w in w_ids}

    print(f"Running {n_sims} simulations for matchup predictions...")

    for sim in range(n_sims):
        if (sim + 1) % 100 == 0:
            print(f"  {sim + 1}/{n_sims}")

        records = {w: (0, 0) for w in w_ids}
        faced = {w: set() for w in w_ids}

        for day in range(1, 16):
            matchups = generate_realistic_torikumi(wrestlers_df, day, records, faced, 'jsa')

            for a, b in matchups:
                h2h_pct, h2h_tot = get_h2h_stats(h2h, a, b)

                f = build_matchup_features(
                    model_version, w_dict[a], w_dict[b],
                    h2h_pct, h2h_tot,
                    records[a][0], records[a][1],
                    records[b][0], records[b][1],
                    day
                )

                try:
                    X = pd.DataFrame([f])[features]
                    prob = model.predict_proba(X)[0][1]
                except Exception:
                    prob = 0.5

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

        for w_id, (wins, _) in records.items():
            total_wins[w_id] += wins

    results = wrestlers_df[['wrestler_id', 'rikishi']].copy()
    results['matchup_wins'] = results['wrestler_id'].map(lambda w: total_wins[w] / n_sims)

    return results


def hybrid_prediction(direct_bundle: dict, matchup_bundle: dict,
                       historical_df: pd.DataFrame, match_history_df: pd.DataFrame,
                       basho_id: int, direct_weight: float = 0.7) -> pd.DataFrame:
    """
    Generate hybrid predictions combining direct and matchup models.

    Args:
        direct_weight: Weight for direct model (0.0-1.0). Remainder goes to matchup model.
                      Default 0.7 as direct model has lower MAE.
    """
    print(f"\n=== HYBRID PREDICTION FOR BASHO {basho_id} ===")
    print(f"Weights: Direct={direct_weight:.1%}, Matchup={1-direct_weight:.1%}")

    # Scrape current banzuke
    new_basho_df = scrape_banzuke(basho_id)
    if new_basho_df.empty:
        print("Failed to scrape banzuke.")
        return pd.DataFrame()

    is_completed = not pd.to_numeric(new_basho_df['w'], errors='coerce').isnull().all()

    # Filter to top divisions
    def is_top_div(rank_str):
        div, _, _ = _parse_rank(rank_str)
        return div in ['Y', 'O', 'S', 'K', 'M', 'J']

    new_basho_df = new_basho_df[new_basho_df['rank'].apply(is_top_div)].copy()

    # Prepare for direct model
    historical_df['wrestler_id'] = pd.to_numeric(historical_df['wrestler_id'], errors='coerce')
    new_basho_df['wrestler_id'] = pd.to_numeric(new_basho_df['wrestler_id'], errors='coerce')

    historical_df['tournament_id'] = pd.to_numeric(historical_df['tournament_id'], errors='coerce')
    historical_df = historical_df[historical_df['tournament_id'] != basho_id]

    combined_df = pd.concat([historical_df, new_basho_df], ignore_index=True)

    print("\nPreprocessing data for direct model...")
    df_processed = preprocess_data(combined_df.copy(), match_history_df=match_history_df)
    df_to_predict = df_processed[df_processed['tournament_id'] == basho_id].copy()

    # Direct predictions
    print("\nGenerating direct model predictions...")
    direct_results = predict_direct_wins(direct_bundle, df_to_predict)

    # Matchup predictions
    print("\nGenerating matchup model predictions...")

    # Need to include the new basho in the combined data for prepare_wrestlers
    combined_for_matchup = pd.concat([historical_df, new_basho_df], ignore_index=True)
    wrestlers_df = prepare_wrestlers(combined_for_matchup, match_history_df, basho_id)
    if wrestlers_df.empty:
        print("Failed to prepare wrestlers for matchup model.")
        return pd.DataFrame()

    # Merge scraped data for names
    wrestlers_df = wrestlers_df.merge(
        new_basho_df[['wrestler_id', 'rikishi']].drop_duplicates(),
        on='wrestler_id', how='left', suffixes=('', '_scraped')
    )
    if 'rikishi_scraped' in wrestlers_df.columns:
        wrestlers_df['rikishi'] = wrestlers_df['rikishi_scraped'].fillna(wrestlers_df['rikishi'])
        wrestlers_df = wrestlers_df.drop(columns=['rikishi_scraped'])

    h2h = compute_h2h(match_history_df, basho_id)
    matchup_results = predict_matchup_wins(matchup_bundle, wrestlers_df, h2h, n_sims=300)

    # Combine predictions
    print("\nCombining predictions...")
    combined = pd.merge(direct_results, matchup_results, on=['wrestler_id', 'rikishi'], how='outer')

    # Fill missing with the other model's prediction
    combined['direct_wins'] = combined['direct_wins'].fillna(combined['matchup_wins'])
    combined['matchup_wins'] = combined['matchup_wins'].fillna(combined['direct_wins'])

    # Hybrid prediction
    combined['hybrid_wins'] = (
        direct_weight * combined['direct_wins'] +
        (1 - direct_weight) * combined['matchup_wins']
    )

    # Merge rank and division info
    rank_info = df_to_predict[['wrestler_id', 'rank', 'division_numeric', 'rank_in_division', 'heya']].drop_duplicates()
    combined = combined.merge(rank_info, on='wrestler_id', how='left')

    # Add actual wins if completed
    if is_completed:
        actuals = new_basho_df[['wrestler_id', 'w']].rename(columns={'w': 'actual_wins'})
        combined = combined.merge(actuals, on='wrestler_id', how='left')
    else:
        combined['actual_wins'] = np.nan

    return combined


def calculate_fantasy_scores(results_df: pd.DataFrame, model_mae: float = 1.7) -> pd.DataFrame:
    """Calculate fantasy scores from hybrid predictions."""
    df = results_df.copy()
    df['predicted_wins'] = df['hybrid_wins']

    # Win points
    df['wins_pts'] = df['predicted_wins'].fillna(0)

    # KK/MK probability
    prob_kk = 1 - norm.cdf(7.5, loc=df['predicted_wins'], scale=model_mae)
    prob_mk = norm.cdf(7.5, loc=df['predicted_wins'], scale=model_mae)
    df['kk_mk_pts'] = (prob_kk * 1.0) - (prob_mk * 0.5)
    df['kk_mk_pts'] = df['kk_mk_pts'].fillna(0)

    # Yusho/Jun-yusho (Monte Carlo)
    df['yusho_pts'] = 0.0
    df['jun_yusho_pts'] = 0.0

    makuuchi_mask = (df['division_numeric'] == 2) & (df['predicted_wins'].notna())
    makuuchi_df = df[makuuchi_mask]

    if not makuuchi_df.empty:
        n_sims = 1000
        locs = makuuchi_df['predicted_wins'].values
        scales = np.full_like(locs, model_mae)
        simulated_wins = np.random.normal(loc=locs, scale=scales, size=(n_sims, len(locs)))
        simulated_wins = np.round(simulated_wins).astype(int)

        yusho_counts = np.zeros(len(locs))
        jun_yusho_counts = np.zeros(len(locs))

        for i in range(n_sims):
            wins_this_sim = simulated_wins[i, :]
            yusho_wins = np.max(wins_this_sim)
            yusho_winners = np.where(wins_this_sim == yusho_wins)[0]
            yusho_counts[yusho_winners] += 1

            if len(yusho_winners) == 1:
                second_wins = max(w for j, w in enumerate(wins_this_sim) if j not in yusho_winners)
                jun_winners = np.where(wins_this_sim == second_wins)[0]
                jun_yusho_counts[jun_winners] += 1 / len(jun_winners)
            else:
                for idx in yusho_winners:
                    jun_yusho_counts[idx] += (len(yusho_winners) - 1) / len(yusho_winners)

        yusho_prob = yusho_counts / n_sims
        jun_yusho_prob = jun_yusho_counts / n_sims

        df.loc[makuuchi_mask, 'yusho_pts'] = yusho_prob * 5.0
        df.loc[makuuchi_mask, 'jun_yusho_pts'] = jun_yusho_prob * 3.0

    # Kinboshi (simplified)
    df['kinboshi_pts'] = 0.0

    # Total fantasy score
    point_cols = ['wins_pts', 'kk_mk_pts', 'yusho_pts', 'jun_yusho_pts', 'kinboshi_pts']
    df['fantasy_score'] = df[point_cols].sum(axis=1)

    return df


def generate_html_report(results_df: pd.DataFrame, basho_id: int) -> str:
    """Generate HTML report."""
    df = results_df.sort_values('fantasy_score', ascending=False)

    def get_draft_category(rank_str):
        div, num, _ = _parse_rank(rank_str)
        if div in ['Y', 'O']:
            return "group-yo"
        if div in ['S', 'K']:
            return "group-sk"
        if div == 'M':
            if num <= 5:
                return "group-m1-5"
            if num <= 10:
                return "group-m6-10"
            return "group-m11"
        if div == 'J':
            return "group-j"
        return ""

    table_rows = ""
    for _, row in df.iterrows():
        group_class = get_draft_category(row['rank'])
        div_class = "division-makuuchi" if row.get('division_numeric') == 2 else "division-juryo"

        actual = f"{int(row['actual_wins'])}" if pd.notna(row.get('actual_wins')) else "-"

        table_rows += f"""
        <tr class="{div_class} {group_class}">
            <td>{row['rikishi']}</td>
            <td>{row['rank']}</td>
            <td><strong>{row['fantasy_score']:.2f}</strong></td>
            <td>{row['predicted_wins']:.1f}</td>
            <td>{row['direct_wins']:.1f}</td>
            <td>{row['matchup_wins']:.1f}</td>
            <td>{row['kk_mk_pts']:.2f}</td>
            <td>{row['yusho_pts']:.2f}</td>
            <td>{actual}</td>
        </tr>"""

    html = f"""
    <!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
    <title>Hybrid Fantasy Sumo Report {basho_id}</title>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; background: #f4f4f9; padding: 20px; }}
        .container {{ max-width: 1200px; margin: auto; background: #fff; padding: 20px; border-radius: 8px; }}
        h1, h2 {{ text-align: center; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 13px; }}
        th, td {{ padding: 8px 10px; border: 1px solid #ddd; text-align: left; }}
        th {{ background-color: #333; color: #fff; cursor: pointer; position: sticky; top: 0; }}
        .group-yo {{ background-color: #ffcccc; }}
        .group-sk {{ background-color: #ffedcc; }}
        .group-m1-5 {{ background-color: #ffffcc; }}
        .group-m6-10 {{ background-color: #ccffcc; }}
        .group-m11 {{ background-color: #ccffff; }}
        .group-j {{ background-color: #e6ccff; }}
    </style>
    </head>
    <body><div class="container">
        <h1>Hybrid Fantasy Sumo Prediction Report</h1>
        <h2>Basho: {basho_id}</h2>
        <p style="text-align:center;">
            <strong>Hybrid Model:</strong> 70% Direct + 30% Matchup Simulation
        </p>
        <table id="reportTable">
            <thead>
                <tr>
                    <th>Rikishi</th>
                    <th>Rank</th>
                    <th>Fantasy Pts</th>
                    <th>Hybrid Wins</th>
                    <th>Direct Wins</th>
                    <th>Matchup Wins</th>
                    <th>KK/MK Pts</th>
                    <th>Yusho Pts</th>
                    <th>Actual Wins</th>
                </tr>
            </thead>
            <tbody>{table_rows}</tbody>
        </table>
    </div></body></html>"""

    filename = f'hybrid_prediction_report_{basho_id}.html'
    with open(filename, 'w') as f:
        f.write(html)
    print(f"Generated HTML report: {filename}")
    return filename


def main(direct_model_path: str, matchup_model_path: str,
         banzuke_path: str, match_history_path: str,
         basho_id: int, direct_weight: float = 0.7):
    """Main entry point."""

    print("Loading models and data...")
    direct_bundle = joblib.load(direct_model_path)
    matchup_bundle = joblib.load(matchup_model_path)
    historical_df = pd.read_csv(banzuke_path)
    match_history_df = pd.read_csv(match_history_path)

    print(f"Direct model MAE: {direct_bundle.get('mae', 'N/A')}")
    print(f"Matchup model accuracy: {matchup_bundle.get('accuracy', 'N/A'):.4f}")

    # Generate hybrid predictions
    results = hybrid_prediction(
        direct_bundle, matchup_bundle,
        historical_df, match_history_df,
        basho_id, direct_weight
    )

    if results.empty:
        print("Failed to generate predictions.")
        return

    # Calculate fantasy scores
    model_mae = direct_bundle.get('mae', 1.7)
    results = calculate_fantasy_scores(results, model_mae)

    # Generate report
    generate_html_report(results, basho_id)

    # Save CSV
    output_file = f'hybrid_predictions_{basho_id}.csv'
    results.to_csv(output_file, index=False)
    print(f"Saved predictions to {output_file}")

    # Print summary
    print(f"\n{'=' * 60}")
    print("TOP 20 PREDICTIONS (by Fantasy Score)")
    print('=' * 60)
    print(results.nlargest(20, 'fantasy_score')[
        ['rikishi', 'rank', 'fantasy_score', 'predicted_wins', 'direct_wins', 'matchup_wins']
    ].to_string(index=False))

    # If completed, show accuracy
    if 'actual_wins' in results.columns and results['actual_wins'].notna().any():
        valid = results[results['actual_wins'].notna()]
        mae_hybrid = np.abs(valid['hybrid_wins'] - valid['actual_wins']).mean()
        mae_direct = np.abs(valid['direct_wins'] - valid['actual_wins']).mean()
        mae_matchup = np.abs(valid['matchup_wins'] - valid['actual_wins']).mean()

        print(f"\n{'=' * 60}")
        print("PREDICTION ACCURACY (vs Actual)")
        print('=' * 60)
        print(f"Hybrid Model MAE:  {mae_hybrid:.3f} wins")
        print(f"Direct Model MAE:  {mae_direct:.3f} wins")
        print(f"Matchup Model MAE: {mae_matchup:.3f} wins")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--basho', type=int, default=202601)
    parser.add_argument('--direct-model', type=str, default="sumo_win_predictor_model.joblib")
    parser.add_argument('--matchup-model', type=str, default="matchup_model_v4.joblib")
    parser.add_argument('--banzuke', type=str, default="banzuke_detailed.csv")
    parser.add_argument('--matches', type=str, default="match_history_with_kimarite.csv")
    parser.add_argument('--direct-weight', type=float, default=0.7)
    args = parser.parse_args()

    main(args.direct_model, args.matchup_model, args.banzuke, args.matches,
         args.basho, args.direct_weight)

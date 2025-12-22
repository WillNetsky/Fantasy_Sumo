# predict_wins.py
import pandas as pd
import numpy as np
import re
import requests
import joblib
import argparse
from bs4 import BeautifulSoup
from scipy.stats import norm
from sumo_utils import preprocess_data, get_divisional_rank_features, _parse_rank

# --- Configuration ---
BASE_URL = "https://sumodb.sumogames.de/"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
BANZUKE_PARAMS = "&heya=-1&shusshin=-1&h=on&w=on&bd=on&hd=on&su=on&hr=on&ho=on&ch=on&cr=on&spr=on&sps=on&c=on&snr=on&sns=on&simple=on"


def get_draft_category(rank_str: str) -> str:
    """
    Categorizes a rikishi's rank string for fantasy sumo drafting.
    """
    division, number, _ = _parse_rank(rank_str)
    if not division:
        return "Other"

    if division in ['Y', 'O']:
        return "Yokozuna/Ozeki"
    if division in ['S', 'K']:
        return "Sekiwake/Komusubi"
    if division == 'M':
        if 1 <= number <= 5:
            return "Maegashira 1-5"
        if 6 <= number <= 10:
            return "Maegashira 6-10"
        if number >= 11:
            return "Maegashira 11+"
    if division == 'J':
        return "Juryo"
    return "Other"


def calculate_fantasy_score(results_df: pd.DataFrame, model_mae: float, n_sims: int = 1000) -> pd.DataFrame:
    """
    Calculates fantasy score components using probabilistic methods and Monte Carlo simulation.
    """
    df = results_df.copy()
    df['numeric_wins'] = pd.to_numeric(df['predicted_wins'], errors='coerce')

    # --- Component 1: Points from Wins ---
    df['wins_pts'] = df['numeric_wins'].fillna(0)

    # --- Component 2: Kachi-koshi / Make-koshi Points ---
    prob_kk = 1 - norm.cdf(7.5, loc=df['numeric_wins'], scale=model_mae)
    prob_mk = norm.cdf(7.5, loc=df['numeric_wins'], scale=model_mae)
    df['kk_mk_pts'] = (prob_kk * 1.0) - (prob_mk * 0.5)
    df['kk_mk_pts'] = df['kk_mk_pts'].fillna(0)

    # --- Component 3 & 4: Yusho & Jun-yusho Points (Monte Carlo) ---
    df['yusho_pts'] = 0.0
    df['jun_yusho_pts'] = 0.0
    
    makuuchi_mask = (df['division_numeric'] == 2) & (df['numeric_wins'].notna())
    makuuchi_df = df[makuuchi_mask]
    
    if not makuuchi_df.empty:
        print(f"Running {n_sims} simulations for Yusho/Jun-yusho probabilities...")
        
        # Create a matrix of simulated wins for all Makuuchi wrestlers
        locs = makuuchi_df['numeric_wins'].values
        scales = np.full_like(locs, model_mae)
        simulated_wins = np.random.normal(loc=locs, scale=scales, size=(n_sims, len(locs)))
        simulated_wins = np.round(simulated_wins).astype(int)
        
        yusho_counts = np.zeros(len(locs))
        jun_yusho_counts = np.zeros(len(locs))

        for i in range(n_sims):
            wins_this_sim = simulated_wins[i, :]
            
            # Find yusho winner(s)
            yusho_wins = np.max(wins_this_sim)
            yusho_winners_indices = np.where(wins_this_sim == yusho_wins)[0]
            yusho_counts[yusho_winners_indices] += 1
            
            # Find jun-yusho winner(s)
            if len(yusho_winners_indices) == 1: # Only one yusho winner
                second_highest_wins = -1
                for j, wins in enumerate(wins_this_sim):
                    if wins < yusho_wins and wins > second_highest_wins:
                        second_highest_wins = wins
                
                if second_highest_wins != -1:
                    jun_yusho_winners_indices = np.where(wins_this_sim == second_highest_wins)[0]
                    jun_yusho_counts[jun_yusho_winners_indices] += 1

        # Calculate probabilities and points
        yusho_prob = yusho_counts / n_sims
        jun_yusho_prob = jun_yusho_counts / n_sims
        
        df.loc[makuuchi_mask, 'yusho_pts'] = yusho_prob * 5.0
        df.loc[makuuchi_mask, 'jun_yusho_pts'] = jun_yusho_prob * 3.0

    # Total score is calculated in the main function after adding kinboshi points
    df = df.drop(columns=['numeric_wins'])
    return df


def get_yokozuna_match_probability(rank_str: str) -> float:
    """
    Estimates the probability of a Maegashira facing a Yokozuna.
    """
    division, rank_num, _ = _parse_rank(rank_str)
    if division != 'M':
        return 0.0

    if 1 <= rank_num <= 2: return 0.95
    if 3 <= rank_num <= 5: return 0.80
    if 6 <= rank_num <= 9: return 0.50
    if 10 <= rank_num <= 13: return 0.25
    return 0.10


def calculate_predicted_kinboshi(df_to_predict: pd.DataFrame, match_history_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the expected number of kinboshi wins for each Maegashira.
    """
    if match_history_df is None or match_history_df.empty:
        return pd.DataFrame(columns=['rikishi', 'predicted_kinboshi_wins'])

    yokozuna_ids = df_to_predict[df_to_predict['rank'].str.startswith('Y', na=False)]['wrestler_id'].tolist()
    maegashira_df = df_to_predict[df_to_predict['rank'].str.startswith('M', na=False)]

    if not yokozuna_ids or maegashira_df.empty:
        return pd.DataFrame(columns=['rikishi', 'predicted_kinboshi_wins'])

    matches = match_history_df.copy()
    matches['w1'] = np.minimum(matches['winner_id'], matches['loser_id'])
    matches['w2'] = np.maximum(matches['winner_id'], matches['loser_id'])
    matches['w1_won'] = (matches['w1'] == matches['winner_id']).astype(int)

    h2h_totals = matches.groupby(['w1', 'w2']).agg(
        w1_total_wins=('w1_won', 'sum'),
        total_matches=('w1', 'size')
    ).reset_index()
    h2h_totals['w2_total_wins'] = h2h_totals['total_matches'] - h2h_totals['w1_total_wins']

    all_kinboshi_predictions = []
    for _, maegashira in maegashira_df.iterrows():
        maegashira_id = maegashira['wrestler_id']
        expected_kinboshi_wins = 0.0
        prob_of_match = get_yokozuna_match_probability(maegashira['rank'])

        for yokozuna_id in yokozuna_ids:
            if pd.isna(maegashira_id) or pd.isna(yokozuna_id): continue

            w1 = min(int(maegashira_id), int(yokozuna_id))
            w2 = max(int(maegashira_id), int(yokozuna_id))

            record = h2h_totals[(h2h_totals['w1'] == w1) & (h2h_totals['w2'] == w2)]

            if record.empty:
                h2h_win_pct = 0.05
            else:
                wins = record['w1_total_wins'].iloc[0] if w1 == maegashira_id else record['w2_total_wins'].iloc[0]
                total_matches = record['total_matches'].iloc[0]
                h2h_win_pct = wins / total_matches if total_matches > 0 else 0

            expected_kinboshi_wins += (h2h_win_pct * prob_of_match)

        all_kinboshi_predictions.append({
            'rikishi': maegashira['rikishi'],
            'predicted_kinboshi_wins': expected_kinboshi_wins
        })

    return pd.DataFrame(all_kinboshi_predictions) if all_kinboshi_predictions else pd.DataFrame(columns=['rikishi', 'predicted_kinboshi_wins'])


def scrape_banzuke(basho_id: int) -> pd.DataFrame:
    """
    Scrapes the banzuke for a given basho.
    """
    url = f"{BASE_URL}Banzuke.aspx?b={basho_id}{BANZUKE_PARAMS}"
    print(f"Scraping banzuke data from: {url}")
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL: {e}")
        return pd.DataFrame()

    soup = BeautifulSoup(response.content, 'lxml')
    banzuke_table = soup.find('table', class_='banzuke')
    if not banzuke_table:
        print("Could not find the banzuke table on the page.")
        return pd.DataFrame()

    header_row = None
    thead = banzuke_table.find('thead')
    if thead: header_row = thead.find('tr')
    if not header_row:
        first_row = banzuke_table.find('tr')
        if first_row and first_row.find('th'): header_row = first_row
    
    if not header_row:
        print("Could not find a valid header row. Banzuke likely not released.")
        return pd.DataFrame()

    headers = [th.get_text(strip=True) for th in header_row.find_all('th')]
    clean_headers = [re.sub(r'[^a-z0-9_]+', '_', h.lower()).strip('_') for h in headers]

    shikona_col_name = 'rikishi'
    try:
        shikona_col_index = clean_headers.index(shikona_col_name)
    except ValueError:
        print(f"Could not find '{shikona_col_name}' in table headers.")
        return pd.DataFrame()

    wrestlers_data = []
    all_rows = banzuke_table.find_all('tr')
    start_index = 1 if all_rows and all_rows[0] == header_row else 0
        
    for row in all_rows[start_index:]:
        if row.find('th'): continue
        cells = row.find_all('td')
        if len(cells) != len(clean_headers): continue
        data = {header: cell.get_text(strip=True) for header, cell in zip(clean_headers, cells)}
        shikona_anchor = cells[shikona_col_index].find('a')
        if shikona_anchor and 'href' in shikona_anchor.attrs:
            data[shikona_col_name] = shikona_anchor.get_text(strip=True)
            id_match = re.search(r'r=(\d+)', shikona_anchor['href'])
            data['wrestler_id'] = id_match.group(1) if id_match else None
        data['tournament_id'] = basho_id
        data['w'] = pd.to_numeric(data.get('w'), errors='coerce')
        data['l'] = pd.to_numeric(data.get('l'), errors='coerce')
        kinboshi_val = pd.to_numeric(data.get('k'), errors='coerce')
        data['k'] = 0 if pd.isna(kinboshi_val) else int(kinboshi_val)
        data['hoshitori'] = np.nan
        wrestlers_data.append(data)

    df = pd.DataFrame(wrestlers_data)
    print(f"Successfully scraped {len(df)} rikishi for basho {basho_id}.")
    return df


def generate_html_report(results_df: pd.DataFrame, basho_id: int, is_completed: bool):
    """
    Generates a styled HTML report with sortable columns and fantasy grouping.
    """
    df = results_df.copy()
    
    # Initial sort
    df = df.sort_values(by='predicted_fantasy_score', ascending=False)

    table_rows = ""
    for _, row in df.iterrows():
        category = get_draft_category(row['rank'])
        group_class = ""
        if category == "Yokozuna/Ozeki": group_class = "group-yo"
        elif category == "Sekiwake/Komusubi": group_class = "group-sk"
        elif category == "Maegashira 1-5": group_class = "group-m1-5"
        elif category == "Maegashira 6-10": group_class = "group-m6-10"
        elif category == "Maegashira 11+": group_class = "group-m11"
        elif category == "Juryo": group_class = "group-j"

        division_class = "division-makuuchi" if row['division_numeric'] == 2 else "division-juryo"

        # Format all point columns
        pred_pts = f"<strong>{row['predicted_fantasy_score']:.2f}</strong>"
        wins_pts = f"{row['wins_pts']:.1f}"
        kk_mk_pts = f"{row['kk_mk_pts']:.2f}"
        yusho_pts = f"{row['yusho_pts']:.2f}"
        jun_yusho_pts = f"{row['jun_yusho_pts']:.2f}"
        kinboshi_pts = f"{row['kinboshi_pts']:.2f}"
        
        actual_wins_val = row.get('actual_wins')
        actual_wins = f"{int(actual_wins_val)}" if pd.notna(actual_wins_val) else "-"
        
        table_rows += f"""
        <tr class="{division_class} {group_class}">
            <td>{row['rikishi']}</td>
            <td>{row['rank']}</td>
            <td>{pred_pts}</td>
            <td>{wins_pts}</td>
            <td>{kk_mk_pts}</td>
            <td>{yusho_pts}</td>
            <td>{jun_yusho_pts}</td>
            <td>{kinboshi_pts}</td>
            <td>{actual_wins}</td>
        </tr>"""

    html_content = f"""
    <!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>Fantasy Sumo Report {basho_id}</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f4f4f9; color: #333; padding: 20px; }}
        .container {{ max-width: 1200px; margin: auto; background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1, h2 {{ text-align: center; color: #444; }}
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
      var table, rows, switching, i, x, y, shouldSwitch, dir, switchcount = 0;
      table = document.getElementById("reportTable");
      switching = true; dir = "asc"; 
      while (switching) {{
        switching = false; rows = table.rows;
        for (i = 1; i < (rows.length - 1); i++) {{
          shouldSwitch = false;
          x = rows[i].getElementsByTagName("TD")[n];
          y = rows[i + 1].getElementsByTagName("TD")[n];
          var xContent = isNaN(parseFloat(x.innerText)) ? x.innerText.toLowerCase() : parseFloat(x.innerText);
          var yContent = isNaN(parseFloat(y.innerText)) ? y.innerText.toLowerCase() : parseFloat(y.innerText);
          if (dir == "asc") {{ if (xContent > yContent) {{ shouldSwitch = true; break; }} }} 
          else if (dir == "desc") {{ if (xContent < yContent) {{ shouldSwitch = true; break; }} }}
        }}
        if (shouldSwitch) {{
          rows[i].parentNode.insertBefore(rows[i + 1], rows[i]);
          switching = true; switchcount ++;      
        }} else {{ if (switchcount == 0 && dir == "asc") {{ dir = "desc"; switching = true; }} }}
      }}
    }}
    function filterDivision(division) {{
        var table = document.getElementById("reportTable");
        var tr = table.getElementsByTagName("tr");
        for (var i = 1; i < tr.length; i++) {{
            if (division === 'all') tr[i].style.display = "";
            else tr[i].style.display = tr[i].classList.contains('division-' + division) ? "" : "none";
        }}
        document.querySelectorAll('.filters button').forEach(btn => btn.classList.remove('active'));
        document.querySelector(`.filters button[onclick="filterDivision('`+division+`')"]`).classList.add('active');
    }}
    window.onload = () => {{ document.querySelector('.filters button[onclick="filterDivision(\\'all\\')"]').classList.add('active'); }};
    </script>
    </head>
    <body><div class="container">
        <h1>Fantasy Sumo Prediction Report</h1>
        <h2>Basho: {basho_id}</h2>
        <div class="filters">
            <button onclick="filterDivision('all')">All</button>
            <button onclick="filterDivision('makuuchi')">Makuuchi</button>
            <button onclick="filterDivision('juryo')">Juryo</button>
        </div>
        <table id="reportTable">
            <thead>
                <tr>
                    <th onclick="sortTable(0)">Rikishi</th>
                    <th onclick="sortTable(1)">Rank</th>
                    <th onclick="sortTable(2)">Total Pts</th>
                    <th onclick="sortTable(3)">Win Pts</th>
                    <th onclick="sortTable(4)">KK/MK Pts</th>
                    <th onclick="sortTable(5)">Yusho Pts</th>
                    <th onclick="sortTable(6)">J-Yusho Pts</th>
                    <th onclick="sortTable(7)">Kinboshi Pts</th>
                    <th onclick="sortTable(8)">Actual Wins</th>
                </tr>
            </thead>
            <tbody>{table_rows}</tbody>
        </table>
    </div></body></html>"""

    output_filename = f'prediction_report_{basho_id}.html'
    with open(output_filename, 'w', encoding='utf-8') as f: f.write(html_content)
    print(f"\nGenerated HTML report: '{output_filename}'")


def predict_basho(model_bundle: dict, historical_df: pd.DataFrame, match_history_df: pd.DataFrame,
                  basho_id_to_predict: int):
    print(f"\n--- Processing Basho: {basho_id_to_predict} ---")
    model, features = model_bundle['model'], model_bundle['features']
    print(f"Model loaded. Predicting with {len(features)} features.")

    new_basho_df = scrape_banzuke(basho_id_to_predict)
    if new_basho_df.empty: return

    is_completed_basho = not pd.to_numeric(new_basho_df['w'], errors='coerce').isnull().all()
    print("Detected completed basho." if is_completed_basho else "Detected upcoming basho.")

    def is_top_division(rank_str):
        div, _, _ = _parse_rank(rank_str)
        return div in ['Y', 'O', 'S', 'K', 'M', 'J']
    new_basho_df = new_basho_df[new_basho_df['rank'].apply(is_top_division)].copy()
    
    if new_basho_df.empty:
        print("No ranked rikishi found in top divisions.")
        return

    historical_df['wrestler_id'] = pd.to_numeric(historical_df['wrestler_id'], errors='coerce')
    new_basho_df['wrestler_id'] = pd.to_numeric(new_basho_df['wrestler_id'], errors='coerce')
    
    historical_df['tournament_id'] = pd.to_numeric(historical_df['tournament_id'], errors='coerce')
    historical_df = historical_df[historical_df['tournament_id'] != basho_id_to_predict]

    combined_df = pd.concat([historical_df, new_basho_df], ignore_index=True)

    print("\nPreprocessing basho data...")
    df_processed = preprocess_data(combined_df.copy(), match_history_df=match_history_df)
    df_to_predict = df_processed[df_processed['tournament_id'] == basho_id_to_predict].copy()

    X_new_basho = df_to_predict.reindex(columns=features)
    X_new_basho.dropna(inplace=True)
    df_to_predict.dropna(subset=features, inplace=True)

    if X_new_basho.empty:
        print("\nNo valid records for prediction.")
        return

    print(f"\nMaking predictions for {len(X_new_basho)} rikishi...")
    predictions = model.predict(X_new_basho)

    results_df = df_to_predict[['rikishi', 'rank', 'heya', 'age', 'bmi', 'division_numeric', 'rank_in_division']].copy()
    results_df['predicted_wins'] = np.round(predictions, 1)

    # --- Calculate Fantasy Score Components ---
    MODEL_MAE = 2.0 # Using the MAE from our last stacking model run
    results_df = calculate_fantasy_score(results_df, model_mae=MODEL_MAE)

    kinboshi_predictions_df = calculate_predicted_kinboshi(df_to_predict, match_history_df)
    results_df = pd.merge(results_df, kinboshi_predictions_df, on='rikishi', how='left')
    results_df['predicted_kinboshi_wins'].fillna(0.0, inplace=True)
    results_df['kinboshi_pts'] = results_df['predicted_kinboshi_wins'] * 2.0

    # --- Calculate Total Fantasy Score ---
    point_cols = ['wins_pts', 'kk_mk_pts', 'yusho_pts', 'jun_yusho_pts', 'kinboshi_pts']
    results_df['predicted_fantasy_score'] = results_df[point_cols].sum(axis=1)
    
    if is_completed_basho:
        actuals = new_basho_df[['rikishi', 'w', 'k']].rename(columns={'w': 'actual_wins', 'k': 'actual_kinboshi'})
        results_df = pd.merge(results_df, actuals, on='rikishi', how='left')
    else:
        results_df['actual_wins'] = np.nan

    generate_html_report(results_df, basho_id_to_predict, is_completed=is_completed_basho)
    
    output_filename = f'prediction_report_{basho_id_to_predict}.csv' if is_completed_basho else f'next_basho_predictions_{basho_id_to_predict}.csv'
    results_df.to_csv(output_filename, index=False)
    print(f"Results saved to '{output_filename}'")


if __name__ == "__main__":
    BASHO_TO_PREDICT = 202511
    parser = argparse.ArgumentParser(description="Predict wins for a sumo tournament.")
    parser.add_argument('--data-file', type=str, default="banzuke_detailed.csv")
    parser.add_argument('--model-file', type=str, default="sumo_win_predictor_model.joblib")
    args = parser.parse_args()

    try:
        model_bundle = joblib.load(args.model_file)
        historical_df = pd.read_csv(args.data_file)
        match_history_df = pd.read_csv("match_history_with_kimarite.csv")
    except FileNotFoundError as e:
        print(f"Error: A required file was not found. {e}")
        exit()

    predict_basho(model_bundle, historical_df, match_history_df, BASHO_TO_PREDICT)

# compare_predictions.py
"""
Compare prediction accuracy between original predictions (202601_FINAL.html)
and actual tournament results.
"""

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from scipy.stats import spearmanr

def parse_html_predictions(html_path: str) -> pd.DataFrame:
    """Parse predictions from HTML report."""
    with open(html_path, 'r') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')

    rows = []
    table = soup.find('table', id='reportTable')
    if not table:
        return pd.DataFrame()

    for tr in table.find_all('tr')[1:]:  # Skip header
        cells = tr.find_all('td')
        if len(cells) >= 9:
            rows.append({
                'rikishi': cells[0].get_text(strip=True),
                'rank': cells[1].get_text(strip=True),
                'predicted_fantasy_score': float(cells[2].get_text(strip=True).replace('<strong>', '').replace('</strong>', '')),
                'predicted_wins': float(cells[3].get_text(strip=True)),
            })

    return pd.DataFrame(rows)


def main():
    # Load predictions from FINAL HTML
    print("Loading original predictions from FINAL report...")
    original_preds = parse_html_predictions('prediction_report_202601_FINAL.html')
    print(f"Loaded {len(original_preds)} predictions")

    # Load actual results from CSV
    print("\nLoading actual results...")
    actual_results = pd.read_csv('outputs/prediction_report_202601.csv')
    print(f"Loaded {len(actual_results)} results")

    # Merge on rikishi name
    merged = pd.merge(original_preds, actual_results[['rikishi', 'actual_wins']], on='rikishi', how='inner')
    print(f"\nMatched {len(merged)} wrestlers")

    # Filter to those with actual results
    merged = merged[merged['actual_wins'].notna()]
    print(f"With actual results: {len(merged)}")

    if merged.empty:
        print("No data to compare!")
        return

    # Calculate metrics
    print("\n" + "=" * 60)
    print("PREDICTION ACCURACY ANALYSIS")
    print("=" * 60)

    # MAE for win predictions
    mae = np.abs(merged['predicted_wins'] - merged['actual_wins']).mean()
    print(f"\nMean Absolute Error (MAE): {mae:.3f} wins")

    # RMSE
    rmse = np.sqrt(((merged['predicted_wins'] - merged['actual_wins']) ** 2).mean())
    print(f"Root Mean Square Error (RMSE): {rmse:.3f} wins")

    # Correlation
    corr = merged['predicted_wins'].corr(merged['actual_wins'])
    print(f"Pearson Correlation: {corr:.3f}")

    # Spearman rank correlation
    spearman_corr, _ = spearmanr(merged['predicted_wins'], merged['actual_wins'])
    print(f"Spearman Rank Correlation: {spearman_corr:.3f}")

    # Prediction accuracy by category
    print("\n" + "-" * 60)
    print("ACCURACY BY RANK TIER")
    print("-" * 60)

    def get_tier(rank):
        if rank.startswith(('Y', 'O')):
            return 'Yokozuna/Ozeki'
        elif rank.startswith(('S', 'K')):
            return 'Sekiwake/Komusubi'
        elif rank.startswith('M'):
            num = int(''.join(filter(str.isdigit, rank)) or '0')
            if num <= 5:
                return 'Maegashira 1-5'
            elif num <= 10:
                return 'Maegashira 6-10'
            else:
                return 'Maegashira 11+'
        elif rank.startswith('J'):
            return 'Juryo'
        return 'Other'

    merged['tier'] = merged['rank'].apply(get_tier)

    for tier in ['Yokozuna/Ozeki', 'Sekiwake/Komusubi', 'Maegashira 1-5',
                  'Maegashira 6-10', 'Maegashira 11+', 'Juryo']:
        tier_data = merged[merged['tier'] == tier]
        if len(tier_data) > 0:
            tier_mae = np.abs(tier_data['predicted_wins'] - tier_data['actual_wins']).mean()
            print(f"{tier}: MAE = {tier_mae:.2f} wins (n={len(tier_data)})")

    # Top/bottom performers
    print("\n" + "-" * 60)
    print("BIGGEST PREDICTION MISSES (Over-predicted)")
    print("-" * 60)
    merged['error'] = merged['predicted_wins'] - merged['actual_wins']
    over_pred = merged.nlargest(10, 'error')[['rikishi', 'rank', 'predicted_wins', 'actual_wins', 'error']]
    print(over_pred.to_string(index=False))

    print("\n" + "-" * 60)
    print("BIGGEST PREDICTION MISSES (Under-predicted)")
    print("-" * 60)
    under_pred = merged.nsmallest(10, 'error')[['rikishi', 'rank', 'predicted_wins', 'actual_wins', 'error']]
    print(under_pred.to_string(index=False))

    # Most accurate predictions
    print("\n" + "-" * 60)
    print("MOST ACCURATE PREDICTIONS")
    print("-" * 60)
    merged['abs_error'] = np.abs(merged['error'])
    accurate = merged.nsmallest(10, 'abs_error')[['rikishi', 'rank', 'predicted_wins', 'actual_wins', 'error']]
    print(accurate.to_string(index=False))

    # Yusho/Jun-yusho analysis
    print("\n" + "-" * 60)
    print("YUSHO CONTENDERS ANALYSIS")
    print("-" * 60)

    # Who actually won the yusho?
    makuuchi = merged[~merged['rank'].str.startswith('J')]
    yusho_winner = makuuchi.loc[makuuchi['actual_wins'].idxmax()]
    print(f"Actual Yusho Winner: {yusho_winner['rikishi']} ({yusho_winner['actual_wins']} wins)")

    # Who did we predict to win?
    pred_winner = makuuchi.loc[makuuchi['predicted_wins'].idxmax()]
    print(f"Predicted Top: {pred_winner['rikishi']} ({pred_winner['predicted_wins']:.1f} pred wins, {pred_winner['actual_wins']} actual)")

    # Where did actual winner rank in our predictions?
    makuuchi_sorted = makuuchi.sort_values('predicted_wins', ascending=False)
    actual_winner_rank = makuuchi_sorted['rikishi'].tolist().index(yusho_winner['rikishi']) + 1
    print(f"Actual yusho winner was ranked #{actual_winner_rank} in our predictions")

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total wrestlers predicted: {len(merged)}")
    print(f"Mean Absolute Error: {mae:.3f} wins")
    print(f"Spearman Rank Correlation: {spearman_corr:.3f}")
    print(f"Predictions within 2 wins: {(merged['abs_error'] <= 2).sum()} ({(merged['abs_error'] <= 2).mean()*100:.1f}%)")
    print(f"Predictions within 3 wins: {(merged['abs_error'] <= 3).sum()} ({(merged['abs_error'] <= 3).mean()*100:.1f}%)")

    return merged


if __name__ == "__main__":
    main()

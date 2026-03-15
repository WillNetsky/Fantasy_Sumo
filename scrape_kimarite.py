import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import time
import argparse
import os
from tqdm import tqdm

# --- Configuration ---
BASE_URL = "https://sumodb.sumogames.de/"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}


def parse_shikona_cell(cell: BeautifulSoup) -> (int, str):
    """Extracts the wrestler ID and shikona (name) from a table cell."""
    anchor = cell.find('a')
    if not (anchor and anchor.has_attr('href')):
        return None, None

    id_match = re.search(r'r=(\d+)', anchor['href'])
    if not id_match:
        return None, None

    wrestler_id = int(id_match.group(1))
    shikona = anchor.get_text(strip=True)
    return wrestler_id, shikona


def scrape_day_results(session: requests.Session, basho_id: int, day: int) -> list:
    """
    Scrapes the results for a single day of a tournament, capturing
    winner, loser, ranks, and kimarite for each match.
    """
    url = f"{BASE_URL}Results.aspx?b={basho_id}&d={day}&simple=on"
    try:
        response = session.get(url, headers=HEADERS, timeout=20)
        response.raise_for_status()
    except requests.exceptions.RequestException:
        # This is expected for days that haven't happened yet or for older, shorter tournaments.
        return []

    soup = BeautifulSoup(response.content, 'lxml')
    # The main results are in the 'tk_table'
    results_table = soup.find('table', class_='tk_table')
    if not results_table:
        return []

    matches = []
    for row in results_table.find_all('tr'):
        cells = row.find_all('td')
        # A valid match row has exactly 12 columns
        if len(cells) != 12:
            continue

        # Extract data for both wrestlers
        east_rank = cells[3].get_text(strip=True)
        east_id, _ = parse_shikona_cell(cells[4])
        east_result = cells[6].get_text(strip=True)

        kimarite = cells[7].get_text(strip=True)

        west_rank = cells[9].get_text(strip=True)
        west_id, _ = parse_shikona_cell(cells[10])

        # Skip if we couldn't parse IDs for both wrestlers
        if not east_id or not west_id:
            continue

        # Determine winner and loser based on the 'W' marker
        if east_result == 'W':
            winner_id = east_id
            winner_rank = east_rank
            loser_id = west_id
            loser_rank = west_rank
        else:
            winner_id = west_id
            winner_rank = west_rank
            loser_id = east_id
            loser_rank = east_rank

        matches.append({
            'tournament_id': basho_id,
            'day': day,
            'winner_id': winner_id,
            'winner_rank': winner_rank,
            'loser_id': loser_id,
            'loser_rank': loser_rank,
            'kimarite': kimarite
        })
    return matches


def main(historical_data_path: str, output_path: str):
    """
    Scrapes full match data (including kimarite) for all tournaments
    found in the historical data file and saves the results to a new CSV.
    Now supports incremental updates.
    """
    print(f"Reading historical basho IDs from '{historical_data_path}'...")
    try:
        df_historical = pd.read_csv(historical_data_path)
    except FileNotFoundError:
        print(f"Error: Historical data file not found at '{historical_data_path}'")
        return

    all_basho_ids = sorted(df_historical['tournament_id'].unique())
    print(f"Found {len(all_basho_ids)} unique tournaments in source data.")

    # --- Incremental Logic ---
    existing_matches_df = pd.DataFrame()
    existing_basho_ids = set()

    if os.path.exists(output_path):
        print(f"Checking for existing match data in '{output_path}'...")
        try:
            existing_matches_df = pd.read_csv(output_path)
            if not existing_matches_df.empty and 'tournament_id' in existing_matches_df.columns:
                existing_basho_ids = set(existing_matches_df['tournament_id'].unique())
                print(f"Found {len(existing_basho_ids)} tournaments already scraped.")
        except Exception as e:
            print(f"Error reading existing file: {e}. Will start fresh.")
            existing_matches_df = pd.DataFrame()

    # Identify new tournaments to scrape
    basho_ids_to_scrape = [bid for bid in all_basho_ids if bid not in existing_basho_ids]
    
    if not basho_ids_to_scrape:
        print("No new tournaments to scrape. Match history is up to date.")
        return

    print(f"Identified {len(basho_ids_to_scrape)} new tournaments to scrape.")
    # --- End Incremental Logic ---

    new_match_data = []
    session = requests.Session()

    # Use tqdm for a nice progress bar
    for basho_id in tqdm(basho_ids_to_scrape, desc="Scraping Match History"):
        for day in range(1, 16):
            day_matches = scrape_day_results(session, basho_id, day)
            if day_matches:
                new_match_data.extend(day_matches)

        # Be polite to the server by adding a small delay
        time.sleep(0.25)

    if not new_match_data:
        print("No new match data was scraped.")
        return

    new_matches_df = pd.DataFrame(new_match_data)
    print(f"\nScraping complete. Found {len(new_matches_df)} new matches.")

    # Combine and Save
    if not existing_matches_df.empty:
        print("Merging with existing data...")
        combined_df = pd.concat([existing_matches_df, new_matches_df], ignore_index=True)
    else:
        combined_df = new_matches_df

    # Sort for tidiness
    combined_df.sort_values(by=['tournament_id', 'day'], inplace=True)

    print(f"Saving full match history ({len(combined_df)} records) to '{output_path}'...")
    combined_df.to_csv(output_path, index=False)
    print("Save complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape historical match data (with kimarite) from SumoDB.")
    parser.add_argument(
        '--data-file',
        type=str,
        default="data/banzuke_detailed.csv",
        help="Path to the historical banzuke data CSV to source tournament IDs."
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default="data/match_history_with_kimarite.csv",
        help="Path to save the scraped match data."
    )
    args = parser.parse_args()

    main(historical_data_path=args.data_file, output_path=args.output_file)

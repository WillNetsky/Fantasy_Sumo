import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time
import os

# --- Configuration ---
BASE_URL = "https://sumodb.sumogames.de/"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
OUTPUT_FILE = "banzuke_detailed.csv"

# These parameters will be used to fetch the banzuke pages.
# The new dynamic scraper will adapt to whatever columns these params generate.
BANZUKE_PARAMS = "&heya=-1&shusshin=-1&h=on&w=on&bd=on&hd=on&su=on&hr=on&ho=on&ch=on&cr=on&spr=on&sps=on&c=on&snr=on&sns=on&simple=on"


def scrape_tournaments(url: str, start_date_str: str | None = None, end_date_str: str | None = None) -> list[str] | None:
    """
    Scrapes a list of tournament IDs (YYYYMM) from the main Basho page,
    filtered by the specified date range.
    """
    print(f"Attempting to fetch tournament list from {url}")
    try:
        session = requests.Session()
        session.headers.update(HEADERS)
        response = session.get(url, timeout=10)
        response.raise_for_status()
        print("Successfully fetched the page.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the page: {e}")
        return None

    soup = BeautifulSoup(response.content, 'lxml')
    tournament_ids = []
    start_filter = int(start_date_str) if start_date_str else None
    end_filter = int(end_date_str) if end_date_str else None

    content_div = soup.find('div', class_='simplecontent')
    if not content_div:
        print("Warning: Could not find the main content div ('div.simplecontent').")
        return []

    tournament_links = content_div.select("a[href*='Banzuke.aspx?b=']")
    if not tournament_links:
        print("Warning: No tournament links found.")
        return []

    for link in tournament_links:
        href = link.get('href')
        if not href:
            continue
        match = re.search(r'b=(\d{6})', href)
        if match:
            tournament_id = match.group(1)
            tournament_id_int = int(tournament_id)
            if start_filter and tournament_id_int < start_filter:
                continue
            if end_filter and tournament_id_int > end_filter:
                continue
            tournament_ids.append(tournament_id)

    print(f"Found {len(tournament_ids)} tournament IDs within the specified range.")
    return sorted(tournament_ids, reverse=True)


def scrape_banzuke_data(tournament_id: str) -> list[dict]:
    """
    Scrapes detailed wrestler data for a single tournament by dynamically
    reading the table headers, making it robust to column changes.
    """
    banzuke_url = f"{BASE_URL}Banzuke.aspx?b={tournament_id}{BANZUKE_PARAMS}"
    print(f"  -> Scraping Banzuke for {tournament_id}")

    try:
        response = requests.get(banzuke_url, headers=HEADERS, timeout=15)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"  !! Error fetching banzuke for {tournament_id}: {e}")
        return []

    soup = BeautifulSoup(response.content, 'lxml')
    wrestlers_data = []

    # The 'simple=on' parameter provides a clean table with the 'banzuke' class
    banzuke_table = soup.find('table', class_='banzuke')
    if not banzuke_table:
        print(f"  !! No banzuke table found for tournament {tournament_id}")
        return []

    # --- Dynamic Header Parsing ---
    header_row = banzuke_table.find('thead').find('tr')
    if not header_row:
        print(f"  !! No header row found in banzuke table for {tournament_id}")
        return []

    # Dynamically read headers and clean them for use as column names
    headers = [th.get_text(strip=True) for th in header_row.find_all('th')]
    clean_headers = [re.sub(r'[^a-z0-9_]+', '_', h.lower()).strip('_') for h in headers]

    # The column for the wrestler's name is 'rikishi' in the simple view
    shikona_col_name = 'rikishi'
    try:
        shikona_col_index = clean_headers.index(shikona_col_name)
    except ValueError:
        print(f"  !! '{shikona_col_name}' column not found in headers for {tournament_id}")
        return []
    # --- End Dynamic Header Parsing ---

    table_body = banzuke_table.find('tbody')
    if not table_body:
        return []  # No data rows to parse

    rows = table_body.find_all('tr')

    for row in rows:
        cells = row.find_all('td')
        if len(cells) != len(clean_headers):
            continue  # Skip malformed rows

        # Create a dictionary by zipping the dynamic headers with cell data
        data = {header: cell.get_text(strip=True) for header, cell in zip(clean_headers, cells)}

        # Extract wrestler ID and clean name from the link in the 'rikishi' cell
        wrestler_id = None
        shikona_anchor = cells[shikona_col_index].find('a')
        if shikona_anchor and shikona_anchor.has_attr('href'):
            # Overwrite the plain text with the cleaner name from the anchor tag
            data[shikona_col_name] = shikona_anchor.get_text(strip=True)
            id_match = re.search(r'r=(\d+)', shikona_anchor['href'])
            if id_match:
                wrestler_id = id_match.group(1)

        data['tournament_id'] = tournament_id
        data['wrestler_id'] = wrestler_id

        wrestlers_data.append(data)

    return wrestlers_data


if __name__ == "__main__":
    # --- Main Execution ---
    # 1. Define the date range for the tournaments you want to scrape.
    # Default start if no file exists
    default_start_basho = "199107"
    # End basho: set to a future date or the specific target.
    # Setting it to something far in the future (e.g., 2030) allows it to just catch everything new.
    end_basho = "202511"

    # 2. Set a delay between requests to be polite to the server.
    delay_seconds = 2

    # 3. For testing: limit the number of tournaments to scrape.
    max_tournaments_to_scrape = None

    print(f"Checking for existing data in '{OUTPUT_FILE}'...")
    existing_df = pd.DataFrame()
    start_basho = default_start_basho
    existing_tournament_ids = set()

    if os.path.exists(OUTPUT_FILE):
        try:
            existing_df = pd.read_csv(OUTPUT_FILE)
            if not existing_df.empty and 'tournament_id' in existing_df.columns:
                existing_tournament_ids = set(existing_df['tournament_id'].astype(str).unique())
                last_basho = max(existing_tournament_ids)
                print(f"Found existing data. Last tournament ID: {last_basho}")
                # We still use the default start_basho for the scrape_tournaments call to get the list,
                # but we will filter the list against existing_tournament_ids.
                # Alternatively, we could try to optimize the start_basho, but filtering is safer.
        except Exception as e:
            print(f"Error reading existing file: {e}. Will start fresh.")
            existing_df = pd.DataFrame()

    print(f"Starting Sumo DB scrape. Target range: {start_basho} to {end_basho}.")
    print(f"Request delay: {delay_seconds}s. Max tournaments: {max_tournaments_to_scrape or 'All'}.")

    # Stage 1: Get all tournament IDs in the desired range
    tournament_ids = scrape_tournaments(f"{BASE_URL}Basho.aspx", start_date_str=start_basho, end_date_str=end_basho)

    if tournament_ids:
        # Filter out tournaments we already have
        new_tournament_ids = [tid for tid in tournament_ids if tid not in existing_tournament_ids]
        print(f"Identified {len(new_tournament_ids)} new tournaments to scrape.")

        if not new_tournament_ids:
            print("No new tournaments found. Data is up to date.")
        else:
            # Apply the max_tournaments limit if it's set
            if max_tournaments_to_scrape is not None:
                new_tournament_ids = new_tournament_ids[:max_tournaments_to_scrape]
                print(f"Limiting run to the {len(new_tournament_ids)} most recent new tournaments.")

            # Stage 2: Scrape detailed banzuke data for each new tournament
            new_wrestler_data = []
            for i, tour_id in enumerate(new_tournament_ids):
                print(f"\nProcessing tournament {i + 1}/{len(new_tournament_ids)}...")
                data_for_tournament = scrape_banzuke_data(tour_id)
                if data_for_tournament:
                    new_wrestler_data.extend(data_for_tournament)
                    print(f"  -> Found {len(data_for_tournament)} wrestler entries.")

                # Be a good web citizen: wait before the next request
                if i < len(new_tournament_ids) - 1:
                    time.sleep(delay_seconds)

            # Stage 3: Process and save the data
            if new_wrestler_data:
                print("\nScraping complete. Merging with existing data...")
                new_df = pd.DataFrame(new_wrestler_data)
                
                # Combine with existing data
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)

                # Reorder columns for better readability
                first_cols = ['tournament_id', 'wrestler_id', 'rikishi', 'rank']
                existing_first_cols = [col for col in first_cols if col in combined_df.columns]
                other_cols = [col for col in combined_df.columns if col not in existing_first_cols]
                combined_df = combined_df[existing_first_cols + other_cols]

                # Deduplicate just in case (based on wrestler_id and tournament_id)
                # Note: wrestler_id might be null for some, so we might need a better subset or just trust the append.
                # Let's trust the append for now but sort.
                combined_df.sort_values(by=['tournament_id', 'rank'], ascending=[False, True], inplace=True)

                # Save to CSV
                combined_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

                print(f"\nSuccessfully saved {len(combined_df)} records ({len(new_df)} new) to '{OUTPUT_FILE}'")
                print("Sample of the new data:")
                print(new_df.head())
            else:
                print("\nNo detailed wrestler data was scraped for the new tournaments.")
    else:
        print("\nCould not retrieve tournament list. Halting script.")

"""Append the official 202605 banzuke to data/banzuke_detailed.csv.

Pulls Makuuchi + Juryo from sumo-api.com, maps shikonaEn to SumoDB
wrestler_id via latest banzuke_detailed name match, and carries forward
per-wrestler metadata (heya, birth_date, height_weight, university,
career-best rank, etc.) from each rikishi's most recent prior entry.
"""

import requests
import pandas as pd
from sumo_utils import get_absolute_rank_score

TOURNAMENT_ID = 202605
BANZUKE_PATH = 'data/banzuke_detailed.csv'

RANK_CODES = {'Yokozuna': 'Y', 'Ozeki': 'O', 'Sekiwake': 'S',
              'Komusubi': 'K', 'Maegashira': 'M', 'Juryo': 'J'}


def to_short_rank(rank_long: str) -> str:
    p = rank_long.split()
    return f"{RANK_CODES[p[0]]}{p[1]}{'e' if p[2] == 'East' else 'w'}"


def fetch_official() -> pd.DataFrame:
    rows = []
    for division in ('Makuuchi', 'Juryo'):
        r = requests.get(
            f"https://sumo-api.com/api/basho/{TOURNAMENT_ID}/banzuke/{division}",
            timeout=15,
        ).json()
        for side in ('east', 'west'):
            for w in r.get(side, []):
                rows.append({
                    'shikona': w['shikonaEn'],
                    'rank': to_short_rank(w['rank']),
                })
    return pd.DataFrame(rows)


def build_rows(official: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    latest_by_name = (history.sort_values('tournament_id')
                              .drop_duplicates('rikishi', keep='last')
                              [['rikishi', 'wrestler_id']])
    official = official.merge(latest_by_name, left_on='shikona', right_on='rikishi', how='left')
    if official['wrestler_id'].isna().any():
        missing = official[official['wrestler_id'].isna()]['shikona'].tolist()
        raise RuntimeError(f"Could not map shikona to wrestler_id: {missing}")

    new_rows = []
    for _, r in official.iterrows():
        w_id = int(r['wrestler_id'])
        prior = history[history['wrestler_id'] == w_id].sort_values('tournament_id')
        if prior.empty:
            raise RuntimeError(f"No prior banzuke entry for {r['shikona']} (id={w_id})")
        last = prior.iloc[-1]

        rank_score_now = get_absolute_rank_score(r['rank'])
        prior_high_score = get_absolute_rank_score(last.get('high')) if pd.notna(last.get('high')) else 0
        new_high = last['high'] if prior_high_score >= rank_score_now else r['rank']

        new_rows.append({
            'tournament_id': TOURNAMENT_ID,
            'wrestler_id':   w_id,
            'rikishi':       r['shikona'],
            'rank':          r['rank'],
            'heya':          last['heya'],
            'birth_date':    last['birth_date'],
            'hatsu':         last['hatsu'],
            'university':    last['university'],
            'height_weight': last['height_weight'],
            'high':          new_high,
            'hoshitori':     pd.NA,
            'career':        last['career'],
            'career_record': last['career_record'],
            'prev':          last['rank'],
            'prev_w':        last['w'],
            'prev_l':        last['l'],
            'w':             pd.NA,
            'l':             pd.NA,
            'next':          pd.NA,
            'next_w':        pd.NA,
            'next_l':        pd.NA,
            'Has_university_sumo': last['Has_university_sumo'],
        })
    return pd.DataFrame(new_rows)


def main():
    history = pd.read_csv(BANZUKE_PATH)
    if (history['tournament_id'] == TOURNAMENT_ID).any():
        print(f"Tournament {TOURNAMENT_ID} already present in {BANZUKE_PATH}. Aborting.")
        return

    official = fetch_official()
    print(f"Pulled {len(official)} wrestlers from sumo-api ({TOURNAMENT_ID}).")

    new_rows = build_rows(official, history)
    print(f"Built {len(new_rows)} new banzuke rows.")

    combined = pd.concat([history, new_rows[history.columns.tolist()]], ignore_index=True)
    combined.to_csv(BANZUKE_PATH, index=False)
    print(f"Appended to {BANZUKE_PATH} (now {len(combined)} rows).")


if __name__ == '__main__':
    main()

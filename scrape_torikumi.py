"""
scrape_torikumi.py
Scrapes daily torikumi (matchups) and results from the JSA official JSON API.
Results are live — judge field updates as each bout completes.
"""
import requests

BASE_URL = "https://sumo.or.jp"
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

KAKU_ID_TO_NAME = {
    1: 'Makuuchi',
    2: 'Juryo',
    3: 'Makushita',
    4: 'Sandanme',
    5: 'Jonidan',
    6: 'Jonokuchi',
}
DIVISION_ORDER = ['Makuuchi', 'Juryo', 'Makushita', 'Sandanme', 'Jonidan', 'Jonokuchi']


def scrape_day(basho_id: int, day: int) -> dict:
    """
    Scrape matchups and results for one day of a basho using the JSA API.

    basho_id (YYYYMM) is accepted for interface compatibility but the JSA
    auto-detects the current basho.

    Returns:
        {
            'basho_id': int,
            'day': int,
            'days_available': [1..15],
            'divisions': {
                'Makuuchi': [bout, ...],
                ...
            },
            'error': str or None
        }

    Each bout dict:
        east_id (int, JSA rikishi_id), east_name, east_rank, east_record,
        west_id (int, JSA rikishi_id), west_name, west_rank, west_record,
        h2h (''), kimarite, winner ('east'/'west'/None)
    """
    divisions = {}

    for kaku_id, div_name in KAKU_ID_TO_NAME.items():
        url = f"{BASE_URL}/EnHonbashoMain/torikumiAjax/{kaku_id}/{day}/"
        try:
            r = requests.post(
                url,
                headers=HEADERS,
                timeout=15,
                data={'basho_id': '', 'kakuzuke_id': str(kaku_id), 'day': str(day)},
            )
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            return {
                'basho_id': basho_id, 'day': day,
                'days_available': [], 'divisions': {}, 'error': str(e),
            }

        if data.get('Result') != '1':
            continue

        bouts = []
        all_bouts = data.get('TorikumiData', []) + data.get('FinalMuch', [])
        for t in all_bouts:
            east = t['east']
            west = t['west']

            judge = int(t.get('judge', 0) or 0)
            if judge == 1:
                winner = 'east'
            elif judge == 2:
                winner = 'west'
            else:
                winner = None

            rest_e = east.get('rest_eng', '')
            rest_w = west.get('rest_eng', '')
            east_record = f"{east['won_number']}-{east['lost_number']}{rest_e}"
            west_record = f"{west['won_number']}-{west['lost_number']}{rest_w}"

            kimarite = t.get('technic_name_eng', '').strip()
            if kimarite in ('&nbsp;', '', ' ', '\xa0'):
                kimarite = None

            bouts.append({
                'east_id':     east['rikishi_id'],
                'east_name':   east['shikona_eng'],
                'east_rank':   east['banzuke_name_eng'],
                'east_record': east_record,
                'east_wins':   int(east.get('won_number', 0)),
                'east_losses': int(east.get('lost_number', 0)),
                'west_id':     west['rikishi_id'],
                'west_name':   west['shikona_eng'],
                'west_rank':   west['banzuke_name_eng'],
                'west_record': west_record,
                'west_wins':   int(west.get('won_number', 0)),
                'west_losses': int(west.get('lost_number', 0)),
                'h2h':         '',
                'kimarite':    kimarite,
                'winner':      winner,
            })

        if bouts:
            divisions[div_name] = bouts

    return {
        'basho_id':       basho_id,
        'day':            day,
        'days_available': list(range(1, 16)),
        'divisions':      divisions,
        'error':          None,
    }

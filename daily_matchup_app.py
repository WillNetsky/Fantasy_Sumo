"""
daily_matchup_app.py
Flask web app showing daily sumo matchup predictions vs live results.

Usage:
    venv/bin/python daily_matchup_app.py [--basho 202603] [--port 5000]

Then open http://localhost:5000 in your browser.
"""
import argparse
import concurrent.futures
import math
import threading
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import requests as _http
import shap
from bs4 import BeautifulSoup as _BS
from flask import Flask, redirect, render_template_string, request, url_for

from scrape_torikumi import scrape_day, DIVISION_ORDER
from simulate_tournament_v4 import (
    build_match_features_v9 as build_match_features,
    compute_h2h,
    compute_h2h_recent,
    get_h2h_stats,
    get_h2h_recent_stats,
    predict_match_v9 as predict_match_v8,
    prepare_wrestlers_v9 as prepare_wrestlers,
)
from sumo_utils import get_absolute_rank_score

app = Flask(__name__)

# --- Global state loaded at startup ---
BASHO_ID = 202603
MODEL_BUNDLE = None
WRESTLER_DICT = None   # {wrestler_id: feature_dict}
NAME_TO_ID = None      # {lowercase_name: wrestler_id} for JSA name→SumoDB ID matching
H2H = None
H2H_RECENT = None      # {(w1,w2): recent win dict} for v9
EXPLAINER = None       # SHAP TreeExplainer
DAY_CACHE = {}         # {(basho_id, day): list_of_bouts} for summary page
WRESTLER_FLAG = {}     # {wrestler_id: flag_emoji or ''} - filled async at startup

_JAPANESE_PREFS = {
    'aichi', 'akita', 'aomori', 'chiba', 'ehime', 'fukui', 'fukuoka',
    'fukushima', 'gifu', 'gunma', 'hiroshima', 'hokkaido', 'hyogo',
    'ibaraki', 'ishikawa', 'iwate', 'kagawa', 'kagoshima', 'kanagawa',
    'kochi', 'kumamoto', 'kyoto', 'mie', 'miyagi', 'miyazaki', 'nagano',
    'nagasaki', 'nara', 'niigata', 'oita', 'okayama', 'okinawa', 'osaka',
    'saga', 'saitama', 'shiga', 'shimane', 'shizuoka', 'tochigi',
    'tokushima', 'tokyo', 'tottori', 'toyama', 'wakayama', 'yamagata',
    'yamaguchi', 'yamanashi',
}

_COUNTRY_FLAG = {
    'mongolia':      '🇲🇳',
    'kazakhstan':    '🇰🇿',
    'georgia':       '🇬🇪',
    'russia':        '🇷🇺',
    'china':         '🇨🇳',
    'south korea':   '🇰🇷',
    'korea':         '🇰🇷',
    'brazil':        '🇧🇷',
    'egypt':         '🇪🇬',
    'philippines':   '🇵🇭',
    'usa':           '🇺🇸',
    'united states': '🇺🇸',
    'ukraine':       '🇺🇦',
    'kyrgyzstan':    '🇰🇬',
    'uzbekistan':    '🇺🇿',
    'tonga':         '🇹🇴',
    'bulgaria':      '🇧🇬',
    'czech republic': '🇨🇿',
    'czechia':       '🇨🇿',
    'hungary':       '🇭🇺',
    'estonia':       '🇪🇪',
    'romania':       '🇷🇴',
}

FEATURE_LABELS = {
    'rank_diff':              'Rank',
    'rank_in_div_diff':       'Rank in division',
    'sanyaku_diff':           'Sanyaku rank',
    'elo_diff':               'ELO rating',
    'elo_expected_A':         'ELO win expectancy',
    'elo_x_age_diff':         'ELO × age',
    'h2h_win_pct_A':          'H2H record',
    'h2h_total_matches':      'H2H matches',
    'has_h2h_history':        'H2H history',
    'win_pct_last_3_A':       'Recent form (east)',
    'win_pct_last_3_B':       'Recent form (west)',
    'win_pct_last_3_diff':    'Recent form diff',
    'win_pct_last_6_diff':    'Form (6 basho)',
    'career_win_pct_diff':    'Career win rate',
    'experience_diff':        'Experience',
    'weighted_recent_diff':   'Weighted recent form',
    'rank_x_form_diff':       'Rank × form',
    'rank_momentum_diff':     'Rank trajectory',
    'rank_momentum_A':        'Trajectory (east)',
    'rank_momentum_B':        'Trajectory (west)',
    'basho_since_peak_diff':  'Basho since peak',
    'rank_gap_diff':          'Peak rank gap',
    'height_diff':            'Height',
    'weight_diff':            'Weight',
    'bmi_diff':               'BMI',
    'age_diff':               'Age',
    'style_advantage_diff':   'Style match',
    'oshi_advantage_diff':    'Push technique',
    'yotsu_advantage_diff':   'Grip technique',
    'pull_advantage_diff':    'Pull technique',
    'leg_advantage_diff':     'Leg technique',
    'okuri_advantage_diff':   'Sending technique',
    'height_x_oshi_diff':     'Height × push style',
    'height_x_yotsu_diff':    'Height × grip style',
    'weight_x_yotsu_diff':    'Weight × grip style',
    'bmi_x_pull_diff':        'BMI × pull style',
    'age_x_oshi_diff':        'Age × push style',
    'age_x_yotsu_diff':       'Age × grip style',
    'record_diff':            'Tournament record',
    'pressure_diff':          'KK/MK pressure',
    'kk_pressure_diff':       'Kachi-koshi pressure',
    'mk_pressure_diff':       'Make-koshi pressure',
    'vuln_oshi_A':            'Push vuln (east)',
    'vuln_oshi_B':            'Push vuln (west)',
    'vuln_yotsu_A':           'Grip vuln (east)',
    'vuln_yotsu_B':           'Grip vuln (west)',
    'vuln_pull_A':            'Pull vuln (east)',
    'vuln_pull_B':            'Pull vuln (west)',
    'vuln_leg_A':             'Leg vuln (east)',
    'vuln_leg_B':             'Leg vuln (west)',
    'vuln_okuri_A':           'Sending vuln (east)',
    'vuln_okuri_B':           'Sending vuln (west)',
    'oshi_ratio_A':           'Push style (east)',
    'oshi_ratio_B':           'Push style (west)',
    'yotsu_ratio_A':          'Grip style (east)',
    'yotsu_ratio_B':          'Grip style (west)',
    'pull_ratio_A':           'Pull style (east)',
    'pull_ratio_B':           'Pull style (west)',
    'leg_ratio_A':            'Leg style (east)',
    'leg_ratio_B':            'Leg style (west)',
    'okuri_ratio_A':          'Sending style (east)',
    'okuri_ratio_B':          'Sending style (west)',
}


KIMARITE_PROBS = {}   # {category: [(kimarite_name, prob), ...]} computed from history


def _compute_kimarite_probs(match_history: pd.DataFrame) -> dict:
    """Compute P(kimarite | technique_category) from historical match data."""
    from simulate_tournament_v4 import TECHNIQUE_TO_CATEGORY
    counts: dict = {}
    for km in match_history['kimarite'].dropna():
        km = str(km).strip().lower()
        cat = TECHNIQUE_TO_CATEGORY.get(km)
        if cat:
            counts.setdefault(cat, {})
            counts[cat][km] = counts[cat].get(km, 0) + 1
    result = {}
    for cat, km_counts in counts.items():
        total = sum(km_counts.values())
        result[cat] = sorted(
            [(km, n / total) for km, n in km_counts.items()],
            key=lambda x: -x[1],
        )
    return result


def predict_kimarite(winner_feats: dict, loser_feats: dict) -> list:
    """Return top predicted kimarites as [(name, probability), ...].

    Category probabilities are derived from winner's attack ratios × loser's
    vulnerabilities.  Within each category the historical frequency distribution
    is used to pick specific kimarites.
    """
    from simulate_tournament_v4 import TECHNIQUE_CATEGORIES
    cat_scores = {
        cat: winner_feats.get(f'{cat}_ratio', 0.2) * loser_feats.get(f'vuln_{cat}', 0.2)
        for cat in TECHNIQUE_CATEGORIES
    }
    total = sum(cat_scores.values()) or 1
    cat_probs = {cat: s / total for cat, s in cat_scores.items()}

    km_probs: dict = {}
    for cat, cat_p in cat_probs.items():
        for km, within_p in KIMARITE_PROBS.get(cat, []):
            km_probs[km] = km_probs.get(km, 0) + cat_p * within_p

    return sorted(km_probs.items(), key=lambda x: -x[1])[:3]


def _fetch_flag(wrestler_id) -> str:
    """Scrape SumoDB profile for Shusshin (birthplace); return flag emoji or '' for Japanese."""
    try:
        url = f'https://sumodb.sumogames.de/Rikishi.aspx?r={int(wrestler_id)}'
        r = _http.get(url, timeout=5, headers={'User-Agent': 'Mozilla/5.0'})
        if r.status_code != 200:
            return ''
        soup = _BS(r.content, 'lxml')
        # Parse as plain text — avoids brittle HTML table cell assumptions.
        # SumoDB shows "Shusshin    Mongolia, Ulaanbaatar" or "Shusshin\nMongolia, Ulaanbaatar".
        lines = [ln.strip() for ln in soup.get_text('\n').split('\n') if ln.strip()]
        origin = ''
        for i, line in enumerate(lines):
            lo = line.lower()
            if lo == 'shusshin':
                origin = lines[i + 1].lower() if i + 1 < len(lines) else ''
                break
            if lo.startswith('shusshin') and len(line) > len('shusshin'):
                origin = lo[len('shusshin'):].lstrip(' \t:').strip()
                break
        if not origin:
            return ''
        if any(pref in origin for pref in _JAPANESE_PREFS):
            return ''
        # "Mongolia, Ulaanbaatar" → country = "Mongolia"
        country = origin.split(',')[0].strip()
        return _COUNTRY_FLAG.get(country, '')
    except Exception:
        pass
    return ''


def _is_top_division(rank: str) -> bool:
    if not rank:
        return False
    r = rank.strip()
    c = r[0].upper()
    if c in ('Y', 'O', 'S', 'K', 'M'):
        return True
    # Juryo: starts with J followed by a digit (not Jd = Jonidan)
    return c == 'J' and len(r) > 1 and r[1].isdigit()


def _load_flags_async(wrestler_ids):
    """Fetch flags for a set of wrestlers concurrently; stores into WRESTLER_FLAG."""
    global WRESTLER_FLAG
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as ex:
        futures = {ex.submit(_fetch_flag, wid): wid for wid in wrestler_ids}
        for fut in concurrent.futures.as_completed(futures):
            wid = futures[fut]
            try:
                WRESTLER_FLAG[wid] = fut.result()
            except Exception:
                WRESTLER_FLAG[wid] = ''
    foreign = sum(1 for v in WRESTLER_FLAG.values() if v)
    print(f'[flags] Loaded — {foreign} non-Japanese wrestlers found.')


def _flag_for_name(name: str) -> str:
    """Return flag emoji for a wrestler name, or '' if Japanese / not found."""
    wid = _lookup_id(name)
    return WRESTLER_FLAG.get(wid, '') if wid is not None else ''


app.jinja_env.globals['flag_for'] = _flag_for_name


def load_resources(basho_id: int, model_file: str = 'matchup_model_v9.joblib'):
    global MODEL_BUNDLE, WRESTLER_DICT, NAME_TO_ID, H2H, H2H_RECENT, BASHO_ID, EXPLAINER, KIMARITE_PROBS
    BASHO_ID = basho_id
    print(f"Loading model and data for basho {basho_id}...")

    banzuke = pd.read_csv('data/banzuke_detailed.csv')
    match_history = pd.read_csv('data/match_history_with_kimarite.csv')
    for df in [banzuke, match_history]:
        for col in ['wrestler_id', 'winner_id', 'loser_id', 'tournament_id']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

    MODEL_BUNDLE = joblib.load(model_file)
    wrestlers_df = prepare_wrestlers(banzuke, match_history, basho_id, all_divisions=True)
    WRESTLER_DICT = wrestlers_df.set_index('wrestler_id').to_dict('index')
    H2H = compute_h2h(match_history, basho_id)
    H2H_RECENT = compute_h2h_recent(match_history, basho_id)

    # Build name→SumoDB ID lookup (JSA uses English shikonas, SumoDB uses 'rikishi' column)
    NAME_TO_ID = {}
    for wid, feat in WRESTLER_DICT.items():
        name = feat.get('rikishi', '')
        if name:
            NAME_TO_ID[name.lower()] = wid

    KIMARITE_PROBS = _compute_kimarite_probs(match_history)
    EXPLAINER = shap.TreeExplainer(MODEL_BUNDLE['model'])
    print(f"Ready — {len(WRESTLER_DICT)} wrestlers loaded.")

    # Fetch nationality flags in the background (top divisions only, ~80 wrestlers)
    top_ids = [wid for wid, feat in WRESTLER_DICT.items()
               if _is_top_division(feat.get('rank', ''))]
    threading.Thread(target=_load_flags_async, args=(top_ids,), daemon=True).start()


def _normalize_name(name: str) -> str:
    """Normalize romanization variants: 'ou'→'o', 'uu'→'u', strip spaces."""
    return name.lower().replace('ou', 'o').replace('uu', 'u').replace(' ', '')


def _lookup_id(name: str):
    """Look up wrestler_id by name, trying exact then normalized match."""
    key = name.lower()
    if key in NAME_TO_ID:
        return NAME_TO_ID[key]
    norm = _normalize_name(name)
    for k, v in NAME_TO_ID.items():
        if _normalize_name(k) == norm:
            return v
    return None


def _blend_win_pct(historical_pct: float, current_wins: int, current_losses: int) -> float:
    """Blend historical win_pct_last_3 with current basho performance.

    Weights current basho results against ~45 historical matches (3 basho × 15 days),
    so early-tournament results have little effect but accumulate over the basho.
    """
    matches_played = current_wins + current_losses
    if matches_played == 0:
        return historical_pct
    current_pct = current_wins / matches_played
    hist_weight = 45  # approximate weight of win_pct_last_3 baseline
    return (hist_weight * historical_pct + matches_played * current_pct) / (hist_weight + matches_played)


def get_prediction(east_name, west_name,
                   east_wins=None, east_losses=None,
                   west_wins=None, west_losses=None):
    """Return (prob, shap_features, meta) or (None, None, None) if either wrestler unknown.

    east_wins/east_losses: current basho record, used to blend live form into predictions.
    shap_features is a list of (label, shap_val) tuples for the top features,
    sorted by absolute value descending. Positive = favours east, negative = west.
    meta is a dict with rank_diff, form_diff, h2h_pct for aggregate analysis.
    """
    east_id = _lookup_id(east_name)
    west_id = _lookup_id(west_name)
    if east_id is None or west_id is None:
        return None, None, None
    if east_id not in WRESTLER_DICT or west_id not in WRESTLER_DICT:
        return None, None, None

    # Copy wrestler feature dicts so we can blend in current-basho form without
    # mutating the global WRESTLER_DICT (which the pre-tournament simulator relies on).
    a = dict(WRESTLER_DICT[east_id])
    b = dict(WRESTLER_DICT[west_id])
    if east_wins is not None and east_losses is not None:
        a['win_pct_last_3'] = _blend_win_pct(a['win_pct_last_3'], east_wins, east_losses)
    if west_wins is not None and west_losses is not None:
        b['win_pct_last_3'] = _blend_win_pct(b['win_pct_last_3'], west_wins, west_losses)

    h2h_pct, h2h_tot = get_h2h_stats(H2H, east_id, west_id)
    h2h_recent_pct, h2h_recent_tot = get_h2h_recent_stats(H2H_RECENT, east_id, west_id)
    _wa = east_wins or 0
    _la = east_losses or 0
    _wb = west_wins or 0
    _lb = west_losses or 0
    prob = predict_match_v8(
        MODEL_BUNDLE['model'],
        MODEL_BUNDLE['features'],
        a, b,
        h2h_pct, h2h_tot,
        wins_a=_wa, losses_a=_la, wins_b=_wb, losses_b=_lb,
        h2h_recent_pct_a=h2h_recent_pct, h2h_recent_total=h2h_recent_tot,
    )

    # SHAP feature importance for this specific bout
    try:
        f = build_match_features(a, b, h2h_pct, h2h_tot,
                                 wins_a=_wa, losses_a=_la, wins_b=_wb, losses_b=_lb,
                                 h2h_recent_pct_a=h2h_recent_pct, h2h_recent_total=h2h_recent_tot)

        import warnings
        X = pd.DataFrame([f])[MODEL_BUNDLE['features']]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            sv = EXPLAINER.shap_values(X)
        vals = sv[1][0] if isinstance(sv, list) else sv[0]

        # In-tournament features only make sense once bouts have been fought.
        # Suppress them from SHAP display when records are still 0-0 (pre-bout / day 1).
        _no_record = (_wa == 0 and _la == 0 and _wb == 0 and _lb == 0)
        _in_tournament_features = {
            'wins_before_A', 'wins_before_B', 'losses_before_A', 'losses_before_B',
            'record_diff', 'day', 'is_kk_match_A', 'is_kk_match_B',
            'is_mk_match_A', 'is_mk_match_B', 'kk_pressure_diff',
            'mk_pressure_diff', 'pressure_diff',
        }

        sigmoid_deriv = prob * (1 - prob)
        pairs = sorted(zip(MODEL_BUNDLE['features'], vals), key=lambda x: abs(x[1]), reverse=True)
        top = [(FEATURE_LABELS.get(k, k), float(v) * sigmoid_deriv)
               for k, v in pairs[:5]
               if abs(v) * sigmoid_deriv > 0.003
               and not (_no_record and k in _in_tournament_features)]
        shap_features = top
    except Exception:
        shap_features = []

    # Kimarite prediction: use predicted winner's attack vs loser's vulnerability
    winner_feats, loser_feats = (a, b) if prob >= 0.5 else (b, a)
    kimarite_pred = predict_kimarite(winner_feats, loser_feats) if KIMARITE_PROBS else []

    meta = {
        'rank_diff': a['rank_score'] - b['rank_score'],      # positive = east ranked higher
        'form_diff': a['win_pct_last_3'] - b['win_pct_last_3'],  # positive = east better form
        'h2h_pct': h2h_pct,
        'h2h_tot': h2h_tot,
        'kimarite_pred': kimarite_pred,
        'rank_A': a.get('rank'),
        'rank_B': b.get('rank'),
    }

    return prob, shap_features, meta


def _predict_prob_only(east_name, west_name,
                       east_wins=None, east_losses=None,
                       west_wins=None, west_losses=None):
    """Like get_prediction but skips SHAP entirely for speed.

    Returns (prob, east_rank, west_rank) or (None, None, None) if either wrestler unknown.
    """
    east_id = _lookup_id(east_name)
    west_id = _lookup_id(west_name)
    if east_id is None or west_id is None:
        return None, None, None
    if east_id not in WRESTLER_DICT or west_id not in WRESTLER_DICT:
        return None, None, None

    a = dict(WRESTLER_DICT[east_id])
    b = dict(WRESTLER_DICT[west_id])
    if east_wins is not None and east_losses is not None:
        a['win_pct_last_3'] = _blend_win_pct(a['win_pct_last_3'], east_wins, east_losses)
    if west_wins is not None and west_losses is not None:
        b['win_pct_last_3'] = _blend_win_pct(b['win_pct_last_3'], west_wins, west_losses)

    h2h_pct, h2h_tot = get_h2h_stats(H2H, east_id, west_id)
    h2h_recent_pct, h2h_recent_tot = get_h2h_recent_stats(H2H_RECENT, east_id, west_id)
    prob = predict_match_v8(
        MODEL_BUNDLE['model'],
        MODEL_BUNDLE['features'],
        a, b,
        h2h_pct, h2h_tot,
        wins_a=east_wins or 0, losses_a=east_losses or 0,
        wins_b=west_wins or 0, losses_b=west_losses or 0,
        h2h_recent_pct_a=h2h_recent_pct, h2h_recent_total=h2h_recent_tot,
    )

    east_rank = WRESTLER_DICT[east_id].get('rank')
    west_rank = WRESTLER_DICT[west_id].get('rank')
    return prob, east_rank, west_rank


def _pre_bout_records(bout):
    """Return (east_wins, east_losses, west_wins, west_losses) BEFORE the current bout.

    The JSA API's won_number/lost_number include the result of the current bout for
    completed bouts. Subtract back so the model sees the record as it was pre-bout.
    """
    ew = int(bout.get('east_wins') or 0)
    el = int(bout.get('east_losses') or 0)
    ww = int(bout.get('west_wins') or 0)
    wl = int(bout.get('west_losses') or 0)
    winner = bout.get('winner')
    if winner == 'east':
        ew = max(0, ew - 1)
        wl = max(0, wl - 1)
    elif winner == 'west':
        ww = max(0, ww - 1)
        el = max(0, el - 1)
    return ew, el, ww, wl


def _enrich_bouts_for_summary(basho_id, day):
    """Scrape day data, attach prob+ranks via _predict_prob_only, return flat bout list.

    Caches results in DAY_CACHE only when at least one bout has a resolved winner.
    """
    cache_key = (basho_id, day)
    if cache_key in DAY_CACHE:
        return DAY_CACHE[cache_key]

    data = scrape_day(basho_id, day)
    flat_bouts = []
    has_result = False

    for div_name, div_bouts in data.get('divisions', {}).items():
        for bout in div_bouts:
            ew, el, ww, wl = _pre_bout_records(bout)
            prob, east_rank, west_rank = _predict_prob_only(
                bout['east_name'], bout['west_name'],
                east_wins=ew, east_losses=el,
                west_wins=ww, west_losses=wl,
            )
            bout['prob'] = round(prob, 3) if prob is not None else None
            bout['east_rank'] = bout.get('east_rank') or east_rank or ''
            bout['west_rank'] = bout.get('west_rank') or west_rank or ''
            bout['division'] = div_name
            if bout.get('winner') is not None:
                has_result = True
            flat_bouts.append(bout)

    if has_result:
        DAY_CACHE[cache_key] = flat_bouts

    return flat_bouts


def _rank_division_label(rank: str) -> str:
    """Return the division name for a rank string (for grouping in summary table)."""
    if not rank:
        return 'Unknown'
    r = rank.strip().upper()
    if r.startswith('Y') or r.startswith('O') or r.startswith('S') or r.startswith('K') or r.startswith('M'):
        return 'Makuuchi'
    if r.startswith('J'):
        return 'Juryo'
    if r.startswith('MS') or r.startswith('SD') or r.startswith('JD') or r.startswith('JK') or r.startswith('MK'):
        return 'Lower Divisions'
    return 'Unknown'


def build_summary(basho_id: int):
    """Build the basho summary page across all 15 days."""
    day_stats = []
    rikishi_stats = {}  # {name: {correct, total, pl, rank, division}}
    days_available = []
    total_days_with_results = 0

    for day in range(1, 16):
        bouts = _enrich_bouts_for_summary(basho_id, day)
        if not bouts:
            break

        days_available.append(day)

        log_loss_sum = 0.0
        log_loss_count = 0
        has_results = False
        div_stats = {}  # {div_name: {'correct': int, 'total': int, 'pl': float, 'max_pl': float}}
        day_max_pl = 0.0
        day_conf_correct = 0
        day_conf_total = 0

        for bout in bouts:
            prob = bout.get('prob')
            winner = bout.get('winner')
            east_name = bout.get('east_name', '')
            west_name = bout.get('west_name', '')
            east_rank = bout.get('east_rank', '')
            west_rank = bout.get('west_rank', '')
            division = bout.get('division', 'Unknown')

            if prob is None:
                continue

            stake = abs(prob - 0.5) * 2
            is_fusen = str(bout.get('kimarite') or '').lower() == 'fusen'

            if winner is not None:
                has_results = True
            if winner is not None and not is_fusen:
                east_won = winner == 'east'
                hit = east_won == (prob >= 0.5)

                actual = 1 if east_won else 0
                p_clipped = max(0.001, min(0.999, prob))
                log_loss_sum += -(actual * math.log(p_clipped) + (1 - actual) * math.log(1 - p_clipped))
                log_loss_count += 1

                if abs(prob - 0.5) >= 0.1:  # 60%+
                    day_conf_total += 1
                    if hit:
                        day_conf_correct += 1

                if division not in div_stats:
                    div_stats[division] = {'correct': 0, 'total': 0, 'pl': 0.0, 'max_pl': 0.0}
                ds = div_stats[division]
                ds['total'] += 1
                ds['max_pl'] += stake
                day_max_pl += stake
                if hit:
                    ds['correct'] += 1
                ds['pl'] += stake if hit else -stake

                # Rikishi tracking
                for name, rank, is_east in [(east_name, east_rank, True), (west_name, west_rank, False)]:
                    if not name:
                        continue
                    if name not in rikishi_stats:
                        rikishi_stats[name] = {
                            'correct': 0, 'total': 0, 'pl': 0.0, 'max_pl': 0.0,
                            'rank': rank, 'division': division,
                        }
                    rs = rikishi_stats[name]
                    rs['total'] += 1
                    rs['max_pl'] += stake
                    wrestler_won = (winner == 'east') == is_east
                    wrestler_hit = wrestler_won == ((prob >= 0.5) == is_east)
                    if wrestler_hit:
                        rs['correct'] += 1
                    rs['pl'] += stake if wrestler_hit else -stake

        if has_results:
            total_days_with_results += 1

        day_total_correct = sum(d['correct'] for d in div_stats.values())
        day_total_bouts = sum(d['total'] for d in div_stats.values())
        day_total_pl = sum(d['pl'] for d in div_stats.values())

        log_loss = round(log_loss_sum / log_loss_count, 3) if log_loss_count > 0 else None
        day_stats.append({
            'day': day,
            'div_stats': div_stats,
            'log_loss': log_loss,
            'has_results': has_results,
            'day_correct': day_total_correct,
            'day_total': day_total_bouts,
            'day_pl': day_total_pl,
            'day_max_pl': day_max_pl,
            'conf_correct': day_conf_correct,
            'conf_total': day_conf_total,
        })

    # Sort rikishi by rank score descending (higher = better rank = first)
    def _rank_sort_key(item):
        rank = item[1].get('rank', '')
        try:
            score = get_absolute_rank_score(rank)
        except Exception:
            score = -999
        return (-score, item[0])

    sorted_rikishi = sorted(rikishi_stats.items(), key=_rank_sort_key)
    sorted_rikishi = [(n, s) for n, s in sorted_rikishi if s['total'] > 0]

    # Collect division order (preserve DIVISION_ORDER where possible)
    seen_divs = []
    for ds in day_stats:
        for d in ds['div_stats']:
            if d not in seen_divs:
                seen_divs.append(d)
    all_divisions = [d for d in DIVISION_ORDER if d in seen_divs] + [d for d in seen_divs if d not in DIVISION_ORDER]

    # Compute overall totals across all days
    overall_correct = sum(ds['day_correct'] for ds in day_stats if ds['has_results'])
    overall_total = sum(ds['day_total'] for ds in day_stats if ds['has_results'])
    overall_pl = sum(ds['day_pl'] for ds in day_stats if ds['has_results'])
    overall_max_pl = sum(ds['day_max_pl'] for ds in day_stats if ds['has_results'])
    overall_conf_correct = sum(ds['conf_correct'] for ds in day_stats if ds['has_results'])
    overall_conf_total = sum(ds['conf_total'] for ds in day_stats if ds['has_results'])
    overall_div_totals = {}
    for ds in day_stats:
        if not ds['has_results']:
            continue
        for div, d in ds['div_stats'].items():
            if div not in overall_div_totals:
                overall_div_totals[div] = {'correct': 0, 'total': 0, 'pl': 0.0, 'max_pl': 0.0}
            overall_div_totals[div]['correct'] += d['correct']
            overall_div_totals[div]['total'] += d['total']
            overall_div_totals[div]['pl'] += d['pl']
            overall_div_totals[div]['max_pl'] += d['max_pl']

    return render_template_string(
        SUMMARY_TEMPLATE,
        basho_id=basho_id,
        day_stats=day_stats,
        all_divisions=all_divisions,
        rikishi_stats=sorted_rikishi,
        days_available=days_available,
        total_days_with_results=total_days_with_results,
        overall_correct=overall_correct,
        overall_total=overall_total,
        overall_pl=overall_pl,
        overall_max_pl=overall_max_pl,
        overall_div_totals=overall_div_totals,
        overall_conf_correct=overall_conf_correct,
        overall_conf_total=overall_conf_total,
    )


SUMMARY_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sumo {{ basho_id }} — Summary</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: #111;
    color: #ddd;
    font-family: 'Segoe UI', system-ui, sans-serif;
    padding: 20px;
    max-width: 960px;
    margin: 0 auto;
  }

  h1 { color: #c9a84c; font-size: 1.4em; margin-bottom: 16px; }
  h2 { color: #c9a84c; font-size: 1.05em; margin: 28px 0 12px; letter-spacing: 0.04em; }

  .day-nav { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 24px; }
  .day-btn {
    padding: 5px 12px; border-radius: 4px;
    text-decoration: none; color: #aaa; background: #222;
    border: 1px solid #333; font-size: 0.9em;
  }
  .day-btn:hover { background: #2a2a2a; color: #fff; }
  .day-btn.active { background: #c9a84c; color: #111; font-weight: bold; border-color: #c9a84c; }

  table {
    width: 100%; border-collapse: collapse; font-size: 0.88em;
    margin-bottom: 8px;
  }
  thead th {
    text-align: left; color: #777; font-size: 0.75em; text-transform: uppercase;
    letter-spacing: 0.06em; padding: 6px 10px; border-bottom: 1px solid #2a2a2a;
    font-weight: 600;
  }
  tbody tr { border-bottom: 1px solid #1c1c1c; }
  tbody tr:hover { background: #181818; }
  tbody td { padding: 7px 10px; color: #ccc; }

  .div-header-row td {
    background: #1a1a1a; color: #888; font-size: 0.72em; text-transform: uppercase;
    letter-spacing: 0.08em; padding: 5px 10px; border-bottom: 1px solid #222;
    font-weight: 700;
  }

  .pl-pos { color: #4ade80; }
  .pl-neg { color: #f87171; }
  .pl-zero { color: #666; }

  thead th.sortable { cursor: pointer; user-select: none; }
  thead th.sortable:hover { color: #aaa; }
  thead th.sort-asc::after { content: ' ▴'; color: #c9a84c; }
  thead th.sort-desc::after { content: ' ▾'; color: #c9a84c; }

  .footer { color: #444; font-size: 0.75em; margin-top: 32px; text-align: center; }
  .dash { color: #444; }
</style>
</head>
<body>

<h1>Haru {{ basho_id }} &mdash; Summary</h1>

<nav class="day-nav">
  <a href="/summary" class="day-btn active">Summary</a>
  {% for d in days_available %}
  <a href="/day/{{ d }}" class="day-btn">Day {{ d }}</a>
  {% endfor %}
</nav>

<h2>Day-by-Day Model Performance</h2>
<table>
  <thead>
    <tr>
      <th>Day</th>
      <th>P&amp;L</th>
      {% for div in all_divisions %}<th>{{ div }}</th>{% endfor %}
      <th>Accuracy</th>
      <th>60%+ Acc</th>
      <th>Log Loss</th>
    </tr>
  </thead>
  <tbody>
  {% for ds in day_stats %}
  {% if ds.has_results %}
  <tr>
    <td><a href="/day/{{ ds.day }}" style="color:#c9a84c;text-decoration:none;">Day {{ ds.day }}</a></td>
    <td class="{% if ds.day_pl > 0 %}pl-pos{% elif ds.day_pl < 0 %}pl-neg{% else %}pl-zero{% endif %}">
      {{ "%+.1f"|format(ds.day_pl) }}u
      {% if ds.day_max_pl > 0 %}<span style="color:#555;font-size:0.8em">&nbsp;({{ "%.0f"|format(ds.day_pl / ds.day_max_pl * 100) }}%)</span>{% endif %}
    </td>
    {% for div in all_divisions %}
    {% set d = ds.div_stats.get(div) %}
    <td>
      {% if d and d.total > 0 %}
        {% set pct = (d.correct / d.total * 100) | round | int %}
        <span style="color:#ccc">{{ pct }}%</span>
        <span class="{% if d.pl > 0 %}pl-pos{% elif d.pl < 0 %}pl-neg{% else %}pl-zero{% endif %}" style="font-size:0.85em">({{ "%+.1f"|format(d.pl) }}u)</span>
      {% else %}
        <span class="dash">—</span>
      {% endif %}
    </td>
    {% endfor %}
    <td>
      {% if ds.day_total > 0 %}
        {% set pct = (ds.day_correct / ds.day_total * 100) | round | int %}
        <span style="color:#ccc">{{ pct }}%</span>
        <span style="color:#555;font-size:0.85em">&nbsp;({{ ds.day_correct }}/{{ ds.day_total }})</span>
      {% else %}<span class="dash">—</span>{% endif %}
    </td>
    <td>
      {% if ds.conf_total > 0 %}
        {% set cpct = (ds.conf_correct / ds.conf_total * 100) | round | int %}
        <span style="color:#ccc">{{ cpct }}%</span>
        <span style="color:#555;font-size:0.85em">&nbsp;({{ ds.conf_correct }}/{{ ds.conf_total }})</span>
      {% else %}<span class="dash">—</span>{% endif %}
    </td>
    <td>{% if ds.log_loss is not none %}{{ "%.3f"|format(ds.log_loss) }}{% else %}<span class="dash">—</span>{% endif %}</td>
  </tr>
  {% endif %}
  {% endfor %}
  {% if overall_total > 0 %}
  <tr style="border-top: 2px solid #333; font-weight: 600; color: #c9a84c;">
    <td>Overall</td>
    <td class="{% if overall_pl > 0 %}pl-pos{% elif overall_pl < 0 %}pl-neg{% else %}pl-zero{% endif %}">
      {{ "%+.1f"|format(overall_pl) }}u
      {% if overall_max_pl > 0 %}<span style="color:#888;font-size:0.85em">&nbsp;({{ "%.0f"|format(overall_pl / overall_max_pl * 100) }}%)</span>{% endif %}
    </td>
    {% for div in all_divisions %}
    {% set od = overall_div_totals.get(div) %}
    <td>
      {% if od and od.total > 0 %}
        {% set pct = (od.correct / od.total * 100) | round | int %}
        <span>{{ pct }}%</span>
        <span class="{% if od.pl > 0 %}pl-pos{% elif od.pl < 0 %}pl-neg{% else %}pl-zero{% endif %}" style="font-size:0.85em">({{ "%+.1f"|format(od.pl) }}u)</span>
      {% else %}
        <span class="dash">—</span>
      {% endif %}
    </td>
    {% endfor %}
    <td>
      {% if overall_total > 0 %}
        {% set pct = (overall_correct / overall_total * 100) | round | int %}
        <span>{{ pct }}%</span>
        <span style="color:#888;font-size:0.85em">&nbsp;({{ overall_correct }}/{{ overall_total }})</span>
      {% else %}<span class="dash">—</span>{% endif %}
    </td>
    <td>
      {% if overall_conf_total > 0 %}
        {% set cpct = (overall_conf_correct / overall_conf_total * 100) | round | int %}
        <span>{{ cpct }}%</span>
        <span style="color:#888;font-size:0.85em">&nbsp;({{ overall_conf_correct }}/{{ overall_conf_total }})</span>
      {% else %}<span class="dash">—</span>{% endif %}
    </td>
    <td style="color:#888">—</td>
  </tr>
  {% endif %}
  </tbody>
</table>

<h2>Rikishi Prediction Accuracy</h2>
<table id="rikishi-table">
  <thead>
    <tr>
      <th class="sortable" data-col="0" data-type="str">Rikishi</th>
      <th class="sortable" data-col="1" data-type="rank">Rank</th>
      <th class="sortable sort-desc" data-col="2" data-type="num">Correct/Total</th>
      <th class="sortable" data-col="3" data-type="num">Accuracy</th>
      <th class="sortable" data-col="4" data-type="num">P&amp;L</th>
    </tr>
  </thead>
  <tbody>
  {% for name, stats in rikishi_stats %}
  <tr>
    <td data-val="{{ name }}">{% set rf = flag_for(name) %}{% if rf %}{{ rf }} {% endif %}{{ name }}</td>
    <td data-val="{{ stats.rank or '' }}" style="color:#666;font-size:0.85em;">{{ stats.rank or '—' }}</td>
    <td data-val="{{ stats.total }}">{{ stats.correct }}/{{ stats.total }}</td>
    <td data-val="{{ (stats.correct / stats.total * 100) | round | int if stats.total > 0 else 0 }}">{% if stats.total > 0 %}{{ "%.0f"|format(stats.correct / stats.total * 100) }}%{% else %}<span class="dash">—</span>{% endif %}</td>
    <td data-val="{{ stats.pl | round(2) }}" class="{% if stats.pl > 0 %}pl-pos{% elif stats.pl < 0 %}pl-neg{% else %}pl-zero{% endif %}">{{ "%+.2f"|format(stats.pl) }}u</td>
  </tr>
  {% endfor %}
  </tbody>
</table>

<script>
(function() {
  var rankOrder = { 'Y':0,'O':1,'S':2,'K':3,'M':4,'J':5 };
  function rankScore(r) {
    if (!r) return 9999;
    var m = r.match(/^([A-Za-z]+)(\\d+)?([ew])?/);
    if (!m) return 9999;
    var div = rankOrder[m[1]] !== undefined ? rankOrder[m[1]] : 6;
    var num = m[2] ? parseInt(m[2]) : 0;
    var side = m[3] === 'e' ? 0 : 1;
    return div * 1000 + num * 2 + side;
  }

  var table = document.getElementById('rikishi-table');
  var tbody = table.querySelector('tbody');
  var ths = table.querySelectorAll('thead th.sortable');
  var sortCol = 2, sortDir = -1;  // default: total desc

  function sortTable(col, type, dir) {
    var rows = Array.from(tbody.querySelectorAll('tr'));
    rows.sort(function(a, b) {
      var av = a.cells[col].dataset.val;
      var bv = b.cells[col].dataset.val;
      if (type === 'rank') { return dir * (rankScore(av) - rankScore(bv)); }
      if (type === 'num')  { return dir * (parseFloat(av) - parseFloat(bv)); }
      return dir * av.localeCompare(bv);
    });
    rows.forEach(function(r) { tbody.appendChild(r); });
  }

  ths.forEach(function(th) {
    th.addEventListener('click', function() {
      var col = parseInt(th.dataset.col);
      var type = th.dataset.type;
      if (sortCol === col) { sortDir *= -1; }
      else { sortCol = col; sortDir = type === 'str' ? 1 : -1; }
      ths.forEach(function(t) { t.classList.remove('sort-asc','sort-desc'); });
      th.classList.add(sortDir === 1 ? 'sort-asc' : 'sort-desc');
      sortTable(col, type, sortDir);
    });
  });

  sortTable(sortCol, 'num', sortDir);
})();
</script>

<p class="footer">{{ total_days_with_results }} day(s) with results &nbsp;|&nbsp; {{ rikishi_stats|length }} rikishi tracked</p>

</body>
</html>
"""


TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sumo {{ basho_id }} — Day {{ day }}</title>
<meta http-equiv="refresh" content="15">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: #111;
    color: #ddd;
    font-family: 'Segoe UI', system-ui, sans-serif;
    padding: 20px;
    max-width: 960px;
    margin: 0 auto;
  }

  h1 { color: #c9a84c; font-size: 1.4em; margin-bottom: 16px; }

  /* Day navigation */
  .day-nav { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 20px; }
  .day-btn {
    padding: 5px 12px; border-radius: 4px;
    text-decoration: none; color: #aaa; background: #222;
    border: 1px solid #333; font-size: 0.9em;
  }
  .day-btn:hover { background: #2a2a2a; color: #fff; }
  .day-btn.active { background: #c9a84c; color: #111; font-weight: bold; border-color: #c9a84c; }
  .flip-btn {
    padding: 5px 12px; border-radius: 4px;
    text-decoration: none; color: #aaa; background: #222;
    border: 1px solid #333; font-size: 0.9em; margin-left: auto;
  }
  .flip-btn:hover { background: #2a2a2a; color: #fff; }
  .flip-btn.active { background: #4a4a4a; color: #fff; border-color: #666; }

  /* Stats bar */
  .stats-bar {
    background: #1e1e1e; border: 1px solid #333;
    border-radius: 6px; padding: 10px 16px;
    display: flex; gap: 24px; flex-wrap: wrap; align-items: center; font-size: 0.9em;
    position: fixed; top: 12px;
    left: 50%; transform: translateX(-50%);
    width: calc(min(960px, 100%) - 40px);
    z-index: 10;
    box-shadow: 0 2px 12px rgba(0,0,0,0.6);
  }
  .stats-bar.open { border-radius: 6px 6px 0 0; border-bottom-color: #2a2a2a; }
  .stats-spacer { height: 64px; }
  .stat-item { display: flex; flex-direction: column; gap: 2px; }
  .stat-label { color: #777; font-size: 0.75em; text-transform: uppercase; }
  .stat-value { color: #fff; font-size: 1.1em; font-weight: 600; }
  .stats-toggle-btn {
    margin-left: auto; background: none; border: 1px solid #333; color: #777;
    font-size: 0.78em; padding: 3px 9px; border-radius: 4px; cursor: pointer; white-space: nowrap;
  }
  .stats-toggle-btn:hover { color: #ccc; border-color: #555; }
  .stats-bar.open .stats-toggle-btn { color: #c9a84c; border-color: #c9a84c; border-radius: 4px 4px 0 0; border-bottom: none; background: #2a2a2a; }
  .stats-dropdown {
    display: none; position: absolute; top: 100%; left: -1px; right: -1px;
    background: #1a1a1a; border: 1px solid #333; border-top: none;
    border-radius: 0 0 6px 6px;
    padding: 14px 16px;
    box-shadow: 0 6px 16px rgba(0,0,0,0.5);
  }
  .stats-bar.open .stats-dropdown { display: block; }

  /* Division header */
  .div-header {
    font-size: 1em; font-weight: bold; color: #c9a84c;
    border-bottom: 1px solid #333; padding: 12px 0 6px;
    margin-top: 8px; letter-spacing: 0.05em;
    cursor: pointer; user-select: none;
  }
  .div-header:hover { color: #e0be6a; }
  .div-toggle { font-size: 0.85em; margin-right: 6px; display: inline-block; transition: transform 0.15s; }

  /* Bout row */
  .bout {
    display: grid;
    grid-template-columns: 52px 1fr 180px 1fr 52px;
    align-items: center;
    gap: 6px;
    padding: 8px 4px;
    border-bottom: 1px solid #1e1e1e;
  }
  .bout:hover { background: #181818; }

  /* Wrestler cells */
  .wrestler { text-align: center; padding: 4px; border-radius: 4px; }
  .wrestler .name { font-size: 0.95em; line-height: 1.3; }
  .wrestler .rank { font-size: 0.72em; color: #777; }
  .wrestler .record { font-size: 0.75em; color: #999; }

  .winner .name { color: #4ade80; font-weight: 700; }
  .loser  .name { color: #555; }
  .winner { background: rgba(74,222,128,0.05); }

  /* Centre prediction column */
  .centre { display: flex; flex-direction: column; gap: 4px; align-items: center; }

  /* Probability bar */
  .prob-bar {
    display: flex; width: 100%; height: 26px;
    border-radius: 3px; overflow: hidden;
  }
  .prob-east, .prob-west {
    display: flex; align-items: center;
    font-size: 0.75em; font-weight: 700;
    transition: width 0.3s;
  }
  .prob-east { justify-content: flex-end; padding-right: 5px; }
  .prob-west { justify-content: flex-start; padding-left: 5px; }

  /* Outcome line (kimarite + correct/wrong) */
  .outcome { font-size: 0.72em; color: #666; text-align: center; }
  .h2h-label { font-size: 0.7em; color: #555; text-align: center; }
  .km-pred { font-size: 0.7em; color: #4a4a4a; text-align: center; font-style: italic; }
  .tick  { color: #4ade80; font-weight: bold; }
  .cross { color: #f87171; font-weight: bold; }
  .no-pred { color: #444; font-size: 0.8em; text-align: center; }

  /* SHAP factors inline under wrestler names */
  .shap-factors { display: none; flex-direction: column; gap: 2px; margin-top: 4px; align-items: center; }
  .bout.open .shap-factors { display: flex; }
  .bout.has-shap { cursor: pointer; }
  .bout.has-shap:hover { background: #181818; }
  .shap-item {
    font-size: 0.65em; display: inline-flex; align-items: center; gap: 3px;
    padding: 2px 6px; border-radius: 10px; white-space: nowrap;
  }
  .shap-east-item {
    background: rgba(59,130,246,0.12); color: #7db3f0;
    border: 1px solid rgba(59,130,246,0.25);
  }
  .shap-west-item {
    background: rgba(249,115,22,0.12); color: #f0a87d;
    border: 1px solid rgba(249,115,22,0.25);
  }
  .shap-arrow { font-size: 0.85em; }
  .shap-val { opacity: 0.6; font-size: 0.9em; margin-left: 2px; }

  /* Footer */
  .footer { color: #444; font-size: 0.75em; margin-top: 28px; text-align: center; }

  /* Error banner */
  .error { background: #7f1d1d; color: #fca5a5; padding: 12px; border-radius: 6px; margin-bottom: 16px; }

  .stats-section { margin-bottom: 12px; }
  .stats-section-title { font-size: 0.7em; text-transform: uppercase; color: #555; letter-spacing: 0.08em; margin-bottom: 6px; }
  .stats-grid { display: flex; flex-wrap: wrap; gap: 8px 20px; }
  .stats-cell { font-size: 0.82em; }
  .stats-cell .sc-label { color: #666; }
  .stats-cell .sc-val { color: #ccc; font-weight: 600; margin-left: 4px; }
  .stats-cell .sc-pct { color: #888; font-size: 0.9em; }
  .stats-cell .sc-calib { font-size: 0.85em; font-weight: 600; margin-left: 2px; }
</style>
</head>
<body>

<h1>Haru {{ basho_id }} &mdash; Day {{ day }}</h1>

<nav class="day-nav">
  <a href="/summary" class="day-btn">Summary</a>
  {% for d in days_available %}
  <a href="/day/{{ d }}{% if not flip %}?flip=0{% endif %}" class="day-btn {% if d == day %}active{% endif %}">Day {{ d }}</a>
  {% endfor %}
  <a href="/day/{{ day }}{% if flip %}?flip=0{% else %}?flip=1{% endif %}" class="flip-btn {% if flip %}active{% endif %}" title="Toggle bout order">
    {% if flip %}▲ Upcoming last{% else %}▼ Upcoming first{% endif %}
  </a>
</nav>

{% if error %}
<div class="error">Error fetching data: {{ error }}</div>
{% endif %}

{% if total_bouts > 0 %}
<div class="stats-bar" id="stats-bar">
  <div class="stat-item">
    <span class="stat-label">Overall</span>
    <span class="stat-value">{% if total_predicted > 0 %}{{ correct }}/{{ total_predicted }} ({{ "%.0f"|format(correct / total_predicted * 100) }}%){% else %}—{% endif %}</span>
  </div>
  <div class="stat-item">
    <span class="stat-label">Confident (60%+)</span>
    <span class="stat-value">{% if confident_total > 0 %}{{ confident_correct }}/{{ confident_total }} ({{ "%.0f"|format(confident_correct / confident_total * 100) }}%){% else %}—{% endif %}</span>
  </div>
  <div class="stat-item">
    <span class="stat-label">Log Loss</span>
    <span class="stat-value">{% if log_loss is not none %}{{ "%.3f"|format(log_loss) }}{% else %}—{% endif %}</span>
  </div>
  <div class="stat-item">
    <span class="stat-label">Bet P&amp;L</span>
    <span class="stat-value" style="color:{% if bet_pl > 0 %}#4ade80{% elif bet_pl < 0 %}#f87171{% else %}#fff{% endif %}">{% if total_predicted > 0 %}{{ "%+.1f"|format(bet_pl) }}u{% else %}—{% endif %}</span>
  </div>
  <div class="stat-item">
    <span class="stat-label">Remaining</span>
    <span class="stat-value">{{ total_bouts - total_with_result }}</span>
  </div>
  <button class="stats-toggle-btn" onclick="document.getElementById('stats-bar').classList.toggle('open')">Model Stats ▾</button>

  <div class="stats-dropdown">

    <div class="stats-section">
      <div class="stats-section-title">Accuracy by confidence</div>
      <div class="stats-grid">
        {% for label, bc, mid in [('50–60%', buckets['50-60'], 55), ('60–70%', buckets['60-70'], 65), ('70–80%', buckets['70-80'], 75), ('80%+', buckets['80+'], 85)] %}
        <div class="stats-cell">
          <span class="sc-label">{{ label }}</span>
          <span class="sc-val">{% if bc[1] > 0 %}{{ bc[0] }}/{{ bc[1] }}{% else %}—{% endif %}</span>
          {% if bc[1] > 0 %}
            {% set actual_rate = bc[0] / bc[1] * 100 %}
            {% set gap = actual_rate - mid %}
            <span class="sc-pct">({{ "%.0f"|format(actual_rate) }}%)</span>
            <span class="sc-calib" style="color:{% if gap > -3 and gap < 5 %}#4ade80{% elif gap < -10 %}#f87171{% else %}#fb923c{% endif %}">
              {{ "%+.0f"|format(gap) }}%
            </span>
          {% endif %}
        </div>
        {% endfor %}
      </div>
    </div>

    <div class="stats-section">
      <div class="stats-section-title">Favourite win rate</div>
      <div class="stats-grid">
        <div class="stats-cell">
          <span class="sc-label">Model fav wins</span>
          <span class="sc-val">{% if fav_total > 0 %}{{ fav_correct }}/{{ fav_total }}{% else %}—{% endif %}</span>
          {% if fav_total > 0 %}<span class="sc-pct">({{ "%.0f"|format(fav_correct / fav_total * 100) }}%)</span>{% endif %}
        </div>
        <div class="stats-cell">
          <span class="sc-label">No prediction</span>
          <span class="sc-val">{{ no_pred_count }} bouts</span>
        </div>
      </div>
    </div>

    <div class="stats-section">
      <div class="stats-section-title">When rank & form disagree</div>
      <div class="stats-grid">
        <div class="stats-cell">
          <span class="sc-label">Model correct</span>
          <span class="sc-val">{% if rank_form_dis_total > 0 %}{{ rank_form_dis_correct }}/{{ rank_form_dis_total }}{% else %}—{% endif %}</span>
          {% if rank_form_dis_total > 0 %}<span class="sc-pct">({{ "%.0f"|format(rank_form_dis_correct / rank_form_dis_total * 100) }}%)</span>{% endif %}
        </div>
      </div>
    </div>

    <div class="stats-section">
      <div class="stats-section-title">When H2H contradicts model</div>
      <div class="stats-grid">
        <div class="stats-cell">
          <span class="sc-label">Model right</span>
          <span class="sc-val">{% if h2h_dis_total > 0 %}{{ h2h_model_correct }}/{{ h2h_dis_total }}{% else %}—{% endif %}</span>
          {% if h2h_dis_total > 0 %}<span class="sc-pct">({{ "%.0f"|format(h2h_model_correct / h2h_dis_total * 100) }}%)</span>{% endif %}
        </div>
        <div class="stats-cell">
          <span class="sc-label">H2H right</span>
          <span class="sc-val">{% if h2h_dis_total > 0 %}{{ h2h_h2h_correct }}/{{ h2h_dis_total }}{% else %}—{% endif %}</span>
          {% if h2h_dis_total > 0 %}<span class="sc-pct">({{ "%.0f"|format(h2h_h2h_correct / h2h_dis_total * 100) }}%)</span>{% endif %}
        </div>
      </div>
    </div>

  </div>
</div>
<div class="stats-spacer"></div>
{% endif %}

{% set ns = namespace(idx=0) %}
{% for division, bouts in divisions.items() %}
{% set ds = div_stats.get(division, {}) %}
<div class="div-header" data-div="{{ division }}">
  <span class="div-toggle">▾</span>{{ division }}
  {% if ds.predicted > 0 %}
  {% set pl = ds.pl %}
  <span style="font-size:0.75em; font-weight:normal; color:#888; margin-left:10px;">
    {{ ds.correct }}/{{ ds.predicted }}
    ({{ "%.0f"|format(ds.correct / ds.predicted * 100) }}%)
    <span style="color:{% if pl > 0 %}#4ade80{% elif pl < 0 %}#f87171{% else %}#888{% endif %}">{{ "%+.1f"|format(pl) }}u</span>
  </span>
  {% endif %}
</div>
<div class="div-bouts" data-div="{{ division }}">

{% for b in bouts %}
{% set ns.idx = ns.idx + 1 %}
{% set prob = b.prob %}
{% set winner = b.winner %}
{% set east_pct = ((prob * 100) | round(1)) if prob is not none else 50 %}
{% set west_pct = ((100 - east_pct) | round(1)) %}

{#
  Compute bar colours dynamically from advantage magnitude.
  adv  = 0 at 50/50, 1 at 100/0
  Favoured side: blue (pending), green (correct), red (wrong)
    → saturation 40→95%, lightness 16→40% as adv grows
  Underdog side: same hue but
    → saturation 40→8%, lightness 16→7% (near-black) as adv grows
  Text colour fades to near-invisible on the underdog side.
#}
{% if prob is not none %}
  {% set adv = ((prob - 0.5) * 2) | abs %}
  {% set fav_s = (40 + adv * 55) | int %}
  {% set fav_l = (16 + adv * 24) | int %}
  {% set unf_s = (40 - adv * 32) | int %}
  {% set unf_l = (16 - adv * 10) | int %}
  {% set fav_text = "rgba(255,255,255,0.9)" %}
  {% set unf_text = "rgba(255,255,255," ~ [(0.35 - adv * 0.25), 0.1] | max ~ ")" %}

  {% if winner is none %}
    {# Pending — blue hue #}
    {% if prob >= 0.5 %}
      {% set ebg = "hsl(215," ~ fav_s ~ "%," ~ fav_l ~ "%)" %}
      {% set wbg = "hsl(220," ~ unf_s ~ "%," ~ unf_l ~ "%)" %}
      {% set etc = fav_text %}{% set wtc = unf_text %}
    {% else %}
      {% set ebg = "hsl(220," ~ unf_s ~ "%," ~ unf_l ~ "%)" %}
      {% set wbg = "hsl(215," ~ fav_s ~ "%," ~ fav_l ~ "%)" %}
      {% set etc = unf_text %}{% set wtc = fav_text %}
    {% endif %}
    {% set badge = '' %}

  {% elif winner == 'east' %}
    {% set correct = prob >= 0.5 %}
    {% set badge = '<span class="tick">✓</span>' if correct else '<span class="cross">✗</span>' %}
    {% if correct %}
      {# East favoured and won — green east, red west #}
      {% set ebg = "hsl(142," ~ fav_s ~ "%," ~ fav_l ~ "%)" %}
      {% set wbg = "hsl(0," ~ unf_s ~ "%," ~ unf_l ~ "%)" %}
    {% else %}
      {# East was underdog and won — still green east, but lower intensity #}
      {% set ebg = "hsl(142," ~ unf_s ~ "%," ~ unf_l ~ "%)" %}
      {% set wbg = "hsl(0," ~ fav_s ~ "%," ~ fav_l ~ "%)" %}
    {% endif %}
    {% set etc = fav_text %}{% set wtc = unf_text %}

  {% else %}
    {# West won #}
    {% set correct = prob < 0.5 %}
    {% set badge = '<span class="tick">✓</span>' if correct else '<span class="cross">✗</span>' %}
    {% if correct %}
      {# West favoured and won #}
      {% set ebg = "hsl(0," ~ unf_s ~ "%," ~ unf_l ~ "%)" %}
      {% set wbg = "hsl(142," ~ fav_s ~ "%," ~ fav_l ~ "%)" %}
    {% else %}
      {# West was underdog and won #}
      {% set ebg = "hsl(0," ~ fav_s ~ "%," ~ fav_l ~ "%)" %}
      {% set wbg = "hsl(142," ~ unf_s ~ "%," ~ unf_l ~ "%)" %}
    {% endif %}
    {% set etc = unf_text %}{% set wtc = fav_text %}
  {% endif %}
{% endif %}
{% set is_fusen = b.kimarite is not none and (b.kimarite|string|lower) == 'fusen' %}
{% set bout_pl = none %}
{% if winner is not none and prob is not none and not is_fusen %}
  {% set bout_pl = adv if correct else -adv %}
{% endif %}

<div class="bout {% if b.shap %}has-shap{% endif %}" data-idx="{{ ns.idx }}">
  {# East rank #}
  <div style="text-align:right;font-size:0.75em;color:#555">{{ b.east_rank }}</div>

  {# East wrestler #}
  <div class="wrestler {% if winner == 'east' %}winner{% elif winner == 'west' %}loser{% endif %}">
    <div class="name">{% set ef = flag_for(b.east_name) %}{% if ef %}<span title="{{ b.east_name }}">{{ ef }}</span> {% endif %}{{ b.east_name }}{% if winner == 'east' and bout_pl is not none %} <span style="font-size:0.7em;font-weight:normal;color:{% if bout_pl > 0 %}#4ade80{% else %}#f87171{% endif %}">{{ "%+.2f"|format(bout_pl) }}u</span>{% endif %}</div>
    <div class="record">{{ b.east_record }}</div>
    {% if b.shap %}
    <div class="shap-factors">
      {% for label, val in b.shap %}{% if val > 0 %}
      <span class="shap-item shap-east-item"><span class="shap-arrow">↑</span>{{ label }}<span class="shap-val">+{{ "%.1f"|format(val * 100) }}%</span></span>
      {% endif %}{% endfor %}
    </div>
    {% endif %}
  </div>

  {# Centre: probability bar + outcome #}
  <div class="centre">
    {% if prob is not none %}
    <div class="prob-bar">
      <div class="prob-east" style="width:{{ east_pct }}%; background:{{ ebg }}; color:{{ etc }}">
        {% if east_pct >= 20 %}{{ east_pct }}%{% endif %}
      </div>
      <div class="prob-west" style="width:{{ west_pct }}%; background:{{ wbg }}; color:{{ wtc }}">
        {% if west_pct >= 20 %}{{ west_pct }}%{% endif %}
      </div>
    </div>
    <div class="outcome">
      {% if b.kimarite %}{{ b.kimarite }} {% endif %}{{ badge | safe }}
    </div>
    {% if b.h2h and winner is none %}
    <div class="h2h-label">H2H: {{ b.h2h }}</div>
    {% endif %}
    {% if b.kimarite_pred %}
    {% set km_hit = b.kimarite and (b.kimarite_pred[0][0] == b.kimarite or (b.kimarite_pred|length > 1 and b.kimarite_pred[1][0] == b.kimarite)) %}
    <div class="km-pred">~ {{ b.kimarite_pred[0][0] }}{% if b.kimarite_pred|length > 1 %} / {{ b.kimarite_pred[1][0] }}{% endif %}{% if winner is not none %} <span class="{{ 'tick' if km_hit else 'cross' }}">{{ '✓' if km_hit else '✗' }}</span>{% endif %}</div>
    {% endif %}
    {% else %}
    <div class="no-pred">—</div>
    {% if b.kimarite %}<div class="outcome">{{ b.kimarite }}</div>{% endif %}
    {% endif %}
  </div>

  {# West wrestler #}
  <div class="wrestler {% if winner == 'west' %}winner{% elif winner == 'east' %}loser{% endif %}">
    <div class="name">{% set wf = flag_for(b.west_name) %}{% if wf %}<span title="{{ b.west_name }}">{{ wf }}</span> {% endif %}{{ b.west_name }}{% if winner == 'west' and bout_pl is not none %} <span style="font-size:0.7em;font-weight:normal;color:{% if bout_pl > 0 %}#4ade80{% else %}#f87171{% endif %}">{{ "%+.2f"|format(bout_pl) }}u</span>{% endif %}</div>
    <div class="record">{{ b.west_record }}</div>
    {% if b.shap %}
    <div class="shap-factors">
      {% for label, val in b.shap %}{% if val < 0 %}
      <span class="shap-item shap-west-item"><span class="shap-arrow">↑</span>{{ label }}<span class="shap-val">+{{ "%.1f"|format(val|abs * 100) }}%</span></span>
      {% endif %}{% endfor %}
    </div>
    {% endif %}
  </div>

  {# West rank #}
  <div style="font-size:0.75em;color:#555">{{ b.west_rank }}</div>
</div>
{% endfor %}
</div>
{% endfor %}


<script>
document.querySelectorAll('.bout.has-shap').forEach(function(row) {
  row.addEventListener('click', function() { this.classList.toggle('open'); });
});

{% if flip %}
(function() {
  var allBouts = Array.from(document.querySelectorAll('.bout'));
  var lastCompletedIdx = -1;
  allBouts.forEach(function(b, i) {
    if (b.querySelector('.winner')) lastCompletedIdx = i;
  });
  var nextIdx = -1;
  for (var i = lastCompletedIdx + 1; i < allBouts.length; i++) {
    if (allBouts[i].classList.contains('has-shap')) {
      allBouts[i].classList.add('open');
      nextIdx = i;
      break;
    }
  }
  // Scroll to the next upcoming bout
  var scrollTarget = nextIdx >= 0 ? allBouts[nextIdx] : (lastCompletedIdx + 1 < allBouts.length ? allBouts[lastCompletedIdx + 1] : null);
  if (scrollTarget) {
    scrollTarget.scrollIntoView({ block: 'center' });
  }
})();
{% endif %}

// Collapsible divisions
(function() {
  var headers = document.querySelectorAll('.div-header[data-div]');
  headers.forEach(function(hdr) {
    var divName = hdr.getAttribute('data-div');
    var boutsEl = document.querySelector('.div-bouts[data-div="' + divName + '"]');
    var toggle = hdr.querySelector('.div-toggle');
    if (!boutsEl) return;

    // Restore collapsed state from sessionStorage
    var key = 'div-collapsed-' + divName;
    if (sessionStorage.getItem(key) === '1') {
      boutsEl.style.display = 'none';
      if (toggle) toggle.textContent = '▸';
    }

    hdr.addEventListener('click', function() {
      var collapsed = boutsEl.style.display === 'none';
      if (collapsed) {
        boutsEl.style.display = '';
        if (toggle) toggle.textContent = '▾';
        sessionStorage.removeItem(key);
      } else {
        boutsEl.style.display = 'none';
        if (toggle) toggle.textContent = '▸';
        sessionStorage.setItem(key, '1');
      }
    });
  });
})();
</script>

<p class="footer">Auto-refreshes every 15s &nbsp;|&nbsp; Last updated: {{ last_updated }}</p>

</body>
</html>
"""


def build_page(basho_id: int, day: int, flip: bool = False):
    data = scrape_day(basho_id, day)

    # Attach predictions to each bout; track overall and per-division stats
    correct = 0
    total_predicted = 0
    total_bouts = 0
    total_with_result = 0
    div_stats = {}  # {division: {'correct': int, 'predicted': int, 'pl': float}}

    for div_name, div_bouts in data['divisions'].items():
        dc, dp = 0, 0
        d_pl = 0.0
        for bout in div_bouts:
            total_bouts += 1
            ew, el, ww, wl = _pre_bout_records(bout)
            prob, shap_features, meta = get_prediction(
                bout['east_name'], bout['west_name'],
                east_wins=ew, east_losses=el,
                west_wins=ww, west_losses=wl,
            )
            bout['prob'] = round(prob, 3) if prob is not None else None
            bout['shap'] = shap_features or []
            bout['meta'] = meta or {}
            if meta:
                if meta.get('rank_A'):
                    bout['east_rank'] = meta['rank_A']
                if meta.get('rank_B'):
                    bout['west_rank'] = meta['rank_B']

            bout['kimarite_pred'] = meta.get('kimarite_pred', []) if meta else []

            if meta and meta.get('h2h_tot', 0) > 0:
                pct = meta['h2h_pct']
                tot = meta['h2h_tot']
                east_w = round(pct * tot)
                bout['h2h'] = f"{east_w}–{tot - east_w} east"
            else:
                bout['h2h'] = ''

            is_fusen = str(bout.get('kimarite') or '').lower() == 'fusen'
            if bout['winner']:
                total_with_result += 1
            if bout['winner'] and prob is not None and not is_fusen:
                total_predicted += 1
                dp += 1
                east_won = bout['winner'] == 'east'
                pred_east = prob >= 0.5
                hit = east_won == pred_east
                stake = abs(prob - 0.5) * 2
                d_pl += stake if hit else -stake
                if hit:
                    correct += 1
                    dc += 1
        div_stats[div_name] = {'correct': dc, 'predicted': dp, 'pl': d_pl}

    # Compute aggregate stats from completed bouts with predictions
    buckets = {'50-60': [0, 0], '60-70': [0, 0], '70-80': [0, 0], '80+': [0, 0]}
    fav_correct = 0
    fav_total = 0
    rank_form_dis_correct = 0
    rank_form_dis_total = 0
    h2h_model_correct = 0
    h2h_h2h_correct = 0
    h2h_dis_total = 0
    log_loss_sum = 0.0
    log_loss_count = 0
    confident_correct = 0
    confident_total = 0
    bet_pl = 0.0

    for div_bouts in data['divisions'].values():
        for bout in div_bouts:
            prob = bout['prob']
            meta = bout['meta']
            if bout['winner'] is None or prob is None:
                continue
            if (bout.get('kimarite') or '').lower() == 'fusen':
                continue
            hit = (bout['winner'] == 'east') == (prob >= 0.5)
            actual = 1 if bout['winner'] == 'east' else 0
            p_clipped = max(0.001, min(0.999, prob))
            log_loss_sum += -(actual * math.log(p_clipped) + (1 - actual) * math.log(1 - p_clipped))
            log_loss_count += 1
            if abs(prob - 0.5) >= 0.1:  # 60%+
                confident_total += 1
                if hit:
                    confident_correct += 1
            stake = abs(prob - 0.5) * 2  # 0 at 50/50, 1 at 100%
            bet_pl += stake if hit else -stake
            confidence = abs(prob - 0.5)  # 0=50/50, 0.5=100/0
            if confidence < 0.1:
                buckets['50-60'][1] += 1
                if hit: buckets['50-60'][0] += 1
            elif confidence < 0.2:
                buckets['60-70'][1] += 1
                if hit: buckets['60-70'][0] += 1
            elif confidence < 0.3:
                buckets['70-80'][1] += 1
                if hit: buckets['70-80'][0] += 1
            else:
                buckets['80+'][1] += 1
                if hit: buckets['80+'][0] += 1
            fav_total += 1
            if hit:
                fav_correct += 1
            if meta:
                rank_diff = meta.get('rank_diff', 0)
                form_diff = meta.get('form_diff', 0)
                h2h_pct = meta.get('h2h_pct', 0.5)
                # Rank vs form disagreement
                if abs(rank_diff) > 0 and abs(form_diff) > 0.05:
                    if (rank_diff > 0) != (form_diff > 0):  # opposite directions
                        rank_form_dis_total += 1
                        if hit:
                            rank_form_dis_correct += 1
                # H2H vs model disagreement
                if h2h_pct != 0.5 and abs(h2h_pct - 0.5) > 0.05:
                    h2h_picks_east = h2h_pct > 0.5
                    model_picks_east = prob >= 0.5
                    if h2h_picks_east != model_picks_east:
                        h2h_dis_total += 1
                        if hit:
                            h2h_model_correct += 1
                        if h2h_picks_east == (bout['winner'] == 'east'):
                            h2h_h2h_correct += 1

    no_pred_count = sum(1 for db in data['divisions'].values() for b in db if b['prob'] is None)
    log_loss = round(log_loss_sum / log_loss_count, 3) if log_loss_count > 0 else None

    # Normal: Makuuchi first, bouts reversed (Yokozuna at top)
    # Flip:   Jonokuchi first, bouts in fight order (chronological log)
    div_order = list(reversed(DIVISION_ORDER)) if flip else DIVISION_ORDER
    ordered = {}
    for key in div_order:
        if key in data['divisions']:
            bouts = data['divisions'][key]
            if not flip:
                bouts = list(reversed(bouts))
            ordered[key] = bouts

    return render_template_string(
        TEMPLATE,
        basho_id=basho_id,
        day=day,
        days_available=data.get('days_available', [day]),
        divisions=ordered,
        div_stats=div_stats,
        correct=correct,
        total_predicted=total_predicted,
        total_bouts=total_bouts,
        total_with_result=total_with_result,
        buckets=buckets,
        fav_correct=fav_correct,
        fav_total=fav_total,
        rank_form_dis_correct=rank_form_dis_correct,
        rank_form_dis_total=rank_form_dis_total,
        h2h_model_correct=h2h_model_correct,
        h2h_h2h_correct=h2h_h2h_correct,
        h2h_dis_total=h2h_dis_total,
        no_pred_count=no_pred_count,
        log_loss=log_loss,
        confident_correct=confident_correct,
        confident_total=confident_total,
        bet_pl=bet_pl,
        last_updated=datetime.now().strftime('%H:%M:%S'),
        error=data.get('error'),
        flip=flip,
    )


@app.route('/')
def index():
    return redirect(url_for('summary_view'))


@app.route('/summary')
def summary_view():
    return build_summary(BASHO_ID)


@app.route('/day/<int:day>')
def day_view(day):
    flip = request.args.get('flip', '1') != '0'
    return build_page(BASHO_ID, day, flip=flip)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--basho', type=int, default=202603)
    parser.add_argument('--port',  type=int, default=5000)
    parser.add_argument('--model', default='matchup_model_v9.joblib',
                        help='Model file to load (default: matchup_model_v9.joblib)')
    args = parser.parse_args()

    load_resources(args.basho, args.model)
    app.run(debug=True, port=args.port, host='0.0.0.0')

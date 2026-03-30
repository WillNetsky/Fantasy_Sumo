"""
banzuke_202605_builder.py

Predicts the May 2026 (Natsu) Banzuke for Makuuchi and Juryo
based on the March 2026 (Haru) 202603 tournament results.

Key 202603 storylines:
- Aonishiki 12-3 (Ozeki, dominant), Kotozakura 8-7 → both stay Ozeki
- Kirishima 11-4 at S1e → builds Ozeki run (22W over 2B at Sekiwake)
- Atamifuji 12-3 at K1w → promoted to Sekiwake; opens K slot
- Takayasu 8-7 at S1w → bare KK, retained (3rd Sekiwake)
- Wakatakakage 9-6 M1e, Fujinokawa 10-5 M2e → both promoted to Komusubi
- Wakamotoharu 8-7 K1e → retained (3rd Komusubi)
- Result: 10-man sanyaku (3S + 3K) → 32 Maegashira slots (M1-M16)
- Wakanosho 12-3 J3w, Asasuiryu 10-5 J2e, Sadanoumi 9-6 J1w → Makuuchi promotions
- Kinbozan 4-11 M16w, Nishikifuji 6-6-3ab M14w, Midorifuji 6-9 M15e → Juryo demotions

Methodology: JSA rank change table + special sanyaku rules + division border logic
"""

import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from sumo_utils import get_absolute_rank_score, _parse_rank

TOURNAMENT_ID = 202605

# ── 202603 results ──────────────────────────────────────────────────────────
# (rank_202603, rikishi, wrestler_id, wins, losses, absences)
RESULTS_202603 = [
    ('Y1e',  'Hoshoryu',      12451, 10,  5, 0),
    ('Y1w',  'Onosato',       12836, 10,  5, 0),
    ('O1e',  'Aonishiki',     12839, 12,  3, 0),
    ('O1w',  'Kotozakura',    12270,  8,  7, 0),
    ('S1e',  'Kirishima',     12231, 11,  4, 0),
    ('S1w',  'Takayasu',       6480,  8,  7, 0),
    ('K1e',  'Wakamotoharu',  11980,  8,  7, 0),
    ('K1w',  'Atamifuji',     12664, 12,  3, 0),
    ('M1e',  'Wakatakakage',  12370,  9,  6, 0),
    ('M1w',  'Yoshinofuji',   12884,  8,  7, 0),
    ('M2e',  'Fujinokawa',    12800, 10,  5, 0),
    ('M2w',  'Churanoumi',    12320,  9,  6, 0),
    ('M3e',  'Hiradoumi',     12314,  9,  6, 0),
    ('M3w',  'Oho',           12453,  4, 11, 0),
    ('M4e',  'Daieisho',      11985,  7,  8, 0),
    ('M4w',  'Takanosho',     11855,  5, 10, 0),
    ('M5e',  'Abi',           12094, 10,  5, 0),
    ('M5w',  'Kotoshoho',     12449,  9,  6, 0),
    ('M6e',  'Ichiyamamoto',  12362,  4, 11, 0),
    ('M6w',  'Onokatsu',      12840,  7,  8, 0),
    ('M7e',  'Oshoma',        12717,  7,  8, 0),
    ('M7w',  'Hakunofuji',    12796,  5,  8, 2),  # 5-8, 2 kyujo
    ('M8e',  'Ura',           12226,  4, 11, 0),
    ('M8w',  'Shodai',        12130,  7,  8, 0),
    ('M9e',  'Tokihayate',    12542,  8,  7, 0),
    ('M9w',  'Tamawashi',      5944,  5, 10, 0),
    ('M10e', 'Gonoyama',      12688,  7,  8, 0),
    ('M10w', 'Roga',          12516,  7,  8, 0),
    ('M11e', 'Shishi',        12599,  9,  6, 0),
    ('M11w', 'Oshoumi',       12634, 10,  5, 0),
    ('M12e', 'Asakoryu',      12710,  9,  6, 0),
    ('M12w', 'Asanoyama',     12291,  9,  6, 0),
    ('M13e', 'Tobizaru',      12203,  7,  8, 0),
    ('M13w', 'Fujiseiun',     12702, 11,  4, 0),
    ('M14e', 'Chiyoshoma',    11785,  6,  9, 0),
    ('M14w', 'Nishikifuji',   12351,  6,  6, 3),  # kyujo; treat as 6-9 effective
    ('M15e', 'Midorifuji',    12352,  6,  9, 0),
    ('M15w', 'Mitakeumi',     12210,  7,  8, 0),
    ('M16e', 'Asahakuryu',    12784,  8,  7, 0),
    ('M16w', 'Kinbozan',      12721,  4, 11, 0),
    ('M17e', 'Fujiryoga',     12945,  9,  6, 0),
    ('M17w', 'Kotoeiho',      12729,  8,  7, 0),
]

RESULTS_JURYO_202603 = [
    ('J1e',  'Ryuden',        6594,   6,  9, 0),
    ('J1w',  'Sadanoumi',     2879,   9,  6, 0),
    ('J2e',  'Asasuiryu',     12894, 10,  5, 0),
    ('J2w',  'Tomokaze',      12427,  4, 11, 0),
    ('J3e',  'Daiseizan',     12725,  7,  8, 0),
    ('J3w',  'Wakanosho',     12730, 12,  3, 0),
    ('J4e',  'Nishinoryu',    12523,  9,  6, 0),
    ('J4w',  'Takerufuji',    12780,  8,  7, 0),
    ('J5e',  'Kagayaki',      11845,  6,  9, 0),
    ('J5w',  'Shirokuma',     12773,  6,  9, 0),
    ('J6e',  'Hitoshi',       12704,  9,  6, 0),
    ('J6w',  'Kayo',          12774,  9,  6, 0),
    ('J7e',  'Meisei',        11946,  8,  7, 0),
    ('J7w',  'Kyokukaiyu',    12841, 10,  5, 0),
    ('J8e',  'Shonannoumi',   12162,  5, 10, 0),
    ('J8w',  'Kitanowaka',    12548,  7,  8, 0),
    ('J9e',  'Tamashoho',     11976,  5, 10, 0),
    ('J9w',  'Hatsuyama',     12732,  2, 13, 0),
    ('J10e', 'Dewanoryu',     12592,  8,  7, 0),
    ('J10w', 'Tohakuryu',     12575,  7,  8, 0),
    ('J11e', 'Kazuma',        12896,  8,  7, 0),
    ('J11w', 'Toshinofuji',   12852,  5,  2, 8),  # 5-2 with 8 kyujo; eff. 5-10
    ('J12e', 'Nishikigi',      6596,  7,  8, 0),
    ('J12w', 'Tsurugisho',    12113,  6,  9, 0),
    ('J13e', 'Shimazuumi',    12013,  6,  1, 8),  # 6-1 with 8 kyujo; eff. 6-9
    ('J13w', 'Fujitensei',    12946,  5,  2, 8),  # 5-2 with 8 kyujo; eff. 5-10
    ('J14e', 'Kazekeno',      12767,  5, 10, 0),
    ('J14w', 'Kotokuzan',     11809,  3, 12, 0),
]

# Top Makushita wrestlers for Juryo promotion
# Only Ms1-Ms6 with kachi-koshi (4-3 or better) typically qualify
MAKUSHITA_TOP_202603 = [
    ('Ms6e', 'Nobehara',     None,  7, 0),   # perfect 7-0!
    ('Ms2w', 'Okaryu',       None,  6, 1),
    ('Ms4e', 'Enho',         None,  6, 1),
    ('Ms1e', 'Himukamaru',   None,  4, 3),   # borderline
]

# Heya lookup for CSV output
HEYA = {
    12451: 'Tatsunami',  12836: 'Nishonoseki', 12839: 'Ajigawa',   12270: 'Sadogatake',
    12231: 'Otowayama', 6480:  'Tagonoura',    12664: 'Isegahama',  11980: 'Arashio',
    12370: 'Arashio',   12884: 'Isegahama',    12800: 'Isenoumi',   12320: 'Kise',
    12314: 'Sakaigawa', 12453: 'Otake',        11985: 'Oitekaze',   11855: 'Minatogawa',
    12094: 'Shikoroyama',12449: 'Sadogatake',  12362: 'Hanaregoma', 12840: 'Onomatsu',
    12717: 'Naruto',    12796: 'Isegahama',    12226: 'Kise',       12130: 'Tokitsukaze',
    12542: 'Tokitsukaze',5944: 'Kataonami',    12688: 'Takekuma',   12516: 'Futagoyama',
    12599: 'Ikazuchi',  12634: 'Naruto',       12710: 'Takasago',   12291: 'Takasago',
    12203: 'Oitekaze',  12702: 'Fujishima',    11785: 'Kokonoe',    12351: 'Isegahama',
    12352: 'Isegahama', 12210: 'Dewanoumi',    12784: 'Takasago',   12721: 'Kise',
    12945: 'Fujishima', 12729: 'Sadogatake',
    6594:  'Takadagawa', 2879: 'Sakaigawa',    12894: 'Takasago',   12427: 'Nakamura',
    12725: 'Arashio',   12730: 'Tokiwayama',   12523: 'Sakaigawa',  12780: 'Isegahama',
    11845: 'Takadagawa',12773: 'Nishonoseki',  12704: 'Oitekaze',   12774: 'Nakamura',
    11946: 'Tatsunami', 12841: 'Oshima',       12162: 'Onomatsu',   12548: 'Hakkaku',
    11976: 'Kataonami', 12732: 'Tamanoi',      12592: 'Dewanoumi',  12575: 'Tamanoi',
    12896: 'Kise',      12852: 'Isegahama',    6596:  'Isenoumi',   12113: 'Oitekaze',
    12013: 'Hanaregoma',12946: 'Fujishima',    12767: 'Oshiogawa',  11809: 'Arashio',
}


# ── Score calculation ────────────────────────────────────────────────────────
def expected_score(rank: str, wins: int, effective_losses: int) -> float:
    """
    Expected new rank score after tournament.
    Uses JSA rank-change table + small adjustment for rank level.
    """
    current = get_absolute_rank_score(rank)
    if np.isnan(current):
        return 0.0

    change_table = {
        15: 14.0, 14: 11.0, 13: 9.0, 12: 7.0, 11: 5.0, 10: 3.5,
        9: 2.0, 8: 1.0, 7: -1.0, 6: -2.0, 5: -3.5, 4: -5.0,
        3: -7.0, 2: -9.0, 1: -11.0, 0: -14.0
    }
    base = change_table.get(wins, (wins - 7.5) * 1.0)

    # High rank: diminishing returns moving up
    if current >= 40 and base > 0:
        base *= 0.8
    # Low rank: slightly amplified movement
    elif current <= 25 and abs(base) > 2:
        base *= 1.1

    return current + base


# ── Sanyaku for 202605 ───────────────────────────────────────────────────────
# 3 Sekiwake + 3 Komusubi = 10 sanyaku → 32 Maegashira slots (M1e–M16w)
#
# Kirishima S1e   11-4  → stays (Ozeki run: 22W over 2B at Sekiwake)
# Atamifuji K1w   12-3  → promoted to Sekiwake (S1w)
# Takayasu  S1w    8-7  → retained Sekiwake (8-7 at S almost always stays; 3rd S)
# Wakatakakage M1e 9-6  → promoted to Komusubi (K1e)
# Fujinokawa M2e  10-5  → promoted to Komusubi (K1w; 10-5 edge over 9-6)
# Wakamotoharu K1e 8-7  → retained Komusubi (K2e; 3rd K)
SANYAKU_202605 = [
    ('Y1e', 'Hoshoryu',     12451),
    ('Y1w', 'Onosato',      12836),
    ('O1e', 'Aonishiki',    12839),
    ('O1w', 'Kotozakura',   12270),
    ('S1e', 'Kirishima',    12231),
    ('S1w', 'Atamifuji',    12664),
    ('S2e', 'Takayasu',      6480),
    ('K1e', 'Wakatakakage', 12370),
    ('K1w', 'Fujinokawa',   12800),
    ('K2e', 'Wakamotoharu', 11980),
]
SANYAKU_IDS = {wid for _, _, wid in SANYAKU_202605}


# ── Makuuchi Maegashira ──────────────────────────────────────────────────────
def build_maegashira():
    """
    Returns sorted list of Maegashira candidates with expected scores.
    Excludes sanyaku and handles Juryo promotees with JSA-calibrated scores.
    """
    candidates = []

    # Existing Makuuchi wrestlers (excluding sanyaku)
    for rank, rikishi, wid, w, l, ab in RESULTS_202603:
        if wid in SANYAKU_IDS:
            continue
        div = rank[0]
        if div != 'M':
            continue
        eff_losses = l + ab  # absences count as losses
        score = expected_score(rank, w, eff_losses)
        candidates.append(dict(rikishi=rikishi, wid=wid, prev_rank=rank,
                                wins=w, losses=l, ab=ab, score=score,
                                note=''))

    # Juryo → Makuuchi promotions
    # Placement based on historical JSA patterns for comparable records:
    #   12-3 from J3 → M13-M15 range  (score ~27)
    #   10-5 from J2 → M14-M16 range  (score ~26)
    #    9-6 from J1 → M16 borderline (score ~25)
    juryo_promos = [
        ('J3w', 'Wakanosho',  12730, 12, 3, 27.0),
        ('J2e', 'Asasuiryu',  12894, 10, 5, 26.0),
        ('J1w', 'Sadanoumi',   2879,  9, 6, 25.0),
    ]
    for rank, rikishi, wid, w, l, promo_score in juryo_promos:
        candidates.append(dict(rikishi=rikishi, wid=wid, prev_rank=rank,
                                wins=w, losses=l, ab=0, score=promo_score,
                                note='↑ from Juryo'))

    # Sort by expected score descending; 32 slots = M1e through M16w
    candidates.sort(key=lambda x: x['score'], reverse=True)
    return candidates[:32]


# ── Juryo ────────────────────────────────────────────────────────────────────
def build_juryo(mak_placed_ids):
    """
    Returns sorted list of Juryo candidates with expected scores.
    mak_placed_ids: set of wrestler_ids already placed in Makuuchi.
    """
    # Clear demotions from Juryo to Makushita:
    # Hatsuyama J9w 2-13, Kotokuzan J14w 3-12, Tomokaze J2w 4-11
    juryo_demoted_ids = {12732, 11809, 12427}

    candidates = []

    # Remaining Juryo wrestlers
    for rank, rikishi, wid, w, l, ab in RESULTS_JURYO_202603:
        if wid in mak_placed_ids or wid in juryo_demoted_ids:
            continue
        eff_losses = l + ab
        score = expected_score(rank, w, eff_losses)
        # Cap Juryo scores to prevent accidental Makuuchi placement
        score = min(score, 24.9)
        candidates.append(dict(rikishi=rikishi, wid=wid, prev_rank=rank,
                                wins=w, losses=l, ab=ab, score=score,
                                note=''))

    # Makuuchi → Juryo demotions (wrestlers not placed in Makuuchi)
    mak_ids = {wid for _, _, wid, _, _, _ in RESULTS_202603}
    for rank, rikishi, wid, w, l, ab in RESULTS_202603:
        if wid in mak_placed_ids or wid in SANYAKU_IDS:
            continue
        # These fell below M16 in new ranking
        eff_losses = l + ab
        score = expected_score(rank, w, eff_losses)
        # Makuuchi-to-Juryo wrestlers slot in upper Juryo
        score = max(score, 20.0)
        candidates.append(dict(rikishi=rikishi, wid=wid, prev_rank=rank,
                                wins=w, losses=l, ab=ab, score=score,
                                note='↓ from Makuuchi'))

    # Makushita → Juryo promotions
    # Top Makushita wrestlers with kachi-koshi (4-3+) at Ms1-Ms6
    # Placement: upper Juryo area (J1-J4 range, score ~22+)
    ms_promos = [
        ('Ms6e', 'Nobehara',   None,  7, 0, 23.0),  # 7-0: near-certain, jumps to upper J
        ('Ms2w', 'Okaryu',     None,  6, 1, 22.5),
        ('Ms4e', 'Enho',       None,  6, 1, 22.5),
        ('Ms1e', 'Himukamaru', None,  4, 3, 22.0),  # borderline
    ]
    for ms_rank, rikishi, wid, w, l, promo_score in ms_promos:
        candidates.append(dict(rikishi=rikishi, wid=wid, prev_rank=ms_rank,
                                wins=w, losses=l, ab=0, score=promo_score,
                                note='↑ from Makushita'))

    candidates.sort(key=lambda x: x['score'], reverse=True)
    return candidates[:28]


# ── Main ─────────────────────────────────────────────────────────────────────
def build_predicted_banzuke():
    mak_wrestlers = build_maegashira()
    mak_placed_ids = {r['wid'] for r in mak_wrestlers} | SANYAKU_IDS

    juryo_wrestlers = build_juryo(mak_placed_ids)

    # ── Assign ranks ──────────────────────────────────────────────────────
    def assign_ranks(wrestlers, prefix):
        ranked = []
        for i, w in enumerate(wrestlers):
            num = i // 2 + 1
            side = 'e' if i % 2 == 0 else 'w'
            ranked.append({**w, 'rank': f"{prefix}{num}{side}"})
        return ranked

    maegashira_ranked = assign_ranks(mak_wrestlers, 'M')
    juryo_ranked = assign_ranks(juryo_wrestlers, 'J')

    # ── Print ─────────────────────────────────────────────────────────────
    print("=" * 72)
    print(f"  PREDICTED BANZUKE: May 2026 — Natsu Basho (202605)")
    print(f"  Based on March 2026 — Haru Basho (202603) results")
    print("=" * 72)

    # Lookup for 202603 records
    rec = {wid: (rank, w, l, ab)
           for rank, rikishi, wid, w, l, ab in RESULTS_202603 + RESULTS_JURYO_202603}

    def fmt_record(wid, w, l, ab):
        if ab:
            return f"{w}-{l}({ab}ab)"
        return f"{w}-{l}"

    def print_section(title, wrestlers, show_prev=True):
        print(f"\n{title}")
        print(f"  {'Rank':<8} {'Rikishi':<22} {'Prev':>8}  {'Record':>10}  {'Note'}")
        print("  " + "-" * 66)
        for r in wrestlers:
            wid = r.get('wrestler_id') or r.get('wid')
            prev = r.get('prev_rank', '')
            w = r.get('prev_wins', r.get('wins', 0))
            l = r.get('prev_losses', r.get('losses', 0))
            ab = r.get('ab', 0)
            record = fmt_record(wid, w, l, ab)
            note = r.get('note', '')
            print(f"  {r['rank']:<8} {r['rikishi']:<22} {prev:>8}  {record:>10}  {note}")

    # Sanyaku
    print("\nMAKUUCHI — SANYAKU (10 wrestlers: 2Y + 2O + 3S + 3K)")
    print(f"  {'Rank':<8} {'Rikishi':<22} {'Prev':>8}  {'Record':>10}  {'Note'}")
    print("  " + "-" * 66)

    sanyaku_notes = {
        12231: '★ Ozeki run builds (22W/2B @ Sekiwake)',
        12664: '↑ promoted from Komusubi (12-3)',
        6480:  '(retained; 3rd Sekiwake)',
        12370: '↑ promoted from M1 (9-6)',
        12800: '↑ promoted from M2 (10-5)',
        11980: '(retained; 3rd Komusubi)',
    }

    for new_rank, rikishi, wid in SANYAKU_202605:
        if wid in rec:
            prev_rank, w, l, ab = rec[wid]
        else:
            prev_rank, w, l, ab = '—', 0, 0, 0
        record = fmt_record(wid, w, l, ab)
        note = sanyaku_notes.get(wid, '')
        print(f"  {new_rank:<8} {rikishi:<22} {prev_rank:>8}  {record:>10}  {note}")

    # Maegashira
    print(f"\nMAKUUCHI — MAEGASHIRA (32 slots, M1e–M16w)")
    print(f"  {'Rank':<8} {'Rikishi':<22} {'Prev':>8}  {'Record':>10}  {'Note'}")
    print("  " + "-" * 66)
    for r in maegashira_ranked:
        prev = r['prev_rank']
        w, l, ab = r['wins'], r['losses'], r.get('ab', 0)
        record = fmt_record(r['wid'], w, l, ab)
        note = r.get('note', '')
        print(f"  {r['rank']:<8} {r['rikishi']:<22} {prev:>8}  {record:>10}  {note}")

    # Juryo
    print(f"\nJURYO (28 slots, J1e–J14w)")
    print(f"  {'Rank':<8} {'Rikishi':<22} {'Prev':>8}  {'Record':>10}  {'Note'}")
    print("  " + "-" * 66)
    for r in juryo_ranked:
        prev = r['prev_rank']
        w, l, ab = r['wins'], r['losses'], r.get('ab', 0)
        record = fmt_record(r['wid'], w, l, ab)
        note = r.get('note', '')
        print(f"  {r['rank']:<8} {r['rikishi']:<22} {prev:>8}  {record:>10}  {note}")

    # Notes
    print("\nNOTES / EDGE CASES:")
    print("  ★ Kirishima (S1e): strong Ozeki candidate — 11-4 twice at S1e")
    print("    Formal promotion requires ~33W over 3 consecutive basho at S/K level")
    print("    Current count: 22W over 2B @ Sekiwake; case strengthens after 202605")
    print()
    print("  SANYAKU:")
    print("  • 3 Sekiwake: S2e = extra slot for Takayasu (8-7 at S almost always retained)")
    print("  • 3 Komusubi: K2e = extra slot for Wakamotoharu (8-7 retained) +")
    print("    K1e/K1w = Wakatakakage (9-6 M1e) and Fujinokawa (10-5 M2e) both promoted")
    print("  • 10-man sanyaku → 32 Maegashira slots (M1e–M16w), M17 disappears")
    print()
    print("  DEMOTIONS TO JURYO:")
    print("  • Kinbozan (M16w 4-11): certain demotion → predicted J4e")
    print("  • Midorifuji (M15e 6-9): make-koshi from lower M → J1e")
    print()
    print("  BORDERLINE / UNCERTAIN:")
    print("  • Nishikifuji (M14w 6-6-3ab): effective 6-9 due to kyujo → stays M15w")
    print("    borderline: could drop to Juryo depending on JSA judgment on absences")
    print("  • Kotoeiho (M17w 8-7): displaced from M17 by sanyaku expansion → M16w")
    print("    stays in Makuuchi only because Sadanoumi (J1w 9-6) didn't clear the bar")
    print("  • Sadanoumi (J1w 9-6): stays in Juryo at J3e (tied at score cutoff)")
    print("    borderline Makuuchi; could land M16-M17 if JSA favors 9-6 from J1")
    print()
    print("  JURYO CHANGES:")
    print("  • Nobehara (Ms6e 7-0): perfect record → J1w debut (rare achievement)")
    print("  • Enho (Ms4e 6-1): Makushita veteran returning to Juryo")
    print("  • Okaryu (Ms2w 6-1): strong Makushita promotee to upper Juryo")
    print("  • Toshinofuji/Shimazuumi/Fujitensei: kyujo-heavy records (eff. 5-10/6-9)")
    print("    drop to J13–J14; borderline for Makushita demotion in 202607")
    print()
    print("  NOT IN PREDICTED BANZUKE (demoted to Makushita):")
    print("    Hatsuyama (J9w 2-13) — certain")
    print("    Kotokuzan (J14w 3-12) — certain")
    print("    Tomokaze (J2w 4-11) — certain")

    # ── Save CSV ──────────────────────────────────────────────────────────
    rows = []
    all_wrestlers = (
        [(r, n, wid) for r, n, wid in SANYAKU_202605] +
        [(r['rank'], r['rikishi'], r['wid']) for r in maegashira_ranked] +
        [(r['rank'], r['rikishi'], r['wid']) for r in juryo_ranked]
    )
    for rank, rikishi, wid in all_wrestlers:
        rows.append({
            'tournament_id': TOURNAMENT_ID,
            'wrestler_id':   wid if wid else float('nan'),
            'rikishi':       rikishi,
            'rank':          rank,
            'heya':          HEYA.get(wid, ''),
            'w':             float('nan'),
            'l':             float('nan'),
            'k':             0,
            'hoshitori':     float('nan'),
            'university':    float('nan'),
        })

    df = pd.DataFrame(rows)
    out = 'data/banzuke_202605_predicted.csv'
    df.to_csv(out, index=False, encoding='utf-8-sig')
    print(f"\nSaved {len(df)} wrestlers ({sum(r['rank'][0]!='J' for r in rows)} Makuuchi,"
          f" {sum(r['rank'][0]=='J' for r in rows)} Juryo) → '{out}'")
    return df


if __name__ == '__main__':
    build_predicted_banzuke()

"""
banzuke_202603_builder.py

Builds a banzuke CSV for the March 2026 (Haru) tournament before it's available on SumoDB.
Wrestler IDs are sourced from banzuke_detailed.csv historical data.

Run this script to generate data/banzuke_202603.csv, then run:
  python predict_wins.py --banzuke-csv data/banzuke_202603.csv

Note: Makuuchi banzuke is from the official JSA announcement (2026-02-24).
      Juryo banzuke is approximate, based on 202601 results and known promotions.
      Hakuoho changed shikona to Hakunofuji beginning Hatsu 2026.
      Fukuzaki changed shikona to Fujitensei for Haru 2026.
      Seihakuho changed shikona to Toshinofuji for Haru 2026.
"""

import pandas as pd

TOURNAMENT_ID = 202603

# Columns: (rank, rikishi, heya, wrestler_id)
# wrestler_id sourced from banzuke_detailed.csv historical records

MAKUUCHI = [
    ('Y1e', 'Hoshoryu',     'Tatsunami',   12451),
    ('Y1w', 'Onosato',      'Nishonoseki', 12836),
    ('O1e', 'Aonishiki',    'Ajigawa',     12839),
    ('O1w', 'Kotozakura',   'Sadogatake',  12270),
    ('S1e', 'Kirishima',    'Otowayama',   12231),
    ('S1w', 'Takayasu',     'Tagonoura',    6480),
    ('K1e', 'Wakamotoharu', 'Arashio',     11980),
    ('K1w', 'Atamifuji',    'Isegahama',   12664),
    ('M1e', 'Wakatakakage', 'Arashio',     12370),
    ('M1w', 'Yoshinofuji',  'Isegahama',   12884),
    ('M2e', 'Fujinokawa',   'Isenoumi',    12800),
    ('M2w', 'Churanoumi',   'Kise',        12320),
    ('M3e', 'Hiradoumi',    'Sakaigawa',   12314),
    ('M3w', 'Oho',          'Otake',       12453),
    ('M4e', 'Daieisho',     'Oitekaze',    11985),
    ('M4w', 'Takanosho',    'Minatogawa',  11855),
    ('M5e', 'Abi',          'Shikoroyama', 12094),
    ('M5w', 'Kotoshoho',    'Sadogatake',  12449),
    ('M6e', 'Ichiyamamoto', 'Hanaregoma',  12362),
    ('M6w', 'Onokatsu',     'Onomatsu',    12840),
    ('M7e', 'Oshoma',       'Naruto',      12717),
    ('M7w', 'Hakunofuji',   'Isegahama',   12796),  # formerly Hakuoho
    ('M8e', 'Ura',          'Kise',        12226),
    ('M8w', 'Shodai',       'Tokitsukaze', 12130),
    ('M9e', 'Tokihayate',   'Tokitsukaze', 12542),
    ('M9w', 'Tamawashi',    'Kataonami',    5944),
    ('M10e', 'Gonoyama',    'Takekuma',    12688),
    ('M10w', 'Roga',        'Futagoyama',  12516),
    ('M11e', 'Shishi',      'Ikazuchi',    12599),
    ('M11w', 'Oshoumi',     'Naruto',      12634),
    ('M12e', 'Asakoryu',    'Takasago',    12710),
    ('M12w', 'Asanoyama',   'Takasago',    12291),
    ('M13e', 'Tobizaru',    'Oitekaze',    12203),
    ('M13w', 'Fujiseiun',   'Fujishima',   12702),
    ('M14e', 'Chiyoshoma',  'Kokonoe',     11785),
    ('M14w', 'Nishikifuji', 'Isegahama',   12351),
    ('M15e', 'Midorifuji',  'Isegahama',   12352),
    ('M15w', 'Mitakeumi',   'Dewanoumi',   12210),
    ('M16e', 'Asahakuryu',  'Takasago',    12784),
    ('M16w', 'Kinbozan',    'Kise',        12721),
    ('M17e', 'Fujiryoga',   'Fujishima',   12945),
    ('M17w', 'Kotoeiho',    'Sadogatake',  12729),
]

# Approximate Juryo banzuke based on 202601 results + known movements.
# 3 relegated from Makuuchi: Tomokaze (4-11), Ryuden (6-9), Hatsuyama (2-13)
# 3 from Makushita: Toshinofuji (fmr Seihakuho), Fujitensei (fmr Fukuzaki), Shimazuumi
JURYO = [
    ('J1e',  'Asasuiryu',   'Takasago',    12894),
    ('J1w',  'Sadanoumi',   'Sakaigawa',    2879),
    ('J2e',  'Kyokukaiyu',  'Oshima',      12841),
    ('J2w',  'Tomokaze',    'Nakamura',    12427),  # relegated from M13w
    ('J3e',  'Hitoshi',     'Oitekaze',    12704),
    ('J3w',  'Ryuden',      'Takadagawa',   6594),  # relegated from M15e
    ('J4e',  'Kayo',        'Nakamura',    12774),
    ('J4w',  'Nishinoryu',  'Sakaigawa',   12523),
    ('J5e',  'Takerufuji',  'Isegahama',   12780),
    ('J5w',  'Wakanosho',   'Tokiwayama',  12730),
    ('J6e',  'Daiseizan',   'Arashio',     12725),
    ('J6w',  'Meisei',      'Tatsunami',   11946),
    ('J7e',  'Hatsuyama',   'Tamanoi',     12732),  # relegated from M17w
    ('J7w',  'Kagayaki',    'Takadagawa',  11845),
    ('J8e',  'Dewanoryu',   'Dewanoumi',   12592),
    ('J8w',  'Kitanowaka',  'Hakkaku',     12548),
    ('J9e',  'Shirokuma',   'Nishonoseki', 12773),
    ('J9w',  'Tohakuryu',   'Tamanoi',     12575),
    ('J10e', 'Nishikigi',   'Isenoumi',     6596),
    ('J10w', 'Kazuma',      'Kise',        12896),
    ('J11e', 'Toshinofuji', 'Isegahama',   12852),  # fmr Seihakuho, promoted from Ms
    ('J11w', 'Fujitensei',  'Fujishima',   12946),  # fmr Fukuzaki, promoted from Ms
    ('J12e', 'Shimazuumi',  'Hanaregoma',  12013),  # returning from Ms
    ('J12w', 'Tsurugisho',  'Oitekaze',    12113),
    ('J13e', 'Tamashoho',   'Kataonami',   11976),
    ('J13w', 'Tochitaikai', 'Kasugano',    12448),
    ('J14e', 'Kotokuzan',   'Arashio',     11809),
    ('J14w', 'Kazekeno',    'Oshiogawa',   12767),
]


def build_banzuke(output_file: str = 'data/banzuke_202603.csv') -> pd.DataFrame:
    rows = []
    for rank, rikishi, heya, wrestler_id in MAKUUCHI + JURYO:
        rows.append({
            'tournament_id': TOURNAMENT_ID,
            'wrestler_id':   wrestler_id,
            'rikishi':       rikishi,
            'rank':          rank,
            'heya':          heya,
            'w':             float('nan'),
            'l':             float('nan'),
            'k':             0,
            'hoshitori':     float('nan'),
            'university':    float('nan'),
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"Saved {len(df)} wrestlers ({len(MAKUUCHI)} Makuuchi, {len(JURYO)} Juryo) to '{output_file}'")
    print("  Makuuchi banzuke: official (JSA announcement 2026-02-24)")
    print("  Juryo banzuke:    approximate (based on 202601 results)")
    return df


if __name__ == '__main__':
    build_banzuke()

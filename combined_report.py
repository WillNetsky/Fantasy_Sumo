"""Render a combined HTML report comparing the direct LightGBM win predictor
against the v9 matchup-model simulator for a given basho.

Reads:
  outputs/prediction_report_<basho>.csv   (predict_wins.py output)
  outputs/simulation_v4_<basho>.csv       (simulate_tournament_v4.py output)

Writes combined_report_<basho>.html alongside the two single-model reports.
"""

import argparse
import pandas as pd

from sumo_utils import _parse_rank


YUSHO_PTS_WEIGHT = 5.0  # predict_wins.py: yusho_pts = yusho_prob * 5.0
JUN_YUSHO_PTS_WEIGHT = 3.0


def draft_category(rank: str) -> str:
    division, number, _ = _parse_rank(rank)
    if division in ('Y', 'O'):
        return "Yokozuna/Ozeki"
    if division in ('S', 'K'):
        return "Sekiwake/Komusubi"
    if division == 'M':
        if 1 <= number <= 5:
            return "Maegashira 1-5"
        if 6 <= number <= 10:
            return "Maegashira 6-10"
        return "Maegashira 11+"
    if division == 'J':
        return "Juryo"
    return "Other"


def division_for(rank: str) -> str:
    d, _, _ = _parse_rank(rank)
    if d in ('Y', 'O', 'S', 'K', 'M'):
        return 'makuuchi'
    if d == 'J':
        return 'juryo'
    return 'other'


GROUP_CLASS = {
    "Yokozuna/Ozeki": "group-yo",
    "Sekiwake/Komusubi": "group-sk",
    "Maegashira 1-5": "group-m1-5",
    "Maegashira 6-10": "group-m6-10",
    "Maegashira 11+": "group-m11",
    "Juryo": "group-j",
}


def build_combined(basho_id: int) -> pd.DataFrame:
    direct = pd.read_csv(f'outputs/prediction_report_{basho_id}.csv')
    sim = pd.read_csv(f'outputs/simulation_v4_{basho_id}.csv')

    direct = direct.rename(columns={
        'predicted_wins': 'direct_wins',
        'predicted_fantasy_score': 'fantasy_pts',
    })
    direct['direct_yusho_prob'] = direct['yusho_pts'] / YUSHO_PTS_WEIGHT
    direct['direct_jun_yusho_prob'] = direct['jun_yusho_pts'] / JUN_YUSHO_PTS_WEIGHT

    sim = sim.rename(columns={
        'expected_wins': 'sim_wins',
        'yusho_prob': 'sim_yusho_prob',
        'jun_yusho_prob': 'sim_jun_yusho_prob',
        'kachi_koshi_prob': 'sim_kk_prob',
    })

    merged = direct[['rikishi', 'rank', 'direct_wins', 'fantasy_pts',
                     'direct_yusho_prob', 'direct_jun_yusho_prob']].merge(
        sim[['rikishi', 'sim_wins', 'sim_kk_prob',
             'sim_yusho_prob', 'sim_jun_yusho_prob']],
        on='rikishi', how='outer',
    )

    merged['yusho_delta'] = merged['sim_yusho_prob'] - merged['direct_yusho_prob']
    merged['wins_delta'] = merged['sim_wins'] - merged['direct_wins']
    merged['division'] = merged['rank'].apply(division_for)
    return merged


def render(basho_id: int, df: pd.DataFrame, n_sims: int) -> str:
    df = df.sort_values('fantasy_pts', ascending=False)

    rows = []
    for _, r in df.iterrows():
        cat = draft_category(r['rank'])
        group_cls = GROUP_CLASS.get(cat, '')
        div_cls = f"division-{r['division']}"

        def pct(v):
            return f"{v * 100:.2f}%" if pd.notna(v) else "-"

        def num(v, fmt='.2f'):
            return format(v, fmt) if pd.notna(v) else "-"

        delta = r['yusho_delta']
        delta_cls = ''
        if pd.notna(delta):
            if delta > 0.02:
                delta_cls = 'pos'
            elif delta < -0.02:
                delta_cls = 'neg'

        rows.append(
            f"<tr class='{div_cls} {group_cls}'>"
            f"<td>{r['rikishi']}</td>"
            f"<td>{r['rank']}</td>"
            f"<td>{num(r['direct_wins'])}</td>"
            f"<td>{num(r['sim_wins'])}</td>"
            f"<td>{num(r['wins_delta'], '+.2f')}</td>"
            f"<td><strong>{num(r['fantasy_pts'])}</strong></td>"
            f"<td>{pct(r['direct_yusho_prob'])}</td>"
            f"<td>{pct(r['sim_yusho_prob'])}</td>"
            f"<td class='{delta_cls}'>{num(delta * 100, '+.2f') if pd.notna(delta) else '-'}{'pp' if pd.notna(delta) else ''}</td>"
            f"<td>{pct(r['sim_kk_prob'])}</td>"
            f"<td>{pct(r['sim_jun_yusho_prob'])}</td>"
            f"</tr>"
        )
    table_rows = "\n".join(rows)

    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>Sumo Combined Report — {basho_id}</title>
<style>
body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f4f4f9; color: #333; padding: 20px; }}
.container {{ max-width: 1300px; margin: auto; background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
h1, h2, .meta {{ text-align: center; color: #444; }}
.meta {{ color: #777; margin-top: -8px; font-size: 14px; }}
table {{ width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 12.5px; }}
th, td {{ padding: 6px 8px; border: 1px solid #ddd; text-align: left; }}
th {{ background-color: #333; color: #fff; cursor: pointer; user-select: none; position: sticky; top: 0; font-size: 11px; }}
.group-header th {{ background-color: #555; text-align: center; }}
.filters {{ text-align: center; margin: 15px 0; }}
.filters button {{ padding: 8px 15px; margin: 0 5px; border: 1px solid #ccc; background: #eee; cursor: pointer; border-radius: 4px; }}
.filters button.active {{ background: #333; color: #fff; border-color: #333; }}
.group-yo {{ background-color: #ffcccc; }} .group-sk {{ background-color: #ffedcc; }}
.group-m1-5 {{ background-color: #ffffcc; }} .group-m6-10 {{ background-color: #ccffcc; }}
.group-m11 {{ background-color: #ccffff; }} .group-j {{ background-color: #e6ccff; }}
.pos {{ color: #1a7f1a; font-weight: bold; }}
.neg {{ color: #b03030; font-weight: bold; }}
.note {{ font-size: 12px; color: #666; margin-top: 10px; }}
</style>
<script>
function sortTable(n) {{
  var table = document.getElementById("reportTable");
  var rows, switching = true, i, x, y, shouldSwitch, dir = "asc", switchcount = 0;
  while (switching) {{
    switching = false; rows = table.rows;
    for (i = 2; i < rows.length - 1; i++) {{
      shouldSwitch = false;
      x = rows[i].getElementsByTagName("TD")[n];
      y = rows[i + 1].getElementsByTagName("TD")[n];
      var xt = x.innerText.replace('%','').replace('pp','').replace('+','');
      var yt = y.innerText.replace('%','').replace('pp','').replace('+','');
      var xc = isNaN(parseFloat(xt)) ? xt.toLowerCase() : parseFloat(xt);
      var yc = isNaN(parseFloat(yt)) ? yt.toLowerCase() : parseFloat(yt);
      if (dir === "asc" ? xc > yc : xc < yc) {{ shouldSwitch = true; break; }}
    }}
    if (shouldSwitch) {{ rows[i].parentNode.insertBefore(rows[i + 1], rows[i]); switching = true; switchcount++; }}
    else if (switchcount === 0 && dir === "asc") {{ dir = "desc"; switching = true; }}
  }}
}}
function filterDivision(division) {{
  var rows = document.getElementById("reportTable").getElementsByTagName("tr");
  for (var i = 2; i < rows.length; i++) {{
    rows[i].style.display = (division === 'all' || rows[i].classList.contains('division-' + division)) ? "" : "none";
  }}
  document.querySelectorAll('.filters button').forEach(b => b.classList.remove('active'));
  document.querySelector(".filters button[data-div='" + division + "']").classList.add('active');
}}
window.onload = () => {{ document.querySelector(".filters button[data-div='all']").classList.add('active'); }};
</script>
</head><body><div class="container">
<h1>Fantasy Sumo Combined Report</h1>
<h2>Basho: {basho_id}</h2>
<div class="meta">Direct LightGBM win predictor · v9 matchup-model simulator ({n_sims} sims)</div>
<div class="filters">
  <button data-div="all" onclick="filterDivision('all')">All</button>
  <button data-div="makuuchi" onclick="filterDivision('makuuchi')">Makuuchi</button>
  <button data-div="juryo" onclick="filterDivision('juryo')">Juryo</button>
</div>
<table id="reportTable">
<thead>
<tr class="group-header">
  <th colspan="2"></th>
  <th colspan="3">Expected Wins</th>
  <th colspan="3">Yusho Probability</th>
  <th></th>
  <th colspan="2">V9 Simulator Only</th>
</tr>
<tr>
  <th onclick="sortTable(0)">Rikishi</th>
  <th onclick="sortTable(1)">Rank</th>
  <th onclick="sortTable(2)">Direct</th>
  <th onclick="sortTable(3)">V9 Sim</th>
  <th onclick="sortTable(4)">Δ</th>
  <th onclick="sortTable(5)">Fantasy Pts</th>
  <th onclick="sortTable(6)">Direct</th>
  <th onclick="sortTable(7)">V9 Sim</th>
  <th onclick="sortTable(8)">Δ</th>
  <th onclick="sortTable(9)">KK %</th>
  <th onclick="sortTable(10)">Jun-yusho %</th>
</tr>
</thead>
<tbody>{table_rows}</tbody>
</table>
<div class="note">
  Δ columns show V9 sim minus Direct. Yusho probability for the direct model is derived from
  <code>yusho_pts / {YUSHO_PTS_WEIGHT}</code> (the fantasy scoring weight). Fantasy Pts is the
  direct model's combined score. Sort any column by clicking its header.
</div>
</div></body></html>"""


def main():
    parser = argparse.ArgumentParser(description="Combined direct + v9 sim report")
    parser.add_argument('--basho', type=int, required=True, help="Tournament ID (YYYYMM)")
    parser.add_argument('--n-sims', type=int, default=5000, help="Sim count for the meta line")
    args = parser.parse_args()

    df = build_combined(args.basho)
    html = render(args.basho, df, args.n_sims)
    out = f"combined_report_{args.basho}.html"
    with open(out, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Wrote: {out} ({len(df)} wrestlers)")


if __name__ == '__main__':
    main()

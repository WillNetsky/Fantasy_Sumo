# Fantasy Sumo Model Features
*As of March 24, 2026 — both models retrained on data through 202603 (Haru)*

---

## Overview

Two models are in active use:

| Model | File | Task | Performance |
|-------|------|------|-------------|
| **LightGBM Win Predictor** | `sumo_win_predictor_model.joblib` | Predicts total wins for a wrestler in a coming basho (regression) | MAE ~1.79 wins |
| **Matchup Model v9** | `matchup_model_v9.joblib` | Predicts head-to-head match outcome (binary classifier) | Test accuracy 63.85%, AUC 0.704 |

Features are engineered in `sumo_utils.py` (win predictor) and `train_matchup_model_v9.py` (matchup model). Most features are computed from `banzuke_detailed.csv` and `match_history_with_kimarite.csv`.

---

## Model 1: LightGBM Win Predictor (29 features)

**Target:** `w` — total wins in the tournament.

Features are computed per wrestler per basho, reflecting their state *entering* that basho.

### Rank & Division

| Feature | Description |
|---------|-------------|
| `division_numeric` | Current division encoded as a number (Makuuchi=2, Juryo=1, other=0). Captures which tier the wrestler is competing in. |
| `rank_in_division` | Numeric rank within the current division (higher = better rank). Sanyaku get fixed high scores; Maegashira/Juryo are calculated from rank number. |
| `prev_division_numeric` | Division in the *previous* basho. Helps detect wrestlers who just got promoted or relegated. |
| `prev_rank_in_division` | Rank within division in the previous basho. Combined with current rank captures trajectory. |
| `rank_gap` | Difference between the wrestler's all-time highest rank score and their current rank score. Positive = currently below career peak; captures "fallen elite" vs steady performers. |

### Previous Tournament Results

| Feature | Description |
|---------|-------------|
| `prev_w` | Wins in the prior basho. Direct recent performance signal. |
| `prev_l` | Losses in the prior basho. |

### Physical & Background

| Feature | Description |
|---------|-------------|
| `age` | Age at time of basho in decimal years. Performance tends to peak in late 20s and decline after ~33. |
| `bmi` | Body Mass Index (weight_kg / height_m²). Heavier-for-height wrestlers tend to have more staying power but may decline faster. |
| `has_uni_sumo` | Binary flag: 1 if the wrestler competed in university sumo before going pro. University wrestlers typically debut more polished. |

### Stable & Division Context

| Feature | Description |
|---------|-------------|
| `heya_strength` | Rolling win percentage of the wrestler's stable (heya) over recent basho. Proxies for training environment quality and sparring partner strength. |
| `division_strength` | Average rank score of opponents the wrestler faces, reflecting schedule difficulty. |

### Consistency & Streaks

| Feature | Description |
|---------|-------------|
| `win_consistency` | Rolling standard deviation of win totals over past several basho (lower = more consistent). Wrestlers with erratic records are harder to predict. |
| `kachi_koshi_streak` | Number of consecutive basho with a winning record (8+ wins). Captures sustained form. |

### Fighting Style

| Feature | Description |
|---------|-------------|
| `oshi_ratio` | Fraction of career wins achieved via oshi (pushing) techniques. Distinguishes pushers from belt wrestlers. |
| `avg_opponent_oshi_ratio` | Average oshi_ratio of opponents faced recently. A pusher facing a schedule of belt wrestlers may be at a disadvantage. |

### Head-to-Head

| Feature | Description |
|---------|-------------|
| `avg_h2h_win_pct` | Wrestler's average head-to-head win rate against the opponents they typically face in their rank band. Reflects proven matchup edges. |

### Injury / Kyujo Risk

| Feature | Description |
|---------|-------------|
| `was_kyujo_last_basho` | Binary flag: 1 if the wrestler had any kyujo absences in the immediately prior basho. Coarser counterpart to `prev_absences`. |
| `prev_absences` | Number of scheduled bouts missed (kyujo) in the immediately prior basho. |
| `absences_last_3` | Total bouts missed across the previous 3 basho. Captures chronic injury risk. |
| `completion_rate_last_3` | Fraction of scheduled bouts actually competed in over the prior 3 basho. Inverse of absence rate; 1.0 = fully healthy across all three. |
| `age_x_prev_absences` | Interaction of age × prior absences. Older wrestlers with recent kyujo are at higher re-injury risk than younger wrestlers with the same absences. |
| `basho_since_full` | Number of basho since the wrestler last completed all 15 bouts without any absences. Captures lingering / recurring injury patterns. |

### Elo & Momentum

All Elo features are computed from the full match history using a standard Elo rating system (K=32, starting rating 1500).

| Feature | Description |
|---------|-------------|
| `elo` | Current Elo rating entering the basho. The single best summary of overall match-level performance history. |
| `elo_change_last_2_bouts` | Net Elo change over the wrestler's last 2 individual bouts. Very short-term momentum — captures hot/cold streaks within the current or most recent basho. |
| `elo_change_last_5_bouts` | Net Elo change over the last 5 bouts. The strongest Elo window per feature importance; balances recency and noise. |
| `elo_change_last_15_bouts` | Net Elo change over the last 15 bouts (~1 full basho worth). Medium-term form. |
| `elo_change_last_1_basho` | Net Elo change over bouts in the most recent basho. Basho-level momentum. |
| `elo_change_last_2_basho` | Net Elo change over the last 2 basho. Two-tournament trend; consistently the second-strongest Elo window. |

---

## Model 2: Matchup Model v9 (100 features)

**Target:** `A_wins` — binary, whether wrestler A beats wrestler B in a specific bout.

Features are computed per *matchup* (wrestler pair × tournament × day). For each feature, versions `_A`, `_B`, and `_diff` (A minus B) are included where applicable. The training set is doubled by creating symmetric examples (swap A/B, flip target and diff signs).

### Core Ranking

| Feature | Description |
|---------|-------------|
| `rank_diff` | Difference in absolute rank scores (A minus B). The #1 predictor in earlier model versions. Higher-ranked wrestlers win ~52.6% just from rank alone. |
| `rank_in_div_diff` | Difference in rank-within-division scores. Finer-grained than rank_diff; distinguishes M1 from M5 even though both are "Maegashira." |
| `sanyaku_diff` | 1 if A is in sanyaku (Y/O/S/K), 0 otherwise, minus same for B. Captures the sanyaku vs Maegashira structural advantage. |

### Elo

| Feature | Description |
|---------|-------------|
| `elo_diff` | Difference in Elo ratings. Summarizes cumulative head-to-head performance history. |
| `elo_expected_A` | Elo win probability for A: `1 / (1 + 10^((elo_B - elo_A)/400))`. The standard Elo expected score. |

### Physical

| Feature | Description |
|---------|-------------|
| `bmi_diff` | BMI difference (A minus B). Heavier-for-height often advantageous in chest-to-chest contact. |
| `height_diff` | Height difference in cm. Taller wrestlers have leverage advantages in some techniques. |
| `weight_diff` | Weight difference in kg. Raw mass matters for pushing and resisting throws. |
| `age_diff` | Age difference in years. |

### Recent Form

| Feature | Description |
|---------|-------------|
| `win_pct_last_3_A/B` | Each wrestler's win percentage averaged over the prior 3 basho. Absolute form level. |
| `win_pct_last_6_A/B` | Each wrestler's win percentage over the prior 6 basho. Longer-window form. |
| `win_pct_last_3_diff` | Difference in 3-basho win percentage (A minus B). |
| `win_pct_last_6_diff` | Difference in 6-basho win percentage. Consistently among the top features by importance. |

### Career

| Feature | Description |
|---------|-------------|
| `career_win_pct_A/B` | Each wrestler's cumulative career win percentage prior to this tournament. One of the strongest predictors overall. |
| `career_win_pct_diff` | Career win percentage differential. |
| `experience_diff` | Difference in number of career tournaments participated in. |
| `weighted_recent_10_A/B` | Exponentially weighted average of per-basho win % over the last ~3 basho (span=3 EWM). Decays older results; a smoother "hot hand" signal. |
| `weighted_recent_diff` | Difference in weighted recent form. |

### Head-to-Head (Career)

| Feature | Description |
|---------|-------------|
| `h2h_win_pct_A` | A's historical win rate against B over all prior career meetings. Defaults to 0.5 if no prior meetings. |
| `h2h_total_matches` | Total number of prior career bouts between A and B. |
| `has_h2h_history` | Binary flag: 1 if the pair have met before. |

### Head-to-Head (Recent) *(v9 new)*

| Feature | Description |
|---------|-------------|
| `h2h_recent_win_pct_A` | A's win rate over the last 10 encounters with B specifically. More reactive than career H2H; captures recent style shifts or dominance streaks. |

### Rank Trajectory

| Feature | Description |
|---------|-------------|
| `rank_momentum_A/B` | Rolling mean of rank score changes over the last 3 basho (positive = rising). Identifies wrestlers on a hot streak vs. those in decline. |
| `rank_momentum_diff` | Momentum differential. |
| `basho_since_peak_A/B` | Number of basho since the wrestler was last at or near their career-best rank. A low number means they're near their peak; a high number signals a wrestler who has been declining for a while. |
| `basho_since_peak_diff` | Difference in basho-since-peak. |
| `rank_gap_diff` | Difference in rank_gap (current rank vs. career best). A wrestler farther below their peak may be more motivated or may be permanently declining. |

### Rank Volatility *(v9 new)*

| Feature | Description |
|---------|-------------|
| `rank_volatility_A/B` | Rolling standard deviation of rank score changes over the last 6 basho. High volatility = yo-yo wrestler with inconsistent results. Low volatility = reliable rank-holder. The second and fourth most important v9 features. |
| `rank_volatility_diff` | Volatility differential. |

### Interactions

| Feature | Description |
|---------|-------------|
| `rank_x_form_diff` | (rank_score × win_pct_last_3) differential. Combines where a wrestler ranks with how well they're currently fighting — rewards high-ranked wrestlers in good form doubly. |
| `elo_x_age_diff` | Elo weighted by an age-decay sigmoid (peaks at ~30, drops off on either side), then differenced. Penalizes high-Elo wrestlers who are very old or very young relative to their prime. |

### In-Tournament Record *(v9 new)*

| Feature | Description |
|---------|-------------|
| `wins_before_A/B` | Each wrestler's win count within this basho *prior to* the current bout. |
| `losses_before_A/B` | Loss count within the basho prior to the current bout. |
| `record_diff` | (wins−losses) differential going into the match. A wrestler who is 7-0 faces very different pressure than one who is 4-3. The strongest single v9 feature. |
| `day` | The day of the basho (1–15). Later days have more loaded matchups and more pressure-driven outcomes. |

### In-Tournament Streak *(v9 new)*

| Feature | Description |
|---------|-------------|
| `current_streak_A/B` | Signed win/loss streak going into the bout. Positive = current consecutive win count; negative = current consecutive loss count (e.g., +3 means 3 straight wins, −2 means 2 straight losses). |
| `current_streak_diff` | Streak differential. |

### Injury / Kyujo *(v9 new)*

| Feature | Description |
|---------|-------------|
| `absences_last_3_A/B` | Total bouts missed across the prior 3 basho. |
| `absences_last_3_diff` | Absence differential. |
| `completion_rate_last_3_A/B` | Fraction of scheduled bouts completed over the prior 3 basho. |
| `completion_rate_diff` | Completion rate differential. |

### Elo Momentum *(v9 new)*

Per-wrestler Elo momentum signals, each in A, B, and diff form (15 features total):

| Feature | Description |
|---------|-------------|
| `elo_change_last_2_bouts_A/B` | Elo change over the wrestler's last 2 bouts. Very short-term hot/cold signal. |
| `elo_change_last_5_bouts_A/B` | Elo change over last 5 bouts. Best signal-to-noise ratio per ablation. |
| `elo_change_last_15_bouts_A/B` | Elo change over last 15 bouts (~1 basho). Medium-term. |
| `elo_change_last_1_basho_A/B` | Elo change over the most recent basho's bouts. Most important of the Elo momentum features per v9 importance ranking. |
| `elo_change_last_2_basho_A/B` | Elo change over the last 2 basho. Captures two-tournament trend. |
| `*_diff` variants | A minus B for each of the above. |

### Fighting Style Profiles

Style ratios are computed from cumulative career kimarite (winning technique) history prior to each basho. Five technique categories are used:

- **oshi**: pushing techniques (oshidashi, tsukidashi, oshitaoshi, tsukitaoshi)
- **yotsu**: belt/grip techniques (yorikiri, uwatenage, shitatenage, and ~12 others)
- **pull**: pulling/slap-down techniques (hatakikomi, hikiotoshi, and ~5 others)
- **leg**: leg trip/sweep techniques (ketaguri, sotogake, and ~15 others)
- **okuri**: rear-push techniques (okuridashi, okuritaoshi, and ~3 others)

| Feature | Description |
|---------|-------------|
| `{cat}_ratio_A/B` | Fraction of career wins achieved via each technique category. Identifies a wrestler's primary style. 10 features total (5 cats × A/B). |
| `vuln_{cat}_A/B` | Fraction of career *losses* suffered via each technique category. Identifies style-specific vulnerabilities. 10 features total. |
| `{cat}_advantage_diff` | `(cat_ratio_A × vuln_cat_B) − (cat_ratio_B × vuln_cat_A)`. Measures whether A's preferred attack matches B's known weakness, minus the reverse. 5 features total. |
| `style_advantage_diff` | Sum of all five `{cat}_advantage_diff` values. Net style matchup edge for A. |

### Physical × Style Interactions

Capture whether physical traits amplify or suppress a wrestler's style effectiveness:

| Feature | Description |
|---------|-------------|
| `height_x_oshi_diff` | (height × oshi_ratio) differential. Tall pushers have more leverage on forward pressure attacks. |
| `height_x_yotsu_diff` | (height × yotsu_ratio) differential. Taller wrestlers fighting from the belt have long-arm advantages. |
| `weight_x_yotsu_diff` | (weight × yotsu_ratio) differential. Heavier belt wrestlers are harder to lift and throw. |
| `bmi_x_pull_diff` | (BMI × pull_ratio) differential. Compact, dense wrestlers who use pull-downs may gain extra from body proportion. |
| `age_x_oshi_diff` | (age × oshi_ratio) differential. Experienced pushers who rely on oshi may hold that style well into older age; young oshi wrestlers may still be developing technique. |
| `age_x_yotsu_diff` | (age × yotsu_ratio) differential. Belt technique often improves with experience (up to a point). |

---

## Notes

- The matchup model uses symmetric training: each bout generates two examples (A wins + B wins), so all _diff features are negated in the flipped copy. This doubles training data and forces the model to learn direction-agnostic representations.
- Features with near-zero importance after ablation have been dropped: `consec_upset_losses` (subsumed by Elo), `was_kyujo_last_basho` (subsumed by `absences_last_3`), `has_recent_h2h` (importance ~5), and all pressure features (`is_kk_match`, `is_mk_match`).
- Elo windows: `last_5_bouts` consistently ranks highest among bout-window Elo features. `last_15_bouts` is the weakest bout-window signal (dead zone between the 5-bout signal and the 1-basho signal).

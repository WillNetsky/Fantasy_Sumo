# Fantasy Sumo Win Predictor

This project is a data science pipeline designed to predict the performance of sumo wrestlers (rikishi) in upcoming tournaments (basho). Using historical data scraped from [SumoDB](https://sumodb.sumogames.de/), it trains a machine learning model to forecast the number of wins a rikishi will achieve in a 15-day tournament.

The core of this project lies in its comprehensive feature engineering, which transforms raw banzuke (rankings) and historical performance data into a rich feature set for the model.

---
  
## Feature Engineering

The predictive power of the model is derived from a wide array of engineered features that aim to capture a rikishi's current form, career trajectory, physical attributes, and the context of the tournament environment.

### 1. Core Rikishi Attributes

These are fundamental, relatively static attributes of a wrestler.

-   **Physicality**: `age`, `height_m`, `weight_kg`, and `bmi` are calculated for each tournament to track physical development over a rikishi's career.
-   **Background**: `has_uni_sumo` is a binary flag indicating whether a rikishi has a university sumo background, which often correlates with a higher initial skill level.

### 2. Rank-Based Features

Rank is the single most important factor in sumo. These features aim to quantify it in various ways.

-   **`absolute_rank_score`**: A single, continuous numerical score that represents a rikishi's standing across all six divisions. The scoring is carefully weighted to reflect the vast skill gaps between divisions (e.g., the gap between Makuuchi and Juryo is much larger than between Juryo and Makushita).
-   **`division_numeric` & `rank_in_division`**: The rank is split into two components: a numerical score for the division (e.g., Makuuchi=2, Juryo=1) and a score for the rank *within* that division. This allows the model to learn the distinct effects of being, for example, a high-ranked Maegashira versus a low-ranked Komusubi, even if their absolute scores are similar.
-   **`rank_gap`**: The difference between a rikishi's current `absolute_rank_score` and their historical `highest_rank_score`. A large positive gap can indicate a rikishi is in a slump or decline, while a gap of zero means they are at their career-best rank.
-   **Previous Rank Features**: `prev_division_numeric` and `prev_rank_in_division` from the prior tournament are included to capture rank momentum.

### 3. Performance & Momentum Features

These features track a rikishi's recent performance to gauge their current form.

-   **`prev_w` & `prev_l`**: The number of wins and losses from the previous basho. This is a powerful short-term predictor.
-   **`was_kyujo_last_basho`**: A binary flag indicating if the rikishi was absent (due to injury) in the last tournament. Returning from injury is a significant, often negative, performance factor.
-   **`kachi_koshi_streak`**: A count of consecutive tournaments with a winning record (8 wins or more). This is a key indicator of a rikishi's momentum and confidence.
-   **`win_consistency`**: The standard deviation of wins over the last six tournaments. A low value indicates a highly consistent performer, while a high value suggests volatility.

### 4. Contextual & Relational Features

A rikishi's performance is also influenced by their environment and the quality of their competition.

-   **`heya_strength`**: The mean `absolute_rank_score` of all rikishi from the same stable (heya) in a given tournament. This serves as a proxy for the quality of their training environment and partners.
-   **`division_strength`**: The mean `absolute_rank_score` of all *other* rikishi in the same division. This feature quantifies the toughness of the competition a rikishi will face.

### 5. Advanced Features from Match History

By incorporating detailed match-by-match history, we can generate sophisticated features about fighting style and head-to-head matchups.

-   **`oshi_ratio`**: A measure of a rikishi's fighting style, calculated as the career ratio of "pushing/thrusting" wins (Oshi-sumo) to "grappling/belt" wins (Yotsu-sumo). This helps the model understand style matchups (e.g., does a pusher struggle against a belt specialist?).
-   **`avg_opponent_oshi_ratio`**: The average `oshi_ratio` of all other rikishi in the division. This provides context for how a rikishi's personal style might fare against the division's meta.
-   **`avg_h2h_win_pct`**: For each rikishi, this is their average historical head-to-head win percentage against every other rikishi in their division for the upcoming tournament. This is a powerful, direct measure of their expected performance against the specific field of competitors.

---

## Modeling & Prediction

The engineered features are used to train a `RandomForestRegressor` model, with the number of wins (`w`) as the target variable. The trained model is then used in the prediction pipeline to:

1.  Generate a **Fantasy Sumo Draft Helper** for upcoming tournaments, which includes a `predicted_fantasy_score` that probabilistically accounts for points from winning records (kachi-koshi) and potential upsets (kinboshi).
2.  Produce a **Prediction Report** for completed tournaments to evaluate model performance against actual results.

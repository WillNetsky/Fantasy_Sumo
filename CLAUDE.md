# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Fantasy Sumo is a data science pipeline for predicting sumo wrestler performance. It scrapes historical data from SumoDB, trains ML models to predict wins, and generates fantasy sumo draft helpers with probabilistic scoring.

## Commands

### Running Predictions
```bash
python predict_wins.py --data-file data/banzuke_detailed.csv --model-file sumo_win_predictor_model.joblib
```

### Training Models
```bash
# Direct win prediction model (LightGBM)
python train_model_lgbm.py

# Matchup prediction model (current: v8, uses Elo momentum features)
python train_matchup_model_v8.py
```

### Tournament Simulation
```bash
python simulate_tournament_v4.py --tournament 202603 --simulations 500
```

### Data Scraping
```bash
python sumodb_scrape.py  # Scrapes banzuke data from SumoDB
```

### Dependencies
```bash
pip install -r requirements.txt
```

## Architecture

### Data Flow
```
SumoDB (sumodb.sumogames.de)
    ↓ sumodb_scrape.py
banzuke_detailed.csv + match_history_with_kimarite.csv
    ↓ sumo_utils.py (feature engineering)
    ├→ train_model_lgbm.py → sumo_win_predictor_model.joblib
    │                           ↓
    │                       predict_wins.py → prediction_report_YYYYMM.{csv,html}
    │
    └→ train_matchup_model_v8.py → matchup_model_v8.joblib
                                       ↓
                                   simulate_tournament_v4.py → simulation results
```

### Key Files

- **sumo_utils.py**: Central feature engineering module. Contains `preprocess_data()` for all feature generation, `_parse_rank()` for rank string parsing, and `get_absolute_rank_score()` for rank-to-score conversion.

- **predict_wins.py**: Main prediction pipeline. Scrapes upcoming basho, generates predictions with fantasy scoring (wins, kachi-koshi bonus, yusho/jun-yusho probability, kinboshi).

- **train_model_lgbm.py**: Trains direct win prediction model (1000 estimators, lr=0.01, num_leaves=20).

- **train_matchup_model_v8.py**: Trains match outcome predictor. Uses Elo momentum features from `sumo_utils.compute_elo_features()`. Old versions archived in `archive/`.

## Domain Concepts

### Sumo Terminology
- **Basho**: Tournament (6 per year, 15 days each)
- **Banzuke**: Official rankings published before each tournament
- **Kachi-koshi**: Winning record (8+ wins in 15 matches)
- **Kyujo**: Tournament withdrawal due to injury
- **Kinboshi**: Upset bonus when Maegashira defeats Yokozuna
- **Yusho/Jun-yusho**: Tournament championship/runner-up
- **Heya**: Sumo stable (training facility)

### Rank Hierarchy (highest to lowest)
Yokozuna (Y) → Ozeki (O) → Sekiwake (S) → Komusubi (K) → Maegashira (M) → Juryo (J)

Ranks include position and side: "M1e" = Maegashira 1 East

### Tournament ID Format
YYYYMM (e.g., 202601 = January 2026)

## Model Performance

| Approach | Win Prediction MAE | Notes |
|----------|-------------------|-------|
| Direct Model (LightGBM) | ~1.82 | Best for win prediction (with Elo features) |
| Matchup Simulation V8 | — | 59.6%+ match accuracy, compounding errors |

The direct model outperforms matchup simulation due to error accumulation over 15 simulated days.

## Feature Categories

Features are generated in `sumo_utils.preprocess_data()`:

1. **Core attributes**: age, height_m, weight_kg, bmi, has_uni_sumo
2. **Rank-based**: absolute_rank_score, rank_gap, rank momentum
3. **Performance**: prev_w/l, kachi_koshi_streak, win_consistency, was_kyujo_last_basho
4. **Contextual**: heya_strength, division_strength
5. **Advanced**: oshi_ratio, fighting style profiles, avg_h2h_win_pct

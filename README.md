# Fantasy Sumo Win Predictor

A data science pipeline for predicting sumo wrestler performance. Scrapes historical data from [SumoDB](https://sumodb.sumogames.de/), trains ML models to predict wins, and generates fantasy sumo draft helpers with probabilistic scoring.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run predictions for an upcoming tournament
python predict_wins.py --data-file banzuke_detailed.csv --model-file sumo_win_predictor_model.joblib

# Train the direct win prediction model
python train_model_lgbm.py

# Run tournament simulation (v4 with multi-segment scheduling)
python simulate_tournament_v4.py --tournament 202601 --simulations 500 --scheduling jsa_v2

# Predict next tournament's banzuke
python simulate_tournament_v4.py --tournament 202601 --simulations 500 --predict-banzuke
```

## Features

- **Win Prediction**: Predicts total wins for each wrestler in a 15-day tournament
- **Fantasy Scoring**: Calculates expected fantasy points including kachi-koshi bonus, kinboshi, and yusho probability
- **Matchup Prediction**: Predicts individual bout outcomes with 63% accuracy
- **Tournament Simulation**: Monte Carlo simulation with realistic JSA-style scheduling
- **Multi-Segment Scheduling**: 5-segment day groupings (1-3, 4-7, 8-10, 11-12, 13-15) for realistic torikumi
- **Banzuke Prediction**: Predicts future tournament rankings based on simulated results
- **HTML Reports**: Generates styled draft helper reports

## Model Performance

### Direct Win Prediction (LightGBM)
| Metric | Value |
|--------|-------|
| Mean Absolute Error | 1.61 wins |
| Within 2 wins | 67% |
| Within 3 wins | 89% |
| Pearson Correlation | 0.415 |

### Matchup Prediction
| Model Version | Accuracy | AUC | Key Features |
|---------------|----------|-----|--------------|
| V2 (baseline) | 59.6% | 0.644 | Rank, style, H2H |
| V3 | 62.4% | 0.671 | + ELO ratings |
| V4 (best) | **63.0%** | **0.691** | + Stacking ensemble |
| V5 | 62.5% | 0.680 | + Glicko ratings |
| V6 | 62.8% | 0.685 | + Career features |

### Accuracy by Rank Tier
| Tier | MAE (wins) |
|------|------------|
| Yokozuna/Ozeki | 0.73 |
| Maegashira 6-10 | 1.01 |
| Maegashira 11+ | 1.59 |
| Juryo | 1.71 |
| Maegashira 1-5 | 1.96 |
| Sekiwake/Komusubi | 2.42 |

## Architecture

### Data Flow
```
SumoDB (sumodb.sumogames.de)
    |
    v sumodb_scrape.py
banzuke_detailed.csv + match_history_with_kimarite.csv
    |
    v sumo_utils.py (feature engineering)
    |
    +---> train_model_lgbm.py --> sumo_win_predictor_model.joblib
    |                                 |
    |                                 v
    |                             predict_wins.py --> prediction_report_YYYYMM.{csv,html}
    |
    +---> train_matchup_model_v4.py --> matchup_model_v4.joblib
                                            |
                                            v
                                        simulate_tournament_v4.py
                                            |
                                            +---> simulation results
                                            |
                                            v banzuke_predictor.py
                                        predicted_banzuke_YYYYMM.csv
```

### Key Files

| File | Description |
|------|-------------|
| `sumo_utils.py` | Feature engineering module with `preprocess_data()` |
| `predict_wins.py` | Main prediction pipeline with fantasy scoring |
| `train_model_lgbm.py` | Direct win prediction model training |
| `train_matchup_model_v4.py` | Best matchup model (63% accuracy) |
| `simulate_tournament_v4.py` | Tournament simulation with multi-segment scheduling |
| `torikumi_scheduler.py` | JSA-style matchup scheduling (5 day segments) |
| `banzuke_predictor.py` | Rank prediction for future tournaments |
| `predict_wins_hybrid.py` | Hybrid approach combining both models |
| `sumodb_scrape.py` | Data scraping from SumoDB |
| `compare_predictions.py` | Prediction accuracy analysis |

## Feature Engineering

Features are generated in `sumo_utils.preprocess_data()`:

### Core Attributes
- `age`, `height_m`, `weight_kg`, `bmi`
- `has_uni_sumo` - university sumo background

### Rank-Based Features
- `absolute_rank_score` - continuous score across all divisions
- `rank_gap` - current rank vs career-best rank
- `division_numeric`, `rank_in_division`
- Previous rank features for momentum

### Performance Features
- `prev_w`, `prev_l` - previous tournament record
- `kachi_koshi_streak` - consecutive winning records
- `win_consistency` - standard deviation over last 6 basho
- `was_kyujo_last_basho` - injury return flag

### Contextual Features
- `heya_strength` - stable's average rank
- `division_strength` - division's average rank

### Fighting Style (Matchup Model)
- 5 technique categories: oshi, yotsu, pull, leg, okuri
- Offensive ratios (`{cat}_ratio`) and defensive vulnerabilities (`vuln_{cat}`)
- Style advantage features (A's strength vs B's weakness)

### Advanced Features
- `oshi_ratio` - pushing vs grappling style
- `avg_h2h_win_pct` - head-to-head win percentage
- ELO/Glicko ratings (matchup models V3+)

## Usage

### Data Scraping
```bash
# Scrape banzuke data from SumoDB
python sumodb_scrape.py

# Scrape match history with kimarite
python scrape_kimarite.py
```

### Training Models
```bash
# Direct win prediction model (recommended)
python train_model_lgbm.py

# Matchup prediction models
python train_matchup_model_v4.py  # Best accuracy (63%)
```

### Running Predictions
```bash
# Generate predictions for upcoming tournament
python predict_wins.py --data-file banzuke_detailed.csv --model-file sumo_win_predictor_model.joblib

# Hybrid prediction (combines direct + simulation)
python predict_wins_hybrid.py --tournament 202601
```

### Tournament Simulation
```bash
# Run Monte Carlo simulation with multi-segment scheduling (recommended)
python simulate_tournament_v4.py --tournament 202601 --simulations 500 --scheduling jsa_v2

# Scheduling options:
#   jsa_v2: 5-segment (days 1-3, 4-7, 8-10, 11-12, 13-15) - most realistic
#   jsa: 2-segment (days 1-7, 8-15) - simpler
#   simple: basic greedy matching

# Predict next tournament's banzuke based on simulation
python simulate_tournament_v4.py --tournament 202601 --simulations 500 --predict-banzuke
```

### Comparing Predictions
```bash
# Analyze prediction accuracy against actual results
python compare_predictions.py --predictions prediction_report_202601.csv --actuals actual_results.csv
```

## Domain Concepts

### Sumo Terminology
| Term | Description |
|------|-------------|
| Basho | Tournament (6 per year, 15 days each) |
| Banzuke | Official rankings published before each tournament |
| Kachi-koshi | Winning record (8+ wins in 15 matches) |
| Kyujo | Tournament withdrawal due to injury |
| Kinboshi | Upset bonus when Maegashira defeats Yokozuna |
| Yusho | Tournament championship |
| Jun-yusho | Tournament runner-up |
| Heya | Sumo stable (training facility) |

### Rank Hierarchy (highest to lowest)
```
Yokozuna (Y) > Ozeki (O) > Sekiwake (S) > Komusubi (K) > Maegashira (M) > Juryo (J)
```

Ranks include position and side: "M1e" = Maegashira 1 East

### Tournament ID Format
`YYYYMM` (e.g., 202601 = January 2026)

## File Structure

```
Fantasy_Sumo/
├── sumo_utils.py                 # Feature engineering
├── sumodb_scrape.py              # Data scraping
├── scrape_kimarite.py            # Match history scraping
│
├── train_model_lgbm.py           # Direct win model training
├── train_matchup_model.py        # Matchup model V1
├── train_matchup_model_v2.py     # Matchup model V2
├── train_matchup_model_v3.py     # + ELO ratings
├── train_matchup_model_v4.py     # + Stacking (best)
├── train_matchup_model_v5.py     # + Glicko ratings
├── train_matchup_model_v6.py     # + Career features
│
├── predict_wins.py               # Main prediction pipeline
├── predict_wins_hybrid.py        # Hybrid predictions
├── compare_predictions.py        # Accuracy analysis
│
├── simulate_tournament.py        # Tournament sim V1
├── simulate_tournament_v2.py     # Tournament sim V2
├── simulate_tournament_v3.py     # + JSA 2-segment scheduling
├── simulate_tournament_v4.py     # + Multi-segment scheduling + banzuke prediction
│
├── torikumi_scheduler.py         # JSA-style matchup scheduling module
├── banzuke_predictor.py          # Rank prediction module
│
├── banzuke_detailed.csv          # Wrestler data
├── match_history_with_kimarite.csv  # Match history
├── sumo_win_predictor_model.joblib  # Trained direct model
├── matchup_model_v4.joblib       # Trained matchup model
│
└── Notebooks/                    # Jupyter notebooks
```

## Known Limitations

1. **Matchup accuracy ceiling (~63%)** - Sumo has high inherent variance
2. **Injuries unpredictable** - Cannot predict mid-tournament withdrawals
3. **Sparse H2H data** - 65% of wrestler pairs have only 1 previous meeting
4. **Scheduling uncertainty** - Exact matchups unknown before tournament
5. **Error accumulation** - Simulating 15 days compounds prediction errors

## Future Improvements

### High Priority
- [ ] **Injury/health tracking** - Add pre-tournament health indicators to predict kyujo risk
- [x] **In-tournament updates** - Day-by-day prediction updates using current record (implemented in v4 scheduling)
- [x] **Better torikumi prediction** - Multi-segment scheduling with 5 day groupings (torikumi_scheduler.py)
- [x] **Banzuke prediction** - Predict future tournament rankings (banzuke_predictor.py)
- [ ] **Ensemble approach** - Combine direct model + simulation predictions with learned weights

### Medium Priority
- [ ] **Neural network models** - Explore deep learning for non-linear pattern capture
- [ ] **Physical matchup interactions** - Model how height/weight combinations affect outcomes
- [ ] **Day-of-tournament features** - Some wrestlers perform differently on certain days
- [ ] **Heya relationship features** - Training partners may know each other's weaknesses
- [ ] **Pre-tournament news integration** - Form updates between basho

### Lower Priority
- [ ] **Betting odds integration** - Use market data as a feature (if available)
- [ ] **Venue/location features** - Home vs away effects
- [ ] **Time decay on H2H** - Weight recent matches more heavily
- [ ] **Technique sequence analysis** - Model how bouts unfold, not just outcomes

### Data Improvements
- [ ] **Expand historical data** - Include more lower division matches
- [ ] **Add wrestler metadata** - Injury history, training camp reports
- [ ] **Real-time data pipeline** - Automated scraping during tournaments

### Infrastructure
- [ ] **Web interface** - Interactive dashboard for predictions (consider hosting on Render.com — free tier supports Flask, deploy via GitHub repo with `gunicorn`)
- [ ] **API service** - Serve predictions via REST API
- [ ] **Automated retraining** - Update models after each tournament

## References

- SumoDB: https://sumodb.sumogames.de/
- Sumo Reference: http://sumoreference.com/
- Japan Sumo Association: https://www.sumo.or.jp/

## License

This project is for educational and personal use. Data is scraped from SumoDB with respect to their terms of service.

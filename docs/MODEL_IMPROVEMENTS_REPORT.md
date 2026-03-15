# Fantasy Sumo Model Improvements Report

## Executive Summary

This report documents improvements made to the Fantasy Sumo prediction system, including new matchup models and analysis of prediction accuracy for the January 2026 (202601) tournament.

## Original Prediction Performance (202601 Tournament)

### Overall Accuracy
- **Mean Absolute Error (MAE)**: 1.609 wins
- **RMSE**: 2.057 wins
- **Pearson Correlation**: 0.415
- **Spearman Rank Correlation**: 0.400
- **Predictions within 2 wins**: 67.1%
- **Predictions within 3 wins**: 88.6%

### Accuracy by Rank Tier
| Tier | MAE (wins) | Count |
|------|------------|-------|
| Yokozuna/Ozeki | 0.73 | 4 |
| Sekiwake/Komusubi | 2.42 | 4 |
| Maegashira 6-10 | 1.01 | 10 |
| Maegashira 11+ | 1.59 | 14 |
| Maegashira 1-5 | 1.96 | 10 |
| Juryo | 1.71 | 28 |

### Key Findings
1. **Yokozuna/Ozeki predictions were most accurate** (MAE: 0.73)
2. **Sekiwake/Komusubi were hardest to predict** (MAE: 2.42)
3. **Yusho winner (Aonishiki, 12 wins) was ranked #2 in predictions**
4. **Biggest over-predictions**: Hatsuyama (pred 7.2, actual 2), Tochitaikai (pred 7.7, actual 3)
5. **Biggest under-predictions**: Wakanosho (pred 7.2, actual 12), Atamifuji (pred 7.5, actual 12)

## New Matchup Model Development

### Model Versions Tested

| Version | Accuracy | Key Features |
|---------|----------|--------------|
| V2 (baseline) | 59.6% | Basic rank, style, H2H |
| V3 | 62.36% | + ELO ratings, momentum |
| V4 | **62.99%** | + Stacking ensemble, within-basho performance |
| V5 | 62.52% | + Glicko ratings, weighted form |
| V6 | 62.81% | + Career features, interaction terms |

### V4 Model (Best Performing)
- **Test Accuracy**: 62.99%
- **AUC**: 0.6905
- **Brier Score**: 0.2194
- **Improvement over baseline**: +3.4%

### Key Features (by importance)
1. `age_diff` - Age difference between wrestlers
2. `elo_diff` - ELO rating difference
3. `win_pct_last_6` - Recent 6-basho form
4. `elo_expected_A` - ELO-based win probability
5. `okuri_ratio` - Rear technique usage
6. `rank_diff` - Rank score difference

## New Components Added

### 1. Improved Matchup Models (train_matchup_model_v3.py, v4.py, v5.py)
- ELO/Glicko rating systems
- Momentum and weighted form features
- Within-tournament performance tracking
- Style matchup analysis (oshi vs yotsu vs pull)

### 2. Realistic Tournament Simulation (simulate_tournament_v3.py)
- JSA-style torikumi scheduling
- Same heya exclusion rules
- Record-based matchup selection (Days 8-15)
- Rank-based matchup selection (Days 1-7)
- Sanyaku priority scheduling

### 3. Hybrid Prediction System (predict_wins_hybrid.py)
- Combines direct win model (70%) + matchup simulation (30%)
- Leverages strengths of both approaches:
  - Direct model: Lower MAE for total win prediction
  - Matchup model: Captures opponent-specific dynamics

## Limitations

### Match Prediction Accuracy Ceiling
The matchup model appears to have a ceiling around 63% accuracy. This is likely due to:
1. **Inherent unpredictability of sumo** - Individual matches have significant variance
2. **Limited feature availability** - Some factors (injuries, mental state) not in data
3. **Scheduling uncertainty** - Exact matchups unknown before tournament

### Model Performance Reality
- **Direct win prediction**: MAE ~1.7-2.0 wins (strong performance)
- **Match outcome prediction**: ~63% accuracy (vs 52.6% baseline)
- **Error accumulation**: Simulating 15 days compounds prediction errors

## Recommendations

### For Fantasy Drafting
1. **Trust Yokozuna/Ozeki predictions** - Highest accuracy tier
2. **Be cautious with Sekiwake/Komusubi** - Most volatile predictions
3. **Watch for breakout candidates** - Model tends to underestimate surging wrestlers
4. **Consider injury risk** - Hatsuyama's 5.2 win miss likely injury-related

### For Model Improvement
1. **Add injury/health data** - Would improve kyujo prediction
2. **Include pre-tournament news** - Form updates between basho
3. **Ensemble with neural networks** - May capture non-linear patterns
4. **Day-by-day prediction** - Instead of full tournament simulation

## Files Created

```
train_matchup_model_v3.py   # ELO + momentum model (62.36%)
train_matchup_model_v4.py   # Stacking ensemble (62.99%)
train_matchup_model_v5.py   # Glicko + weighted form (62.52%)
train_matchup_model_v6.py   # Career features + interactions (62.81%)
simulate_tournament_v3.py   # Improved torikumi scheduling
predict_wins_hybrid.py      # Hybrid prediction system
compare_predictions.py      # Prediction accuracy analysis
```

## Model Files Generated

```
matchup_model_v3.joblib     # V3 model
matchup_model_v4.joblib     # V4 model (recommended)
matchup_model_v5.joblib     # V5 model
matchup_model_v6.joblib     # V6 model
```

## Conclusion

The prediction system achieved reasonable accuracy for the 202601 tournament:
- **67% of predictions within 2 wins**
- **89% within 3 wins**
- **Yusho winner ranked #2 in predictions**

The new matchup models improved single-match prediction accuracy from 59.6% to 63%, though this falls short of the 65% target. The hybrid approach combining direct win prediction with matchup simulation offers the best overall performance for fantasy scoring purposes.

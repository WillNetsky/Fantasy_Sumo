# Sumo Matchup Prediction Model

A machine learning model that predicts the outcome of individual sumo wrestling bouts. Given two wrestlers, it predicts P(Wrestler A defeats Wrestler B).

## Current Performance

| Metric | Baseline | V1 Model | V2 Model |
|--------|----------|----------|----------|
| Accuracy | 52.6% | 57.4% | **59.6%** |
| AUC | 0.500 | 0.609 | **0.644** |
| Improvement | - | +4.8% | **+7.0%** |

Tested on held-out data from the last 10 tournaments (~45,000 matches).

### V2 Improvements
- Expanded technique categories (oshi/yotsu/pull/leg/okuri)
- Added vulnerability profiles (what techniques each wrestler loses to)
- Style advantage features (A's strength vs B's weakness)

## Features

### V2 Model (37 features)

The improved model uses 37 features capturing matchup dynamics:

### Rank-Based Features
| Feature | Description | Importance |
|---------|-------------|------------|
| `rank_diff` | Difference in absolute rank scores (A - B) | 1396 |
| `rank_in_div_diff` | Difference in position within division | 904 |

### Physical Attributes
| Feature | Description | Importance |
|---------|-------------|------------|
| `age_diff` | Age difference in years | 1814 |
| `weight_diff` | Weight difference in kg | 667 |
| `height_diff` | Height difference in cm | 603 |
| `bmi_diff` | BMI difference | 323 |

### Fighting Style (V2 - 5 categories)
| Category | Techniques | Example |
|----------|------------|---------|
| `oshi` | Pushing/thrusting | oshidashi, tsukidashi |
| `yotsu` | Belt grappling | yorikiri, uwatenage |
| `pull` | Pull-down/slap | hatakikomi, hikiotoshi |
| `leg` | Leg attacks | sotogake, ketaguri |
| `okuri` | Rear attacks | okuridashi, okuritaoshi |

Each wrestler has `{cat}_ratio` (offensive) and `vuln_{cat}` (defensive) for each category.

### Style Advantage Features (V2 - NEW)
| Feature | Description | Importance |
|---------|-------------|------------|
| `vuln_oshi_A/B` | Vulnerability to pushing | 1015/1009 |
| `okuri_ratio_A/B` | Rear attack tendency | 988/959 |
| `vuln_pull_A/B` | Vulnerability to pull-downs | 977/952 |
| `{cat}_advantage_diff` | A's style vs B's weakness | varies |

### Form & History
| Feature | Description | Importance |
|---------|-------------|------------|
| `win_pct_last_3_A` | A's win percentage in last 3 basho | **2041** |
| `win_pct_last_3_B` | B's win percentage in last 3 basho | **2025** |
| `win_pct_last_3_diff` | Difference in recent form | 1289 |
| `h2h_win_pct_A` | A's historical win % against B | 747 |
| `h2h_total_matches` | Number of previous meetings | 784 |

**Key finding:** Recent form (`win_pct_last_3`) is the most predictive feature, more important than rank.

## Data

- **Training data:** 535,695 historical matches (1991-2025)
- **Source:** SumoDB (sumodb.sumogames.de)
- **Coverage:** All divisions, but primarily useful for Makuuchi/Juryo

### Head-to-Head Data Availability
| H2H Matches | Wrestler Pairs | Notes |
|-------------|----------------|-------|
| 1 match | 198,834 | No H2H signal |
| 2-5 matches | 97,931 | Limited signal |
| 6-10 matches | 5,880 | Usable |
| 10+ matches | 3,186 | Good signal |
| 20+ matches | 609 | Excellent signal |

## Usage

### Training the Model

```bash
python train_matchup_model.py
```

Options:
- `--banzuke-file`: Path to banzuke CSV (default: `banzuke_detailed.csv`)
- `--match-file`: Path to match history CSV (default: `match_history_with_kimarite.csv`)
- `--model-file`: Output model path (default: `matchup_model.joblib`)

### Using the Model

```python
import joblib
import pandas as pd

# Load model
bundle = joblib.load('matchup_model.joblib')
model = bundle['model']
features = bundle['features']

# Prepare matchup features
matchup = {
    'rank_diff': 5.0,           # A is 5 ranks higher
    'rank_in_div_diff': 3.0,
    'bmi_diff': 2.5,
    'height_diff': 3.0,
    'weight_diff': 15.0,
    'age_diff': -2.0,           # A is 2 years younger
    'oshi_ratio_A': 0.6,        # A prefers pushing
    'oshi_ratio_B': 0.3,        # B prefers grappling
    'oshi_ratio_diff': 0.3,
    'h2h_win_pct_A': 0.6,       # A has won 60% of previous meetings
    'h2h_total_matches': 5,
    'win_pct_last_3_A': 0.55,
    'win_pct_last_3_B': 0.48,
    'win_pct_last_3_diff': 0.07,
}

X = pd.DataFrame([matchup])[features]
prob_A_wins = model.predict_proba(X)[0][1]
print(f"P(A wins) = {prob_A_wins:.3f}")
```

### Tournament Simulation

```bash
python simulate_tournament.py --tournament 202511 --simulations 500
```

Options:
- `--tournament`: Tournament ID (YYYYMM format)
- `--simulations`: Number of Monte Carlo simulations
- `--matchup-model`: Path to trained model

## Model Architecture

- **Algorithm:** LightGBM Classifier
- **Hyperparameters:**
  - `n_estimators`: 500
  - `learning_rate`: 0.02
  - `num_leaves`: 31
  - `max_depth`: 6
  - `reg_alpha`: 0.1
  - `reg_lambda`: 0.1

## Known Limitations

1. **57.4% accuracy is modest** - Sumo has high inherent variance; many matches are close to 50-50.

2. **Injuries unpredictable** - Model cannot predict mid-tournament withdrawals (kyujo).

3. **Sparse H2H data** - 65% of wrestler pairs have only 1 previous meeting.

4. **Style features limited** - Only captures oshi vs yotsu; misses henka, leg trips, etc.

5. **No in-tournament momentum** - Current version doesn't use day-by-day performance within a tournament.

## Future Improvements

### High Priority (COMPLETED in V2)
- [x] Add kimarite-specific matchup features (vulnerability profiles)
- [x] Improve technique categories (oshi/yotsu/pull/leg/okuri)
- [x] Style advantage features (A's strength vs B's weakness)

### Next Priority
- [ ] Add current tournament record features for days 8-15 predictions
- [ ] Better torikumi (scheduling) prediction algorithm
- [ ] Ensemble: combine direct model + simulation predictions

### Medium Priority
- [ ] Add physical matchup interactions (tall vs short, heavy vs light)
- [ ] Include day-of-tournament features (some wrestlers perform better on certain days)
- [ ] Add heya relationship features (training partners may know each other's weaknesses)

### Low Priority
- [ ] Incorporate betting odds as a feature (if available)
- [ ] Add venue/location features
- [ ] Time decay on H2H records (recent matches weighted more)

## File Structure

```
Fantasy_Sumo/
├── train_matchup_model.py      # V1 model training
├── train_matchup_model_v2.py   # V2 model training (improved)
├── simulate_tournament.py      # V1 tournament simulation
├── simulate_tournament_v2.py   # V2 tournament simulation
├── matchup_model.joblib        # V1 trained model
├── matchup_model_v2.joblib     # V2 trained model
├── match_history_with_kimarite.csv  # Training data
└── banzuke_detailed.csv        # Wrestler attributes
```

## Performance Notes

### Comparison with Direct Win Prediction (202511 Basho)

| Approach | Match Acc. | MAE | Correlation | Within 2 Wins |
|----------|------------|-----|-------------|---------------|
| Direct Model | N/A | **1.71** | **0.488** | **65.7%** |
| Matchup Sim V1 | 57.4% | 1.93 | 0.325 | 60.0% |
| Matchup Sim V2 | 59.6% | 1.90 | 0.347 | 60.0% |

The direct model still outperforms matchup simulation because:
- Compounding errors over 15 days accumulate
- Imperfect matchup scheduling algorithm
- Even 59.6% match accuracy compounds to significant tournament-level error

**Recommendation:** Need >65% match accuracy before simulation becomes competitive, or develop hybrid approach.

## References

- SumoDB: https://sumodb.sumogames.de/
- Sumo Reference: http://sumoreference.com/

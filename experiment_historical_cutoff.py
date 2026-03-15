#!/usr/bin/env python3
"""
Quick experiment: Test how different historical data cutoffs affect matchup model accuracy.
Uses a fast model config to iterate quickly.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score
import time
from train_matchup_model_v2 import (
    load_data, compute_wrestler_profiles, compute_style_profiles,
    compute_vulnerability_profiles, compute_h2h_history,
    TECHNIQUE_CATEGORIES
)

# Fast model params for experimentation
FAST_LGBM_PARAMS = {
    'n_estimators': 150,  # Reduced from 800
    'learning_rate': 0.05,  # Higher LR for faster convergence
    'num_leaves': 31,
    'max_depth': 5,
    'min_child_samples': 100,
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': -1
}

# Test cutoffs
CUTOFF_YEARS = [1991, 1995, 2000, 2005, 2010, 2015]

# Fixed test set: last 10 tournaments (same across all experiments)
TEST_TOURNAMENTS = 10


def build_features_fast(banzuke: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Simplified feature building - uses core features only for speed."""

    # Get profiles
    wrestler_profiles = compute_wrestler_profiles(banzuke, matches)
    style_profiles = compute_style_profiles(matches)
    vuln_profiles = compute_vulnerability_profiles(matches)
    h2h_data = compute_h2h_history(matches)

    # Merge
    wrestler_profiles = wrestler_profiles.merge(
        style_profiles, on=['wrestler_id', 'tournament_id'], how='left'
    )
    wrestler_profiles = wrestler_profiles.merge(
        vuln_profiles, on=['wrestler_id', 'tournament_id'], how='left'
    )

    # Fill missing
    for cat in TECHNIQUE_CATEGORIES.keys():
        wrestler_profiles[f'{cat}_ratio'] = wrestler_profiles[f'{cat}_ratio'].fillna(0.2)
        wrestler_profiles[f'vuln_{cat}'] = wrestler_profiles[f'vuln_{cat}'].fillna(0.2)

    # Build matchups
    matchups = h2h_data.copy()

    profile_cols = ['wrestler_id', 'tournament_id', 'rank_score', 'bmi', 'height',
                    'weight', 'age', 'win_pct_last_3']
    profile_cols += [f'{cat}_ratio' for cat in TECHNIQUE_CATEGORIES.keys()]
    profile_cols += [f'vuln_{cat}' for cat in TECHNIQUE_CATEGORIES.keys()]

    # Merge winner features
    matchups = matchups.merge(
        wrestler_profiles[profile_cols],
        left_on=['winner_id', 'tournament_id'],
        right_on=['wrestler_id', 'tournament_id'],
        how='left'
    ).drop(columns=['wrestler_id'])

    rename_a = {c: f'{c}_A' for c in profile_cols if c not in ['wrestler_id', 'tournament_id']}
    matchups.rename(columns=rename_a, inplace=True)

    # Merge loser features
    matchups = matchups.merge(
        wrestler_profiles[profile_cols],
        left_on=['loser_id', 'tournament_id'],
        right_on=['wrestler_id', 'tournament_id'],
        how='left'
    ).drop(columns=['wrestler_id'])

    rename_b = {c: f'{c}_B' for c in profile_cols if c not in ['wrestler_id', 'tournament_id']}
    matchups.rename(columns=rename_b, inplace=True)

    # H2H
    matchups['winner_is_w1'] = (matchups['winner_id'] == matchups['w1']).astype(int)
    matchups['h2h_wins_A'] = np.where(
        matchups['winner_is_w1'] == 1,
        matchups['w1_prev_wins'],
        matchups['w2_prev_wins']
    )
    matchups['h2h_total_matches'] = matchups['prev_total']
    matchups['h2h_win_pct_A'] = np.where(
        matchups['h2h_total_matches'] > 0,
        matchups['h2h_wins_A'] / matchups['h2h_total_matches'],
        0.5
    )

    # Difference features
    matchups['rank_diff'] = matchups['rank_score_A'] - matchups['rank_score_B']
    matchups['bmi_diff'] = matchups['bmi_A'] - matchups['bmi_B']
    matchups['weight_diff'] = matchups['weight_A'] - matchups['weight_B']
    matchups['age_diff'] = matchups['age_A'] - matchups['age_B']
    matchups['win_pct_diff'] = matchups['win_pct_last_3_A'] - matchups['win_pct_last_3_B']

    # Style advantage
    for cat in TECHNIQUE_CATEGORIES.keys():
        matchups[f'{cat}_adv'] = matchups[f'{cat}_ratio_A'] * matchups[f'vuln_{cat}_B'] - \
                                  matchups[f'{cat}_ratio_B'] * matchups[f'vuln_{cat}_A']

    matchups['A_wins'] = 1

    # Create symmetric examples
    matchups_flip = matchups.copy()
    a_cols = [c for c in matchups.columns if c.endswith('_A')]
    b_cols = [c for c in matchups.columns if c.endswith('_B')]

    for a_col in a_cols:
        b_col = a_col.replace('_A', '_B')
        if b_col in matchups.columns:
            matchups_flip[a_col], matchups_flip[b_col] = matchups[b_col].copy(), matchups[a_col].copy()

    for col in [c for c in matchups.columns if '_diff' in c or '_adv' in c]:
        matchups_flip[col] = -matchups[col]

    matchups_flip['h2h_win_pct_A'] = 1 - matchups['h2h_win_pct_A']
    matchups_flip['A_wins'] = 0

    return pd.concat([matchups, matchups_flip], ignore_index=True)


def run_experiment(banzuke: pd.DataFrame, matches: pd.DataFrame, cutoff_year: int,
                   test_tournaments: list) -> dict:
    """Run a single cutoff experiment."""

    cutoff_id = cutoff_year * 100 + 1  # e.g., 200001 for year 2000

    # Filter data to cutoff
    banzuke_filtered = banzuke[banzuke['tournament_id'] >= cutoff_id].copy()
    matches_filtered = matches[matches['tournament_id'] >= cutoff_id].copy()

    n_tournaments = banzuke_filtered['tournament_id'].nunique()
    n_matches = len(matches_filtered)

    print(f"\n{'='*60}")
    print(f"CUTOFF: {cutoff_year} | Tournaments: {n_tournaments} | Matches: {n_matches:,}")
    print('='*60)

    if n_tournaments < TEST_TOURNAMENTS + 5:
        print("  ⚠️  Not enough tournaments for meaningful train/test split. Skipping.")
        return {'cutoff': cutoff_year, 'accuracy': None, 'auc': None,
                'n_tournaments': n_tournaments, 'n_matches': n_matches}

    start = time.time()

    # Build features
    print("  Building features...", end=" ", flush=True)
    matchups = build_features_fast(banzuke_filtered, matches_filtered)
    print(f"({time.time() - start:.1f}s)")

    # Features list
    features = ['rank_diff', 'bmi_diff', 'weight_diff', 'age_diff', 'win_pct_diff',
                'h2h_win_pct_A', 'h2h_total_matches']
    features += [f'{cat}_adv' for cat in TECHNIQUE_CATEGORIES.keys()]

    # Clean
    df = matchups.dropna(subset=features + ['A_wins'])

    # Split - use fixed test tournaments
    train_mask = ~df['tournament_id'].isin(test_tournaments)
    test_mask = df['tournament_id'].isin(test_tournaments)

    X_train, X_test = df.loc[train_mask, features], df.loc[test_mask, features]
    y_train, y_test = df.loc[train_mask, 'A_wins'], df.loc[test_mask, 'A_wins']

    if len(X_test) == 0:
        print("  ⚠️  No test data available. Skipping.")
        return {'cutoff': cutoff_year, 'accuracy': None, 'auc': None,
                'n_tournaments': n_tournaments, 'n_matches': n_matches}

    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")

    # Train
    print("  Training...", end=" ", flush=True)
    model = lgb.LGBMClassifier(**FAST_LGBM_PARAMS)
    model.fit(X_train, y_train)
    print(f"({time.time() - start:.1f}s total)")

    # Evaluate
    pred_proba = model.predict_proba(X_test)[:, 1]
    pred = (pred_proba > 0.5).astype(int)

    acc = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, pred_proba)

    print(f"\n  ✓ Accuracy: {acc:.4f} | AUC: {auc:.4f}")

    return {
        'cutoff': cutoff_year,
        'accuracy': acc,
        'auc': auc,
        'n_tournaments': n_tournaments,
        'n_matches': n_matches,
        'train_size': len(X_train),
        'test_size': len(X_test)
    }


def main():
    print("Loading data...")
    banzuke, matches = load_data("data/banzuke_detailed.csv", "data/match_history_with_kimarite.csv")

    # Determine test tournaments (most recent 10)
    all_tournaments = sorted(matches['tournament_id'].unique())
    test_tournaments = all_tournaments[-TEST_TOURNAMENTS:]
    print(f"\nTest set: {test_tournaments[0]} to {test_tournaments[-1]} ({len(test_tournaments)} tournaments)")

    results = []
    for cutoff in CUTOFF_YEARS:
        result = run_experiment(banzuke, matches, cutoff, test_tournaments)
        results.append(result)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Historical Cutoff Experiment")
    print("="*70)
    print(f"{'Cutoff':<10} {'Tournaments':<12} {'Matches':<12} {'Accuracy':<12} {'AUC':<10}")
    print("-"*70)

    best_acc = 0
    best_cutoff = None

    for r in results:
        acc_str = f"{r['accuracy']:.4f}" if r['accuracy'] else "N/A"
        auc_str = f"{r['auc']:.4f}" if r['auc'] else "N/A"
        print(f"{r['cutoff']:<10} {r['n_tournaments']:<12} {r['n_matches']:<12,} {acc_str:<12} {auc_str:<10}")

        if r['accuracy'] and r['accuracy'] > best_acc:
            best_acc = r['accuracy']
            best_cutoff = r['cutoff']

    print("-"*70)
    if best_cutoff:
        print(f"Best cutoff: {best_cutoff} with accuracy {best_acc:.4f}")

    print("\nNote: Using fast model config (150 trees). Full model may show different patterns.")


if __name__ == "__main__":
    main()

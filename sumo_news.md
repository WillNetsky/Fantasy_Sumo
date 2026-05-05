# Sumo News & Trends

A running log of retirements, promotions, injuries, notable storylines, and other happenings
relevant to the Fantasy Sumo pipeline. Organized by tournament cycle.

---

## Retirements

### Post-Haru 2026 (202603)

| Rikishi       | Last Rank (202603) | Stable      | Notes                                  |
|---------------|--------------------|-------------|----------------------------------------|
| Chiyomaru     | Sd1w               | Kokonoe     | Former Makuuchi; peaked M2             |
| Chiyosakae    | Jd3w               | Kokonoe     | Lower division career                  |
| Chiyotaiko    | Sd73e              | Kokonoe     | Lower division career                  |
| Daishoho      | Sd16w              | Oitekaze    | Former Juryo; peaked J7                |
| Hidenoumi     | Ms3w               | Isenoumi    | Former Makuuchi; peaked M7. Was on Juryo bubble. |
| Higohikari    | Jk22e (202511)     | —           | Jonokuchi; effectively retired mid-2025 |
| Kotokenryu    | Ms46w              | Sadogatake  | Deep Makushita; kyujo 202603           |
| Minatoryu     | Jk18e (202601)     | Minatogawa  | Jonokuchi; effectively retired early 2026 |
| Ogitora       | Jd21w              | —           | Jonidan                                |
| Shiden        | Ms21e              | Shikoroyama | Makushita; 1-6 in final basho          |
| Yamenosato    | Jd78w              | —           | Deep Jonidan                           |

**Pipeline impact:** None on the May 2026 (202605) banzuke builder. All wrestlers were
below Juryo. Hidenoumi (Ms3w) was the closest to Juryo contention but was not in the
top-6 Makushita promotee pool for 202605.

Lower-division ripple: ~11 slots freed up across Sandanme/Jonidan/Makushita, which will
gradually open paths upward for wrestlers below Ms6 — relevant for 202607 and beyond.

---

## Ozeki / Sanyaku Watch

### Kirishima — Ozeki Run (active as of 202605)
- S1e 202601: ~11W (run starts)
- S1e 202603: 11-4 → **22W over 2 basho at Sekiwake**
- Needs ~11W at 202605 to clear the informal ~33W/3B bar
- Strong candidate for promotion after Natsu 2026

### Atamifuji — Promoted to Sekiwake (202605)
- 12-3 from K1w in 202603 → jumps to S1w
- Young, aggressive wrestler on an upward trajectory

### Sanyaku size (202605)
- 10-man sanyaku: 2Y + 2O + 3S + 3K
- Results in 32 Maegashira slots (M1e–M16w) — one extra row vs. standard 30-slot banzuke

---

## Yokozuna / Top Division

### Hoshoryu & Onosato (Y1e / Y1w)
- Both went 10-5 in 202603 — solid but not dominant
- No retirement discussions as of March 2026

### Aonishiki (O1e)
- 12-3 in 202603; dominant Ozeki performance
- Primary yusho threat going into Natsu 2026

---

## Notable Division Movements (202603 → 202605)

### Promotions to Makuuchi
| Rikishi    | From  | To (predicted) | Record  |
|------------|-------|----------------|---------|
| Wakanosho  | J3w   | ~M13-M15       | 12-3    |
| Asasuiryu  | J2e   | ~M14-M16       | 10-5    |
| Sadanoumi  | J1w   | borderline     | 9-6     |

### Demotions from Makuuchi
| Rikishi     | Rank    | Record   | Predicted new rank |
|-------------|---------|----------|--------------------|
| Kinbozan    | M16w    | 4-11     | ~J4e               |
| Midorifuji  | M15e    | 6-9      | ~J1e               |
| Nishikifuji | M14w    | 6-6-3ab  | borderline; ~J1w   |

### Promotions to Juryo
| Rikishi    | From  | Record | Notes                    |
|------------|-------|--------|--------------------------|
| Nobehara   | Ms6e  | 7-0    | Perfect record; jumps to upper Juryo |
| Okaryu     | Ms2w  | 6-1    | Strong upper-Juryo placement |
| Enho       | Ms4e  | 6-1    | Veteran returning to Juryo |
| Himukamaru | Ms1e  | 4-3    | Borderline                |

### Demotions from Juryo
| Rikishi    | Rank  | Record    | Notes             |
|------------|-------|-----------|-------------------|
| Hatsuyama  | J9w   | 2-13      | Certain demotion  |
| Kotokuzan  | J14w  | 3-12      | Certain demotion  |
| Tomokaze   | J2w   | 4-11      | Certain demotion  |

---

## Stable (Heya) Notes

### Kokonoe Stable
- Lost three wrestlers in the post-Haru 2026 retirement wave (Chiyomaru, Chiyosakae, Chiyotaiko)
- No active Makuuchi/Juryo representatives as of 202605

---

## Basho Results Summary

### 202603 — Haru Basho (March 2026, Osaka)
- **Yusho:** Aonishiki (O1e) 12-3
- **Jun-yusho:** Atamifuji (K1w) 12-3, Fujiseiun (M13w) 11-4 tied
- **Ginosho (Technique):** TBD
- **Kantosho (Fighting Spirit):** TBD
- **Kinboshi:** TBD (any M defeating Y)
- **Notable:** Kirishima builds Ozeki run to 22W/2B; Atamifuji promoted to Sekiwake

---

## Pipeline / Model Notes

### 2026-05-05 — Win predictor MAE regression fixed
- Found that commit `0ff9257` (blanket Elo ffill across all wrestler-tournament rows) regressed win-predictor MAE from 1.79 → 1.99 by filling historical kyujo gaps with each wrestler's pre-absence Elo, training the model on "high Elo + 0 wins" noise pairs.
- Fix in `3938780`: ffill is now scoped to tournaments after the last tournament present in match_history. Upcoming-basho inference still works; training data is unaffected.
- Verified MAE back to 1.7903 with `--no-tune` defaults.
- Matchup model v9 retrain confirmed: 63.96% accuracy, AUC 0.7040, Brier 0.215.

### 2026-05-05 — May 2026 (202605) predictions regenerated
- Run against fixed win predictor and `data/banzuke_202605_predicted.csv` (built from the 202605 banzuke builder).
- 66 rikishi predicted; predicted-wins range 5.4–11.6, mean 7.32.
- **Model's top yusho contenders for Natsu 2026:**

| Rikishi   | Rank | Predicted wins | Yusho prob |
|-----------|------|---------------:|-----------:|
| Aonishiki | O1e  | 11.6           | 28.2%      |
| Onosato   | Y1w  | 9.7            | 8.8%       |
| Hoshoryu  | Y1e  | 9.7            | 7.3%       |
| Atamifuji | S1w  | 9.3            | 5.1%       |
| Kirishima | S1e  | 8.3            | 2.7%       |

- Aonishiki is the runaway favorite, consistent with his 12-3 Haru run and Ozeki promotion.

---

*Last updated: 2026-05-05*

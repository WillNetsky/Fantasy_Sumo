# banzuke_predictor.py
"""
Banzuke (ranking) prediction engine.

Predicts future tournament rankings based on:
1. Current rank
2. Tournament record (wins/losses)
3. Special status (Ozeki kadoban, Yokozuna, etc.)
4. Promotion/demotion rules for division borders

Key JSA Rules Implemented:
- Kachi-koshi (8-7): Rise ~1.5 ranks
- Make-koshi (7-8): Fall ~1.5 ranks
- Strong records scale proportionally
- Ozeki kadoban system (must get 8 wins or face demotion)
- Yokozuna never demoted (only retire)
- Sekiwake/Komusubi: need strong records to stay
- Division border handling (Juryo/Makuuchi exchange)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
import numpy as np
import pandas as pd
from sumo_utils import _parse_rank, get_absolute_rank_score


class RankDivision(Enum):
    """Sumo divisions in order of prestige."""
    YOKOZUNA = "Y"
    OZEKI = "O"
    SEKIWAKE = "S"
    KOMUSUBI = "K"
    MAEGASHIRA = "M"
    JURYO = "J"
    MAKUSHITA = "Ms"
    SANDANME = "Sd"
    JONIDAN = "Jd"
    JONOKUCHI = "Jk"


# Division hierarchy for comparisons
DIVISION_ORDER = {
    RankDivision.YOKOZUNA: 6,
    RankDivision.OZEKI: 5,
    RankDivision.SEKIWAKE: 4,
    RankDivision.KOMUSUBI: 3,
    RankDivision.MAEGASHIRA: 2,
    RankDivision.JURYO: 1,
    RankDivision.MAKUSHITA: 0,
}


@dataclass
class RankPosition:
    """Represents a wrestler's rank."""
    division: RankDivision
    number: int
    side: str  # 'e' (east) or 'w' (west)

    @classmethod
    def from_string(cls, rank_str: str) -> Optional['RankPosition']:
        """Parse a rank string into a RankPosition object."""
        div_str, num, side = _parse_rank(rank_str)
        if not div_str:
            return None
        try:
            division = RankDivision(div_str)
        except ValueError:
            return None
        return cls(division=division, number=num or 1, side=side or 'e')

    def to_string(self) -> str:
        """Convert back to a rank string."""
        if self.division in [RankDivision.YOKOZUNA, RankDivision.OZEKI,
                             RankDivision.SEKIWAKE, RankDivision.KOMUSUBI]:
            return f"{self.division.value}{self.number}{self.side}"
        return f"{self.division.value}{self.number}{self.side}"

    def to_score(self) -> float:
        """Get the absolute rank score."""
        return get_absolute_rank_score(self.to_string())

    def __lt__(self, other: 'RankPosition') -> bool:
        """Compare ranks (lower score = lower rank)."""
        return self.to_score() < other.to_score()


@dataclass
class WrestlerBanzukeRecord:
    """Tracks a wrestler's status for banzuke calculation."""
    wrestler_id: int
    rikishi: str
    current_rank: RankPosition
    wins: int
    losses: int
    absences: int = 0

    # Special flags
    is_ozeki_kadoban: bool = False
    ozeki_wins_last_3: int = 0  # For Ozeki promotion tracking
    consecutive_kk_at_sekiwake: int = 0

    # Calculated fields
    new_rank: Optional[RankPosition] = None
    rank_change_score: float = 0.0  # Change in rank score

    @property
    def record_diff(self) -> int:
        """Wins minus losses, not counting absences."""
        return self.wins - self.losses

    @property
    def is_kachi_koshi(self) -> bool:
        """Has a winning record (8+ wins)."""
        return self.wins >= 8

    @property
    def is_make_koshi(self) -> bool:
        """Has a losing record (8+ losses)."""
        return self.losses >= 8

    @property
    def total_bouts(self) -> int:
        """Total bouts completed."""
        return self.wins + self.losses


# Rank change lookup based on historical JSA patterns
# Format: wins -> typical rank score change (positive = promotion)
RANK_CHANGE_TABLE = {
    15: 14.0,   # Zensho yusho - massive jump
    14: 11.0,
    13: 9.0,
    12: 7.0,
    11: 5.0,
    10: 3.5,
    9: 2.0,
    8: 1.0,     # Bare kachi-koshi
    7: -1.0,    # Bare make-koshi
    6: -2.0,
    5: -3.5,
    4: -5.0,
    3: -7.0,
    2: -9.0,
    1: -11.0,
    0: -14.0,
}


def calculate_rank_change(record: WrestlerBanzukeRecord) -> float:
    """
    Calculate the expected rank score change based on tournament record.

    Uses historical patterns:
    - Each win above 7.5 = roughly +0.5 to +1 rank score
    - Each loss above 7.5 = roughly -0.5 to -1 rank score
    - Yusho winner gets extra boost
    - Strong records at high ranks have multiplied effect

    Returns:
        Float representing rank score change (positive = promotion)
    """
    wins = record.wins
    total = record.total_bouts

    # Handle incomplete tournaments (kyujo)
    if total < 15:
        # Absences counted as losses for rank change calculation
        effective_losses = 15 - wins
        wins = record.wins
    else:
        effective_losses = record.losses

    # Look up base change from table
    if wins in RANK_CHANGE_TABLE:
        base_change = RANK_CHANGE_TABLE[wins]
    else:
        # Interpolate for edge cases
        base_change = (wins - 7.5) * 1.0

    # Adjust for rank - high-ranked wrestlers with strong records move less
    # (there's less room to move up), low-ranked move more
    current_score = record.current_rank.to_score()

    if current_score >= 40:  # Upper Maegashira/Sanyaku
        # Diminishing returns at top
        if base_change > 0:
            base_change *= 0.8
    elif current_score <= 25:  # Lower Maegashira/Juryo
        # More movement at bottom
        if abs(base_change) > 2:
            base_change *= 1.1

    return base_change


def handle_yokozuna(record: WrestlerBanzukeRecord) -> None:
    """
    Handle Yokozuna ranking rules.

    Yokozuna never demoted - they either stay or retire.
    Poor performance usually leads to retirement, not demotion.
    """
    # Yokozuna always stays Yokozuna (or retires)
    record.new_rank = record.current_rank
    record.rank_change_score = 0.0


def handle_ozeki(record: WrestlerBanzukeRecord, kadoban_status: Dict[int, bool]) -> None:
    """
    Handle Ozeki ranking rules with kadoban system.

    Rules:
    - Ozeki with make-koshi becomes kadoban
    - Kadoban Ozeki with make-koshi drops to Sekiwake
    - Can regain Ozeki with 10+ wins next basho (special status)
    """
    w_id = record.wrestler_id

    if record.is_make_koshi:
        if record.is_ozeki_kadoban or kadoban_status.get(w_id, False):
            # Kadoban Ozeki with make-koshi -> Sekiwake
            record.new_rank = RankPosition(RankDivision.SEKIWAKE, 1, 'e')
            record.rank_change_score = -5.0
            kadoban_status[w_id] = False  # No longer kadoban (demoted)
        else:
            # First make-koshi -> becomes kadoban but stays Ozeki
            record.new_rank = record.current_rank
            record.rank_change_score = 0.0
            kadoban_status[w_id] = True
            record.is_ozeki_kadoban = True
    else:
        # Kachi-koshi clears kadoban, stays Ozeki
        record.new_rank = record.current_rank
        record.rank_change_score = 0.0
        kadoban_status[w_id] = False
        record.is_ozeki_kadoban = False


def handle_sekiwake(
    record: WrestlerBanzukeRecord,
    ozeki_promotion_tracker: Dict[int, int]
) -> None:
    """
    Handle Sekiwake ranking rules.

    Rules:
    - Ozeki promotion: ~33 wins over 3 basho at Sekiwake or better
    - Kachi-koshi: Stay at Sekiwake
    - Make-koshi: Drop to Komusubi or lower
    """
    w_id = record.wrestler_id

    if record.is_kachi_koshi:
        # Update Ozeki promotion tracker
        current_total = ozeki_promotion_tracker.get(w_id, 0)
        new_total = current_total + record.wins
        ozeki_promotion_tracker[w_id] = new_total

        # Check for Ozeki promotion (33 wins over 3 basho is the guideline)
        # Also need to be at S/K level for those basho
        if record.wins >= 13 or (record.wins >= 11 and new_total >= 33):
            record.new_rank = RankPosition(RankDivision.OZEKI, 2, 'e')
            record.rank_change_score = 5.0
        else:
            # Stay at Sekiwake
            record.new_rank = record.current_rank
            record.rank_change_score = 0.0
    else:
        # Make-koshi at Sekiwake -> drop
        ozeki_promotion_tracker[w_id] = 0  # Reset tracker

        if record.losses <= 9:
            # Mild make-koshi -> Komusubi
            record.new_rank = RankPosition(RankDivision.KOMUSUBI, 1, 'e')
            record.rank_change_score = -3.0
        else:
            # Severe make-koshi -> Upper Maegashira
            drop_to = min(5, record.losses - 7)
            record.new_rank = RankPosition(RankDivision.MAEGASHIRA, drop_to, 'e')
            record.rank_change_score = calculate_rank_change(record)


def handle_komusubi(record: WrestlerBanzukeRecord) -> None:
    """
    Handle Komusubi ranking rules.

    Rules:
    - Strong kachi-koshi (10+): Promotion to Sekiwake
    - Kachi-koshi (8-9): Stay at Komusubi
    - Make-koshi: Drop to Maegashira
    """
    if record.is_kachi_koshi:
        if record.wins >= 10:
            record.new_rank = RankPosition(RankDivision.SEKIWAKE, 2, 'e')
            record.rank_change_score = 3.0
        else:
            # Stay at Komusubi
            record.new_rank = record.current_rank
            record.rank_change_score = 0.0
    else:
        # Make-koshi -> Maegashira
        drop_to = max(1, 9 - record.wins)
        record.new_rank = RankPosition(RankDivision.MAEGASHIRA, drop_to, 'e')
        record.rank_change_score = calculate_rank_change(record)


def handle_division_border(
    records: List[WrestlerBanzukeRecord],
    upper_division: RankDivision,
    lower_division: RankDivision,
    upper_max_rank: int,
    lower_promotion_threshold: int
) -> None:
    """
    Handle promotion/demotion at division borders.

    For Juryo/Makuuchi border:
    - Top Juryo with 8+ wins usually promoted
    - Bottom Makuuchi with 7 or fewer wins may be demoted
    - Exchange depends on available slots
    """
    # Find wrestlers at the border
    lower_div_wrestlers = [
        r for r in records
        if r.current_rank.division == lower_division
        and r.current_rank.number <= lower_promotion_threshold
        and r.new_rank is None  # Not already processed
    ]

    upper_div_bottom = [
        r for r in records
        if r.current_rank.division == upper_division
        and r.current_rank.number >= upper_max_rank - 4  # Bottom ~5 ranks
        and r.new_rank is None
    ]

    # Calculate promotion candidates (kachi-koshi in top of lower division)
    promotion_candidates = [
        r for r in lower_div_wrestlers
        if r.is_kachi_koshi
    ]
    promotion_candidates.sort(
        key=lambda r: (r.wins, -r.current_rank.number),
        reverse=True
    )

    # Calculate demotion candidates (make-koshi at bottom of upper division)
    demotion_candidates = [
        r for r in upper_div_bottom
        if r.is_make_koshi
    ]
    demotion_candidates.sort(
        key=lambda r: (r.losses, r.current_rank.number),
        reverse=True
    )

    # Match promotions with demotions
    n_exchanges = min(len(promotion_candidates), len(demotion_candidates))

    for i in range(n_exchanges):
        promoted = promotion_candidates[i]
        demoted = demotion_candidates[i]

        # Promoted wrestler goes to bottom of upper division
        promoted.new_rank = RankPosition(
            upper_division,
            upper_max_rank - i,
            'e' if i % 2 == 0 else 'w'
        )
        promoted.rank_change_score = 20.0  # Big jump for promotion

        # Demoted wrestler goes to top of lower division
        demoted.new_rank = RankPosition(
            lower_division,
            1 + i,
            'e' if i % 2 == 0 else 'w'
        )
        demoted.rank_change_score = -20.0


def assign_remaining_ranks(
    records: List[WrestlerBanzukeRecord],
    division: RankDivision,
    max_wrestlers: int = 42
) -> None:
    """
    Assign final rank positions for wrestlers in a division.

    Sorts by expected new score and assigns sequential ranks,
    ensuring no duplicate positions.
    """
    # Get unassigned wrestlers in this division
    div_records = [
        r for r in records
        if r.new_rank is None and r.current_rank.division == division
    ]

    if not div_records:
        return

    # Calculate expected new scores
    for record in div_records:
        expected_score = record.current_rank.to_score() + calculate_rank_change(record)
        record._expected_score = expected_score
        record.rank_change_score = calculate_rank_change(record)

    # Sort by expected score (descending)
    div_records.sort(key=lambda r: r._expected_score, reverse=True)

    # Assign ranks
    for i, record in enumerate(div_records):
        rank_num = i // 2 + 1  # 1, 1, 2, 2, 3, 3, ...
        side = 'e' if i % 2 == 0 else 'w'

        if rank_num > max_wrestlers // 2:
            # Would exceed division size - push to lower division
            if division == RankDivision.MAEGASHIRA:
                record.new_rank = RankPosition(RankDivision.JURYO, 1, side)
            else:
                record.new_rank = RankPosition(division, max_wrestlers // 2, side)
        else:
            record.new_rank = RankPosition(division, rank_num, side)


def count_sanyaku_slots(records: List[WrestlerBanzukeRecord]) -> Dict[RankDivision, int]:
    """
    Count how many wrestlers should be at each sanyaku rank.

    Returns minimum counts based on records:
    - Yokozuna: based on existing Yokozuna
    - Ozeki: existing + promotions
    - Sekiwake: at least 2 (E/W)
    - Komusubi: at least 2 (E/W)
    """
    slots = {
        RankDivision.YOKOZUNA: 0,
        RankDivision.OZEKI: 0,
        RankDivision.SEKIWAKE: 2,
        RankDivision.KOMUSUBI: 2,
    }

    for r in records:
        if r.new_rank:
            div = r.new_rank.division
            if div in slots:
                slots[div] = max(slots[div], r.new_rank.number * 2)

    return slots


def predict_next_banzuke(
    current_banzuke: pd.DataFrame,
    tournament_results: Dict[int, Tuple[int, int]],  # wrestler_id -> (wins, losses)
    yusho_winner_id: Optional[int] = None,
    jun_yusho_ids: List[int] = None,
    kadoban_status: Dict[int, bool] = None,
    ozeki_promotion_tracker: Dict[int, int] = None
) -> Tuple[pd.DataFrame, Dict[int, bool], Dict[int, int]]:
    """
    Predict the next tournament's banzuke based on results.

    Args:
        current_banzuke: DataFrame with current rankings
        tournament_results: Dict mapping wrestler_id to (wins, losses)
        yusho_winner_id: ID of tournament winner (gets extra boost)
        jun_yusho_ids: IDs of runner-up(s)
        kadoban_status: Dict of Ozeki kadoban status (mutated)
        ozeki_promotion_tracker: Cumulative wins for Ozeki promotion (mutated)

    Returns:
        Tuple of:
        - DataFrame with predicted next banzuke
        - Updated kadoban_status dict
        - Updated ozeki_promotion_tracker dict
    """
    if kadoban_status is None:
        kadoban_status = {}
    if ozeki_promotion_tracker is None:
        ozeki_promotion_tracker = {}
    if jun_yusho_ids is None:
        jun_yusho_ids = []

    # Build wrestler records
    records = []
    for _, row in current_banzuke.iterrows():
        w_id = row['wrestler_id']
        wins, losses = tournament_results.get(w_id, (0, 0))

        rank_pos = RankPosition.from_string(row['rank'])
        if rank_pos is None:
            continue

        record = WrestlerBanzukeRecord(
            wrestler_id=w_id,
            rikishi=row['rikishi'],
            current_rank=rank_pos,
            wins=wins,
            losses=losses,
            is_ozeki_kadoban=kadoban_status.get(w_id, False),
            ozeki_wins_last_3=ozeki_promotion_tracker.get(w_id, 0)
        )
        records.append(record)

    # Step 1: Handle sanyaku special rules (order matters!)

    # Yokozuna first
    for record in records:
        if record.current_rank.division == RankDivision.YOKOZUNA:
            handle_yokozuna(record)

    # Ozeki second
    for record in records:
        if record.current_rank.division == RankDivision.OZEKI:
            handle_ozeki(record, kadoban_status)

    # Sekiwake third
    for record in records:
        if record.current_rank.division == RankDivision.SEKIWAKE:
            handle_sekiwake(record, ozeki_promotion_tracker)

    # Komusubi fourth
    for record in records:
        if record.current_rank.division == RankDivision.KOMUSUBI:
            handle_komusubi(record)

    # Step 2: Handle Juryo/Makuuchi border
    handle_division_border(
        records,
        RankDivision.MAEGASHIRA, RankDivision.JURYO,
        upper_max_rank=17, lower_promotion_threshold=5
    )

    # Step 3: Assign remaining Maegashira ranks
    assign_remaining_ranks(records, RankDivision.MAEGASHIRA, max_wrestlers=34)

    # Step 4: Assign remaining Juryo ranks
    assign_remaining_ranks(records, RankDivision.JURYO, max_wrestlers=28)

    # Step 5: Handle yusho winner bonus (extra consideration)
    if yusho_winner_id:
        for record in records:
            if record.wrestler_id == yusho_winner_id:
                # Yusho winners typically get extra promotion consideration
                # This is already partially handled by their high win count
                # Add a small bonus to ensure they're ranked appropriately
                if record.new_rank and record.current_rank.division == RankDivision.MAEGASHIRA:
                    current_num = record.new_rank.number
                    if current_num > 1:
                        # Bump up slightly for yusho
                        record.new_rank = RankPosition(
                            record.new_rank.division,
                            max(1, current_num - 1),
                            record.new_rank.side
                        )

    # Build output DataFrame
    output_data = []
    for record in records:
        if record.new_rank:
            output_data.append({
                'wrestler_id': record.wrestler_id,
                'rikishi': record.rikishi,
                'rank': record.new_rank.to_string(),
                'rank_score': record.new_rank.to_score(),
                'prev_rank': record.current_rank.to_string(),
                'prev_wins': record.wins,
                'prev_losses': record.losses,
                'rank_change_score': record.rank_change_score,
                'is_ozeki_kadoban': record.is_ozeki_kadoban
            })

    result_df = pd.DataFrame(output_data)

    # Sort by rank score descending
    if not result_df.empty:
        result_df = result_df.sort_values('rank_score', ascending=False).reset_index(drop=True)

    return result_df, kadoban_status, ozeki_promotion_tracker


def simulate_multiple_tournaments(
    initial_banzuke: pd.DataFrame,
    match_history: pd.DataFrame,
    model_bundle: dict,
    n_tournaments: int = 6,
    n_sims_per_tournament: int = 100,
    verbose: bool = True
) -> List[pd.DataFrame]:
    """
    Chain tournament simulations to predict future rankings.

    Flow:
    1. Simulate tournament N with current banzuke
    2. Predict banzuke for tournament N+1
    3. Repeat

    Args:
        initial_banzuke: Starting banzuke DataFrame
        match_history: Historical match data
        model_bundle: Trained matchup model
        n_tournaments: Number of tournaments to simulate
        n_sims_per_tournament: Simulations per tournament
        verbose: Print progress

    Returns:
        List of predicted banzuke DataFrames for each tournament
    """
    # Import here to avoid circular dependency
    from simulate_tournament_v4 import simulate_tournament, prepare_wrestlers, compute_h2h

    predicted_banzukes = [initial_banzuke.copy()]
    current_banzuke = initial_banzuke.copy()

    # Track state that persists across tournaments
    kadoban_status = {}
    ozeki_tracker = {}

    for t in range(n_tournaments):
        if verbose:
            print(f"\nSimulating tournament {t + 1}/{n_tournaments}...")

        # Get tournament ID (approximate - just for data prep)
        base_tournament = int(current_banzuke['tournament_id'].max()) if 'tournament_id' in current_banzuke.columns else 202601
        next_tournament = base_tournament + (1 if base_tournament % 100 < 11 else 89)  # Increment month

        # Prepare wrestlers from current banzuke
        wrestlers_df = prepare_wrestlers(current_banzuke, match_history, next_tournament)

        if wrestlers_df.empty:
            if verbose:
                print("  No wrestlers found, stopping.")
            break

        # Compute H2H history
        h2h = compute_h2h(match_history, next_tournament)

        # Run simulation
        results = simulate_tournament(
            model_bundle, wrestlers_df, h2h,
            n_sims=n_sims_per_tournament
        )

        # Convert simulation results to tournament outcomes
        # Use expected wins rounded to simulate "actual" results
        tournament_results = {}
        for _, row in results.iterrows():
            wins = int(round(row['expected_wins']))
            losses = 15 - wins
            tournament_results[row['wrestler_id']] = (wins, losses)

        # Find yusho winner (highest expected wins)
        yusho_winner = results.loc[results['expected_wins'].idxmax()]['wrestler_id']

        # Predict next banzuke
        next_banzuke, kadoban_status, ozeki_tracker = predict_next_banzuke(
            current_banzuke,
            tournament_results,
            yusho_winner_id=yusho_winner,
            kadoban_status=kadoban_status,
            ozeki_promotion_tracker=ozeki_tracker
        )

        if verbose:
            print(f"  Predicted {len(next_banzuke)} wrestlers for next banzuke")

        predicted_banzukes.append(next_banzuke)
        current_banzuke = next_banzuke

    return predicted_banzukes


def compare_predicted_vs_actual(
    predicted: pd.DataFrame,
    actual: pd.DataFrame
) -> Dict[str, float]:
    """
    Compare predicted banzuke against actual.

    Returns dict with accuracy metrics:
    - mae_rank_score: Mean absolute error of rank scores
    - division_accuracy: % correct division
    - within_2_ranks: % within 2 rank positions
    - within_5_ranks: % within 5 rank positions
    """
    merged = predicted.merge(
        actual[['wrestler_id', 'rank']].rename(columns={'rank': 'actual_rank'}),
        on='wrestler_id',
        how='inner'
    )

    if merged.empty:
        return {'error': 'No matching wrestlers'}

    # Calculate actual rank scores
    merged['actual_score'] = merged['actual_rank'].apply(get_absolute_rank_score)
    merged['score_diff'] = abs(merged['rank_score'] - merged['actual_score'])

    # Division match
    merged['pred_div'] = merged['rank'].str[0]
    merged['actual_div'] = merged['actual_rank'].str[0]
    merged['div_match'] = merged['pred_div'] == merged['actual_div']

    return {
        'mae_rank_score': merged['score_diff'].mean(),
        'division_accuracy': merged['div_match'].mean(),
        'within_2_ranks': (merged['score_diff'] <= 2).mean(),
        'within_5_ranks': (merged['score_diff'] <= 5).mean(),
        'n_wrestlers': len(merged)
    }

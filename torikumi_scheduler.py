# torikumi_scheduler.py
"""
JSA-style torikumi (match schedule) generation module.

Implements realistic scheduling rules across 5 day segments:
- Days 1-3: Rank-based, establishing initial matchups
- Days 4-7: Rank-influenced with early record adjustments
- Days 8-10: Record becomes primary, rank secondary
- Days 11-12: Critical kachi-koshi/make-koshi battles
- Days 13-15: Yusho race + sanyaku finalization + musubi

Key JSA rules implemented:
- Same heya wrestlers never face each other
- No rematches within a tournament
- All sanyaku must face each other at least once
- Day 15 musubi no ichiban features top yokozuna
- Winners face winners, losers face losers (days 8+)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional
from enum import Enum
import pandas as pd


class SchedulingSegment(Enum):
    """Tournament day segments with different scheduling priorities."""
    EARLY = "days_1_3"       # Days 1-3: Pure rank-based
    MID_EARLY = "days_4_7"   # Days 4-7: Rank primary, record emerging
    MID_LATE = "days_8_10"   # Days 8-10: Record primary, rank secondary
    CRITICAL = "days_11_12"  # Days 11-12: Kachi-koshi battles
    FINAL = "days_13_15"     # Days 13-15: Yusho race, sanyaku completion


@dataclass
class SchedulingConfig:
    """Configuration weights for each scheduling segment."""
    rank_weight: float        # How much rank matters (0-1)
    record_weight: float      # How much current record matters (0-1)
    sanyaku_priority: float   # Bonus for sanyaku-vs-sanyaku matchups
    yusho_race_priority: float # Bonus for matching leaders
    critical_record_bonus: float  # Bonus for 7-X vs X-7 matchups
    rank_proximity_bonus: float   # Bonus for similar ranks


# Tuned configurations for each segment
SEGMENT_CONFIGS = {
    SchedulingSegment.EARLY: SchedulingConfig(
        rank_weight=0.95,
        record_weight=0.05,
        sanyaku_priority=3.0,
        yusho_race_priority=0.0,
        critical_record_bonus=0.0,
        rank_proximity_bonus=2.0
    ),
    SchedulingSegment.MID_EARLY: SchedulingConfig(
        rank_weight=0.65,
        record_weight=0.35,
        sanyaku_priority=5.0,
        yusho_race_priority=1.0,
        critical_record_bonus=0.0,
        rank_proximity_bonus=1.5
    ),
    SchedulingSegment.MID_LATE: SchedulingConfig(
        rank_weight=0.25,
        record_weight=0.75,
        sanyaku_priority=3.0,
        yusho_race_priority=3.0,
        critical_record_bonus=2.0,
        rank_proximity_bonus=0.5
    ),
    SchedulingSegment.CRITICAL: SchedulingConfig(
        rank_weight=0.15,
        record_weight=0.85,
        sanyaku_priority=2.0,
        yusho_race_priority=4.0,
        critical_record_bonus=8.0,  # High priority for kachi-koshi battles
        rank_proximity_bonus=0.3
    ),
    SchedulingSegment.FINAL: SchedulingConfig(
        rank_weight=0.10,
        record_weight=0.90,
        sanyaku_priority=10.0,  # Must complete sanyaku matchups
        yusho_race_priority=12.0,  # Yusho contenders face each other
        critical_record_bonus=5.0,
        rank_proximity_bonus=0.2
    ),
}


@dataclass
class WrestlerState:
    """Tracks wrestler state during tournament simulation."""
    wrestler_id: int
    rikishi: str
    rank: str
    rank_score: float
    division: str  # 'makuuchi' or 'juryo'
    heya: str
    is_sanyaku: bool
    wins: int = 0
    losses: int = 0
    faced: Set[int] = field(default_factory=set)
    sanyaku_faced: Set[int] = field(default_factory=set)

    @property
    def record_diff(self) -> int:
        """Current wins minus losses."""
        return self.wins - self.losses

    @property
    def total_bouts(self) -> int:
        """Total bouts fought so far."""
        return self.wins + self.losses

    @property
    def is_kachi_koshi_line(self) -> bool:
        """True if wrestler is at 7-X or X-7 (critical for kk/mk)."""
        return self.wins == 7 or self.losses == 7

    @property
    def needs_one_more_win(self) -> bool:
        """True if wrestler needs 1 win for kachi-koshi."""
        return self.wins == 7 and self.losses < 8

    @property
    def needs_one_more_loss(self) -> bool:
        """True if wrestler faces make-koshi with 1 more loss."""
        return self.losses == 7 and self.wins < 8

    @property
    def is_yusho_contender(self) -> bool:
        """True if wrestler has 2 or fewer losses after day 10."""
        return self.losses <= 2

    def has_secured_kachi_koshi(self) -> bool:
        """True if wrestler already has 8+ wins."""
        return self.wins >= 8

    def has_make_koshi(self) -> bool:
        """True if wrestler already has 8+ losses."""
        return self.losses >= 8


def get_segment(day: int) -> SchedulingSegment:
    """Determine which scheduling segment a day belongs to."""
    if day <= 3:
        return SchedulingSegment.EARLY
    elif day <= 7:
        return SchedulingSegment.MID_EARLY
    elif day <= 10:
        return SchedulingSegment.MID_LATE
    elif day <= 12:
        return SchedulingSegment.CRITICAL
    else:
        return SchedulingSegment.FINAL


def calculate_matchup_score(
    a: WrestlerState,
    b: WrestlerState,
    day: int,
    config: SchedulingConfig,
    sanyaku_ids: Set[int],
    leader_wins: int,
    unfaced_sanyaku_pairs: Set[Tuple[int, int]]
) -> float:
    """
    Calculate desirability score for matching wrestlers A and B.
    Higher score = more desirable matchup.

    Args:
        a, b: The two wrestlers being considered
        day: Current tournament day (1-15)
        config: Scheduling configuration for current segment
        sanyaku_ids: Set of all sanyaku wrestler IDs
        leader_wins: Current win count of tournament leader(s)
        unfaced_sanyaku_pairs: Set of sanyaku pairs that haven't faced yet

    Returns:
        Float score (higher = better matchup)
    """
    score = 0.0

    # 1. Rank proximity component
    rank_diff = abs(a.rank_score - b.rank_score)
    # Scale: 0-5 rank diff is great, 5-15 is okay, 15+ is poor
    rank_score = max(0, 15 - rank_diff) * config.rank_proximity_bonus * config.rank_weight
    score += rank_score

    # 2. Record proximity component (stronger in later segments)
    record_diff = abs(a.record_diff - b.record_diff)
    # Scale: matching records is ideal, each point of difference is penalized
    record_score = max(0, 8 - record_diff * 2) * config.record_weight
    score += record_score

    # 3. Sanyaku vs sanyaku bonus
    if a.is_sanyaku and b.is_sanyaku:
        pair = (min(a.wrestler_id, b.wrestler_id), max(a.wrestler_id, b.wrestler_id))
        if pair in unfaced_sanyaku_pairs:
            # Extra bonus if they haven't faced each other yet
            score += config.sanyaku_priority * 2
        else:
            score += config.sanyaku_priority * 0.5

    # 4. Yusho race bonus (leaders should face each other in late basho)
    if day >= 10 and config.yusho_race_priority > 0:
        a_is_contender = a.wins >= leader_wins - 1 and a.losses <= 3
        b_is_contender = b.wins >= leader_wins - 1 and b.losses <= 3
        if a_is_contender and b_is_contender:
            score += config.yusho_race_priority

    # 5. Critical record bonus (kachi-koshi/make-koshi battles)
    if config.critical_record_bonus > 0:
        # 7-X vs X-7 matchups are highly valued (dramatic kachi-koshi battles)
        if (a.needs_one_more_win and b.needs_one_more_loss) or \
           (a.needs_one_more_loss and b.needs_one_more_win):
            score += config.critical_record_bonus
        # Similar records near the kachi-koshi line
        elif a.is_kachi_koshi_line and b.is_kachi_koshi_line:
            score += config.critical_record_bonus * 0.7
        elif a.is_kachi_koshi_line or b.is_kachi_koshi_line:
            if abs(a.record_diff - b.record_diff) <= 2:
                score += config.critical_record_bonus * 0.4

    # 6. Day 15 special: musubi no ichiban (final bout)
    if day == 15:
        # Top yokozuna should be in high-profile final bout
        a_is_top_yoko = a.rank.startswith('Y1') or a.rank.startswith('Y2')
        b_is_top_yoko = b.rank.startswith('Y1') or b.rank.startswith('Y2')
        if (a_is_top_yoko or b_is_top_yoko) and a.is_sanyaku and b.is_sanyaku:
            score += 8.0

    # 7. Avoid pairing wrestlers who have already secured/lost kachi-koshi together
    # (unless necessary) - creates more dramatic matchups
    both_decided = (a.has_secured_kachi_koshi() or a.has_make_koshi()) and \
                   (b.has_secured_kachi_koshi() or b.has_make_koshi())
    one_undecided = (a.is_kachi_koshi_line or b.is_kachi_koshi_line)
    if both_decided and not one_undecided and day >= 11:
        score -= 2.0  # Slight penalty for "meaningless" late matchups

    return score


def get_unfaced_sanyaku_pairs(wrestlers: List[WrestlerState]) -> Set[Tuple[int, int]]:
    """Get all sanyaku pairs that haven't faced each other yet."""
    sanyaku = [w for w in wrestlers if w.is_sanyaku and w.division == 'makuuchi']
    unfaced = set()

    for i, a in enumerate(sanyaku):
        for b in sanyaku[i+1:]:
            if b.wrestler_id not in a.faced:
                pair = (min(a.wrestler_id, b.wrestler_id), max(a.wrestler_id, b.wrestler_id))
                unfaced.add(pair)

    return unfaced


def enforce_sanyaku_requirements(
    wrestlers: List[WrestlerState],
    day: int
) -> List[Tuple[int, int]]:
    """
    Generate mandatory sanyaku matchups for the final days.

    JSA rules: All sanyaku must face each other at least once.
    This function identifies sanyaku pairs that haven't met
    and prioritizes them for days 13-15.

    Returns:
        List of (wrestler_id_a, wrestler_id_b) mandatory matchups
    """
    mandatory = []
    sanyaku = [w for w in wrestlers if w.is_sanyaku and w.division == 'makuuchi']

    # Calculate days remaining
    days_remaining = 15 - day + 1

    # Find unfaced pairs
    unfaced_pairs = []
    for i, a in enumerate(sanyaku):
        for b in sanyaku[i+1:]:
            if b.wrestler_id not in a.faced and a.heya != b.heya:
                unfaced_pairs.append((a, b))

    # On day 13+, we must schedule unfaced sanyaku pairs
    # Priority: pairs with fewer remaining chances
    if day >= 13 and unfaced_pairs:
        # Sort by urgency (fewer days remaining = more urgent)
        # All pairs are equally urgent now, so just take them
        for a, b in unfaced_pairs[:days_remaining]:
            mandatory.append((a.wrestler_id, b.wrestler_id))

    return mandatory


def generate_torikumi(
    wrestlers: List[WrestlerState],
    day: int,
) -> List[Tuple[int, int]]:
    """
    Generate matchups for a single day using JSA-style scheduling.

    Algorithm:
    1. Determine scheduling segment and load config
    2. Handle any mandatory matchups (sanyaku requirements)
    3. For each division:
       a. Build candidate matchup pool (excluding invalid pairs)
       b. Score each potential matchup
       c. Greedily select highest-scored valid matchups

    Args:
        wrestlers: List of WrestlerState objects
        day: Tournament day (1-15)

    Returns:
        List of (wrestler_id_a, wrestler_id_b) tuples
    """
    segment = get_segment(day)
    config = SEGMENT_CONFIGS[segment]

    matchups = []
    matched = set()

    # Calculate leader wins for yusho race tracking
    leader_wins = max((w.wins for w in wrestlers), default=0)

    # Get all sanyaku IDs
    sanyaku_ids = {w.wrestler_id for w in wrestlers if w.is_sanyaku}

    # Get unfaced sanyaku pairs
    unfaced_sanyaku_pairs = get_unfaced_sanyaku_pairs(wrestlers)

    # Handle mandatory sanyaku matchups first (days 13-15)
    if day >= 13:
        mandatory = enforce_sanyaku_requirements(wrestlers, day)
        for a_id, b_id in mandatory:
            a = next((w for w in wrestlers if w.wrestler_id == a_id), None)
            b = next((w for w in wrestlers if w.wrestler_id == b_id), None)
            if a and b and a_id not in matched and b_id not in matched:
                matchups.append((a_id, b_id))
                matched.add(a_id)
                matched.add(b_id)

    # Process each division
    for division in ['makuuchi', 'juryo']:
        div_wrestlers = [w for w in wrestlers
                        if w.division == division and w.wrestler_id not in matched]

        if len(div_wrestlers) < 2:
            continue

        # Build all candidate matchups with scores
        candidates = []
        for i, a in enumerate(div_wrestlers):
            for b in div_wrestlers[i+1:]:
                # Skip invalid matchups
                if a.heya == b.heya:  # Same heya
                    continue
                if b.wrestler_id in a.faced:  # Already faced
                    continue

                score = calculate_matchup_score(
                    a, b, day, config, sanyaku_ids, leader_wins, unfaced_sanyaku_pairs
                )
                candidates.append((score, a.wrestler_id, b.wrestler_id))

        # Sort by score descending
        candidates.sort(reverse=True, key=lambda x: x[0])

        # Greedily select matchups
        div_matched = set()
        for score, a_id, b_id in candidates:
            if a_id not in div_matched and b_id not in div_matched:
                matchups.append((a_id, b_id))
                div_matched.add(a_id)
                div_matched.add(b_id)

    return matchups


def generate_full_tournament_schedule(
    wrestlers: List[WrestlerState],
    days: int = 15
) -> Dict[int, List[Tuple[int, int]]]:
    """
    Generate matchups for all days of a tournament.

    Note: This is for preview/testing purposes. In actual simulation,
    matchups should be generated day-by-day as results come in.

    Args:
        wrestlers: List of WrestlerState objects (will be modified)
        days: Number of tournament days (default 15)

    Returns:
        Dict mapping day number to list of matchups
    """
    schedule = {}

    for day in range(1, days + 1):
        matchups = generate_torikumi(wrestlers, day)
        schedule[day] = matchups

        # Update faced sets for next day's scheduling
        for a_id, b_id in matchups:
            a = next((w for w in wrestlers if w.wrestler_id == a_id), None)
            b = next((w for w in wrestlers if w.wrestler_id == b_id), None)
            if a and b:
                a.faced.add(b_id)
                b.faced.add(a_id)
                if a.is_sanyaku and b.is_sanyaku:
                    a.sanyaku_faced.add(b_id)
                    b.sanyaku_faced.add(a_id)

    return schedule


def create_wrestler_states_from_df(wrestlers_df: pd.DataFrame) -> List[WrestlerState]:
    """
    Convert a wrestlers DataFrame to a list of WrestlerState objects.

    Args:
        wrestlers_df: DataFrame with columns: wrestler_id, rikishi, rank,
                      rank_score, division, heya, is_sanyaku

    Returns:
        List of WrestlerState objects
    """
    states = []
    for _, row in wrestlers_df.iterrows():
        state = WrestlerState(
            wrestler_id=int(row['wrestler_id']),
            rikishi=row['rikishi'],
            rank=row['rank'],
            rank_score=float(row['rank_score']),
            division=row['division'],
            heya=row.get('heya', ''),
            is_sanyaku=bool(row.get('is_sanyaku', 0))
        )
        states.append(state)
    return states


def validate_schedule(
    schedule: Dict[int, List[Tuple[int, int]]],
    wrestlers: List[WrestlerState]
) -> Dict[str, any]:
    """
    Validate a generated schedule against JSA rules.

    Returns dict with validation results and statistics.
    """
    w_dict = {w.wrestler_id: w for w in wrestlers}

    issues = []
    stats = {
        'total_bouts': 0,
        'same_heya_violations': 0,
        'rematch_violations': 0,
        'sanyaku_completion': 0.0,
    }

    faced_all = {w.wrestler_id: set() for w in wrestlers}

    for day, matchups in schedule.items():
        stats['total_bouts'] += len(matchups)

        for a_id, b_id in matchups:
            a = w_dict.get(a_id)
            b = w_dict.get(b_id)

            if not a or not b:
                issues.append(f"Day {day}: Unknown wrestler in matchup ({a_id}, {b_id})")
                continue

            # Check same heya
            if a.heya == b.heya and a.heya:
                stats['same_heya_violations'] += 1
                issues.append(f"Day {day}: Same heya violation - {a.rikishi} vs {b.rikishi}")

            # Check rematch
            if b_id in faced_all[a_id]:
                stats['rematch_violations'] += 1
                issues.append(f"Day {day}: Rematch - {a.rikishi} vs {b.rikishi}")

            faced_all[a_id].add(b_id)
            faced_all[b_id].add(a_id)

    # Check sanyaku completion
    sanyaku = [w for w in wrestlers if w.is_sanyaku and w.division == 'makuuchi']
    total_sanyaku_pairs = len(sanyaku) * (len(sanyaku) - 1) // 2
    faced_sanyaku_pairs = 0

    for i, a in enumerate(sanyaku):
        for b in sanyaku[i+1:]:
            if b.wrestler_id in faced_all[a.wrestler_id]:
                faced_sanyaku_pairs += 1

    stats['sanyaku_completion'] = faced_sanyaku_pairs / total_sanyaku_pairs if total_sanyaku_pairs > 0 else 1.0

    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'stats': stats
    }

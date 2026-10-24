"""Attack and defence strength for national teams, from the goals they score and concede.

Purpose:
    Elo only sees win, draw or loss, so two teams that both win 1-0 look
    identical even if one scores freely and the other defends well. This module
    keeps two numbers per team, both as they stood just before each match:

        attack    how many goals the team scores
        defence   how many goals the team concedes (lower is better)

    These go into the win/draw/loss classifiers as input features.

Inputs:
    results: DataFrame from load_data(), after prepare(), with columns
        date, home_team, away_team, home_score, away_score, neutral.
        Must be sorted by date. Rows with a missing score are skipped.

Outputs:
    1. A per-match DataFrame with the same index as ``results``:
           goals_home_attack_pre, goals_home_defence_pre
           goals_away_attack_pre, goals_away_defence_pre
    2. A GoalsState holding each team's decayed totals as of the last match,
       for predicting future fixtures (see rating_as_of).

Method (version 1):
    Each team keeps time-decayed sums of goals for (GF), goals against (GA)
    and match weight (W). Every past match is weighted by
        0.5 ** (days_since_match / HALF_LIFE_DAYS)
    so recent matches count more. Decay runs on calendar days, not match
    counts, because national teams play very irregularly.

    Teams with few matches are pulled toward the global average:
        attack  = (GF + PRIOR_STRENGTH * GLOBAL_GF) / (W + PRIOR_STRENGTH)
        defence = (GA + PRIOR_STRENGTH * GLOBAL_GF) / (W + PRIOR_STRENGTH)
    where GLOBAL_GF is the average goals per team per match (the global rate
    changes over time, so it is itself computed from past matches only).

Method (version 2, behind OPPONENT_ADJUSTED):
    Scoring 3 against a weak defence says less than scoring 3 against a strong
    one. With the flag on, goals scored are divided by the opponent's
    pre-match defence strength, and goals conceded by the opponent's attack.

No look-ahead:
    Features for a match use only matches strictly before it, and the global
    average is computed from past matches only.

Differs from shootouts.py:
    This module uses every match (about 49,000 rows) and feeds the match
    classifiers. shootouts.py uses 682 shootouts and feeds only the simulation.

Tuning:
    HALF_LIFE_DAYS and PRIOR_STRENGTH are starting values. Tune them by
    minimising log loss on held-out matches.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

HALF_LIFE_DAYS = 1095        # about 3 years; matches older than this count half
PRIOR_STRENGTH = 5.0         # pseudo-matches of average form added to every team
OPPONENT_ADJUSTED = False    # version 1 first; switch on once the baseline works


@dataclass
class GoalsState:
    """Decayed goal totals per team after the last processed match.

    Attributes:
        as_of: Date of the last processed match.
        totals: Maps team -> (decayed goals for, decayed goals against, decayed weight).
    """

    as_of: pd.Timestamp
    totals: dict[str, tuple[float, float, float]] = field(default_factory=dict)


def compute_goal_ratings(results: pd.DataFrame) -> tuple[pd.DataFrame, GoalsState]:
    """Replay all matches in date order and record attack/defence before each one.

    Args:
        results: See the module docstring. Must be sorted by date.

    Returns:
        A tuple (match_ratings, state).
        match_ratings has the four goals_*_pre columns and the same index as
        ``results``; skipped rows hold NaN.
        state is the GoalsState after the last match.
    """
    raise NotImplementedError


def rating_as_of(state: GoalsState, team: str, date: pd.Timestamp) -> tuple[float, float]:
    """Attack and defence for a team at a future date.

    Totals keep decaying while a team does not play, so a rating read months
    after the last match is shrunk toward the average.

    Args:
        state: The GoalsState returned by compute_goal_ratings.
        team: Team name, standardised as in load_data().
        date: The date of the fixture being predicted.

    Returns:
        A tuple (attack, defence). An unseen team returns the global average.
    """
    raise NotImplementedError
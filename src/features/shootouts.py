"""Penalty-shootout skill for national teams, with heavy shrinkage toward 50%.

Purpose:
    Estimate how likely each team is to win a penalty shootout, and from that
    the probability that team A beats team B in one. Shootouts are not part of
    the match result in results.csv (the score is after extra time, without
    penalties), so this model only matters when a knockout game ends level:

        P(A advances) = P(A wins) + P(draw) * P(A wins shootout)

    Decision to revisit: the output feeds the tournament simulation only. It is
    not an input to the win/draw/loss classifiers. Exposing it as a classifier
    feature is a small change in build_features.py if wanted.

Inputs:
    shootouts: DataFrame from load_data() with columns
        date, home_team, away_team, winner, first_shooter.
        first_shooter is mostly missing, so it is ignored in version 1.

Outputs:
    1. A per-shootout DataFrame with the same index as ``shootouts``:
           so_home_rating_pre, so_away_rating_pre   (shrunken win rates)
           so_home_win_prob                         (prediction made before the shootout)
    2. A ShootoutState holding each team's record after the last shootout.

Method:
    The data is tiny: 682 shootouts in total, so most teams have a handful or
    none, and a raw record such as 2 wins from 2 means almost nothing. Each team's
    rating is a shrunken win rate:
        rating = (wins + PRIOR_STRENGTH * 0.5) / (shootouts + PRIOR_STRENGTH)
    which sits at 0.5 for an unseen team and moves only as evidence builds up.

    Head-to-head uses the log5 formula for two win rates a and b:
        P(A beats B) = (a - a*b) / (a + b - 2*a*b)

No look-ahead:
    Each shootout is predicted from shootouts strictly before its date.

Honest check:
    Research suggests shootouts are close to coin flips. evaluate_vs_coin_flip()
    compares this model with a constant 0.5 on log loss, walk-forward, so the
    project can report whether the feature carries any signal at all.

Differs from goals.py:
    This module uses 682 shootouts and feeds only the simulation. goals.py uses
    every match and feeds the match classifiers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

PRIOR_STRENGTH = 10.0   # pseudo-shootouts at 50%; larger means more caution


@dataclass
class ShootoutState:
    """Win/loss record per team after the last processed shootout.

    Attributes:
        as_of: Date of the last processed shootout.
        records: Maps team -> (wins, shootouts played).
    """

    as_of: pd.Timestamp
    records: dict[str, tuple[int, int]] = field(default_factory=dict)


def shootout_rating(wins: int, played: int) -> float:
    """Shrunken shootout win rate.

    Args:
        wins: Shootouts the team has won.
        played: Shootouts the team has taken part in.

    Returns:
        A rating between 0 and 1; exactly 0.5 when played is 0.
    """
    raise NotImplementedError


def p_win_shootout(rating_a: float, rating_b: float) -> float:
    """Probability that team A beats team B in a shootout (log5).

    Args:
        rating_a: Shrunken win rate of team A.
        rating_b: Shrunken win rate of team B.

    Returns:
        A probability between 0 and 1; 0.5 when the ratings are equal.
    """
    raise NotImplementedError


def compute_shootout_ratings(shootouts: pd.DataFrame) -> tuple[pd.DataFrame, ShootoutState]:
    """Replay all shootouts in date order and predict each one before it happens.

    Args:
        shootouts: See the module docstring.

    Returns:
        A tuple (per_shootout, state).
        per_shootout has so_home_rating_pre, so_away_rating_pre and
        so_home_win_prob, with the same index as ``shootouts``.
        state is the ShootoutState after the last shootout.
    """
    raise NotImplementedError


def evaluate_vs_coin_flip(per_shootout: pd.DataFrame, shootouts: pd.DataFrame) -> dict[str, float]:
    """Compare the model's walk-forward log loss with a constant 0.5.

    Args:
        per_shootout: The first return value of compute_shootout_ratings.
        shootouts: The same shootouts DataFrame, for the actual winners.

    Returns:
        A dict with keys model_log_loss, coin_flip_log_loss and n.
    """
    raise NotImplementedError
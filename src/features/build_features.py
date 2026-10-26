"""
Assemble the per-match feature table that the models train on.

The three feature modules (elo.py, goals.py, shootouts.py) never
import each other and know nothing about the models.

This module is the single place that runs them over the same sorted matches,
joins the results into one table (one row per match), adds the match-level columns,
checks that nothing leaks from the future, and saves the table so during training
we never have to recompute ratings.

Inputs:
    results, shootouts: DataFrames from load_data(), after prepare().

Outputs:
    data/processed/features.parquet (needs pyarrow; use CSV if not installed)
    with one row per played match. Columns:

        Identifiers (never fed to the models):
            match_id, date, home_team, away_team, tournament

        From elo.py:
            elo_home_pre, elo_away_pre, elo_diff

        From goals.py:
            goals_home_attack_pre, goals_home_defence_pre,
            goals_away_attack_pre, goals_away_defence_pre
            plus differences, e.g. attack_diff = home attack - away attack

        Match context:
            neutral         True when played at a neutral venue
            home_is_host    True when the home side is the host nation
            t_group         coarse tournament group (encoded inside the models)

        Target:
            outcome         "home_win", "draw" or "away_win" (full time, no shootout)

Design rules:
    * match_id is the row position after sorting by date (ties keep file order),
      so every feature module lines up with the same rows.
    * Every feature is computed from matches strictly before the match.
    * Differences are home minus away. At neutral venues the home/away label is
      only partly arbitrary (the home label still wins more often in the data),
      so neutral stays a feature.
    * Ratings are computed from the whole history. Restricting the training
      rows to recent decades is a choice made in train.py, because ratings need
      a long burn-in.

Why a separate module:
    * One place to enforce the no-look-ahead rule, with a test.
    * One saved table, so training, evaluation and notebooks all read the same
      thing.
    * It also builds features for fixtures that have not been played, which the
      tournament simulation needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class FeatureState:
    """Everything needed to build features for an unplayed fixture.

    Attributes:
        final_elo: Maps team -> rating after the last match (from elo.py).
        goals_state: GoalsState after the last match (from goals.py).
        shootout_state: ShootoutState after the last shootout (from shootouts.py).
    """

    final_elo: dict
    goals_state: object
    shootout_state: object


def build_features(results: pd.DataFrame, shootouts: pd.DataFrame) -> tuple[pd.DataFrame, FeatureState]:
    """Run all three feature modules and join them into one table.

    Args:
        results: Played and unplayed matches, sorted by date.
        shootouts: Shootouts DataFrame, used for the shootout state.

    Returns:
        A tuple (features, state).
        features is the table described in the module docstring, played
        matches only.
        state is the FeatureState after the last match, for build_fixture_features.
    """
    raise NotImplementedError


def build_fixture_features(state: FeatureState, home: str, away: str, date: pd.Timestamp,
                           neutral: bool = True, home_is_host: bool = False) -> pd.DataFrame:
    """Features for a match that has not been played (used by the simulation).

    Args:
        state: The FeatureState returned by build_features.
        home: Home team name, standardised as in load_data().
        away: Away team name.
        date: Fixture date; goal strengths decay up to this date.
        neutral: Whether the venue is neutral.
        home_is_host: Whether the home side is a host nation.

    Returns:
        A one-row DataFrame with the same feature columns as the saved table.
    """
    raise NotImplementedError


def check_no_lookahead(results: pd.DataFrame, shootouts: pd.DataFrame, cutoff: pd.Timestamp) -> None:
    """Test that no feature uses information from the future.

    Builds features on the full data and again on data truncated at ``cutoff``,
    then asserts every match before the cutoff has identical features in both.
    Also asserts that each team's first match carries the default starting
    values.

    Args:
        results: Full results DataFrame.
        shootouts: Full shootouts DataFrame.
        cutoff: Date at which to truncate.

    Raises:
        AssertionError: If any feature differs between the two builds.
    """
    raise NotImplementedError


def save_features(features: pd.DataFrame, path: Path) -> None:
    """Write the feature table to disk (data/processed/features.parquet).

    Args:
        features: The table returned by build_features.
        path: Destination file.
    """
    raise NotImplementedError


def load_features(path: Path) -> pd.DataFrame:
    """Read a feature table written by save_features.

    Args:
        path: File to read.

    Returns:
        The feature DataFrame.
    """
    raise NotImplementedError
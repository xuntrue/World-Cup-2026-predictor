from __future__ import annotations

import numpy as np
import math
import matplotlib.pyplot as plt
import os
import pandas as pd

INITIAL_RATING = 1000.0
HOME_ADVANTAGE = 100.0   # rating points added to the home side at non-neutral venues
K_BY_GROUP = {           # larger K = ratings react faster to that kind of match
    "World Cup": 60,
    "Continental": 50,
    "Qualifier": 40,
    "Other": 30,
    "Friendly": 20,
}

# Substrings that mark a continental championship in the tournament column.
_CONTINENTAL_KEYWORDS = (
    "UEFA Euro", "Copa Am", "African Cup of Nations", "AFC Asian Cup",
    "Gold Cup", "Confederations Cup", "Oceania Nations Cup",
)

def _fallback_group(tournament: str) -> str:
    """
    Rough tournament grouping, used only if no group_fn is supplied
    Order matters: "FIFA World Cup qualification" contains "World Cup", so qualifiers must be tested before the World Cup itself
    """
    if tournament == "Friendly":
        return "Friendly"
    if "qualification" in tournament:
        return "Qualifier"
    if tournament == "FIFA World Cup":
        return "World Cup"
    if any(key in tournament for key in _CONTINENTAL_KEYWORDS):
        return "Continental"
    return "Other"


def expected_score(rating_diff: float) -> float:
    """ Probability-like expected score for the side with the higher rating """
    return 1.0 / (1.0 + 10.0 ** (-rating_diff / 400.0))

def goal_diff_multiplier(goal_diff: int) -> float:
    """ Scale an update by the margin of victory """
    goal_diff = abs(int(goal_diff))
    if goal_diff <= 1:
        return 1.0
    if goal_diff == 2:
        return 1.5
    return (11.0 + goal_diff) / 8.0   # 3 -> 1.75, 5 -> 2.0, 9 -> 2.5


def k_factor(tournament: str, group_fn: Callable[[str], str] | None = None) -> float:
    group = (group_fn or _fallback_group)(tournament)
    return float(K_BY_GROUP.get(group, K_BY_GROUP["Other"]))

def compute_elo(results: pd.DataFrame, group_fn: Callable[[str], str] | None = None) -> tuple[pd.DataFrame, dict[str, float]]:
    """ Replay all matches in date order and record ratings before each one """
    if not results["date"].is_monotonic_increasing:
        raise ValueError("results must be sorted by date before computing Elo.")

    n = len(results)
    # Work on plain numpy arrays: much faster than .iloc in a 49k-row loop.
    dates = results["date"].to_numpy()
    homes = results["home_team"].to_numpy()
    aways = results["away_team"].to_numpy()
    home_goals = results["home_score"].to_numpy(dtype=float)
    away_goals = results["away_score"].to_numpy(dtype=float)
    tournaments = results["tournament"].to_numpy()
    neutral = results["neutral"].to_numpy(dtype=bool)

    elo_home_pre = np.full(n, np.nan)
    elo_away_pre = np.full(n, np.nan)

    ratings: dict[str, float] = {}
    k_cache: dict[str, float] = {}   # tournament name -> K, so we group each name once

    # Row ranges [start, end) that share the same date.
    cuts = np.flatnonzero(dates[1:] != dates[:-1]) + 1
    starts = np.concatenate(([0], cuts))
    ends = np.concatenate((cuts, [n]))

    for start, end in zip(starts, ends):
        pending: dict[str, float] = {}   # team -> total rating change for this date

        for i in range(start, end):
            if np.isnan(home_goals[i]) or np.isnan(away_goals[i]):
                continue   # no result: leave NaN and do not update

            home, away = homes[i], aways[i]
            r_home = ratings.get(home, INITIAL_RATING)   # start-of-date ratings
            r_away = ratings.get(away, INITIAL_RATING)

            # Record the pre-match values BEFORE any update from this match.
            elo_home_pre[i] = r_home
            elo_away_pre[i] = r_away

            h = 0.0 if neutral[i] else HOME_ADVANTAGE
            expected_home = expected_score(r_home + h - r_away)

            if home_goals[i] > away_goals[i]:
                actual_home = 1.0
            elif home_goals[i] == away_goals[i]:
                actual_home = 0.5
            else:
                actual_home = 0.0

            tournament = tournaments[i]
            if tournament not in k_cache:
                k_cache[tournament] = k_factor(tournament, group_fn)

            g = goal_diff_multiplier(home_goals[i] - away_goals[i])
            delta = k_cache[tournament] * g * (actual_home - expected_home)

            pending[home] = pending.get(home, 0.0) + delta
            pending[away] = pending.get(away, 0.0) - delta

        # Apply the whole date's updates at once.
        for team, delta in pending.items():
            ratings[team] = ratings.get(team, INITIAL_RATING) + delta

    match_ratings = pd.DataFrame(
        {
            "elo_home_pre": elo_home_pre,
            "elo_away_pre": elo_away_pre,
            "elo_diff": elo_home_pre - elo_away_pre,
        },
        index=results.index,
    )
    return match_ratings, ratings


if __name__ == "__main__":
    # Tiny self-test with made-up matches (no data files needed)

    assert abs(expected_score(0) - 0.5) < 1e-12
    assert expected_score(400) > 0.9 > 0.1 > expected_score(-400)
    assert goal_diff_multiplier(0) == goal_diff_multiplier(1) == 1.0
    assert goal_diff_multiplier(5) == 2.0

    toy = pd.DataFrame(
        {
            "date": pd.to_datetime(["2000-01-01", "2000-01-02", "2000-01-02"]),
            "home_team": ["A", "A", "B"],
            "away_team": ["B", "C", "C"],
            "home_score": [2, 1, 0],
            "away_score": [0, 1, 0],
            "tournament": ["Friendly", "Friendly", "FIFA World Cup"],
            "neutral": [False, True, True],
        }
    )
    pre, final = compute_elo(toy)
    print(pre)
    print(final)

    # Every team's first match sees the starting rating.
    assert pre.loc[0, "elo_home_pre"] == INITIAL_RATING

    # Updates are zero-sum, so the total rating is conserved.
    assert abs(sum(final.values()) - 3 * INITIAL_RATING) < 1e-9
    print("self-test passed")
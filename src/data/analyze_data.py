from pathlib import Path
from scipy import stats

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .load_data import load_data

# ===== CONFIG =====
ROOT = Path(__file__).resolve().parents[2]  # repo root
DATA_DIR = ROOT / "vendor" / "international_results"
FIG_DIR = Path("notebooks") / "figures"
VERBOSE = True # Set to True to see detailed loading/standarization output

def prepare(results, shootouts, goalscorers):
    """ Clean up the loaded data """
    for df in (results, shootouts, goalscorers):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    results["year"] = results["date"].dt.year
    results["decade"] = (results["year"] // 10) * 10
    results["total_goals"] = results["home_score"] + results["away_score"]
    results["goal_diff"] = results["home_score"] - results["away_score"]
    results["outcome"] = np.select(
        [results["goal_diff"] > 0, results["goal_diff"] < 0],
        ["home_win", "away_win"],
        default="draw",
    )
    results["t_group"] = results["tournament"].map(tournament_group)
    return results, shootouts, goalscorers

def missing_and_types(results, shootouts, goalscorers):
    section("MISSING VALUES (all columns) & DATE RANGE")
    for name, df in [("results", results), ("shootouts", shootouts), ("goalscorers", goalscorers)]:
        nulls = df.isna().sum()
        print(f"{name}.csv: {len(df):,} rows")
        nonzero = nulls[nulls > 0]
        if nonzero.empty:
            print("  no nulls")
        else:
            for col, n in nonzero.items():
                print(f"  null {col}: {n:,}")
        print(f"  date range: {df['date'].min().date()} -> {df['date'].max().date()}"
              f" | unparseable dates: {df['date'].isna().sum()}")
        print()

    unplayed = results[results[["home_score", "away_score"]].isna().any(axis=1)]
    print(f"\nResults rows with a missing score (unplayed/scheduled?): {len(unplayed)}")
    if len(unplayed):
        print(unplayed[["date", "home_team", "away_team", "tournament"]].tail(10))

    null_c = results[results["country"].isna()]
    print(f"\nnull 'country' rows: {len(null_c)}")
    if len(null_c):
        print("  by decade:\n", null_c["decade"].value_counts().sort_index().to_string())
        print("  top tournaments:\n", null_c["tournament"].value_counts().head(8).to_string())
        print("  top cities:\n", null_c["city"].value_counts().head(8).to_string())

def duplicates(results, shootouts):
    section("DUPLICATES")
    print("results exact duplicate rows:", results.duplicated().sum())

    result_key_dups = results.duplicated(subset=["date", "home_team", "away_team"], keep=False)
    print("\nresults same date+home+away (different details?):", result_key_dups.sum())
    if result_key_dups.any():
        print(results[result_key_dups].sort_values(["date", "home_team"]).to_string()) #.head(10)

    shootout_key_dups = shootouts.duplicated(subset=["date", "home_team", "away_team"], keep=False)
    print("\nshootouts duplicate date+teams:", shootout_key_dups.sum())
    if shootout_key_dups.any():
        print(shootouts[shootout_key_dups].sort_values(["date", "home_team"]).to_string()) #.head(10)

def former_name_window_check(results, former_names):
    """ Re-checked on the already-standardized frames, only reports the mapping table itself and flags names """
    section("FORMER-NAME MAPPING SANITY CHECK")
    fn = former_names.copy()
    fn["start_date"] = pd.to_datetime(fn["start_date"])
    fn["end_date"] = pd.to_datetime(fn["end_date"])
    ambiguous = fn[fn["former"].isin(fn["current"])]
    print(f"{len(fn)} mappings; {len(ambiguous)} where 'former' is also a 'current' name:")
    print(ambiguous.to_string(index=False) if len(ambiguous) else "  none")
    multi = fn.groupby("former")["current"].nunique()
    print("former names mapping to >1 current name:", list(multi[multi > 1].index))

def coverage_over_time(results: pd.DataFrame, bin_years: int = 10, plot: bool = True) -> None:
    section(f"MATCHES PER {bin_years}-YEAR")
    if bin_years < 1:
        raise ValueError("bin_years must be >= 1")

    years = results["year"].dropna().astype(int)
    bin_start = (years // bin_years) * bin_years
    counts = bin_start.groupby(bin_start).size()

    # Include empty bins as 0 so gaps (e.g. wartime) show up instead of vanishing
    all_bins = range(counts.index.min(), counts.index.max() + 1, bin_years)
    counts = counts.reindex(all_bins, fill_value=0)

    total = counts.sum()
    table = pd.DataFrame({"count": counts, "share (%)": counts / total * 100, "cumulative (%)": (counts / total * 100).cumsum()})
    table.index.name = "bin start"
    print(table.to_string(float_format="{:.2f}".format, formatters={"count": "{:,}".format}))
    print(f"{'total':>9}  {total:,}")

    if plot:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(counts.index, counts.values, width=bin_years * 0.9, align="edge")
        ax.set_title(f"Matches per {bin_years}-year bin")
        ax.set_xlabel("year")
        ax.set_axisbelow(True)

        if True: # Linear
            ax.set_ylabel("matches")
            ax.set_ylim(0, counts.max() * 1.05)
        else: # Logarithmic
            ax.set_yscale("log")
            ax.set_ylabel("matches (log scale)")
            ax.set_ylim(0.8, counts.max() * 1.5)

        ax.minorticks_on()
        ax.grid(which="major", linestyle="-", linewidth=0.8, alpha=0.6)
        ax.grid(which="minor", linestyle=":", linewidth=0.5, alpha=0.4)

        save_fig(f"matches_per_{bin_years}y")

def tournaments(results: pd.DataFrame, plot: bool = True) -> None:
    section("TOURNAMENTS")
    print(f"{results['tournament'].nunique()} distinct tournaments\n")

    top_n = 20
    print(f"--- Top {top_n} 'tournament' type(s) ---")
    print(counts_and_shares(results["tournament"]).head(top_n).to_string(float_format="{:.2f}".format))

    print("\n--- Tournament grouping ---")
    print(counts_and_shares(results["t_group"]).to_string(float_format="{:.2f}".format))

    ct = pd.crosstab(results["decade"], results["t_group"])
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100

    print("\n--- Tournament group by decade (counts) ---")
    counts_table = ct.copy()
    counts_table["Total"] = ct.sum(axis=1)
    counts_table.loc["All"] = counts_table.sum()          # bottom totals row
    print(counts_table.to_string(formatters={c: "{:,}".format for c in counts_table.columns}))

    print("\n--- Tournament group by decade (% of that decade's matches) --- ")
    pct_table = ct_pct.copy()
    pct_table.loc["All"] = ct.sum() / ct.values.sum() * 100   # overall mix
    print(pct_table.to_string(float_format="{:.2f}".format))

    if plot:
        ct = pd.crosstab(results["decade"], results["t_group"])

        fig, ax = plt.subplots(figsize=(11, 5))
        ax.set_axisbelow(True)

        if True:  # Raw counts
            ct.plot(kind="bar", stacked=True, ax=ax)
            ax.set_title("Match type mix by decade (counts)")
            ax.set_ylabel("matches")
            ax.set_ylim(0, ct.sum(axis=1).max() * 1.05)
            fig_name = "tournament_mix_by_decade_counts"
        else:     # Share of matches (%)
            ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
            ct_pct.plot(kind="bar", stacked=True, ax=ax)
            ax.set_title("Match type mix by decade (share of matches)")
            ax.set_ylabel("share of matches (%)")
            ax.set_ylim(0, 100)
            fig_name = "tournament_mix_by_decade_share"

        ax.set_xlabel("decade")
        ax.yaxis.minorticks_on()
        ax.grid(which="major", axis="y", linestyle="-", linewidth=0.8, alpha=0.6)
        ax.grid(which="minor", axis="y", linestyle=":", linewidth=0.5, alpha=0.4)
        ax.legend(title="tournament group", bbox_to_anchor=(1.01, 1), loc="upper left")

        save_fig(fig_name) # fig_name is dynamic

def score_distributions(results):
    section("SCORES & OUTCOMES")
    played = results.dropna(subset=["home_score", "away_score"])
    _print_summary_stats(played, plot=False, y_limits=None)
    _print_outcomes(played, plot=False)
    _print_outcomes_by_neutral_counts(played, plot=False)
    _print_outcomes_by_neutral_share(played)
    _print_neutral_effect_test(played, alpha=0.05)
    _print_goals_over_time(played, bin_size=10, plot=False)
    _print_top_scorelines(played, top_n=10)
    _print_biggest_margins(played, top_n=5)
    _print_poisson_check(played, plot=False, min_year=None, max_goals=8)

def _neutral_crosstab(played: pd.DataFrame) -> pd.DataFrame:
    """ Outcome counts split by the neutral flag (shared by two sections)."""
    return pd.crosstab(played["neutral"], played["outcome"])

def _print_summary_stats(played: pd.DataFrame, plot: bool = False, y_limits: tuple | None = None) -> None:
    cols = ["home_score", "away_score", "total_goals", "goal_diff"]
    plot_cols = ["home_score", "away_score", "total_goals"]
    print(played[cols].describe().round(2))

    if plot:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.set_axisbelow(True)   # gridlines BEHIND the boxes

        bp = ax.boxplot(
            [played[c] for c in plot_cols],
            showmeans=True,
            whis=(0, 100),
            meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black", markersize=6),
            flierprops=dict(marker="o", markersize=3, alpha=0.3),
        )
        ax.set_xticks(range(1, len(plot_cols) + 1), labels=plot_cols)

        ax.set_title("Score distributions")
        ax.set_ylabel("goals")
        if y_limits is not None: # Linear y-axis; pass y_limits=(low, high) to zoom in
            ax.set_ylim(*y_limits)
        ax.yaxis.minorticks_on()
        ax.grid(which="major", axis="y", linestyle="-", linewidth=0.8, alpha=0.6)
        ax.grid(which="minor", axis="y", linestyle=":", linewidth=0.5, alpha=0.4)
        ax.legend([bp["medians"][0], bp["means"][0]], ["median", "mean"], loc="upper right")

        save_fig("score_boxplots")

def _print_outcomes(played: pd.DataFrame, plot: bool = False) -> None:
    _subsection("Game Outcome (all)")
    table = counts_and_shares(played["outcome"])
    print(table.to_string(float_format="{:.2f}".format))

    if plot:
        order = [o for o in ("home_win", "draw", "away_win") if o in table.index]
        colors = {
            "home_win": "#4C72B0",
            "draw": "#8C8C8C",
            "away_win": "#DD8452"
        }
        data = table.loc[order]

        labels = [f"{name}\n{row['count']:,.0f}\n({row['share (%)']:.2f}%)" for name, row in data.iterrows()]

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.pie(
            data["count"],
            labels=labels,
            colors=[colors[o] for o in order],
            startangle=90,
            counterclock=False,
            wedgeprops=dict(edgecolor="white", linewidth=1.5),
            textprops=dict(fontsize=11),
        )
        ax.set_title(f"Game outcomes, all matches (n = {data['count'].sum():,.0f})")

        save_fig("outcomes_pie")

def _print_outcomes_by_neutral_counts(played: pd.DataFrame, plot: bool = True) -> None:
    _subsection("Game Outcomes by neutral flag (counts)")
    ct = _neutral_crosstab(played)
    counts_table = ct.copy()
    counts_table["Total"] = ct.sum(axis=1)
    print(counts_table.to_string(formatters={c: "{:,}".format for c in counts_table.columns}))

    if plot:
        order = [o for o in ("away_win", "draw", "home_win") if o in ct.columns]
        ct = ct[order]
        pct = ct.div(ct.sum(axis=1), axis=0) * 100   # share within each venue type

        # Venue colours, deliberately not the blue/orange used for home/away in the pie
        colors = {False: "#1848F0", True: "#C5CFC9"}
        names = {False: "Non-neutral venue", True: "Neutral venue"}

        x = np.arange(len(order))
        width = 0.38

        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.set_axisbelow(True)

        for i, flag in enumerate((False, True)):
            if flag not in ct.index:
                continue
            bars = ax.bar(
                x + (i - 0.5) * width,
                pct.loc[flag],
                width,
                color=colors[flag],
                edgecolor="white",
                label=f"{names[flag]} (n = {ct.loc[flag].sum():,})",
            )
            ax.bar_label(
                bars,
                labels=[f"{p:.2f}%\n({c:,})" for p, c in zip(pct.loc[flag], ct.loc[flag])],
                padding=3,
                fontsize=9,
            )

        ax.set_xticks(x, labels=[o.replace("_", " ") for o in order])
        ax.set_ylabel("share of matches at that venue type (%)")
        ax.set_title("Match outcomes by venue type")
        ax.set_ylim(0, pct.max().max() * 1.18)   # headroom for labels

        ax.yaxis.minorticks_on()
        ax.grid(which="major", axis="y", linestyle="-", linewidth=0.8, alpha=0.6)
        ax.grid(which="minor", axis="y", linestyle=":", linewidth=0.5, alpha=0.4)
        ax.legend(loc="upper left")

        save_fig("outcomes_by_neutral")

def _print_outcomes_by_neutral_share(played: pd.DataFrame) -> None:
    _subsection("Game Outcomes by neutral flag (% of that row)")
    ct = _neutral_crosstab(played)
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    print(ct_pct.to_string(float_format="{:.2f}".format))

def _print_neutral_effect_test(played: pd.DataFrame, alpha: float = 0.05) -> None:
    _subsection("Neutral flag vs outcome: chi-square test of independence")
    order = [o for o in ("away_win", "draw", "home_win") if o in played["outcome"].unique()]
    ct = _neutral_crosstab(played)[order]
    n = ct.values.sum()

    # --- (1) Overall test: does the outcome mix depend on venue type? ---
    chi2, p, dof, expected = stats.chi2_contingency(ct)
    expected = pd.DataFrame(expected, index=ct.index, columns=ct.columns)
    cramers_v = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))

    print("H0: the outcome mix is the same at neutral and non-neutral venues")
    verdict = "reject H0" if p < alpha else "fail to reject H0"
    print(f"chi2 = {chi2:.2f}, dof = {dof}, p = {p:.3g}  ->  {verdict} at alpha = {alpha}")
    print(f"Cramer's V = {cramers_v:.3f} ({_cramers_v_label(cramers_v)} effect)")
    print(f"Smallest expected count = {expected.values.min():.1f} (test assumes >= 5)")

    # --- (2) Which cells drive it? Adjusted standardized residuals ---
    row_p = ct.sum(axis=1) / n
    col_p = ct.sum(axis=0) / n
    adj_resid = (ct - expected) / np.sqrt(expected * np.outer(1 - row_p, 1 - col_p))
    print("\nAdjusted residuals (|value| > 1.96 = that cell differs from independence):")
    print(adj_resid.to_string(float_format="{:+.2f}".format))

    # --- (3) How big is each shift? Neutral minus non-neutral, with 95% CI ---
    n_non, n_neu = ct.loc[False].sum(), ct.loc[True].sum()
    bonferroni = alpha / len(order)   # 3 separate tests, so tighten the threshold
    rows = []
    for outcome in order:
        x1, x2 = ct.loc[False, outcome], ct.loc[True, outcome]
        p1, p2 = x1 / n_non, x2 / n_neu
        diff = p2 - p1

        se_ci = np.sqrt(p1 * (1 - p1) / n_non + p2 * (1 - p2) / n_neu)   # unpooled, for the CI
        p_pool = (x1 + x2) / (n_non + n_neu)
        se_test = np.sqrt(p_pool * (1 - p_pool) * (1 / n_non + 1 / n_neu))  # pooled, for the test
        p_val = 2 * stats.norm.sf(abs(diff / se_test))

        rows.append({
            "outcome": outcome,
            "non-neutral (%)": p1 * 100,
            "neutral (%)": p2 * 100,
            "diff (pp)": diff * 100,
            "95% CI low": (diff - 1.96 * se_ci) * 100,
            "95% CI high": (diff + 1.96 * se_ci) * 100,
            "p-value": p_val,
            "significant": p_val < bonferroni,
        })

    """
    Matches aren’t independent. The same teams play repeatedly, and strong teams win a lot wherever they play.
    Chi-square assumes independent observations, so the p-value is overconfident.
    Treat it as strong evidence of association, not an exact probability.
    """

    print(f"\nShare shift at neutral venues (Bonferroni threshold = {bonferroni:.4f}):")
    print(pd.DataFrame(rows).set_index("outcome").to_string(float_format=lambda v: f"{v:.2f}" if abs(v) >= 0.01 else f"{v:.2e}"))

_BIN_TITLES = {1: "by year", 10: "by decade", 100: "by century"}
def _print_goals_over_time(played: pd.DataFrame, bin_size: int = 10, plot: bool = False) -> None:
    if bin_size < 1:
        raise ValueError("bin_size must be >= 1")
    suffix = _BIN_TITLES.get(bin_size, f"by {bin_size}-year bin")

    _subsection(f"Goals per match {suffix}")
    bins = (played["year"].dropna() // bin_size * bin_size).astype(int) # Floor each match's year to the start of its bin
    desc = played.loc[bins.index, "total_goals"].groupby(bins).describe()
    desc["count"] = desc["count"].astype(int)   # describe() returns count as float, let's change to int
    print(desc.to_string(float_format="{:.2f}".format, formatters={"count": "{:,}".format}))

    if plot: # Reindex over every bin so empty bins become NaN gaps instead of the line joining straight across them (counts are only needed for the CI, so NaN is fine)
        desc = desc.reindex(range(desc.index.min(), desc.index.max() + 1, bin_size))
        x = desc.index
        mean = desc["mean"]
        ci = 1.96 * desc["std"] / np.sqrt(desc["count"])   # 95% CI half-width of the mean

        colors = {"mean": "#4C72B0", "median": "#DD8452"}

        fig, ax = plt.subplots(figsize=(11, 5))
        ax.set_axisbelow(True)

        ax.fill_between(x, mean - ci, mean + ci, color=colors["mean"], alpha=0.2, linewidth=0, label="95% CI of the mean")
        ax.plot(x, mean, marker="o", markersize=4 if len(x) > 40 else 6, color=colors["mean"], linewidth=2, label="mean")
        ax.plot(x, desc["50%"], marker="s", markersize=3 if len(x) > 40 else 4, linestyle="--", color=colors["median"], linewidth=1.2, label="median")

        ax.set_title(f"Goals per match {suffix}")
        ax.set_xlabel("year" if bin_size == 1 else f"start of {bin_size}-year bin")
        ax.set_ylabel("goals per match")
        ax.set_ylim(0, (mean + ci).max() * 1.08)

        # One labelled tick per bin, ONLY when that fits; otherwise let matplotlib choose
        if len(x) <= 25:
            ax.set_xticks(x, labels=[_bin_label(s, bin_size) for s in x], rotation=45)

        ax.yaxis.minorticks_on()
        ax.grid(which="major", axis="y", linestyle="-", linewidth=0.8, alpha=0.6)
        ax.grid(which="minor", axis="y", linestyle=":", linewidth=0.5, alpha=0.4)
        ax.grid(which="major", axis="x", linestyle="-", linewidth=0.5, alpha=0.3)
        ax.legend(loc="upper right")

        save_fig(f"goals_per_match_{bin_size}y")

def _print_top_scorelines(played: pd.DataFrame, top_n: int = 10) -> None:
    _subsection(f"Top {top_n} common scorelines")
    scorelines = (played["home_score"].astype(int).astype(str) + "-"
                  + played["away_score"].astype(int).astype(str))
    print(counts_and_shares(scorelines).head(top_n).to_string(float_format="{:.2f}".format))

def _print_biggest_margins(played: pd.DataFrame, top_n: int = 5) -> None:
    _subsection(f"Top {top_n} biggest margins")
    big = played.assign(abs_diff=played["goal_diff"].abs()).nlargest(top_n, "abs_diff")
    print(big[["date", "home_team", "away_team", "home_score", "away_score",
               "tournament"]].to_string(index=False))

def _print_poisson_check(played: pd.DataFrame, plot: bool = False, min_year: int | None = None, max_goals: int = 8) -> None:
    if min_year is not None:
        played = played[played["year"] >= min_year]
    scope = f"since {min_year}" if min_year is not None else "all years"
    _subsection(f"Poisson check ({scope})")

    fmt = {
        "observed": "{:,}".format,
        "expected": "{:,.1f}".format,
        "obs share (%)": "{:.2f}".format,
        "exp share (%)": "{:.2f}".format,
        "pearson resid": "{:+.1f}".format,
    }

    results_by_side = {}
    for side in ("home_score", "away_score"):
        scores = played[side]
        n, m, v = len(scores), scores.mean(), scores.var()

        # (1) Dispersion test: under Poisson, (n-1) * var / mean follows chi2(n-1)
        disp_stat = (n - 1) * v / m
        p_disp = stats.chi2.sf(disp_stat, n - 1)          # one-sided: H1 = overdispersed

        # (2) Goodness of fit on the whole shape, after pooling sparse tail bins
        table = _poisson_table(scores, max_goals)
        obs_p, exp_p = _pool_small_bins(table["observed"].to_numpy(), table["expected"].to_numpy())
        chi2 = ((obs_p - exp_p) ** 2 / exp_p).sum()
        dof = len(obs_p) - 2  # bins - 1 - 1 (lambda was estimated)
        p_gof = stats.chi2.sf(chi2, dof)

        # (3) Effect size: where is the biggest gap, in percentage points?
        gap = table["obs share (%)"] - table["exp share (%)"]
        worst = gap.abs().idxmax()

        print(f"\n{side}: n={n:,}  mean={m:.2f}  var={v:.2f}  var/mean={v / m:.2f}")
        print(f"  dispersion test:  stat={disp_stat:,.0f}, p={p_disp:.3g}  (H0: Poisson, H1: overdispersed)")
        print(f"  chi-square fit:   chi2={chi2:,.1f}, dof={dof}, p={p_gof:.3g}")
        print(f"  largest gap:      {gap[worst]:+.2f} pp at {worst} goals")
        print(table.to_string(formatters=fmt))

        results_by_side[side] = (table, v / m)

    if plot:
        colors = {"home_score": "#4C72B0", "away_score": "#DD8452"}
        poisson_color = "#973EED"

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        for ax, (side, (table, ratio)) in zip(axes, results_by_side.items()):
            x = np.arange(len(table))
            ax.set_axisbelow(True)
            ax.bar(x, table["obs share (%)"], color=colors[side], edgecolor="white",
                   label="observed")
            ax.plot(x, table["exp share (%)"], color=poisson_color, marker="o",
                    linewidth=1.5, label="Poisson (same mean)")

            ax.set_xticks(x, labels=table.index)
            ax.set_xlabel("goals scored")
            ax.set_title(f"{side}  (var/mean = {ratio:.2f})")

            ax.yaxis.minorticks_on()
            ax.grid(which="major", axis="y", linestyle="-", linewidth=0.8, alpha=0.6)
            ax.grid(which="minor", axis="y", linestyle=":", linewidth=0.5, alpha=0.4)
            ax.legend(loc="upper right")

        axes[0].set_ylabel("share of matches (%)")
        fig.suptitle(f"Observed goals vs Poisson with the same mean ({scope})")

        save_fig(f"poisson_check_{'since_' + str(min_year) if min_year else 'all'}")

def teams(results: pd.DataFrame, plot: bool = True, top_n: int = 10, order: str = "count", recent_start: int = 2000) -> None:
    section("TEAMS")
    by_team = _matches_per_team(results)
    print(f"Total of {len(by_team)} distinct teams\n")

    # All results
    _print_top_teams(by_team, len(results), top_n)  # all results

    # Recent (21st century) window only
    n_recent = len(_filter_years(results, start_year=recent_start))
    _print_top_teams(_matches_per_team(results, start_year=recent_start), n_recent, top_n, scope=_scope_label(recent_start, None)) 

    _print_low_coverage(by_team, results)
    _print_team_name_checks(results, by_team)

    if plot:
        _plot_matches_per_team_hist(by_team)
        _plot_top_teams(by_team, top_n, order)

def _filter_years(results: pd.DataFrame, start_year: int | None = None, end_year: int | None = None) -> pd.DataFrame:
    """ Keep matches with start_year <= year <= end_year (inclusive, None = open on that side) """
    if start_year is not None and end_year is not None and start_year > end_year:
        raise ValueError("start_year must be <= end_year")
    mask = pd.Series(True, index=results.index)
    if start_year is not None:
        mask &= results["year"] >= start_year
    if end_year is not None:
        mask &= results["year"] <= end_year
    return results[mask]

def _scope_label(start_year: int | None, end_year: int | None) -> str:
    if start_year is None and end_year is None:
        return "all years"
    return f"{start_year if start_year is not None else 'start'} to {end_year if end_year is not None else 'present'}"

def _matches_per_team(results: pd.DataFrame, start_year: int | None = None, end_year: int | None = None) -> pd.DataFrame:
    """ One row per team within the year window: matches, share of the window's matches (%), first/last year """
    window = _filter_years(results, start_year, end_year)
    long = pd.concat([
        window[["home_team", "year"]].rename(columns={"home_team": "team"}),
        window[["away_team", "year"]].rename(columns={"away_team": "team"}),
    ])
    by_team = long.groupby("team")["year"].agg(matches="size", first_year="min", last_year="max")
    by_team["share (%)"] = by_team["matches"] / len(window) * 100
    return by_team.sort_values("matches", ascending=False)

def _print_top_teams(by_team: pd.DataFrame, n_matches: int, top_n: int = 10, scope: str = "all years") -> None:
    _subsection(f"Top {top_n} teams by matches played, {scope} (share = % of the {n_matches:,} matches in this window)")
    if by_team.empty:
        print("No matches in this window.")
        return
    print(by_team.head(top_n).to_string(
        float_format="{:.2f}".format,
        formatters={"matches": "{:,}".format, "first_year": "{:.0f}".format, "last_year": "{:.0f}".format},
    ))

def _print_low_coverage(by_team: pd.DataFrame, results: pd.DataFrame, thresholds=(5, 20, 50)) -> None:
    _subsection("Teams with few matches")
    rows = []
    for t in thresholds:
        low = by_team.index[by_team["matches"] < t]
        involved = (results["home_team"].isin(low) | results["away_team"].isin(low)).sum()
        rows.append({
            "threshold": f"< {t}",
            "teams": len(low),
            "teams (%)": len(low) / len(by_team) * 100,
            "matches involved": involved,
            "matches involved (%)": involved / len(results) * 100,
        })
    print(pd.DataFrame(rows).set_index("threshold").to_string(
        float_format="{:.2f}".format,
        formatters={"teams": "{:,}".format, "matches involved": "{:,}".format},
    ))

def _print_team_name_checks(results: pd.DataFrame, by_team: pd.DataFrame) -> None:
    _subsection("Team name integrity")
    print("null team names (home/away):",
          int(results["home_team"].isna().sum()), "/", int(results["away_team"].isna().sum()))
    print("matches where a team plays itself:",
          int((results["home_team"] == results["away_team"]).sum()))

    names = pd.Series(by_team.index.astype(str))
    cleaned = names.str.strip().str.replace(r"\s+", " ", regex=True)
    print("names with leading/trailing/double whitespace:", names[names != cleaned].tolist())
    print("names that collide after normalising case/whitespace:",
          names[cleaned.str.casefold().duplicated(keep=False)].tolist())

def _plot_matches_per_team_hist(by_team: pd.DataFrame, n_bins: int = 50) -> None:
    m = by_team["matches"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.set_axisbelow(True)

    # Linear-spaced bins (linear x-axis)
    bins = np.linspace(0, m.max(), n_bins)
    ax.hist(m, bins=bins, color="#4C72B0", edgecolor="white")
    ax.set_xlabel("matches played")
    
    # Log-spaced bins (log x-axis)
    #bins = np.logspace(0, np.log10(m.max()), n_bins)
    #ax.hist(m, bins=bins, color="#4C72B0", edgecolor="white")
    #ax.set_xscale("log")
    #ax.set_xlabel("matches played (log scale)")

    ax.set_ylabel("number of teams")
    ax.set_title(f"Matches per team ({len(m)} teams)")
    ax.minorticks_on()   # minor ticks on both axes (must come after the x-scale is set)
    ax.grid(which="major", axis="both", linestyle="-", linewidth=0.8, alpha=0.6)
    ax.grid(which="minor", axis="both", linestyle=":", linewidth=0.5, alpha=0.4)
    save_fig("matches_per_team_hist")

def _plot_top_teams(by_team: pd.DataFrame, top_n: int = 20, order: str = "count") -> None:
    top = by_team.head(top_n)            # the top N by matches, whichever order is drawn
    if order == "name":
        top = top.sort_index()

    fig, ax = plt.subplots(figsize=(9, 0.32 * len(top) + 1.5))
    ax.set_axisbelow(True)
    bars = ax.barh(top.index, top["matches"], color="#4C72B0", edgecolor="white")
    ax.bar_label(bars,
                 labels=[f"{m:,} ({s:.2f}%)" for m, s in zip(top["matches"], top["share (%)"])],
                 padding=3, fontsize=8)

    ax.invert_yaxis()                    # first row at the top
    ax.set_xlim(0, top["matches"].max() * 1.18)
    ax.set_xlabel("matches played")
    ax.set_title(f"Top {top_n} teams by matches played "
                 f"({'sorted by count' if order == 'count' else 'sorted A-Z'})")

    ax.xaxis.minorticks_on()
    ax.grid(which="major", axis="x", linestyle="-", linewidth=0.8, alpha=0.6)
    ax.grid(which="minor", axis="x", linestyle=":", linewidth=0.5, alpha=0.4)
    save_fig(f"top_{top_n}_teams_{order}")

def shootout_integrity(results: pd.DataFrame, shootouts: pd.DataFrame, bin_size: int = 10, plot: bool = False) -> None:
    section("SHOOTOUTS vs RESULTS")
    merged = _shootout_match_table(results, shootouts)

    _print_shootout_coverage(merged)
    _print_shootouts_not_draw(merged)
    _print_shootout_winner_check(shootouts)
    _print_first_shooter_missing(shootouts)
    _print_shootouts_over_time(shootouts, bin_size, plot)
    _print_shootout_share(results, shootouts, bin_size, plot)

def _bin_start(years: pd.Series, bin_size: int) -> pd.Series:
    """ Floor each year to the first year of its bin (1987 -> 1980 for bin_size=10) """
    if bin_size < 1:
        raise ValueError("bin_size must be >= 1")
    return (years // bin_size * bin_size).astype(int)

def _apply_bin_ticks(ax, starts, bin_size: int) -> None:
    # One labelled tick per bin only when that fits; otherwise let matplotlib choose
    if len(starts) <= 25:
        ax.set_xticks(list(starts), labels=[_bin_label(s, bin_size) for s in starts], rotation=45)

def _shootout_match_table(results: pd.DataFrame, shootouts: pd.DataFrame) -> pd.DataFrame:
    """ Each shootout joined to its result (order-independent team pair + date) """
    r = results.assign(key=pair_key(results)).drop_duplicates("key")   # avoids row explosion
    s = shootouts.assign(key=pair_key(shootouts))
    return s.merge(r[["key", "home_score", "away_score"]], on="key", how="left", indicator=True)

def _print_shootout_coverage(merged: pd.DataFrame) -> None:
    _subsection("Shootouts matched to a result")
    matched = merged["_merge"] == "both"
    print(f"shootouts: {len(merged):,} | matched: {matched.sum():,} | unmatched: {(~matched).sum():,}")
    if (~matched).any():
        print("unmatched (first 20):")
        print(merged.loc[~matched, ["date", "home_team", "away_team"]].head(20).to_string(index=False))

def _print_shootouts_not_draw(merged: pd.DataFrame) -> None:
    _subsection("Matched shootouts whose result was NOT a draw")
    scored = merged[merged["_merge"] == "both"].dropna(subset=["home_score", "away_score"])
    not_draw = scored[scored["home_score"] != scored["away_score"]]
    print(f"count: {len(not_draw):,} of {len(scored):,} matched shootouts")
    if len(not_draw):
        print(not_draw[["date", "home_team", "away_team", "home_score", "away_score"]]
              .head(20).to_string(index=False))

def _print_shootout_winner_check(shootouts: pd.DataFrame) -> None:
    _subsection("Shootout winner is one of the two teams")
    ok = (shootouts["winner"] == shootouts["home_team"]) | (shootouts["winner"] == shootouts["away_team"])
    print(f"invalid winners: {(~ok).sum():,} of {len(shootouts):,}")
    if (~ok).any():
        print(shootouts.loc[~ok, ["date", "home_team", "away_team", "winner"]].head(20).to_string(index=False))


def _print_first_shooter_missing(shootouts: pd.DataFrame) -> None:
    _subsection("first_shooter missing")
    missing = shootouts["first_shooter"].isna()
    print(f"overall: {missing.sum():,} of {len(shootouts):,} ({missing.mean() * 100:.2f}%)")

    decade = (shootouts["date"].dt.year // 10 * 10).astype("Int64")
    by_decade = pd.DataFrame({
        "shootouts": missing.groupby(decade).size(),
        "missing": missing.groupby(decade).sum(),
        "missing (%)": missing.groupby(decade).mean() * 100,
    })
    by_decade.index.name = "decade"
    print(by_decade.to_string(float_format="{:.2f}".format, formatters={"shootouts": "{:,}".format, "missing": "{:,}".format}))

def _print_shootouts_over_time(shootouts: pd.DataFrame, bin_size: int = 10, plot: bool = False) -> None:
    suffix = _BIN_TITLES.get(bin_size, f"by {bin_size}-year bin")
    _subsection(f"Shootouts {suffix}")

    starts = _bin_start(shootouts["date"].dt.year.dropna(), bin_size)
    counts = starts.groupby(starts).size()
    counts = counts.reindex(range(counts.index.min(), counts.index.max() + 1, bin_size), fill_value=0)

    total = counts.sum()
    table = pd.DataFrame({
        "count": counts,
        "share (%)": counts / total * 100,
        "cumulative (%)": (counts / total * 100).cumsum(),
    })
    table.index.name = "bin start"
    print(table.to_string(float_format="{:.2f}".format, formatters={"count": "{:,}".format}))
    print(f"{'total':>9}  {total:,}")

    if plot:
        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.set_axisbelow(True)
        ax.bar(counts.index, counts.values, width=bin_size * 0.9, align="edge", color="#4C72B0")

        ax.set_title(f"Penalty shootouts {suffix}")
        ax.set_xlabel("year" if bin_size == 1 else f"start of {bin_size}-year bin")
        ax.set_ylabel("shootouts")
        ax.set_ylim(0, counts.max() * 1.05)
        _apply_bin_ticks(ax, counts.index, bin_size)

        ax.minorticks_on()
        ax.grid(which="major", linestyle="-", linewidth=0.8, alpha=0.6)
        ax.grid(which="minor", linestyle=":", linewidth=0.5, alpha=0.4)
        save_fig(f"shootouts_per_{bin_size}y")

def _print_shootout_share(results: pd.DataFrame, shootouts: pd.DataFrame,
                          bin_size: int = 10, plot: bool = False) -> None:
    suffix = _BIN_TITLES.get(bin_size, f"by {bin_size}-year bin")
    _subsection(f"Share of matches that went to a shootout {suffix}")

    first_date = shootouts["date"].min()
    shoot_keys = pair_key(shootouts)

    # Only count matches from the first recorded shootout onwards
    # Before that, 0% would mean "not recorded", not "none happened"
    r = (results.assign(key=pair_key(results))
                .dropna(subset=["year"])
                .loc[lambda d: d["date"] >= first_date]
                .assign(had_shootout=lambda d: d["key"].isin(shoot_keys),
                        is_draw=lambda d: d["home_score"] == d["away_score"],
                        bin=lambda d: _bin_start(d["year"], bin_size)))

    g = r.groupby("bin").agg(matches=("key", "size"),
                             draws=("is_draw", "sum"),
                             shootouts=("had_shootout", "sum"))
    g["% of matches"] = g["shootouts"] / g["matches"] * 100
    g["% of draws"] = g["shootouts"] / g["draws"].replace(0, np.nan) * 100
    g.index.name = "bin start"

    print(f"(matches from {first_date.date()} onwards, the first recorded shootout)")
    print(g.to_string(float_format="{:.2f}".format,
                      formatters={"matches": "{:,}".format, "draws": "{:,}".format,
                                  "shootouts": "{:,}".format}))
    m, d, s = g["matches"].sum(), g["draws"].sum(), g["shootouts"].sum()
    print(f"{'total':>9}  {m:,} matches, {d:,} draws, {s:,} shootouts "
          f"({s / m * 100:.2f}% of matches, {s / d * 100:.2f}% of draws)")

    if plot:
        colors = {"matches": "#4C72B0", "draws": "#DD8452"}
        x = g.index

        fig, ax = plt.subplots(figsize=(11, 5))
        ax.set_axisbelow(True)
        ax.plot(x, g["% of matches"], marker="o", linewidth=2, color=colors["matches"],
                label="% of all matches")
        ax.plot(x, g["% of draws"], marker="s", markersize=5, linestyle="--", linewidth=1.5,
                color=colors["draws"], label="% of drawn matches")

        ax.set_title(f"Share of matches that went to a shootout {suffix}")
        ax.set_xlabel("year" if bin_size == 1 else f"start of {bin_size}-year bin")
        ax.set_ylabel("share (%)")
        ax.set_ylim(0, max(g["% of matches"].max(), g["% of draws"].max()) * 1.1)
        _apply_bin_ticks(ax, x, bin_size)

        ax.minorticks_on()
        ax.grid(which="major", linestyle="-", linewidth=0.8, alpha=0.6)
        ax.grid(which="minor", linestyle=":", linewidth=0.5, alpha=0.4)
        ax.legend(loc="upper left")
        save_fig(f"shootout_share_{bin_size}y")

PERIOD_EDGES = [0, 15, 30, 45, 60, 75, 90, np.inf]
PERIOD_LABELS = ["1-15", "16-30", "31-45", "46-60", "61-75", "76-90", "91+"]
PERIOD_COLORS = ["#DCE9F5", "#B5D1EA", "#8DB8DE", "#6A9FD0", "#4A85BF", "#2F6AA8", "#1B4F8A"]

# own_goal and penalty share the same analysis; only these details differ
FLAG_CONFIG = {
    "own_goal": {"noun": "own goals", "team_col": "opponent", "team_header": "player's team", "color": "#C44E52"},
    "penalty":  {"noun": "penalties", "team_col": "team",     "team_header": "team",          "color": "#4C72B0"},
}


def _prepare_goals(goalscorers: pd.DataFrame) -> pd.DataFrame:
    g = goalscorers.copy()
    g["year"] = g["date"].dt.year
    for col in ("own_goal", "penalty"):
        g[col] = g[col].eq(True)          # NaN -> False, works for bool or object columns
    # The side the goal was scored against. If 'team' is the side credited with an own goal,
    # the player who scored it played for this side (checked in _print_own_goal_sanity).
    g["opponent"] = np.where(g["team"] == g["home_team"], g["away_team"], g["home_team"])
    g["period"] = pd.cut(g["minute"], PERIOD_EDGES, labels=PERIOD_LABELS, include_lowest=True)
    return g


def _print_goal_data_overview(g: pd.DataFrame, results: pd.DataFrame) -> None:
    _subsection("Goal data overview")
    keys = ["date", "home_team", "away_team"]
    print(f"goal rows: {len(g):,} | matches with goal data: {g.groupby(keys).ngroups:,} | "
          f"{g['date'].min().date()} -> {g['date'].max().date()}")

    nulls = g[keys + ["team", "scorer", "minute"]].isna().sum()
    nulls = nulls[nulls > 0]
    print("nulls:", nulls.to_dict() if len(nulls) else "none")

    for flag in ("own_goal", "penalty"):
        n = g[flag].sum()
        print(f"{flag}: {n:,} ({n / len(g) * 100:.2f}% of goal rows)")

    bad_team = ~((g["team"] == g["home_team"]) | (g["team"] == g["away_team"]))
    print(f"rows where 'team' is neither home nor away: {bad_team.sum():,}")
    dups = g.duplicated(subset=keys + ["team", "scorer", "minute"]).sum()
    print(f"rows repeating the same match/team/scorer/minute: {dups:,} (possible duplicates)")

    match_idx = pd.MultiIndex.from_frame(g[keys].drop_duplicates())
    results_idx = pd.MultiIndex.from_frame(results[keys])
    print(f"matches in goalscorers with no row in results: {(~match_idx.isin(results_idx)).sum():,}")


def _print_goal_coverage(results: pd.DataFrame, g: pd.DataFrame) -> None:
    _subsection("Scorer coverage by decade (matches with at least one goal)")
    keys = ["date", "home_team", "away_team"]
    n_rows = g.groupby(keys).size().rename("n_goal_rows")

    m = results.dropna(subset=["total_goals"])
    m = m[m["total_goals"] > 0].join(n_rows, on=keys)   # 0-0 draws have no goal rows by design
    m["covered"] = m["n_goal_rows"].notna()
    m["exact"] = m["n_goal_rows"] == m["total_goals"]
    m["too_many"] = m["n_goal_rows"] > m["total_goals"]

    t = m.groupby("decade").agg(matches=("covered", "size"), covered=("covered", "sum"),
                                exact=("exact", "sum"), too_many=("too_many", "sum"))
    t["coverage (%)"] = t["covered"] / t["matches"] * 100
    t["exact (% of covered)"] = t["exact"] / t["covered"].replace(0, np.nan) * 100
    print(t.to_string(float_format="{:.2f}".format,
                      formatters={c: "{:,}".format for c in ("matches", "covered", "exact", "too_many")}))
    print("too_many = more goal rows than goals in the final score (suspect rows)")


def _print_minute_distribution(g: pd.DataFrame, plot: bool = False, cap: int = 120) -> None:
    _subsection("Goal minute distribution")
    minutes = g["minute"].dropna()
    print(f"minute missing: {g['minute'].isna().sum():,} of {len(g):,} "
          f"({g['minute'].isna().mean() * 100:.2f}%)")
    print(minutes.describe().round(1).to_string())

    counts = g["period"].value_counts(sort=False)          # keeps the period order
    share = counts / counts.sum() * 100
    table = pd.DataFrame({"goals": counts, "share (%)": share, "cumulative (%)": share.cumsum()})
    table.index.name = "minutes"
    print("\n" + table.to_string(float_format="{:.2f}".format, formatters={"goals": "{:,}".format}))

    by_min = minutes.round().astype(int).value_counts()
    for mark in (45, 90):
        neighbours = (by_min.get(mark - 1, 0) + by_min.get(mark + 1, 0)) / 2
        print(f"minute {mark}: {by_min.get(mark, 0):,} goals vs {neighbours:,.0f} on average at "
              f"{mark - 1} and {mark + 1}")

    if plot:
        clipped = minutes.round().astype(int).clip(upper=cap)   # anything later is lumped at the cap
        per_minute = clipped.value_counts().reindex(range(int(clipped.min()), cap + 1), fill_value=0)

        fig, ax = plt.subplots(figsize=(12, 4.5))
        ax.set_axisbelow(True)
        ax.bar(per_minute.index, per_minute.values, width=0.9, color="#4C72B0")
        for i, mark in enumerate((45, 90, 120)):
            ax.axvline(mark, color="#8C8C8C", linestyle="--", linewidth=1,
                       label="45', 90', 120'" if i == 0 else None)

        ax.set_title("When goals are scored")
        ax.set_xlabel(f"minute (goals after {cap}' grouped at {cap}')")
        ax.set_ylabel("goals")
        ax.minorticks_on()
        ax.grid(which="major", linestyle="-", linewidth=0.8, alpha=0.6)
        ax.grid(which="minor", linestyle=":", linewidth=0.5, alpha=0.4)
        ax.legend(loc="upper left")
        save_fig("goal_minute_distribution")


def _print_minute_by_bin(g: pd.DataFrame, bin_size: int = 10, plot: bool = False,
                         min_goals: int = 30) -> None:
    suffix = _BIN_TITLES.get(bin_size, f"by {bin_size}-year bin")
    _subsection(f"Goal minute {suffix}")

    d = g.dropna(subset=["minute", "year"]).copy()
    d["bin"] = _bin_start(d["year"], bin_size)

    summary = d.groupby("bin")["minute"].agg(goals="size", mean="mean", median="median")
    shares = (pd.crosstab(d["bin"], d["period"], normalize="index") * 100)
    shares = shares.reindex(columns=PERIOD_LABELS).fillna(0)
    table = summary.join(shares)
    table.index.name = "bin start"
    print(table.to_string(float_format="{:.2f}".format, formatters={"goals": "{:,}".format}))
    print("(period columns = % of that bin's goals)")

    if plot:
        shown = table[table["goals"] >= min_goals]    # tiny bins give meaningless percentages
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.set_axisbelow(True)
        bottom = np.zeros(len(shown))
        for label, color in zip(PERIOD_LABELS, PERIOD_COLORS):
            ax.bar(shown.index, shown[label], width=bin_size * 0.9, align="edge",
                   bottom=bottom, color=color, edgecolor="white", linewidth=0.5, label=label)
            bottom += shown[label].to_numpy()

        ax.set_title(f"Share of goals by minute range {suffix} (bins with < {min_goals} goals hidden)")
        ax.set_xlabel("year" if bin_size == 1 else f"start of {bin_size}-year bin")
        ax.set_ylabel("share of goals (%)")
        ax.set_ylim(0, 100)
        _apply_bin_ticks(ax, shown.index, bin_size)
        ax.minorticks_on()
        ax.grid(which="major", axis="y", linestyle="-", linewidth=0.8, alpha=0.6)
        ax.grid(which="minor", axis="y", linestyle=":", linewidth=0.5, alpha=0.4)
        ax.legend(title="minute", bbox_to_anchor=(1.01, 1), loc="upper left")
        save_fig(f"goal_minute_by_bin_{bin_size}y")


def _scorer_table(g: pd.DataFrame) -> pd.DataFrame:
    """Goals per (player, team), own goals excluded, sorted by goals then name."""
    t = (g[~g["own_goal"]]
         .groupby(["scorer", "team"])
         .agg(goals=("date", "size"), penalties=("penalty", "sum"),
              first_year=("year", "min"), last_year=("year", "max"))
         .rename_axis(["player", "team"]))
    t["penalties"] = t["penalties"].astype(int)
    t["penalty share (%)"] = t["penalties"] / t["goals"] * 100
    t["share (%)"] = t["goals"] / t["goals"].sum() * 100
    return t.sort_index().sort_values("goals", ascending=False, kind="stable")


def _print_top_scorers(g: pd.DataFrame, top_n: int = 20, plot: bool = False) -> None:
    t = _scorer_table(g)
    _subsection(f"Top {top_n} scorers of all time (own goals excluded; "
                f"share = % of all {int(t['goals'].sum()):,} goals)")
    print(t.head(top_n).to_string(
        float_format="{:.2f}".format,
        formatters={"goals": "{:,}".format, "first_year": "{:.0f}".format, "last_year": "{:.0f}".format}))

    if plot:
        top = t.head(top_n)
        names = [f"{p} ({tm})" for p, tm in top.index]
        fig, ax = plt.subplots(figsize=(9, 0.32 * len(top) + 1.5))
        ax.set_axisbelow(True)
        bars = ax.barh(names, top["goals"], color="#4C72B0", edgecolor="white")
        ax.bar_label(bars, labels=[f"{n:,} ({s:.2f}%)" for n, s in zip(top["goals"], top["share (%)"])],
                     padding=3, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, top["goals"].max() * 1.18)
        ax.set_xlabel("goals")
        ax.set_title(f"Top {top_n} scorers of all time")
        ax.xaxis.minorticks_on()
        ax.grid(which="major", axis="x", linestyle="-", linewidth=0.8, alpha=0.6)
        ax.grid(which="minor", axis="x", linestyle=":", linewidth=0.5, alpha=0.4)
        save_fig(f"top_{top_n}_scorers")


def _print_top_scorers_by_bin(g: pd.DataFrame, top_m: int = 3, bin_size: int = 10) -> None:
    suffix = _BIN_TITLES.get(bin_size, f"by {bin_size}-year bin")
    _subsection(f"Top {top_m} scorers {suffix} (own goals excluded)")

    d = g[~g["own_goal"]].dropna(subset=["year"]).copy()
    d["bin"] = _bin_start(d["year"], bin_size)
    totals = d.groupby("bin").size()

    t = (d.groupby(["bin", "scorer", "team"]).size().rename("goals").reset_index()
          .sort_values(["bin", "goals", "scorer"], ascending=[True, False, True]))
    t["rank"] = t.groupby("bin").cumcount() + 1
    top = t[t["rank"] <= top_m].copy()
    top["share of bin (%)"] = top["goals"] / top["bin"].map(totals) * 100

    top["bin"] = top["bin"].map(lambda b: _bin_label(b, bin_size))
    top.loc[top["rank"] > 1, "bin"] = ""        # show each bin label once
    print(top.rename(columns={"scorer": "player"}).to_string(
        index=False, float_format="{:.2f}".format, formatters={"goals": "{:,}".format}))


def _print_own_goal_sanity(g: pd.DataFrame) -> None:
    _subsection("Own goal convention check")
    print(g.loc[g["own_goal"], ["date", "home_team", "away_team", "team", "scorer", "minute", "opponent"]]
            .head(3).to_string(index=False))
    print("Verify: 'team' should be the side CREDITED with the goal, so the player's own side is 'opponent'.")


def _print_flag_top_players(g: pd.DataFrame, flag: str, top_p: int = 10) -> None:
    cfg = FLAG_CONFIG[flag]
    noun = cfg["noun"]
    flagged = g[g[flag]]
    _subsection(f"Top {top_p} players by {noun} (share = % of all {len(flagged):,} {noun})")

    names = ["player", cfg["team_header"]]
    t = (flagged.groupby(["scorer", cfg["team_col"]])
                .agg(n=("date", "size"), first_year=("year", "min"), last_year=("year", "max"))
                .rename_axis(names).rename(columns={"n": noun}))
    t["share (%)"] = t[noun] / len(flagged) * 100

    formatters = {noun: "{:,}".format, "first_year": "{:.0f}".format, "last_year": "{:.0f}".format}
    if flag == "penalty":
        totals = g[~g["own_goal"]].groupby(["scorer", "team"]).size().rename_axis(names)
        t["all goals"] = totals.reindex(t.index).to_numpy()
        t["% of their goals"] = t[noun] / t["all goals"] * 100
        formatters["all goals"] = "{:,.0f}".format

    t = t.sort_index().sort_values(noun, ascending=False, kind="stable")
    print(t.head(top_p).to_string(float_format="{:.2f}".format, formatters=formatters))


def _print_flag_over_time(g: pd.DataFrame, flag: str, bin_size: int = 10,
                          plot: bool = False, min_goals: int = 100) -> None:
    cfg = FLAG_CONFIG[flag]
    noun = cfg["noun"]
    suffix = _BIN_TITLES.get(bin_size, f"by {bin_size}-year bin")
    _subsection(f"{noun.capitalize()} {suffix}")

    d = g.dropna(subset=["year"]).copy()
    d["bin"] = _bin_start(d["year"], bin_size)
    t = d.groupby("bin").agg(goals=("date", "size"), flagged=(flag, "sum"))
    t["flagged"] = t["flagged"].astype(int)
    t["share of goals (%)"] = t["flagged"] / t["goals"] * 100
    t = t.rename(columns={"flagged": noun})
    t.index.name = "bin start"
    print(t.to_string(float_format="{:.2f}".format, formatters={"goals": "{:,}".format, noun: "{:,}".format}))
    print(f"{'total':>9}  {t['goals'].sum():,} goals, {t[noun].sum():,} {noun} "
          f"({t[noun].sum() / t['goals'].sum() * 100:.2f}%)")

    if plot:
        color = cfg["color"]
        x = t.index
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.set_axisbelow(True)
        ax.bar(x, t[noun], width=bin_size * 0.9, align="edge", color=color, alpha=0.75, label=f"{noun} (count)")
        ax.set_xlabel("year" if bin_size == 1 else f"start of {bin_size}-year bin")
        ax.set_ylabel(f"{noun}")
        ax.set_ylim(0, t[noun].max() * 1.1)
        _apply_bin_ticks(ax, x, bin_size)

        ax2 = ax.twinx()   # share of all goals, hidden where the bin is too small to trust
        share = t["share of goals (%)"].where(t["goals"] >= min_goals)
        ax2.plot(x + bin_size * 0.45, share, color="#222222", marker="o", linewidth=1.8,
                 label=f"% of goals (bins >= {min_goals} goals)")
        ax2.set_ylabel("share of goals (%)")
        ax2.set_ylim(0, share.max() * 1.15)

        ax.set_title(f"{noun.capitalize()} {suffix}")
        ax.minorticks_on()
        ax.grid(which="major", linestyle="-", linewidth=0.8, alpha=0.6)
        ax.grid(which="minor", linestyle=":", linewidth=0.5, alpha=0.4)
        handles = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
        labels = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
        ax.legend(handles, labels, loc="upper left")
        save_fig(f"{flag}_over_time_{bin_size}y")


def goalscorer_integrity(results: pd.DataFrame, goalscorers: pd.DataFrame, bin_size: int = 10, top_n: int = 20, top_m: int = 3, top_p: int = 10, plot: bool = False) -> None:
    section("GOALSCORERS vs RESULTS")
    g = _prepare_goals(goalscorers)

    _print_goal_data_overview(g, results)
    _print_goal_coverage(results, g)
    _print_minute_distribution(g, plot)
    _print_minute_by_bin(g, bin_size, plot)
    _print_top_scorers(g, top_n, plot)
    _print_top_scorers_by_bin(g, top_m, bin_size)
    _print_own_goal_sanity(g)
    _print_flag_top_players(g, "own_goal", top_p)
    _print_flag_over_time(g, "own_goal", bin_size, plot)
    _print_flag_top_players(g, "penalty", top_p)
    _print_flag_over_time(g, "penalty", bin_size, plot)

# --- Helper function(s) ---
def section(title: str) -> None:
    print(f"\n{'=' * 30} {title} {'=' * 30}")

def _subsection(title: str) -> None:
    print(f"\n--- {title} ---")

def save_fig(name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{name}.png", dpi=600)
    plt.close()

def counts_and_shares(series: pd.Series) -> pd.DataFrame:
    """ Counts, share (%) of the full total, and cumulative share (%), sorted by count """
    counts = series.value_counts()
    share = counts / counts.sum() * 100
    return pd.DataFrame({
        "count": counts,
        "share (%)": share,
        "cumulative (%)": share.cumsum(),
    })

def _cramers_v_label(v: float) -> str:
    # Cohen's thresholds for a 2-row table (1 degree of freedom per variable)
    if v < 0.1:
        return "negligible"
    if v < 0.3:
        return "small"
    if v < 0.5:
        return "medium"
    return "large"

def _bin_label(start: int, bin_size: int) -> str:
    if bin_size == 1:
        return f"{start}"
    if bin_size == 10:
        return f"{start}s"
    return f"{start}-{start + bin_size - 1}"

def _poisson_table(scores: pd.Series, max_goals: int = 8) -> pd.DataFrame:
    """Observed vs expected match counts for 0, 1, ..., max_goals, and a max_goals+ tail."""
    n, lam = len(scores), scores.mean()
    k = np.arange(max_goals + 1)

    observed = np.append([(scores == i).sum() for i in k], (scores > max_goals).sum())
    probs = np.append(stats.poisson.pmf(k, lam), stats.poisson.sf(max_goals, lam))
    expected = probs * n

    return pd.DataFrame({
        "observed": observed,
        "expected": expected,
        "obs share (%)": observed / n * 100,
        "exp share (%)": probs * 100,
        "pearson resid": (observed - expected) / np.sqrt(expected),
    }, index=[str(i) for i in k] + [f"{max_goals + 1}+"])

def _pool_small_bins(observed, expected, min_expected: float = 5):
    """Merge the rightmost bins until the last one has expected count >= min_expected."""
    obs, exp = list(observed), list(expected)
    while len(exp) > 2 and exp[-1] < min_expected:
        obs[-2] += obs.pop()
        exp[-2] += exp.pop()
    return np.array(obs), np.array(exp)

def pair_key(df: pd.DataFrame) -> pd.Series:
    """ Order-independent key for a match: date + sorted team pair """
    teams = np.sort(df[["home_team", "away_team"]].astype(str).to_numpy(), axis=1)
    return df["date"].dt.strftime("%Y-%m-%d") + "|" + teams[:, 0] + "|" + teams[:, 1]

def tournament_group(name: str) -> str:
    """ Coarse grouping, useful later for Elo K-factors """
    n = name.lower()
    if n == "fifa world cup":
        return "World Cup"
    if "qualification" in n:
        return "Qualifier"
    if n == "friendly":
        return "Friendly"
    if any(k in n for k in ("euro", "copa am", "african cup", "afc asian", "gold cup", "confederations", "nations cup", "nations league")):
        return "Continental"
    return "Other"

def analyze_data():
    if VERBOSE:
        print("Loading data . . .")
    results, shootouts, goalscorers, former_names = load_data()
    results, shootouts, goalscorers = prepare(results, shootouts, goalscorers)

    missing_and_types(results, shootouts, goalscorers)
    duplicates(results, shootouts)
    former_name_window_check(results, former_names)
    coverage_over_time(results, bin_years=10, plot=False)
    tournaments(results, plot=False)
    score_distributions(results)
    teams(results, plot=False, top_n=20, order="count")
    shootout_integrity(results, shootouts, bin_size=10, plot=False)
    goalscorer_integrity(results, goalscorers, bin_size=10, top_n=10, top_m=3, top_p=10, plot=False)

    if VERBOSE:
        print(f"\nFigures saved to {FIG_DIR.resolve()}")

if __name__ == "__main__":
    analyze_data()
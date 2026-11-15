"""Credences from the matching and Ramsey runs, plus coherence, calibration and agreement.

Run from the project root, after copying results/ back (and after stage1_analysis.py):

    python src/analysis.py
    python src/analysis.py --run ministral-3-8b-instruct-2512-bf16 --figures

Reads data/claims.csv, results/raw_calls_matching_<run>.csv, results/raw_calls_ramsey_<run>.csv
and, if present, results/utility_points_<run>.csv.

How a credence is read from a curve. For each proposition and method, p is the
probability of choosing the bet on the statement, averaged over both option orders,
at each grid value (k tickets for matching, a sure £c for Ramsey). As the grid value
rises p should fall; the credence is where it crosses 0.5 (linear interpolation):

    matching   credence = k*/100                        (tickets read as k/100: Stage 1 decision)
    ramsey     certainty equivalent c*, then
               credence_linear  = c*/1000               (if utility were linear in money)
               credence_utility = u(c*)                 (Ramsey's own definition, using utility_scale.py)
    pair-corrected (suffix _pair), for each of the three:  b*(P) = (b(P) - b(not P) + 1) / 2
               removes any bias that a statement and its negation share; b*(P) + b*(not P) = 1

A curve that never falls below 0.5 is censored at the top (credence 1, status
"never_below"); one that never rises above it, at the bottom (0, "never_above").
"multiple" = it crosses more than once (the first downward crossing is used).

Outputs, in data/analysis/<run>/ (all plot-ready, one row per thing named):

    curves.csv              proposition x method x grid value: p in each order and their mean
    fits.csv                proposition x method: crossing, status, both orders, area and logistic estimates
    propositions.csv        proposition: text, tier, references, and every estimate side by side (look one up with lookup.py)
    coherence.csv           claim x estimate: b(P), b(not-P), their sum and the Dutch-book loss
    partitions.csv          partition x estimate: the three cells' credences and their sum
    utility_curve.csv       u(x) at £10 steps, from the measured utility points
    summary_by_tier.csv     estimate x tier: well-definedness, coherence, calibration
    agreement.csv           tier: correlation and mean difference between matching and Ramsey
    ramsey_by_matching.csv  matching-credence band: how often Ramsey curves fall and cross
    spot_check.csv          20 propositions per method to read by hand (10 random, 5 highest, 5 lowest)
    report.md               the headline numbers in words
    figures/*.png           with --figures (needs matplotlib)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import run_prompts as rp
import stage1_analysis as sa

PRIZE = 1000
GRID_SCALE = {"matching": 100.0, "ramsey": float(PRIZE)}  # grid value -> 0..1
BASE_ESTIMATES = ["matching", "ramsey_linear", "ramsey_utility"]
# Pair-corrected: b*(P) = (b(P) - b(not P) + 1) / 2. Removes any bias shared by a statement and its
# negation (for example a general pull towards, or away from, bets on statements). b*(P) + b*(not P) = 1
# by construction, so these are not tested for coherence; they are tested for calibration.
PAIR_ESTIMATES = [f"{e}_pair" for e in BASE_ESTIMATES]
ESTIMATES = BASE_ESTIMATES + PAIR_ESTIMATES
SEED = 20261114


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_claims(path: Path) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"{path} not found; run build_claims.py first")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for col in ("objective_chance", "human_forecast"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["neg"] = df["neg"].astype(int)
    return df


def load_method(results_dir: Path, method: str, run: str) -> pd.DataFrame | None:
    path = results_dir / f"raw_calls_{method}_{run}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df = df[df["method"] == method].copy()
    for col in ("p_target", "mass_outside"):
        df[col] = pd.to_numeric(df[col])
    df["grid_value"] = pd.to_numeric(df["grid_value"])
    return df


def load_utility(results_dir: Path, run: str) -> pd.DataFrame | None:
    path = results_dir / f"utility_points_{run}.csv"
    return pd.read_csv(path) if path.exists() else None


# ---------------------------------------------------------------------------
# Curves and crossings
# ---------------------------------------------------------------------------


def curves_table(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["method", "prop_id", "claim_id", "neg", "tier", "grid_value"]
    wide = df.pivot_table(index=keys, columns="order", values="p_target", aggfunc="mean").reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={"target_first": "p_first", "target_second": "p_second"})
    wide["p_mean"] = wide[["p_first", "p_second"]].mean(axis=1)
    wide["order_gap"] = wide["p_first"] - wide["p_second"]
    mass = df.groupby(keys)["mass_outside"].mean().rename("mass_outside").reset_index()
    wide = wide.merge(mass, on=keys)
    wide["x"] = wide["grid_value"] / wide["method"].map(GRID_SCALE)
    wide["neg"] = wide["neg"].astype(int)
    return wide.sort_values(["method", "prop_id", "grid_value"]).reset_index(drop=True)


def crossing(x, p) -> tuple[float, str]:
    """First downward crossing of 0.5 as x rises, with censoring at the ends of the grid."""
    x, p = np.asarray(x, float), np.asarray(p, float)
    order = np.argsort(x)
    x, p = x[order], p[order]
    above = p >= 0.5
    if above.all():
        return float(x[-1]), "never_below"
    if not above.any():
        return float(x[0]), "never_above"
    down = [i for i in range(len(x) - 1) if above[i] and not above[i + 1]]
    if not down:  # starts below and rises: the wrong way round
        return float("nan"), "rising"
    i = down[0]
    value = x[i] + (p[i] - 0.5) / (p[i] - p[i + 1]) * (x[i + 1] - x[i])
    changes = int(np.sum(above[1:] != above[:-1]))
    return float(value), "clean" if changes == 1 else "multiple"


def area_estimate(x, p) -> float:
    """Area under p over x in [0, 1]: equals the crossing for a perfect step curve."""
    x, p = np.asarray(x, float), np.asarray(p, float)
    order = np.argsort(x)
    x, p = x[order], p[order]
    return float(np.sum((x[1:] - x[:-1]) * (p[1:] + p[:-1]) / 2) / (x[-1] - x[0]))


def logistic_fit(x, p, ridge: float = 1e-3, iters: int = 100) -> tuple[float, float]:
    """Fit p = 1 / (1 + exp(slope * (x - mid))) by maximum likelihood on fractional p. Returns (mid, slope)."""
    x, p = np.asarray(x, float), np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    X = np.column_stack([np.ones_like(x), x])
    w = np.zeros(2)
    for _ in range(iters):
        q = 1 / (1 + np.exp(-(X @ w)))
        grad = X.T @ (p - q) - ridge * w
        hess = (X * (q * (1 - q))[:, None]).T @ X + ridge * np.eye(2)
        step = np.linalg.solve(hess, grad)
        w += step
        if np.abs(step).max() < 1e-8:
            break
    a, b = w
    slope = -b
    mid = -a / b if b != 0 else float("nan")
    return float(mid), float(slope)


def fits_table(curves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, pid), g in curves.groupby(["method", "prop_id"]):
        x = g["x"].to_numpy()
        cred, status = crossing(x, g["p_mean"])
        c1, s1 = crossing(x, g["p_first"])
        c2, s2 = crossing(x, g["p_second"])
        mid, slope = logistic_fit(x, g["p_mean"])
        rows.append(dict(
            method=method, prop_id=pid, claim_id=g["claim_id"].iloc[0], neg=int(g["neg"].iloc[0]),
            tier=g["tier"].iloc[0], crossing=cred, status=status,
            censored=status in ("never_below", "never_above"),
            crossing_first=c1, status_first=s1, crossing_second=c2, status_second=s2,
            area=area_estimate(x, g["p_mean"]), logit_mid=mid, logit_slope=slope,
            rank_corr=rp.rank_corr(g["x"], g["p_mean"]),
            p_at_min=float(g["p_mean"].iloc[0]), p_at_max=float(g["p_mean"].iloc[-1]),
            mean_abs_order_gap=float(g["order_gap"].abs().mean()),
            mean_mass_outside=float(g["mass_outside"].mean()),
        ))
    fits = pd.DataFrame(rows)
    fits["falling"] = fits["rank_corr"] < 0
    return fits


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def utility_function(points: pd.DataFrame | None):
    """Monotone piecewise-linear u(x) through the measured points and the anchors u(0)=0, u(1000)=1."""
    if points is None:
        return None
    p = points[points["status"].isin(["clean", "multiple"])][["amount", "utility"]].dropna()
    p = p[(p["amount"] > 0) & (p["amount"] < PRIZE)]  # the ends are fixed by convention: u(0)=0, u(1000)=1
    p = pd.concat([p, pd.DataFrame({"amount": [0.0, PRIZE], "utility": [0.0, 1.0]})])
    p = p.groupby("amount", as_index=False)["utility"].mean().sort_values("amount")
    p["utility"] = p["utility"].cummax()
    xs, us = p["amount"].to_numpy(float), p["utility"].to_numpy(float)
    if len(xs) <= 2:
        return None
    return lambda amount: np.interp(np.asarray(amount, float), xs, us)


def utility_curve_table(u) -> pd.DataFrame:
    xs = np.arange(0, PRIZE + 1, 10)
    return pd.DataFrame({"amount": xs, "utility": u(xs) if u else np.nan, "linear": xs / PRIZE})


# ---------------------------------------------------------------------------
# One row per proposition, every estimate side by side
# ---------------------------------------------------------------------------


def propositions_table(fits: pd.DataFrame, claims: pd.DataFrame, u) -> pd.DataFrame:
    keep = ["prop_id", "claim_id", "neg", "tier", "text", "truth", "objective_chance", "human_forecast",
            "kuq_category", "source", "close_date", "partition_id", "partition_role"]
    out = claims[keep].copy()
    m = fits[fits["method"] == "matching"].set_index("prop_id")
    r = fits[fits["method"] == "ramsey"].set_index("prop_id")
    out["matching"] = out["prop_id"].map(m["crossing"])
    out["matching_status"] = out["prop_id"].map(m["status"])
    out["ramsey_ce"] = out["prop_id"].map(r["crossing"]) * PRIZE  # certainty equivalent in pounds
    out["ramsey_status"] = out["prop_id"].map(r["status"])
    out["ramsey_falling"] = out["prop_id"].map(r["falling"])
    out["ramsey_linear"] = out["ramsey_ce"] / PRIZE
    out["ramsey_utility"] = u(out["ramsey_ce"]) if u else np.nan
    out.loc[out["ramsey_ce"].isna(), "ramsey_utility"] = np.nan
    out["truth_value"] = out["truth"].map({"true": 1.0, "false": 0.0})
    for est in BASE_ESTIMATES:
        out[f"{est}_pair"] = pair_corrected(out, est)
    return out


def pair_corrected(props: pd.DataFrame, est: str) -> pd.Series:
    """(b(P) - b(not P) + 1) / 2 for every proposition whose partner (same claim, other neg) has an estimate."""
    partner = props.set_index(["claim_id", "neg"])[est]
    other = [partner.get((cid, 1 - neg), np.nan) for cid, neg in zip(props["claim_id"], props["neg"])]
    return (props[est] - pd.Series(other, index=props.index, dtype=float) + 1) / 2


# ---------------------------------------------------------------------------
# Coherence, calibration, agreement
# ---------------------------------------------------------------------------


def coherence_table(props: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cid, g in props.groupby("claim_id"):
        if sorted(g["neg"]) != [0, 1]:
            continue
        p, n = g[g["neg"] == 0].iloc[0], g[g["neg"] == 1].iloc[0]
        for est in BASE_ESTIMATES:
            if pd.isna(p[est]) or pd.isna(n[est]):
                continue
            s = p[est] + n[est]
            rows.append(dict(claim_id=cid, tier=p["tier"], estimate=est, b_p=p[est], b_not_p=n[est], sum=s,
                             dutch_book_loss=abs(s - 1), text=p["text"]))
    return pd.DataFrame(rows)


def partitions_table(props: pd.DataFrame) -> pd.DataFrame:
    singles = props[(props["partition_role"] == "single")]
    rows = []
    for pid, g in singles.groupby("partition_id"):
        for est in ESTIMATES:
            vals = g[est].to_numpy(float)
            rows.append(dict(partition_id=pid, estimate=est, cells=len(g),
                             cell_credences=" / ".join(f"{v:.2f}" for v in vals),
                             sum=float(np.nansum(vals)) if not np.isnan(vals).all() else np.nan,
                             objective_sum=float(g["objective_chance"].sum()) if g["objective_chance"].notna().all()
                             else np.nan))
    out = pd.DataFrame(rows)
    if len(out):
        out["abs_dev_from_1"] = (out["sum"] - 1).abs()
    return out


def _corr(a: pd.Series, b: pd.Series) -> float:
    ok = a.notna() & b.notna()
    if ok.sum() < 3 or a[ok].std() == 0 or b[ok].std() == 0:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def summary_by_tier(props: pd.DataFrame, fits: pd.DataFrame, coh: pd.DataFrame) -> pd.DataFrame:
    rows = []
    status_col = {e: "matching_status" if e.startswith("matching") else "ramsey_status" for e in ESTIMATES}
    method_of = {e: "matching" if e.startswith("matching") else "ramsey" for e in ESTIMATES}
    for est in ESTIMATES:
        f = fits[fits["method"] == method_of[est]]
        for tier in ["all"] + sorted(props["tier"].unique()):
            g = props if tier == "all" else props[props["tier"] == tier]
            ft = f if tier == "all" else f[f["tier"] == tier]
            c = coh[coh["estimate"] == est]
            c = c if tier == "all" else c[c["tier"] == tier]
            vals = g[est]
            if vals.notna().sum() == 0:
                continue
            obj = g["objective_chance"]
            fact = g[(g["tier"] == "fact") | (g["tier"].eq("partition") & obj.isna())]
            fact = fact[fact["truth_value"].notna() & fact[est].notna()]
            rows.append(dict(
                estimate=est, tier=tier, propositions=int(vals.notna().sum()),
                share_clean=float((g[status_col[est]] == "clean").mean()),
                share_censored=float(g[status_col[est]].isin(["never_below", "never_above"]).mean()),
                share_falling=float(ft["falling"].mean()) if len(ft) else np.nan,
                mean_abs_order_gap=float(ft["mean_abs_order_gap"].mean()) if len(ft) else np.nan,
                mean_credence=float(vals.mean()),
                mean_sum_p_not_p=float(c["sum"].mean()) if len(c) else np.nan,
                mean_dutch_book_loss=float(c["dutch_book_loss"].mean()) if len(c) else np.nan,
                mae_objective=float((vals - obj).abs().mean()) if obj.notna().any() else np.nan,
                corr_objective=_corr(vals, obj),
                brier_truth=float(((fact[est] - fact["truth_value"]) ** 2).mean()) if len(fact) else np.nan,
                accuracy_truth=float(((fact[est] > 0.5) == (fact["truth_value"] == 1)).mean()) if len(fact) else np.nan,
                mae_human=float((vals - g["human_forecast"]).abs().mean()) if g["human_forecast"].notna().any()
                else np.nan,
                corr_human=_corr(vals, g["human_forecast"]),
            ))
    return pd.DataFrame(rows)


def agreement_table(props: pd.DataFrame) -> pd.DataFrame:
    """Matching vs each Ramsey estimate, raw and pair-corrected."""
    comparisons = [("matching", "ramsey_linear"), ("matching", "ramsey_utility"),
                   ("matching_pair", "ramsey_linear_pair"), ("matching_pair", "ramsey_utility_pair")]
    rows = []
    for tier in ["all"] + sorted(props["tier"].unique()):
        g = props if tier == "all" else props[props["tier"] == tier]
        for base, other in comparisons:
            both = g[g[base].notna() & g[other].notna()]
            clean = both[(both["matching_status"] == "clean") & (both["ramsey_status"] == "clean")]
            diff = both[other] - both[base]
            rows.append(dict(tier=tier, compared=base, compared_with=other, propositions=len(both),
                             both_clean=len(clean), corr=_corr(both[base], both[other]),
                             corr_both_clean=_corr(clean[base], clean[other]),
                             mean_difference=float(diff.mean()) if len(both) else np.nan,
                             mean_abs_difference=float(diff.abs().mean()) if len(both) else np.nan))
    return pd.DataFrame(rows)


def ramsey_by_matching(props: pd.DataFrame) -> pd.DataFrame:
    g = props[props["matching"].notna() & props["ramsey_status"].notna()].copy()
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0001]
    labels = ["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1"]
    g["matching_band"] = pd.cut(g["matching"], bins=bins, labels=labels, right=False, include_lowest=True)
    out = g.groupby("matching_band", observed=False).agg(
        propositions=("prop_id", "size"),
        share_ramsey_falling=("ramsey_falling", "mean"),
        share_ramsey_clean=("ramsey_status", lambda s: float((s == "clean").mean()) if len(s) else np.nan),
        share_ramsey_never_below=("ramsey_status", lambda s: float((s == "never_below").mean()) if len(s) else np.nan),
        mean_ramsey_ce=("ramsey_ce", "mean"),
        mean_ramsey_utility=("ramsey_utility", "mean"),
    ).reset_index()
    return out


def spot_check(props: pd.DataFrame, n_random: int = 10, n_extreme: int = 5, seed: int = SEED) -> pd.DataFrame:
    picks = []
    for est in ("matching", "ramsey_linear"):
        g = props[props[est].notna()]
        if g.empty:
            continue
        chosen = pd.concat([
            g.nlargest(n_extreme, est).assign(why="highest"),
            g.nsmallest(n_extreme, est).assign(why="lowest"),
        ])
        rest = g[~g["prop_id"].isin(chosen["prop_id"])]
        chosen = pd.concat([chosen, rest.sample(min(n_random, len(rest)), random_state=seed).assign(why="random")])
        picks.append(chosen.assign(method=est.split("_")[0]))
    cols = ["method", "why", "prop_id", "tier", "text", "truth", "objective_chance", "human_forecast",
            "matching", "matching_status", "ramsey_ce", "ramsey_status", "ramsey_utility"]
    return pd.concat(picks)[cols].reset_index(drop=True) if picks else pd.DataFrame(columns=cols)


# ---------------------------------------------------------------------------
# Report and figures
# ---------------------------------------------------------------------------


def f2(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.2f}"


def report(run: str, out: dict) -> str:
    s = out["summary_by_tier"]
    a = out["agreement"]
    lines = [f"# Credence analysis: {run}", "",
             "Credence = where the order-averaged choice curve crosses 0.5. Censored curves count at the end "
             "of the grid (see status).", "", "## Well-defined and coherent?", "",
             "| Estimate | Propositions | Clean crossing | Censored | Curves falling | Mean b(P)+b(not P) | "
             "Mean Dutch-book loss |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in s[(s["tier"] == "all") & s["estimate"].isin(BASE_ESTIMATES)].itertuples():
        lines.append(f"| {r.estimate} | {r.propositions} | {r.share_clean:.0%} | {r.share_censored:.0%} | "
                     f"{r.share_falling:.0%} | {f2(r.mean_sum_p_not_p)} | {f2(r.mean_dutch_book_loss)} |")
    lines += ["", "Pair-corrected estimates, b*(P) = (b(P) - b(not P) + 1) / 2, sum to 1 by construction; "
              "they appear below with the suffix _pair.", "", "## Calibration on the anchors", "",
              "| Estimate | Chance: MAE vs true chance | Chance: correlation | Facts: Brier | Facts: accuracy | "
              "Forecasts: MAE vs crowd | Forecasts: correlation |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for est in ESTIMATES:
        g = s[s["estimate"] == est].set_index("tier")
        if g.empty:
            continue
        ch, fa, fc = (g.loc[t] if t in g.index else None for t in ("chance", "fact", "forecast"))
        lines.append(f"| {est} | {f2(ch.mae_objective if ch is not None else None)} | "
                     f"{f2(ch.corr_objective if ch is not None else None)} | "
                     f"{f2(fa.brier_truth if fa is not None else None)} | "
                     f"{f2(fa.accuracy_truth if fa is not None else None)} | "
                     f"{f2(fc.mae_human if fc is not None else None)} | {f2(fc.corr_human if fc is not None else None)} |")
    lines += ["", "## Do the methods agree?", "", "| Tier | Compared | With | Propositions | Correlation | "
              "Mean difference (Ramsey - matching) |", "| --- | --- | --- | --- | --- | --- |"]
    for r in a.itertuples():
        lines.append(f"| {r.tier} | {r.compared} | {r.compared_with} | {r.propositions} | {f2(r.corr)} | "
                     f"{f2(r.mean_difference)} |")
    lines += ["", "## Where do Ramsey curves fall?", "", "| Matching credence | Propositions | Ramsey falling | "
              "Ramsey never below 0.5 | Mean certainty equivalent |", "| --- | --- | --- | --- | --- |"]
    for r in out["ramsey_by_matching"].itertuples():
        lines.append(f"| {r.matching_band} | {r.propositions} | {f2(r.share_ramsey_falling)} | "
                     f"{f2(r.share_ramsey_never_below)} | {'n/a' if pd.isna(r.mean_ramsey_ce) else f'£{r.mean_ramsey_ce:.0f}'} |")
    if out["utility_curve"]["utility"].notna().any():
        u = out["utility_curve"].set_index("amount")["utility"]
        lines += ["", "## Utility scale", "",
                  f"u(£500) = {u.loc[500]:.3f}, u(£900) = {u.loc[900]:.3f}, u(£990) = {u.loc[990]:.3f} "
                  "(linear would be 0.5, 0.9, 0.99)."]
    return "\n".join(lines) + "\n"


def make_figures(out: dict, fig_dir: Path) -> list[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: skipping figures (pip install matplotlib)")
        return []
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    curves, props = out["curves"], out["propositions"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, method in zip(axes, ["matching", "ramsey"]):
        c = curves[curves["method"] == method]
        for tier, g in c.groupby("tier"):
            m = g.groupby("x")["p_mean"].mean()
            ax.plot(m.index, m.values, marker=".", label=tier)
        ax.axhline(0.5, color="grey", lw=0.8, ls="--")
        ax.set_title(method)
        ax.set_xlabel("k/100" if method == "matching" else "sure amount / £1000")
    axes[0].set_ylabel("p(choose the bet on the statement)")
    axes[0].legend()
    fig.tight_layout()
    paths.append(fig_dir / "mean_curves_by_tier.png")
    fig.savefig(paths[-1], dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    ch = props[props["objective_chance"].notna()]
    for est, mk in (("matching", "o"), ("ramsey_linear", "x")):
        ax.scatter(ch["objective_chance"], ch[est], marker=mk, label=est, alpha=0.7)
    ax.plot([0, 1], [0, 1], color="grey", lw=0.8)
    ax.set_xlabel("true chance")
    ax.set_ylabel("measured credence")
    ax.legend()
    fig.tight_layout()
    paths.append(fig_dir / "calibration_chance.png")
    fig.savefig(paths[-1], dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    for tier, g in props.groupby("tier"):
        ax.scatter(g["matching"], g["ramsey_linear"], label=tier, alpha=0.6, s=12)
    ax.plot([0, 1], [0, 1], color="grey", lw=0.8)
    ax.set_xlabel("matching credence")
    ax.set_ylabel("Ramsey credence (linear utility)")
    ax.legend()
    fig.tight_layout()
    paths.append(fig_dir / "matching_vs_ramsey.png")
    fig.savefig(paths[-1], dpi=150)
    plt.close(fig)

    uc = out["utility_curve"]
    if uc["utility"].notna().any():
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(uc["amount"], uc["utility"], label="measured")
        ax.plot(uc["amount"], uc["linear"], color="grey", ls="--", label="linear")
        ax.set_xlabel("£")
        ax.set_ylabel("u")
        ax.legend()
        fig.tight_layout()
        paths.append(fig_dir / "utility_curve.png")
        fig.savefig(paths[-1], dpi=150)
        plt.close(fig)
    return paths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def analyse(claims: pd.DataFrame, matching: pd.DataFrame | None, ramsey: pd.DataFrame | None,
            utility_points: pd.DataFrame | None) -> dict:
    frames = [d for d in (matching, ramsey) if d is not None]
    if not frames:
        sys.exit("No matching or Ramsey results found for this run.")
    curves = curves_table(pd.concat(frames, ignore_index=True))
    fits = fits_table(curves)
    u = utility_function(utility_points)
    props = propositions_table(fits, claims, u)
    props = props[props["matching"].notna() | props["ramsey_ce"].notna()].reset_index(drop=True)
    coh = coherence_table(props)
    return {
        "curves": curves, "fits": fits, "propositions": props, "coherence": coh,
        "partitions": partitions_table(props), "utility_curve": utility_curve_table(u),
        "summary_by_tier": summary_by_tier(props, fits, coh), "agreement": agreement_table(props),
        "ramsey_by_matching": ramsey_by_matching(props), "spot_check": spot_check(props),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".", help="project folder")
    ap.add_argument("--results", help="results folder (default: <root>/results), e.g. results_8B")
    ap.add_argument("--run", help="run name (default: the only non-fake one)")
    ap.add_argument("--out-dir", help="default: data/analysis/<run>")
    ap.add_argument("--figures", action="store_true", help="also save PNG figures (needs matplotlib)")
    args = ap.parse_args(argv)

    root = Path(args.root)
    results = Path(args.results) if args.results else root / "results"
    run = sa.run_name(sa.find_run(results, args.run, prefix="matching"), prefix="matching")
    out = analyse(load_claims(root / "data" / "claims.csv"), load_method(results, "matching", run),
                  load_method(results, "ramsey", run), load_utility(results, run))
    out_dir = Path(args.out_dir) if args.out_dir else root / "data" / "analysis" / run
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, table in out.items():
        table.to_csv(out_dir / f"{name}.csv", index=False)
    text = report(run, out)
    (out_dir / "report.md").write_text(text, encoding="utf-8")
    figs = make_figures(out, out_dir / "figures") if args.figures else []
    print(text)
    print(f"Wrote {len(out) + 1 + len(figs)} files to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

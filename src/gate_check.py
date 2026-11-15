"""Gate check after the smoke test: is it worth running the full files for this model?

Friday's lesson: curves can fall in the right direction and still never reach
indifference. This checks the things that matter, on the smoke results:

  1. Dominance at the top of the grid: at k = 100 (a sure £1000, "whichever ticket is
     drawn") and c = £1000 (a sure £1000), the bet on the statement should LOSE:
     order-averaged p(statement bet) below 0.5.
  2. Crossings: the order-averaged curve should cross 0.5 inside the grid.
  3. Position bias: the mean gap between the two orders.
  4. If Stage 1 results are there too: dominance on Stage 1's checks.

Run on the pod (or locally), from the project folder:

    python src/gate_check.py --run llama-3-3-70b-instruct-fp8-dynamic

PASS means the matching method looks usable (Ramsey is reported, but not required).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import analysis as an

MIN_SHARE = 0.8  # share of smoke propositions that must pass each matching check


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for col in ("p_target", "p_A", "mass_outside"):
        df[col] = pd.to_numeric(df[col])
    df["grid_value"] = pd.to_numeric(df["grid_value"], errors="coerce")
    return df


GATE_RULE = ("v2: a proposition whose curve is censored in the opposite direction to its negation's (P never "
             "below 0.5 and not-P never above, or the reverse) counts as a confident belief, not a failure")


def confident_pairs(fits: pd.DataFrame) -> set[str]:
    """Propositions censored in the opposite direction to their negation: a confident, coherent belief.

    A model sure that P is true is indifferent at the top of the grid between "£1000 if P" and a sure £1000,
    so its curve never falls below 0.5 there; the negation's never rises above 0.5. Counting these as
    failures (gate rule v1) mistook certainty for a broken measure. The 8B's failure looked different: P and
    not-P both censored at the top.
    """
    out = set()
    for _, g in fits.groupby("claim_id"):
        st = dict(zip(g["neg"].astype(int), g["status"]))
        ids = dict(zip(g["neg"].astype(int), g["prop_id"]))
        if {st.get(0), st.get(1)} == {"never_below", "never_above"}:
            out |= set(ids.values())
    return out


def gate(smoke: pd.DataFrame, stage1: pd.DataFrame | None = None) -> dict:
    curves = an.curves_table(smoke[smoke["method"].isin(["matching", "ramsey"])])
    fits = an.fits_table(curves)
    out = {"methods": {}, "propositions": [], "rule": GATE_RULE}
    for method, f in fits.groupby("method"):
        confident = confident_pairs(f)
        top = curves[curves["method"] == method]
        top = top[top["grid_value"] == top["grid_value"].max()]
        out["methods"][method] = {
            "propositions": int(len(f)),
            "confident_beliefs": len(confident),
            "share_dominance_at_top": float(((top["p_mean"] < 0.5) | top["prop_id"].isin(confident)).mean()),
            "mean_p_at_top": float(top["p_mean"].mean()),
            "share_crossing": float((f["status"].isin(["clean", "multiple"]) | f["prop_id"].isin(confident)).mean()),
            "mean_abs_order_gap": float(f["mean_abs_order_gap"].mean()),
            "mean_mass_outside": float(f["mean_mass_outside"].mean()),
        }
        for r in f.itertuples():
            out["propositions"].append(dict(method=method, prop_id=r.prop_id, tier=r.tier,
                                            status="confident" if r.prop_id in confident else r.status,
                                            crossing=r.crossing, p_at_top=r.p_at_max))
    if stage1 is not None:
        dom = stage1[stage1["expected_label"] != ""]
        if len(dom):
            chosen = np.where(dom["p_A"] > 0.5, "A", "B")
            out["stage1_dominance"] = float((chosen == dom["expected_label"]).mean())
    m = out["methods"].get("matching", {})
    reasons = []
    if not m:
        reasons.append("no matching results in the smoke test")
    else:
        if m["share_dominance_at_top"] < MIN_SHARE:
            reasons.append("the bet on the statement still wins at the top of the matching grid")
        if m["share_crossing"] < MIN_SHARE:
            reasons.append("matching curves don't cross 0.5")
    if out.get("stage1_dominance", 1.0) < 0.9:
        reasons.append("Stage 1 dominance checks fail")
    out["reasons"] = reasons
    out["pass"] = not reasons
    return out


def report(out: dict, run: str) -> str:
    lines = [f"Gate check: {run}", ""]
    for method, s in out["methods"].items():
        lines.append(f"{method}: {s['propositions']} propositions | dominance at the top of the grid "
                     f"{s['share_dominance_at_top']:.0%} (mean p {s['mean_p_at_top']:.2f}, want < 0.5) | "
                     f"curves crossing 0.5 or confident {s['share_crossing']:.0%} ({s['confident_beliefs']} confident) | "
                     f"mean |order gap| {s['mean_abs_order_gap']:.2f}"
                     f" | outside A/B {s['mean_mass_outside']:.1%}")
    if "stage1_dominance" in out:
        lines.append(f"stage1: dominance respected in {out['stage1_dominance']:.0%} of single-order prompts")
    lines += ["", f"Gate rule {out['rule']}.", "", "Per proposition:"]
    for p in out["propositions"]:
        cross = "" if np.isnan(p["crossing"]) else f"{p['crossing']:.2f}"
        lines.append(f"  {p['method']:9} {p['tier']:9} {p['status']:12} crossing {cross:>5}  "
                     f"p at top {p['p_at_top']:.2f}  {p['prop_id']}")
    lines += ["", "PASS: worth running the full files." if out["pass"] else
              "FAIL: " + "; ".join(out["reasons"]) + ". Don't run the full files; copy results/ back and look first."]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="model short name, as in results/raw_calls_smoke_<run>.csv")
    ap.add_argument("--results", default="results")
    ap.add_argument("--file", default="smoke", help="which results to check: smoke (default), or e.g. matching "
                    "for a cut-down reasoning run")
    args = ap.parse_args(argv)
    results = Path(args.results)
    smoke_path = results / f"raw_calls_{args.file}_{args.run}.csv"
    if not smoke_path.exists():
        sys.exit(f"{smoke_path} not found; run the smoke test first")
    stage1_path = results / f"raw_calls_stage1_{args.run}.csv"
    out = gate(load(smoke_path), load(stage1_path) if stage1_path.exists() else None)
    print(report(out, args.run))
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

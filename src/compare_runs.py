"""Compare runs side by side: one column per model (or condition), one row per check.

Reads what stage1_analysis.py, analysis.py and utility_scale.py have already written, so
run those first for each run. From the project root:

    python src/compare_runs.py
    python src/compare_runs.py --runs ministral-3-8b-instruct-2512-bf16 qwen3-32b qwen3-32b-thinking

With no --runs, every folder in data/analysis/ that holds a summary_by_tier.csv is
included (fake runs left out). Utility-scale summaries are looked for in every
results* folder (results/, results_8B/, ...).

Outputs, in data/analysis/:
    comparison.csv   long format: run, section, check, value (for your own tables and plots)
    comparison.md    the same as one table: checks down the side, runs across the top
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

BASE = ["matching", "ramsey_linear", "ramsey_utility"]
PAIR = ["matching_pair", "ramsey_linear_pair", "ramsey_utility_pair"]
LABEL = {"matching": "Matching", "ramsey_linear": "Ramsey, linear utility", "ramsey_utility": "Ramsey, measured utility",
         "matching_pair": "Matching, pair-corrected", "ramsey_linear_pair": "Ramsey, pair-corrected",
         "ramsey_utility_pair": "Ramsey (measured utility), pair-corrected"}


def find_runs(analysis_dir: Path) -> list[str]:
    return sorted(p.name for p in analysis_dir.iterdir()
                  if (p / "summary_by_tier.csv").exists() and "fake" not in p.name)


def utility_summary(root: Path, run: str) -> dict | None:
    for folder in sorted(root.glob("results*")):
        path = folder / f"utility_summary_{run}.json"
        if path.exists():
            return json.loads(path.read_text())
    return None


def rows_for(root: Path, run: str) -> list[dict]:
    out_dir = root / "data" / "analysis" / run
    rows = []

    def add(section, check, value):
        rows.append(dict(run=run, section=section, check=check, value=value))

    dec_path = out_dir / "stage1_decisions.json"
    if dec_path.exists():
        d = json.loads(dec_path.read_text())
        ne, dom, tr, val = d["neutral_event"], d["dominance"], d.get("transitivity", {}), d.get("validation", {})
        add("Stage 1", "Neutral event N", f"{ne['event']}" + ("" if ne["qualifies"] else " (doesn't qualify)"))
        add("Stage 1", "Ticket scale (script's verdict)", d["ticket_symmetry"]["ticket_scale"])
        add("Stage 1", "Dominance, orders averaged", f"{dom['correct_order_averaged']} of {dom['checks']}")
        add("Stage 1", "Dominance, single orders", f"{dom['correct_single_orders']} of {dom['single_orders']}")
        add("Stage 1", "Tickets vs equal stated chance (mean p)", val.get("tickets_mean_p"))
        add("Stage 1", "Transitivity: cyclic triples", f"{tr.get('cyclic_triples')} of {tr.get('triples')}")
        add("Stage 1", "Sure amount chosen over a bet", tr.get("share_sure_vs_bet_choosing_sure"))
        add("Stage 1", "Pairs agreeing with expected value", tr.get("share_pairs_agreeing_with_ev"))

    u = utility_summary(root, run)
    if u:
        add("Utility scale", "m(1/2), £", u.get("m_half"))
        add("Utility scale", "Points found", f"{u.get('points_found')} of {u.get('points_attempted')}")
        add("Utility scale", "Power-curve rho (1 = linear)", u.get("rho"))

    s = pd.read_csv(out_dir / "summary_by_tier.csv")
    s = s.set_index(["estimate", "tier"])

    def get(est, tier, col):
        try:
            v = s.loc[(est, tier), col]
        except KeyError:
            return None
        return None if pd.isna(v) else float(v)

    for est in BASE:
        sec = LABEL[est]
        add(sec, "Clean crossing", get(est, "all", "share_clean"))
        add(sec, "Censored (never crosses)", get(est, "all", "share_censored"))
        add(sec, "Mean b(P) + b(not P)", get(est, "all", "mean_sum_p_not_p"))
        add(sec, "Chance claims: correlation with true chance", get(est, "chance", "corr_objective"))
        add(sec, "Chance claims: mean absolute error", get(est, "chance", "mae_objective"))
        add(sec, "Facts: accuracy", get(est, "fact", "accuracy_truth"))
        add(sec, "Forecasts: correlation with crowd", get(est, "forecast", "corr_human"))
    for est in PAIR:
        sec = LABEL[est]
        add(sec, "Chance claims: correlation with true chance", get(est, "chance", "corr_objective"))
        add(sec, "Chance claims: mean absolute error", get(est, "chance", "mae_objective"))
        add(sec, "Facts: accuracy", get(est, "fact", "accuracy_truth"))
        add(sec, "Forecasts: correlation with crowd", get(est, "forecast", "corr_human"))
    return rows


def fmt(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def table(long: pd.DataFrame, runs: list[str]) -> str:
    lines = ["| Section | Check | " + " | ".join(runs) + " |", "| --- | --- | " + " | ".join("---" for _ in runs) + " |"]
    order = long.drop_duplicates(["section", "check"])[["section", "check"]]
    wide = long.set_index(["section", "check", "run"])["value"]
    last = None
    for sec, chk in order.itertuples(index=False):
        cells = [fmt(wide.get((sec, chk, r))) for r in runs]
        lines.append(f"| {sec if sec != last else ''} | {chk} | " + " | ".join(cells) + " |")
        last = sec
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--runs", nargs="+", help="run names, in column order (default: all in data/analysis)")
    args = ap.parse_args(argv)
    root = Path(args.root)
    analysis_dir = root / "data" / "analysis"
    runs = args.runs or find_runs(analysis_dir)
    if not runs:
        raise SystemExit("No analysed runs found; run analysis.py first.")
    missing = [r for r in runs if not (analysis_dir / r / "summary_by_tier.csv").exists()]
    if missing:
        raise SystemExit(f"Not analysed yet: {missing}; run  python src/analysis.py --run <name>  for each.")
    long = pd.DataFrame([row for r in runs for row in rows_for(root, r)])
    long.to_csv(analysis_dir / "comparison.csv", index=False)
    md = "# Runs compared\n\n" + table(long, runs) + "\n"
    stale = [r for r in runs if long[(long["run"] == r) & long["section"].isin([LABEL[e] for e in PAIR])]
             ["value"].isna().all()]
    if stale:
        md += (f"\nNo pair-corrected estimates for {', '.join(stale)}: their analysis predates them. Rerun "
               "analysis.py for those runs (add --results results_8B for the 8B run).\n")
    (analysis_dir / "comparison.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"Wrote {analysis_dir / 'comparison.csv'} and {analysis_dir / 'comparison.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

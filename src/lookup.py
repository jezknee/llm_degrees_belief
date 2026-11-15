"""Look up the measured credences for any proposition, by a word or phrase in its text.

Run analysis.py first. Then, from the project root:

    python src/lookup.py "unemployment"
    python src/lookup.py "Canberra" --run qwen3-32b
    python src/lookup.py "die" --tier chance

For each matching proposition it prints the reference value (truth, true chance or
crowd forecast, where there is one) and every estimate with its curve status.

Wording: the values are "credences" only if the run holds a passing certificate from
credence_pipeline.py (reports/<run>/certificate.json). Otherwise they are "betting quotients": real,
reproducible measurements of where the model switches, but not shown to behave like degrees of
belief (coherent, tracking known chances), so not to be read as "the model believes p to degree x".


    Ramsey, pair-corrected   (b(P) - b(not P) + 1) / 2 with Ramsey's linear-utility credences:
                             the estimate to quote ("the model believes p to degree x")
    Ramsey, linear utility   c*/1000, the raw crossing
    Ramsey, measured utility u(c*), using the utility scale
    Matching                 k*/100, the raw crossing, and pair-corrected

Status: clean = the curve crossed 0.5 once; multiple = it wavered; never_below /
never_above = it never crossed (the value is the end of the grid, not a real crossing).
Single values are roughly +/- 5-10 points: the grid moves in steps of 0.05 and the two
option orders often disagree.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

import stage1_analysis as sa


def pct(x) -> str:
    return "  n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:5.0%}"


def reference(r) -> str:
    if not pd.isna(r.objective_chance):
        return f"true chance {r.objective_chance:.0%}"
    if r.truth in ("true", "false"):
        return f"in fact {r.truth}"
    if not pd.isna(r.human_forecast):
        return f"crowd forecast {r.human_forecast:.0%}"
    return "no known answer"


def describe(r, word: str = "credence") -> str:
    return "\n".join([
        f"{r.text}",
        f"  [{r.prop_id} | tier {r.tier} | {reference(r)}]",
        f"  {word.capitalize()} (Ramsey, pair-corrected)   {pct(r.ramsey_linear_pair)}",
        f"  Ramsey, linear utility    {pct(r.ramsey_linear)}   ({r.ramsey_status})",
        f"  Ramsey, measured utility  {pct(r.ramsey_utility)}",
        f"  Matching                  {pct(r.matching)}   ({r.matching_status});  pair-corrected {pct(r.matching_pair)}",
    ])


def certification(root: Path, run: str) -> tuple[bool, str]:
    """(certified, one-line status) from the pipeline's certificate, if there is one."""
    import json
    path = root / "reports" / run / "certificate.json"
    if not path.exists():
        return False, ("No certificate for this run (credence_pipeline.py check makes one): these are betting "
                       "quotients, not shown to be credences.")
    cert = json.loads(path.read_text())
    if cert.get("certified"):
        return True, f"Certified on {cert.get('date')}: these values can be read as the model's credences."
    return False, (f"NOT certified ({cert.get('stopped') or 'failed certification checks'}): these are betting "
                   "quotients, not credences.")


def find(props: pd.DataFrame, text: str, tier: str | None = None) -> pd.DataFrame:
    m = props[props["text"].str.contains(text, case=False, regex=False)]
    return m[m["tier"] == tier] if tier else m


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", help="word or phrase to search for in the propositions")
    ap.add_argument("--run", help="run name (default: the only non-fake run with matching results)")
    ap.add_argument("--tier", help="only this tier: forecast, kuq, fact, chance, partition")
    ap.add_argument("--root", default=".")
    ap.add_argument("--limit", type=int, default=20, help="show at most this many")
    args = ap.parse_args(argv)
    root = Path(args.root)
    run = args.run or sa.run_name(sa.find_run(root / "results", None, prefix="matching"), prefix="matching")
    path = root / "data" / "analysis" / run / "propositions.csv"
    if not path.exists():
        sys.exit(f"{path} not found; run  python src/analysis.py --run {run}  first")
    props = pd.read_csv(path, keep_default_na=False, na_values={c: [""] for c in (
        "objective_chance", "human_forecast", "matching", "ramsey_linear", "ramsey_utility", "matching_pair",
        "ramsey_linear_pair", "ramsey_utility_pair")})
    if "ramsey_linear_pair" not in props.columns:
        sys.exit(f"{path} predates the pair-corrected estimates; rerun analysis.py")
    hits = find(props, args.text, args.tier)
    certified, status = certification(root, run)
    word = "credence" if certified else "betting quotient"
    print(f"Run: {run}. {status}\n{len(hits)} proposition(s) contain {args.text!r}.\n")
    for r in hits.head(args.limit).itertuples():
        print(describe(r, word) + "\n")
    if len(hits) > args.limit:
        print(f"... {len(hits) - args.limit} more; narrow the search or raise --limit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

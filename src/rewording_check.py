"""Rewording check: is the Ramsey failure caused by the words "for certain"?

The Ramsey prompts offer "You receive £c for certain." The model almost never takes
it. This check reruns every Ramsey prompt with only that option reworded, so the
two runs can be compared proposition by proposition.

Variants:
    tickets   (default) adds the matching method's ticket setup and offers
              "You receive £c, whichever ticket is drawn."  - the same wording the
              matching method uses at k = 100, where the model behaves well.
    whatever  no setup; offers "You receive £c, whatever happens."  - the smallest change.

Step 1, locally: write the prompts (no API, free)

    python src/rewording_check.py prompts                 # -> prompts/ramsey_reworded.csv
    python src/rewording_check.py prompts --variant whatever   # -> prompts/ramsey_whatever.csv

Step 2, on the pod: run them like any other prompts file

    python src/run_prompts.py prompts/ramsey_reworded.csv --model ... (same flags as before)

Step 3, locally, after copying results/ back: compare with the original Ramsey run

    python src/rewording_check.py compare
    python src/rewording_check.py compare --variant whatever --run ministral-3-8b-instruct-2512-bf16

Comparison outputs, in data/analysis/<run>/:
    rewording_<variant>_propositions.csv  proposition: matching credence, original and reworded crossings
    rewording_<variant>_summary.csv       wording x tier: curves falling, clean, censored, agreement with matching
    rewording_<variant>_curves.csv        wording x tier x sure amount: mean p(bet) (plot-ready)
    rewording_<variant>_report.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import analysis as an
import make_prompts as mp
import stage1_analysis as sa

VARIANTS = {
    "tickets": dict(file="ramsey_reworded", setup=mp.TICKETS_SETUP),
    "whatever": dict(file="ramsey_whatever", setup=""),
}


def sure_text(c: int, variant: str) -> str:
    if variant == "tickets":
        if c <= 0:
            return "You receive nothing, whichever ticket is drawn."
        return f"You receive {mp.money(c)}, whichever ticket is drawn."
    if variant == "whatever":
        if c <= 0:
            return "You receive nothing, whatever happens."
        return f"You receive {mp.money(c)}, whatever happens."
    raise SystemExit(f"unknown variant {variant!r}; choose from {list(VARIANTS)}")


def reworded_prompts(claims: pd.DataFrame, variant: str = "tickets", c_values=mp.C_VALUES) -> pd.DataFrame:
    """The Ramsey prompts with only the sure option (and, for 'tickets', the setup) changed."""
    setup = VARIANTS[variant]["setup"]
    rows = []
    for _, row in claims.iterrows():
        target = mp.claim_bet(row["text"])
        settle = mp.settlement_sentence(row)
        for c in c_values:
            base = mp.claim_base(row, "ramsey", c, "gbp")
            base["prompt_id"] = f"ramsey_{variant}_{row['prop_id']}_{c:04d}"
            base["step"] = f"reworded_{variant}"
            rows += mp.choice_rows(base, target, sure_text(c, variant), setup=setup, settle=settle)
    return pd.DataFrame(rows, columns=mp.COLUMNS)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def compare(claims: pd.DataFrame, matching: pd.DataFrame, original: pd.DataFrame,
            reworded: pd.DataFrame) -> dict:
    curves = {"original": an.curves_table(original), "reworded": an.curves_table(reworded)}
    fits = {k: an.fits_table(v) for k, v in curves.items()}
    m = an.fits_table(an.curves_table(matching)).set_index("prop_id")

    props = claims[["prop_id", "claim_id", "neg", "tier", "text", "truth", "objective_chance"]].copy()
    props["matching"] = props["prop_id"].map(m["crossing"])
    for k, f in fits.items():
        f = f.set_index("prop_id")
        props[f"{k}_ce"] = props["prop_id"].map(f["crossing"]) * an.PRIZE
        props[f"{k}_status"] = props["prop_id"].map(f["status"])
        props[f"{k}_falling"] = props["prop_id"].map(f["falling"])
        props[f"{k}_p_at_1000"] = props["prop_id"].map(f["p_at_max"])
        props[f"{k}_order_gap"] = props["prop_id"].map(f["mean_abs_order_gap"])
    props = props[props["original_ce"].notna() & props["reworded_ce"].notna()].reset_index(drop=True)
    props["ce_change"] = props["reworded_ce"] - props["original_ce"]

    rows = []
    for wording in ("original", "reworded"):
        for tier in ["all"] + sorted(props["tier"].unique()):
            g = props if tier == "all" else props[props["tier"] == tier]
            lin = g[f"{wording}_ce"] / an.PRIZE
            sums = _pair_sums(g, lin)
            rows.append(dict(
                wording=wording, tier=tier, propositions=len(g),
                share_falling=float(g[f"{wording}_falling"].mean()),
                share_clean=float((g[f"{wording}_status"] == "clean").mean()),
                share_never_below=float((g[f"{wording}_status"] == "never_below").mean()),
                share_takes_bet_over_1000=float((g[f"{wording}_p_at_1000"] > 0.5).mean()),
                mean_abs_order_gap=float(g[f"{wording}_order_gap"].mean()),
                mean_certainty_equivalent=float(g[f"{wording}_ce"].mean()),
                corr_with_matching=an._corr(lin, g["matching"]),
                mean_abs_diff_from_matching=float((lin - g["matching"]).abs().mean()),
                mean_sum_p_not_p=float(np.mean(sums)) if sums else np.nan,
            ))
    summary = pd.DataFrame(rows)

    mean_curves = []
    for wording, c in curves.items():
        c = c[c["prop_id"].isin(props["prop_id"])]
        for tier, g in [("all", c)] + list(c.groupby("tier")):
            mc = g.groupby("grid_value")[["p_first", "p_second", "p_mean"]].mean().reset_index()
            mean_curves.append(mc.assign(wording=wording, tier=tier))
    mean_curves = pd.concat(mean_curves, ignore_index=True)[
        ["wording", "tier", "grid_value", "p_first", "p_second", "p_mean"]]
    return {"propositions": props, "summary": summary, "curves": mean_curves}


def _pair_sums(g: pd.DataFrame, lin: pd.Series) -> list[float]:
    g = g.assign(_lin=lin)
    sums = []
    for _, pair in g.groupby("claim_id"):
        if sorted(pair["neg"]) == [0, 1] and pair["_lin"].notna().all():
            sums.append(float(pair["_lin"].sum()))
    return sums


def report(run: str, variant: str, out: dict) -> str:
    s = out["summary"]
    allrows = s[s["tier"] == "all"].set_index("wording")
    o, r = allrows.loc["original"], allrows.loc["reworded"]
    example = sure_text(500, variant)
    lines = [f"# Rewording check ({variant}): {run}", "",
             f'Original sure option: "You receive £500 for certain."  Reworded: "{example}"', "",
             "| | Original | Reworded |", "| --- | --- | --- |"]
    for label, col, pct in [
        ("Curves falling", "share_falling", True),
        ("Clean crossing", "share_clean", True),
        ("Never below 0.5 (bet beats every sure amount)", "share_never_below", True),
        ("Bet chosen over a sure £1000", "share_takes_bet_over_1000", True),
        ("Mean absolute order gap", "mean_abs_order_gap", False),
        ("Mean certainty equivalent (£)", "mean_certainty_equivalent", False),
        ("Correlation with matching (c*/1000)", "corr_with_matching", False),
        ("Mean absolute difference from matching", "mean_abs_diff_from_matching", False),
        ("Mean b(P) + b(not P)", "mean_sum_p_not_p", False),
    ]:
        fmt = (lambda v: f"{v:.0%}") if pct else (lambda v: f"{v:.0f}" if abs(v) > 10 else f"{v:.2f}")
        lines.append(f"| {label} | {fmt(o[col])} | {fmt(r[col])} |")
    lines += ["", "## By tier: curves falling", "", "| Tier | Original | Reworded |", "| --- | --- | --- |"]
    by = s[s["tier"] != "all"].pivot(index="tier", columns="wording", values="share_falling")
    for tier, row in by.iterrows():
        lines.append(f"| {tier} | {row['original']:.0%} | {row['reworded']:.0%} |")
    verdict = ("The wording matters: far more curves fall once 'for certain' is gone."
               if r["share_falling"] - o["share_falling"] >= 0.2 else
               "The wording makes little difference: the failure is not about the words 'for certain'.")
    lines += ["", f"**Reading:** {verdict}"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["prompts", "compare"])
    ap.add_argument("--variant", choices=list(VARIANTS), default="tickets")
    ap.add_argument("--root", default=".", help="project folder")
    ap.add_argument("--run", help="run name for compare (default: the only non-fake one)")
    args = ap.parse_args(argv)
    root = Path(args.root)
    name = VARIANTS[args.variant]["file"]

    if args.action == "prompts":
        claims = mp.load_claims(root / "data" / "claims.csv")
        df = reworded_prompts(claims, args.variant)
        out = root / "prompts" / f"{name}.csv"
        out.parent.mkdir(exist_ok=True)
        df.to_csv(out, index=False)
        print(f"Wrote {out}: {len(df)} prompts ({df['prop_id'].nunique()} propositions x "
              f"{len(mp.C_VALUES)} amounts x 2 orders).")
        print(f'Example option: "{sure_text(500, args.variant)}"')
        print(f"Copy it to the pod and run:  python src/run_prompts.py prompts/{name}.csv --model ... ")
        return 0

    results = root / "results"
    run = sa.run_name(sa.find_run(results, args.run, prefix="matching"), prefix="matching")
    loaded = {}
    for key, stem in (("matching", "matching"), ("original", "ramsey"), ("reworded", name)):
        path = results / f"raw_calls_{stem}_{run}.csv"
        if not path.exists():
            sys.exit(f"{path} not found; run prompts/{stem}.csv on the pod and copy results/ back.")
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        for col in ("p_target", "mass_outside"):
            df[col] = pd.to_numeric(df[col])
        df["grid_value"] = pd.to_numeric(df["grid_value"])
        loaded[key] = df
    out = compare(an.load_claims(root / "data" / "claims.csv"), loaded["matching"], loaded["original"],
                  loaded["reworded"])
    out_dir = root / "data" / "analysis" / run
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, table in out.items():
        table.to_csv(out_dir / f"rewording_{args.variant}_{key}.csv", index=False)
    text = report(run, args.variant, out)
    (out_dir / f"rewording_{args.variant}_report.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"Wrote 4 files to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

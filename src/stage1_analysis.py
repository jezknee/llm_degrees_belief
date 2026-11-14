"""Stage 1 analysis: neutral event, ticket symmetry, validation, dominance, transitivity.

Reads the Stage 1 results copied back from the pod and writes plot-ready tables,
a short report and the decisions later scripts need (the neutral event N, and
whether tickets can be read as k/100).

Run from the project root, after copying results/ back:

    python src/stage1_analysis.py
    python src/stage1_analysis.py --run ministral-3-8b-instruct-2512-bf16 --tol 0.1

Outputs, in data/analysis/<run>/:

    stage1_choices.csv             every choice, both orders and their mean (the master table)
    stage1_neutral.csv             each candidate event x outcome pair
    stage1_neutral_summary.csv     one row per candidate, with the neutrality verdict
    stage1_symmetry.csv            single-ticket and block pairs
    stage1_validation.csv          tickets vs stated chances, and each candidate vs a 50% chance
    stage1_dominance.csv           each dominance check
    stage1_transitivity_pairs.csv  each pairwise choice, with expected values
    stage1_transitivity_triples.csv  each triple: cyclic or not, agrees with expected value or not
    stage1_position_bias.csv       order gap by step
    stage1_decisions.json          N, ticket scale, and the headline numbers
    stage1_report.md               the same in words

All p values are the probability of choosing the "target" option (the one named
first in the prompt design), renormalised over A and B. p_mean averages the two
orders, which removes position bias to first order.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_TOL = 0.1  # |p - 0.5| allowed for "indifferent"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def find_run(results_dir: Path, run: str | None, prefix: str = "stage1") -> Path:
    """The results file for this run; with no --run, the only non-fake one."""
    if run:
        path = results_dir / f"raw_calls_{prefix}_{run}.csv"
        if not path.exists():
            sys.exit(f"{path} not found")
        return path
    files = sorted(results_dir.glob(f"raw_calls_{prefix}_*.csv"))
    real = [f for f in files if not f.stem.endswith("_fake")]
    candidates = real or files
    if not candidates:
        sys.exit(f"No raw_calls_{prefix}_*.csv in {results_dir}; copy results/ back from the pod first.")
    if len(candidates) > 1:
        names = ", ".join(f.stem.removeprefix(f"raw_calls_{prefix}_") for f in candidates)
        sys.exit(f"Several runs found ({names}); choose one with --run.")
    return candidates[0]


def run_name(path: Path, prefix: str = "stage1") -> str:
    return path.stem.removeprefix(f"raw_calls_{prefix}_")


def load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df = df[df["method"] == "stage1"].copy()
    if df.empty:
        sys.exit(f"{path} has no Stage 1 rows")
    for col in ("p_A", "p_target", "mass_outside"):
        df[col] = pd.to_numeric(df[col])
    return df


def attach_descriptions(df: pd.DataFrame, prompts_path: Path) -> pd.DataFrame:
    if not prompts_path.exists():
        df["target_desc"] = ""
        df["other_desc"] = ""
        return df
    p = pd.read_csv(prompts_path, dtype=str, keep_default_na=False)[["prompt_id", "target_desc", "other_desc"]]
    return df.merge(p, on="prompt_id", how="left").fillna({"target_desc": "", "other_desc": ""})


# ---------------------------------------------------------------------------
# One row per choice, both orders side by side
# ---------------------------------------------------------------------------


def choice_id(prompt_id: str) -> str:
    """'stage1_neutral_die_1000_0_tf' -> 'neutral_die_1000_0'."""
    return re.sub(r"_t[fs]$", "", prompt_id).removeprefix("stage1_")


def choices_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["choice_id"] = df["prompt_id"].map(choice_id)
    wide = df.pivot_table(index="choice_id", columns="order", values="p_target", aggfunc="mean")
    wide = wide.rename(columns={"target_first": "p_first", "target_second": "p_second"})
    for col in ("p_first", "p_second"):
        if col not in wide:
            wide[col] = np.nan
    meta = df.drop_duplicates("choice_id").set_index("choice_id")
    out = pd.DataFrame({
        "step": meta["step"],
        "item_id": meta["item_id"],
        "choice_id": meta.index,
        "target_desc": meta.get("target_desc", ""),
        "other_desc": meta.get("other_desc", ""),
        "expected_target": meta["expected_label"].ne(""),
    }).join(wide[["p_first", "p_second"]])
    out["p_mean"] = out[["p_first", "p_second"]].mean(axis=1)
    out["order_gap"] = out["p_first"] - out["p_second"]
    out["chose_target"] = out["p_mean"] > 0.5
    out["mass_outside"] = df.groupby("choice_id")["mass_outside"].mean()
    return out.reset_index(drop=True).sort_values(["step", "choice_id"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 1. Neutral event
# ---------------------------------------------------------------------------


def neutral_tables(ch: pd.DataFrame, tol: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = ch[ch["step"] == "neutral"].copy()
    parts = n["choice_id"].str.extract(r"^neutral_(?P<candidate>[a-z]+)_(?P<hi>\d+)_(?P<lo>\d+)$")
    n = pd.concat([n, parts], axis=1)
    n["hi"], n["lo"] = n["hi"].astype(int), n["lo"].astype(int)
    n["deviation"] = (n["p_mean"] - 0.5).abs()
    n["indifferent"] = n["deviation"] <= tol
    table = n[["candidate", "hi", "lo", "p_first", "p_second", "p_mean", "order_gap", "deviation",
               "indifferent", "target_desc", "other_desc"]].sort_values(["candidate", "hi", "lo"])

    val = ch[ch["step"] == "validation_neutral"].copy()
    val["candidate"] = val["choice_id"].str.removeprefix("validation_neutral_")
    val = val.set_index("candidate")["p_mean"]

    rows = []
    for cand, g in table.groupby("candidate"):
        rows.append({
            "candidate": cand,
            "pairs": len(g),
            "mean_p": g["p_mean"].mean(),
            "max_deviation": g["deviation"].max(),
            "mean_abs_order_gap": g["order_gap"].abs().mean(),
            "all_pairs_indifferent": bool(g["indifferent"].all()),
            "p_vs_stated_50pct": val.get(cand, np.nan),
        })
    summary = pd.DataFrame(rows).sort_values(["max_deviation", "mean_abs_order_gap"]).reset_index(drop=True)
    summary["rank"] = np.arange(1, len(summary) + 1)
    return table.reset_index(drop=True), summary


def choose_neutral(summary: pd.DataFrame) -> dict:
    if summary.empty:
        return {"event": None, "qualifies": False, "max_deviation": None, "note": "no neutral rows"}
    best = summary.iloc[0]
    ok = summary[summary["all_pairs_indifferent"]]
    if len(ok):
        pick = ok.iloc[0]
        return {"event": pick["candidate"], "qualifies": True, "max_deviation": float(pick["max_deviation"]),
                "note": "indifferent for every outcome pair"}
    return {"event": best["candidate"], "qualifies": False, "max_deviation": float(best["max_deviation"]),
            "note": "no candidate is indifferent for every outcome pair; this is the closest"}


# ---------------------------------------------------------------------------
# 2. Ticket symmetry
# ---------------------------------------------------------------------------


def symmetry_table(ch: pd.DataFrame, tol: float) -> pd.DataFrame:
    s = ch[ch["step"].isin(["symmetry_single", "symmetry_block"])].copy()
    single = s["choice_id"].str.extract(r"^symmetry_single_(\d+)_(\d+)$").astype(float)
    block = s["choice_id"].str.extract(r"^symmetry_block_(\d+)_(\d+)_vs_(\d+)_(\d+)$").astype(float)
    s["target_lo"] = single[0].fillna(block[0])
    s["target_hi"] = single[0].fillna(block[1])
    s["other_lo"] = single[1].fillna(block[2])
    s["other_hi"] = single[1].fillna(block[3])
    s["distance"] = s["other_lo"] - s["target_lo"]
    s["deviation"] = (s["p_mean"] - 0.5).abs()
    s["indifferent"] = s["deviation"] <= tol
    cols = ["step", "choice_id", "target_lo", "target_hi", "other_lo", "other_hi", "distance",
            "p_first", "p_second", "p_mean", "order_gap", "deviation", "indifferent"]
    return s[cols].reset_index(drop=True)


def symmetry_verdict(sym: pd.DataFrame, tol: float) -> dict:
    single = sym[sym["step"] == "symmetry_single"]
    blocks = sym[sym["step"] == "symmetry_block"]
    out = {
        "single_pairs": int(len(single)),
        "single_share_indifferent": float(single["indifferent"].mean()) if len(single) else None,
        "single_mean_p_lower_ticket": float(single["p_mean"].mean()) if len(single) else None,
        "single_corr_with_distance": _corr(single["distance"], single["p_mean"]),
        "blocks": {r.choice_id: round(float(r.p_mean), 3) for r in blocks.itertuples()},
        "blocks_all_indifferent": bool(blocks["indifferent"].all()) if len(blocks) else None,
    }
    ok_single = len(single) == 0 or (out["single_share_indifferent"] >= 0.8
                                     and abs(out["single_mean_p_lower_ticket"] - 0.5) <= tol / 2)
    ok_blocks = len(blocks) == 0 or out["blocks_all_indifferent"]
    out["ticket_scale"] = "k/100" if ok_single and ok_blocks else "measured"
    out["note"] = ("tickets treated as exchangeable: read ticket bets as k/100" if out["ticket_scale"] == "k/100"
                   else "tickets not treated as exchangeable: matching needs a measured reference curve")
    return out


def _mean_or_none(x: pd.Series) -> float | None:
    x = pd.to_numeric(x).dropna()
    return float(x.mean()) if len(x) else None


def _corr(x: pd.Series, y: pd.Series) -> float | None:
    x, y = pd.to_numeric(x), pd.to_numeric(y)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


# ---------------------------------------------------------------------------
# 3. Validation: tickets vs stated chances
# ---------------------------------------------------------------------------


def validation_table(ch: pd.DataFrame, tol: float) -> pd.DataFrame:
    v = ch[ch["step"].isin(["validation_tickets", "validation_neutral"])].copy()
    k = v["choice_id"].str.extract(r"^validation_tickets_(\d+)$")[0]
    v["k"] = pd.to_numeric(k, errors="coerce")
    v["candidate"] = v["choice_id"].where(v["step"] == "validation_neutral", "").str.removeprefix(
        "validation_neutral_")
    v["deviation"] = (v["p_mean"] - 0.5).abs()
    v["indifferent"] = v["deviation"] <= tol
    cols = ["step", "choice_id", "k", "candidate", "p_first", "p_second", "p_mean", "order_gap",
            "deviation", "indifferent", "target_desc", "other_desc"]
    return v[cols].sort_values(["step", "k", "candidate"]).reset_index(drop=True)


def validation_verdict(val: pd.DataFrame) -> dict:
    t = val[val["step"] == "validation_tickets"]
    if t.empty:
        return {}
    return {
        "tickets_mean_p": float(t["p_mean"].mean()),
        "tickets_min_p": float(t["p_mean"].min()),
        "tickets_max_p": float(t["p_mean"].max()),
        "tickets_share_indifferent": float(t["indifferent"].mean()),
        "tickets_corr_with_k": _corr(t["k"], t["p_mean"]),
        "reading": validation_reading(t["p_mean"].mean()),
    }


def validation_reading(mean_p: float, tol: float = DEFAULT_TOL) -> str:
    if abs(mean_p - 0.5) <= tol / 2:
        return "treats tickets and an equal stated chance as equivalent"
    return ("prefers tickets to an equal stated chance" if mean_p > 0.5
            else "prefers a stated chance to equal tickets")


# ---------------------------------------------------------------------------
# 4. Dominance
# ---------------------------------------------------------------------------


def dominance_table(ch: pd.DataFrame) -> pd.DataFrame:
    d = ch[ch["step"] == "dominance"].copy()
    d["correct_first"] = d["p_first"] > 0.5
    d["correct_second"] = d["p_second"] > 0.5
    d["correct_mean"] = d["p_mean"] > 0.5
    cols = ["choice_id", "p_first", "p_second", "p_mean", "order_gap", "correct_first", "correct_second",
            "correct_mean", "target_desc", "other_desc"]
    return d[cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 5. Transitivity
# ---------------------------------------------------------------------------


def expected_value(option: str) -> tuple[float, bool]:
    """'sure450' -> (450, True); 't70x700' -> (490, False)."""
    m = re.fullmatch(r"sure(\d+)", option)
    if m:
        return float(m.group(1)), True
    m = re.fullmatch(r"t(\d+)x(\d+)", option)
    if m:
        return int(m.group(1)) / 100 * int(m.group(2)), False
    raise ValueError(f"unknown transitivity option {option!r}")


def transitivity_tables(ch: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    t = ch[ch["step"] == "transitivity"].copy()
    parts = t["choice_id"].str.extract(r"^transitivity_(?P<triple>\d+)_(?P<x>[a-z0-9]+)_vs_(?P<y>[a-z0-9]+)$")
    t = pd.concat([t, parts], axis=1)
    if t.empty:
        return pd.DataFrame(), pd.DataFrame()
    for side in ("x", "y"):
        ev = [expected_value(o) for o in t[side]]
        t[f"ev_{side}"] = [e for e, _ in ev]
        t[f"{side}_sure"] = [s for _, s in ev]
    t["prefers"] = np.select([t["p_mean"] > 0.5, t["p_mean"] < 0.5], [t["x"], t["y"]], "")  # "" = exact tie
    t["higher_ev"] = np.where(t["ev_x"] > t["ev_y"], t["x"], np.where(t["ev_y"] > t["ev_x"], t["y"], ""))
    t["agrees_with_ev"] = np.where((t["higher_ev"] == "") | (t["prefers"] == ""), np.nan,
                                   (t["prefers"] == t["higher_ev"]).astype(float))
    t["kind"] = np.select([t["x_sure"] & t["y_sure"], t["x_sure"] | t["y_sure"]],
                          ["sure_vs_sure", "sure_vs_bet"], "bet_vs_bet")
    sure = np.where(t["x_sure"] & ~t["y_sure"], t["x"], np.where(t["y_sure"] & ~t["x_sure"], t["y"], ""))
    t["chose_sure"] = np.where((sure == "") | (t["prefers"] == ""), np.nan, (t["prefers"] == sure).astype(float))
    pair_cols = ["triple", "x", "y", "ev_x", "ev_y", "kind", "p_first", "p_second", "p_mean", "order_gap",
                 "prefers", "higher_ev", "agrees_with_ev", "chose_sure"]
    pairs = t[pair_cols].sort_values(["triple", "x", "y"]).reset_index(drop=True)

    rows = []
    for triple, g in pairs.groupby("triple"):
        opts = sorted(set(g["x"]) | set(g["y"]))
        beats = {(r.x, r.y) if r.p_mean > 0.5 else (r.y, r.x) for r in g.itertuples() if r.p_mean != 0.5}
        cyclic = is_cyclic(opts, beats)
        rows.append({
            "triple": triple,
            "options": " / ".join(opts),
            "pairs": len(g),
            "cyclic": cyclic,
            "order": "" if cyclic else " > ".join(rank_order(opts, beats)),
            "pairs_agreeing_with_ev": float(g["agrees_with_ev"].sum()),
            "pairs_with_ev_difference": int(g["agrees_with_ev"].notna().sum()),
            "min_margin": float((g["p_mean"] - 0.5).abs().min()),
        })
    return pairs, pd.DataFrame(rows)


def is_cyclic(options: list[str], beats: set[tuple[str, str]]) -> bool:
    """True if the strict preferences among three options form a cycle."""
    for a, b, c in itertools.permutations(options, 3):
        if (a, b) in beats and (b, c) in beats and (c, a) in beats:
            return True
    return False


def rank_order(options: list[str], beats: set[tuple[str, str]]) -> list[str]:
    wins = {o: sum((o, p) in beats for p in options) for o in options}
    return sorted(options, key=lambda o: -wins[o])


def transitivity_verdict(pairs: pd.DataFrame, triples: pd.DataFrame) -> dict:
    if triples.empty:
        return {}
    repeated = pairs.groupby(["x", "y"])["prefers"].agg(["nunique", "size"])
    repeated = repeated[repeated["size"] > 1]
    sure_vs_bet = pairs[pairs["kind"] == "sure_vs_bet"]
    return {
        "triples": int(len(triples)),
        "cyclic_triples": int(triples["cyclic"].sum()),
        "share_pairs_agreeing_with_ev": _mean_or_none(pairs["agrees_with_ev"]),
        "share_sure_vs_bet_choosing_sure": _mean_or_none(sure_vs_bet["chose_sure"]),
        "repeated_pairs": int(len(repeated)),
        "repeated_pairs_inconsistent": int((repeated["nunique"] > 1).sum()),
    }


# ---------------------------------------------------------------------------
# Position bias, decisions, report
# ---------------------------------------------------------------------------


def position_bias_table(ch: pd.DataFrame) -> pd.DataFrame:
    ch = ch.assign(abs_gap=ch["order_gap"].abs(),
                   flips=((ch["p_first"] > 0.5) != (ch["p_second"] > 0.5)).astype(float))
    g = ch.groupby("step")
    return pd.DataFrame({
        "choices": g.size(),
        "mean_p_first": g["p_first"].mean(),
        "mean_p_second": g["p_second"].mean(),
        "mean_order_gap": g["order_gap"].mean(),
        "mean_abs_order_gap": g["abs_gap"].mean(),
        "share_order_flips": g["flips"].mean(),
        "mean_mass_outside": g["mass_outside"].mean(),
    }).reset_index()


def analyse(df: pd.DataFrame, tol: float = DEFAULT_TOL) -> dict:
    ch = choices_table(df)
    neutral, neutral_summary = neutral_tables(ch, tol)
    sym = symmetry_table(ch, tol)
    val = validation_table(ch, tol)
    dom = dominance_table(ch)
    pairs, triples = transitivity_tables(ch)
    bias = position_bias_table(ch)
    decisions = {
        "tolerance": tol,
        "neutral_event": choose_neutral(neutral_summary),
        "ticket_symmetry": symmetry_verdict(sym, tol),
        "validation": validation_verdict(val),
        "dominance": {
            "checks": int(len(dom)),
            "correct_order_averaged": int(dom["correct_mean"].sum()),
            "correct_single_orders": int(dom["correct_first"].sum() + dom["correct_second"].sum()),
            "single_orders": int(2 * len(dom)),
            "failures": dom.loc[~dom["correct_mean"], "choice_id"].tolist(),
        },
        "transitivity": transitivity_verdict(pairs, triples),
        "position_bias_overall": float(ch["order_gap"].mean()),
    }
    return {
        "stage1_choices": ch, "stage1_neutral": neutral, "stage1_neutral_summary": neutral_summary,
        "stage1_symmetry": sym, "stage1_validation": val, "stage1_dominance": dom,
        "stage1_transitivity_pairs": pairs, "stage1_transitivity_triples": triples,
        "stage1_position_bias": bias, "decisions": decisions,
    }


def fmt(x, digits=2) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{digits}f}"


def pct(x) -> str:
    return "n/a" if x is None else f"{x:.0%}"


def report(out: dict, run: str) -> str:
    d = out["decisions"]
    ne, sym, val, dom, tr = (d["neutral_event"], d["ticket_symmetry"], d["validation"], d["dominance"],
                             d["transitivity"])
    lines = [f"# Stage 1 results: {run}", "",
             f"Indifference tolerance: |p - 0.5| <= {d['tolerance']}. p = probability of choosing the target "
             "option, averaged over both orders.", "",
             "## 1. Neutral event", ""]
    ns = out["stage1_neutral_summary"]
    for r in ns.itertuples():
        lines.append(f"- {r.candidate}: mean p {fmt(r.mean_p)}, worst pair off by {fmt(r.max_deviation)}, "
                     f"mean |order gap| {fmt(r.mean_abs_order_gap)}, vs stated 50%: {fmt(r.p_vs_stated_50pct)}"
                     f"{' (indifferent for every pair)' if r.all_pairs_indifferent else ''}")
    share = sym["single_share_indifferent"]
    share_txt = "n/a" if share is None else f"{share:.0%}"
    lines += ["", f"**N = {ne['event']}** ({ne['note']}).", "", "## 2. Ticket symmetry", "",
              f"- Single tickets: {sym['single_pairs']} pairs, {share_txt} "
              f"indifferent; mean p of the lower-numbered ticket {fmt(sym['single_mean_p_lower_ticket'])}; "
              f"correlation with distance {fmt(sym['single_corr_with_distance'])}.",
              "- Blocks: " + ", ".join(f"{k} {v}" for k, v in sym["blocks"].items()) + ".", "",
              f"**Ticket scale: {sym['ticket_scale']}** ({sym['note']}).", "",
              "## 3. Validation: tickets vs stated chances", ""]
    if val:
        lines += [f"- Mean p(tickets) {fmt(val['tickets_mean_p'])} (range {fmt(val['tickets_min_p'])} to "
                  f"{fmt(val['tickets_max_p'])}); correlation with k {fmt(val['tickets_corr_with_k'])}.",
                  f"- Reading: the model {val['reading']}. A roughly constant preference across k is a framing "
                  "effect, not a distortion of the ticket scale."]
    lines += ["", "## 4. Dominance", "",
              f"- {dom['correct_order_averaged']} of {dom['checks']} correct after averaging orders; "
              f"{dom['correct_single_orders']} of {dom['single_orders']} single-order prompts correct.",
              f"- Failures: {', '.join(dom['failures']) or 'none'}.", "", "## 5. Transitivity", ""]
    if tr:
        lines += [f"- {tr['cyclic_triples']} of {tr['triples']} triples cyclic.",
                  f"- {pct(tr['share_pairs_agreeing_with_ev'])} of pairs with different expected values "
                  "prefer the higher one.",
                  f"- Sure amount vs bet: the sure amount is chosen in "
                  f"{pct(tr['share_sure_vs_bet_choosing_sure'])} of such pairs.",
                  f"- Pairs asked in more than one triple: {tr['repeated_pairs']}; "
                  f"answered inconsistently: {tr['repeated_pairs_inconsistent']}."]
    lines += ["", "## Position bias", "", "| Step | Mean order gap | Share flipping with order |", "| --- | --- | --- |"]
    for r in out["stage1_position_bias"].itertuples():
        lines.append(f"| {r.step} | {r.mean_order_gap:+.2f} | {r.share_order_flips:.0%} |")
    return "\n".join(lines) + "\n"


def write_outputs(out: dict, out_dir: Path, run: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, table in out.items():
        if isinstance(table, pd.DataFrame):
            path = out_dir / f"{name}.csv"
            table.to_csv(path, index=False)
            written.append(path)
    path = out_dir / "stage1_decisions.json"
    path.write_text(json.dumps({"run": run, **out["decisions"]}, indent=2, default=_json_default))
    written.append(path)
    path = out_dir / "stage1_report.md"
    path.write_text(report(out, run), encoding="utf-8")
    written.append(path)
    return written


def _json_default(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return None if np.isnan(x) else float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    raise TypeError(type(x))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".", help="project folder")
    ap.add_argument("--run", help="run name, e.g. ministral-3-8b-instruct-2512-bf16 (default: the only one)")
    ap.add_argument("--tol", type=float, default=DEFAULT_TOL, help="|p - 0.5| counted as indifferent")
    ap.add_argument("--out-dir", help="default: data/analysis/<run>")
    args = ap.parse_args(argv)

    root = Path(args.root)
    path = find_run(root / "results", args.run)
    run = run_name(path)
    df = attach_descriptions(load_results(path), root / "prompts" / "stage1.csv")
    out = analyse(df, args.tol)
    out_dir = Path(args.out_dir) if args.out_dir else root / "data" / "analysis" / run
    written = write_outputs(out, out_dir, run)
    print(report(out, run))
    print(f"Wrote {len(written)} files to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

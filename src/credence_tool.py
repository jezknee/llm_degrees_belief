"""Credence tool, version 2: a light screening test first, then the weighty measurement.

    screen   Can this model's betting quotients be read as credences? A small, fixed battery, cheap enough
             to run before anything else:
               - Stage 1 in full (neutral event, ticket symmetry, validation, dominance, transitivity);
               - about 12 chance claims spread across the probability range, with negations
                 (the prompt fixes the right answer, so no world knowledge is assumed);
               - whole partitions for coherence: every chance partition, plus a few fact partitions;
               - Ramsey's utility scale (part of Ramsey's method, so always run);
               - both methods, both option orders, a thinned grid.
             It stops early if the chance partitions already fail coherence decisively. Same prompts,
             same single choices and the same analysis as credence_pipeline.py: only less of it.
    measure  For a model that passed the screen: its credence in each of your claims, measured carefully
             (full grid, both orders, more samples when thinking is on), pair-corrected.

Usage, on the pod, from the project folder (needs src/, model_configs/ and data/claims.csv):

    python src/credence_tool.py screen model_configs/qwen3-32b-thinking.json
    python src/credence_tool.py measure model_configs/qwen3-32b-thinking.json --claims my_claims.csv

Test anywhere without a GPU:

    python src/credence_tool.py screen model_configs/fake.json

Each config may carry a "screen" block (all optional):

    "screen": {"chance_claims": 12, "fact_partitions": 4, "every": 2, "samples": 3,
               "early_stop_p": 0.001, "measure_samples": 5}

Outputs use the run name <model run>-screen, so nothing mixes with full study runs:
    results/raw_calls_screen_<block>_<run>-screen.csv, results/utility_*_<run>-screen.*
    data/analysis/<run>-screen/                 analysis tables
    reports/<run>-screen/report.md, certificate.json, pipeline.log
    reports/<run>-screen/audit_<claims file>.csv / .md   measurements
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import pandas as pd

import analysis as an
import credence_pipeline as pl
import gate_check as sg
import make_prompts as mp
import run_prompts as rp
import run_reasoning as rr
import stage1_analysis as sa
import utility_scale as us

DEFAULT_SCREEN = {"chance_claims": 12, "fact_partitions": 4, "every": 2, "samples": 3,
                  "early_stop_p": 0.001, "measure_samples": 5}


# ---------------------------------------------------------------------------
# The battery
# ---------------------------------------------------------------------------


def select_battery(claims: pd.DataFrame, n_chance: int, n_fact_parts: int, seed: int = rr.SEED) -> dict:
    """The fixed screening battery from claims.csv, as two blocks of claims (each with its negation).

    block 1: chance claims spread evenly across the true-chance range, and every chance partition;
    block 2: n_fact_parts whole fact partitions (fixed seed).
    """
    pos = claims[claims["neg"] == 0]
    chance = pos[(pos["tier"] == "chance") & pos["objective_chance"].notna()].sort_values("objective_chance")
    if n_chance and len(chance):
        idx = sorted({round(i * (len(chance) - 1) / max(n_chance - 1, 1)) for i in range(min(n_chance, len(chance)))})
        chance_ids = set(chance.iloc[idx]["claim_id"])
    else:
        chance_ids = set()
    part_ids = claims["partition_id"].fillna("").astype(str)
    chance_parts = sorted({p for p in part_ids if p.startswith("part_chance")})
    fact_parts = sorted(rr.choose_partitions(sorted({p for p in part_ids if p.startswith("part_fact")}),
                                             n_fact_parts, seed)) if n_fact_parts else []

    def rows(ids=(), parts=()):
        return claims[claims["claim_id"].isin(set(ids)) | part_ids.isin(set(parts))]
    return {"block1": rows(chance_ids, chance_parts), "block2": rows(parts=fact_parts),
            "chance_partitions": chance_parts, "fact_partitions": fact_parts}


def block_prompts(claims: pd.DataFrame, every: int) -> dict[str, pd.DataFrame]:
    out = {}
    for method, build in (("matching", mp.matching_prompts), ("ramsey", mp.ramsey_prompts)):
        df = build(claims).astype(str)
        out[method] = rr.subset(df, None, every)
    return out


# ---------------------------------------------------------------------------
# Scoring and analysis
# ---------------------------------------------------------------------------


def screen_config(cfg: dict) -> dict:
    scr = {**DEFAULT_SCREEN, **cfg.get("screen", {})}
    c = copy.deepcopy(cfg)
    c["screen"] = scr
    c["reasoning"] = {**c["reasoning"], "samples": scr["samples"]}
    c["run"] = f"{cfg['run']}-screen"
    return c


def score(engine: pl.Engine, cfg: dict, root: Path, name: str, prompts: pd.DataFrame, log) -> pd.DataFrame:
    src = root / "prompts" / f"screen_{name}_{cfg['run']}.csv"
    src.parent.mkdir(exist_ok=True)
    prompts.to_csv(src, index=False)
    out = root / "results" / f"raw_calls_screen_{name}_{cfg['run']}.csv"
    rp.run(src, out, engine.backend_for_run(out.with_suffix(".samples.jsonl")), f"screen_{name}_{cfg['run']}",
           cfg["model"], batch_size=200 if cfg["thinking"] else 1000, log=log)
    df = pd.read_csv(out, dtype=str, keep_default_na=False)
    for col in ("p_target", "mass_outside", "p_A"):
        df[col] = pd.to_numeric(df[col])
    df["grid_value"] = pd.to_numeric(df["grid_value"], errors="coerce")
    return df


def analyse(root: Path, cfg: dict, claims: pd.DataFrame, frames: list[pd.DataFrame]) -> None:
    """Write the analysis for the battery scored so far to data/analysis/<run>/."""
    data = pd.concat(frames, ignore_index=True)
    matching, ramsey = data[data["method"] == "matching"], data[data["method"] == "ramsey"]
    result = an.analyse(claims, matching if len(matching) else None, ramsey if len(ramsey) else None,
                        an.load_utility(root / "results", cfg["run"]))
    out_dir = root / "data" / "analysis" / cfg["run"]
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, table in result.items():
        table.to_csv(out_dir / f"{name}.csv", index=False)
    (out_dir / "report.md").write_text(an.report(cfg["run"], result), encoding="utf-8")


def coherence_p(root: Path, cfg: dict) -> tuple[float | None, int]:
    """Pooled coherence p-value over the partitions analysed so far."""
    adir = root / "data" / "analysis" / cfg["run"]
    props = pd.read_csv(adir / "propositions.csv", dtype={"partition_id": str, "partition_role": str})
    parts = pl.partition_coherence(props, pd.read_csv(adir / "fits.csv"),
                                   pd.read_csv(adir / "curves.csv", usecols=["method", "x"]),
                                   cfg["certify"]["estimate"])
    if parts.empty:
        return None, 0
    return pl.chi2_sf(float((parts["z"] ** 2).sum()), len(parts)), len(parts)


def decisive_fail(p: float | None, threshold: float) -> bool:
    return p is not None and p < threshold


# ---------------------------------------------------------------------------
# screen
# ---------------------------------------------------------------------------


def screen(base_cfg: dict, root: Path, log=print) -> int:
    cfg = screen_config(base_cfg)
    scr = cfg["screen"]
    run = cfg["run"]
    log(f"=== screen: {cfg['model']} (run {run}, thinking {'on' if cfg['thinking'] else 'off'}) ===")
    problems = pl.preflight(cfg, root, log)
    if problems:
        pl.write_report(root, cfg, {}, None, "pre-flight: " + "; ".join(problems), log)
        return 1
    claims = mp.load_claims(root / "data" / "claims.csv")
    claims["objective_chance"] = pd.to_numeric(claims.get("objective_chance", ""), errors="coerce")
    battery = select_battery(claims, scr["chance_claims"], scr["fact_partitions"])
    log(f"Battery: block 1 = {battery['block1']['claim_id'].nunique()} claims (chance claims and chance "
        f"partitions {battery['chance_partitions']}); block 2 = {battery['block2']['claim_id'].nunique()} claims "
        f"(fact partitions {battery['fact_partitions']})")
    full_claims = an.load_claims(root / "data" / "claims.csv")

    # Stage 1 needs the stage1 prompts and the smoke file for the first-answer check.
    outputs = mp.build_all(claims)
    (root / "prompts").mkdir(exist_ok=True)
    for name in ("stage1", "smoke"):
        outputs[name].to_csv(root / "prompts" / f"{name}.csv", index=False)
    engine = pl.Engine(cfg)
    problem = pl.first_prompt_check(engine, cfg, root, log)
    if problem:
        pl.write_report(root, cfg, {}, None, problem, log)
        return 1

    stage1 = score(engine, cfg, root, "stage1", outputs["stage1"], log)
    s1 = sa.analyse(sa.attach_descriptions(stage1, root / "prompts" / "stage1.csv"))
    adir = root / "data" / "analysis" / run
    sa.write_outputs(s1, adir, run)
    neutral = cfg["neutral"] if cfg["neutral"] != "auto" else (s1["decisions"]["neutral_event"]["event"] or "die")

    # The utility scale is part of Ramsey's method: always run it (credence_pipeline's lesson).
    backend = engine.backend_for_run(root / "results" / f"utility_{run}.samples.jsonl")
    ut = us.build_scale(backend, cfg["model"], run, neutral, levels=3, log=log)
    us.write_outputs(ut, us.summary_for(ut, run, cfg["model"], neutral), root / "results", run)

    frames = []
    p1 = block_prompts(battery["block1"], scr["every"])
    for method, df in p1.items():
        frames.append(score(engine, cfg, root, f"block1_{method}", df, log))
    gate = sg.gate(pd.concat([f for f in frames if (f["method"] == "matching").any()]), stage1)
    log(sg.report(gate, run))
    if not gate["pass"]:
        analyse(root, cfg, full_claims, frames)
        pl.write_report(root, cfg, gate, None, "failed the gate; the rest of the battery was not run", log)
        return 1

    analyse(root, cfg, full_claims, frames)
    p, n = coherence_p(root, cfg)
    log(f"Coherence on the chance partitions: p = {p} over {n} partitions")
    if decisive_fail(p, scr["early_stop_p"]):
        cert = pl.certify(root, run, gate, cfg["certify"])
        pl.write_report(root, cfg, gate, cert, f"stopped early: the chance partitions already fail coherence "
                        f"decisively (p = {p:.1e} < {scr['early_stop_p']})", log)
        return 1

    if len(battery["block2"]):
        for method, df in block_prompts(battery["block2"], scr["every"]).items():
            frames.append(score(engine, cfg, root, f"block2_{method}", df, log))
        analyse(root, cfg, full_claims, frames)
    cert = pl.certify(root, run, gate, cfg["certify"])
    pl.write_report(root, cfg, gate, cert, None, log)
    return 0 if cert["passed"] else 1


# ---------------------------------------------------------------------------
# measure
# ---------------------------------------------------------------------------


def measure(base_cfg: dict, root: Path, claims_path: Path, force: bool = False, log=print) -> int:
    """The weighty part: full grid, both orders, more samples; needs a passing screen certificate."""
    cfg = screen_config(base_cfg)
    cfg["reasoning"] = {**cfg["reasoning"], "every": 1, "samples": cfg["screen"]["measure_samples"]}
    return pl.audit(cfg, root, claims_path, force, log)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["screen", "measure"])
    ap.add_argument("config", type=Path, help="model config, e.g. model_configs/qwen3-32b.json")
    ap.add_argument("--claims", type=Path, help="measure: CSV with a 'text' column (optional: negation, settles)")
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--force", action="store_true", help="measure: run without a passing screen")
    args = ap.parse_args(argv)
    cfg = pl.load_config(args.config)
    log_dir = args.root / "reports" / f"{cfg['run']}-screen"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = (log_dir / "pipeline.log").open("a", encoding="utf-8")

    def log(msg=""):
        print(msg)
        log_file.write(str(msg) + "\n")
        log_file.flush()

    try:
        if args.command == "screen":
            return screen(cfg, args.root, log)
        if not args.claims:
            ap.error("measure needs --claims")
        return measure(cfg, args.root, args.claims, args.force, log)
    finally:
        log_file.close()


if __name__ == "__main__":
    sys.exit(main())

"""End-to-end credence pipeline: certify a model (check), then measure its credence in your claims (audit).

    check   Do betting-based credences mean anything for this model? Pre-flight checks, smoke test
            and Stage 1, a gate, then (if it passes) the full matching and Ramsey runs, the utility
            scale and the analysis. Ends with a report card and a certificate. The model is loaded
            once for everything.
    audit   For a model that passed, measure its credence in each of your own claims (with their
            negations), pair-corrected.

Each model has a JSON config in model_configs/ (see model_configs/README.md). Usage, on the pod, from the
project folder (needs src/ and data/claims.csv; the prompts are generated here):

    python src/credence_pipeline.py check model_configs/qwen3-32b.json
    python src/credence_pipeline.py audit model_configs/qwen3-32b.json --claims my_claims.csv

For runs already made with the separate scripts, a report card from the results on disk (no GPU):

    python src/credence_pipeline.py report model_configs/qwen3-32b.json
    python src/credence_pipeline.py report model_configs/ministral-3-8b.json --results results_8B

Test anywhere without a GPU:

    python src/credence_pipeline.py check model_configs/fake.json

Outputs:
    results/raw_calls_<file>_<run>.csv          every scored prompt (as run_prompts.py)
    results/utility_*_<run>.*                   the utility scale (as utility_scale.py)
    data/analysis/<run>/                        Stage 1 and credence analysis (as the analysis scripts)
    reports/<run>/report.md                     the report card
    reports/<run>/certificate.json              pass or fail, with the numbers behind it
    reports/<run>/pipeline.log                  everything printed
    reports/<run>/audit_<name>.csv / .md        audit results
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import shutil
import sys
from pathlib import Path

import pandas as pd

import analysis as an
import compare_runs as cr
import make_prompts as mp
import run_prompts as rp
import run_reasoning as rr
import gate_check as sg
import stage1_analysis as sa
import utility_scale as us

# What a model must show to be certified. Each can be overridden in a config's "certify" block.
# Certification tests the MEASURE, not what the model knows: every check uses information in the prompt
# (chance claims) or internal consistency (partitions), never facts the model may not have learned.
DEFAULT_CERTIFY = {
    "estimate": "ramsey_linear_pair",       # the estimate the certificate vouches for
    "min_chance_corr": 0.7,                 # correlation with the chances the prompts fix
    "max_chance_mae": 0.2,                  # mean absolute error on those chances
    "partition_alpha": 0.05,                # coherence: partitions must be consistent with summing to exactly 1,
                                            # given measurement error (pooled test; fail if p < alpha)
}
DEFAULT_REASONING = {"samples": 5, "max_tokens": 3000, "temperature": 0.6, "top_p": 0.95, "top_k": 20,
                     "claims": 20, "partitions": "all", "every": 2}
JUDGE = "An infallible judge who knows the true answer will settle the bet."


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text())
    if "model" not in cfg:
        raise SystemExit(f"{path}: 'model' is required")
    cfg.setdefault("backend", "vllm")
    cfg.setdefault("thinking", False)
    cfg.setdefault("vllm", {})
    cfg.setdefault("top_k", 20)
    cfg["reasoning"] = {**DEFAULT_REASONING, **cfg.get("reasoning", {})}
    cfg["certify"] = {**DEFAULT_CERTIFY, **cfg.get("certify", {})}
    cfg.setdefault("neutral", "auto")
    cfg.setdefault("gate_min_share", sg.MIN_SHARE)
    base = cfg.get("name") or rp.slug(cfg["model"])
    cfg["run"] = f"{base}-thinking" if cfg["thinking"] and not base.endswith("-thinking") else base
    return cfg


# ---------------------------------------------------------------------------
# One model, loaded once, scored two ways
# ---------------------------------------------------------------------------


class Engine:
    """Loads the model once. score() reads the first token's log-probabilities; reasoning samples."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.fake = cfg["backend"] == "fake"
        if self.fake:
            self.llm = None
            return
        from vllm import LLM  # only on the pod
        self.llm = LLM(model=cfg["model"], **cfg["vllm"])

    def _template(self, thinking: bool) -> dict:
        key = self.cfg.get("thinking_switch", "enable_thinking")  # Qwen-style chat template switch
        kwargs = dict(self.cfg.get("chat_template_kwargs", {}))
        if key:
            kwargs[key] = thinking
        return {"chat_template_kwargs": kwargs} if kwargs else {}

    def logprob_backend(self):
        if self.fake:
            return FakeLogprobBackend()
        from vllm import SamplingParams
        engine = self
        params = SamplingParams(max_tokens=1, temperature=1.0, logprobs=self.cfg["top_k"], seed=0)

        class Backend:
            name = "vllm"

            def score(self, rows):
                outs = engine.llm.chat([rp.build_messages(r) for r in rows], params, use_tqdm=False,
                                       **engine._template(False))
                tok = engine.llm.get_tokenizer()
                return [rp.extract_top(o, tok) for o in outs]
        return Backend()

    def reasoning_backend(self, sample_log: Path):
        r = self.cfg["reasoning"]
        if self.fake:
            generate = rr.fake_generate
        else:
            from vllm import SamplingParams
            engine = self

            def generate(conversations, n):
                params = SamplingParams(n=n, temperature=r["temperature"], top_p=r["top_p"], top_k=r["top_k"],
                                        min_p=0.0, max_tokens=r["max_tokens"], seed=0, skip_special_tokens=False)
                outs = engine.llm.chat(conversations, params, use_tqdm=False, **engine._template(True))
                return [[(c.text, len(c.token_ids)) for c in o.outputs] for o in outs]
        backend = rr.ReasoningBackend(generate, r["samples"], sample_log)
        backend.name = "fake-reasoning" if self.fake else "vllm-reasoning"
        return backend

    def backend_for_run(self, sample_log: Path):
        return self.reasoning_backend(sample_log) if self.cfg["thinking"] else self.logprob_backend()


class FakeLogprobBackend(rp.FakeBackend):
    """run_prompts' fake model, which also gets Stage 1's dominance checks right (for testing the pipeline)."""

    def score(self, rows):
        tops = super().score(rows)
        for i, r in enumerate(rows):
            if r.get("expected_label"):
                other = "B" if r["expected_label"] == "A" else "A"
                tops[i] = [(r["expected_label"], math.log(0.95)), (other, math.log(0.04)), ("I", math.log(0.01))]
        return tops


# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------


def preflight(cfg: dict, root: Path, log) -> list[str]:
    """Checks before loading anything. Returns problems that should stop the run."""
    problems = []
    claims = root / "data" / "claims.csv"
    if not claims.exists():
        problems.append(f"{claims} not found: copy data/claims.csv to the pod")
    if cfg["backend"] == "fake":
        log("Pre-flight: fake backend, no GPU checks.")
        return problems
    try:
        import torch
        if not torch.cuda.is_available():
            problems.append("no CUDA GPU visible")
        else:
            n = torch.cuda.device_count()
            names = {torch.cuda.get_device_name(i) for i in range(n)}
            mem = sum(torch.cuda.get_device_properties(i).total_memory for i in range(n)) / 1e9
            cc = torch.cuda.get_device_capability(0)
            log(f"Pre-flight: {n} GPU(s) {', '.join(names)}, {mem:.0f} GB in all, compute capability {cc[0]}.{cc[1]}")
            tp = int(cfg["vllm"].get("tensor_parallel_size", 1))
            if tp > n:
                problems.append(f"config asks for tensor_parallel_size {tp} but only {n} GPU(s) found")
            if cc[0] >= 10 and "fp8" in cfg["model"].lower():
                log("  Warning: an FP8 model on a Blackwell GPU failed to compile on 13 November; "
                    "prefer BF16 or a Hopper GPU.")
            size = model_size_gb(cfg["model"], log)
            if size and size > 0.92 * mem:
                problems.append(f"model weights ~{size:.0f} GB won't fit in {mem:.0f} GB of GPU memory")
            free = shutil.disk_usage(Path.home()).free / 1e9
            log(f"  Disk free in {Path.home()}: {free:.0f} GB")
            if size and free < size * 1.1:
                problems.append(f"only {free:.0f} GB free on disk for ~{size:.0f} GB of weights")
    except ImportError:
        problems.append("PyTorch not installed")
    return problems


def model_size_gb(model: str, log) -> float | None:
    """Size of the model's weight files on Hugging Face, which also checks access."""
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(model, files_metadata=True)
        size = sum((s.size or 0) for s in info.siblings if s.rfilename.endswith((".safetensors", ".bin"))) / 1e9
        log(f"  Hugging Face: {model} is accessible, weights ~{size:.0f} GB")
        return size or None
    except Exception as e:  # gated, missing, offline: say so, but let vLLM give the final word
        log(f"  Hugging Face check failed for {model}: {type(e).__name__}: {str(e)[:200]}")
        return None


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------


def prepare_prompts(root: Path, log) -> pd.DataFrame:
    claims = mp.load_claims(root / "data" / "claims.csv")
    outputs = mp.build_all(claims)
    manifest = mp.write_all(outputs, root / "prompts", mp.SEED)
    log(f"Prompts written to prompts/: {manifest['rows']}")
    return claims


def score_file(engine: Engine, cfg: dict, root: Path, name: str, log, cut: bool) -> pd.DataFrame:
    """Score prompts/<name>.csv (cut down for reasoning runs) into results/raw_calls_<name>_<run>.csv."""
    src = root / "prompts" / f"{name}.csv"
    prompts = rp.load_prompts(src)
    if cfg["thinking"] and cut:
        r = cfg["reasoning"]
        claim_file = name in ("matching", "ramsey")
        sub = rr.subset(prompts, r["claims"] if claim_file else None, r["every"],
                        partitions=r["partitions"] if claim_file else None)
        src = root / "prompts" / f"{name}_reasoning_subset.csv"
        sub.to_csv(src, index=False)
    out = root / "results" / f"raw_calls_{name}_{cfg['run']}.csv"
    backend = engine.backend_for_run(out.with_suffix(".samples.jsonl"))
    rp.run(src, out, backend, f"{name}_{cfg['run']}", cfg["model"], batch_size=200 if cfg["thinking"] else 1000,
           log=log)
    df = pd.read_csv(out, dtype=str, keep_default_na=False)
    summary = rp.summarize(df)
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    for w in rp.warnings_for(summary):
        log(f"  WARNING: {w}")
    return df


def first_prompt_check(engine: Engine, cfg: dict, root: Path, log) -> str | None:
    """Score one smoke prompt and check the answer comes as A or B. Returns a problem, or None."""
    row = rp.load_prompts(root / "prompts" / "smoke.csv").iloc[0].to_dict()
    if cfg["thinking"]:
        backend = engine.reasoning_backend(None)
        top = backend.score([row])[0]
    else:
        top = engine.logprob_backend().score([row])[0]
    lp = rp.choice_logprobs(top)
    probs = rp.choice_probs(lp["lp_A"], lp["lp_B"], row["target_label"])
    log("First prompt, top tokens: " + ", ".join(f"{t!r} {math.exp(v):.3f}" for t, v in
                                                 sorted(top, key=lambda x: -x[1])[:8]))
    log(f"  mass outside A/B {probs['mass_outside']:.3f}; A missing {lp['a_missing']}, B missing {lp['b_missing']}")
    if probs["mass_outside"] > 0.5 or (lp["a_missing"] and lp["b_missing"]):
        return ("the model doesn't answer with A or B (most probability is on other tokens). Check the "
                "chat template and thinking switch in the config.")
    return None


def grid_sd(curves: pd.DataFrame, method: str) -> float:
    """Measurement error from the grid alone: a crossing is only located within one grid step (uniform)."""
    xs = sorted(curves.loc[curves["method"] == method, "x"].unique())
    step = min(b - a for a, b in zip(xs, xs[1:])) if len(xs) > 1 else 0.05
    return step / math.sqrt(12)


def chi2_sf(x: float, df: int) -> float:
    """Upper tail of a chi-squared distribution (Wilson-Hilferty approximation; no scipy needed)."""
    if df <= 0:
        return float("nan")
    z = ((x / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))
    return 0.5 * math.erfc(z / math.sqrt(2))


def partition_coherence(props: pd.DataFrame, fits: pd.DataFrame, curves: pd.DataFrame, est: str) -> pd.DataFrame:
    """Each partition: sum of its three cells, the measurement error of that sum, and whether the gap from 1
    is bigger than measurement error explains.

    Coherence is exact (a partition's cells must sum to exactly 1); the tolerance is for MEASUREMENT error only.
    Error per proposition = grid resolution plus half the spread between the two option orders' crossings
    (half, because a strong position bias inflates the spread). A pair-corrected cell (b(P) - b(not P) + 1)/2
    has error sqrt(var_P + var_N) / 2.
    """
    method = "matching" if est.startswith("matching") else "ramsey"
    f = fits[fits["method"] == method].set_index("prop_id")
    g = grid_sd(curves, method)
    spread = (f["crossing_first"] - f["crossing_second"]).abs().fillna(0) / 2
    var = g ** 2 + spread ** 2
    rows = []
    cells = props[(props["partition_role"] == "single") & props[est].notna()]
    for pid, c in cells.groupby("partition_id"):
        if len(c) != 3:
            continue
        cell_var = []
        for r in c.itertuples():
            v = var.get(r.prop_id, g ** 2)
            if est.endswith("_pair"):
                partner = props[(props["claim_id"] == r.claim_id) & (props["neg"] != r.neg)]["prop_id"]
                v = (v + (var.get(partner.iloc[0], g ** 2) if len(partner) else g ** 2)) / 4
            cell_var.append(v)
        total = float(c[est].sum())
        sd = math.sqrt(sum(cell_var))
        z = (total - 1) / sd if sd > 0 else float("inf")
        rows.append(dict(partition_id=pid, cells=" / ".join(f"{v:.2f}" for v in c[est]), sum=total,
                         sure_loss_per_pound=abs(total - 1), sd=sd, z=z, within_2sd=abs(z) <= 2))
    return pd.DataFrame(rows)


def sign_test(sums: pd.Series) -> float:
    """Two-sided sign test: are partition sums above and below 1 as often as chance would give?"""
    above, below = int((sums > 1).sum()), int((sums < 1).sum())
    n, k = above + below, max(above, below)
    if n == 0:
        return float("nan")
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def certify(root: Path, run: str, gate: dict, crit: dict) -> dict:
    """Checks on the measure itself, one row per claim (the proposition, not its negation).

    - chance claims: the right credence is fixed by the prompt ("a fair die ..."), so no world knowledge
      is assumed beyond simple probability;
    - partitions: the three cells of each partition must sum to exactly 1 (coherence is a logical condition,
      not an approximation). The test asks whether the measured sums are consistent with exactly 1 given
      the measurement error, pooled over all partitions; the degree of incoherence is reported as the sure
      loss per £1 a Dutch book could extract. Pair correction makes b*(P) + b*(not P) = 1 by construction,
      but nothing forces a partition's cells to add up, so this is a real coherence test;
    - facts are reported, not tested: low accuracy there could mean the model doesn't know, not that the
      measure is wrong.
    Pair-corrected estimates mirror each claim (b*(not P) = 1 - b*(P)), so negations are left out of the
    chance checks to avoid counting each claim twice.
    """
    adir = root / "data" / "analysis" / run
    props = pd.read_csv(adir / "propositions.csv",
                        dtype={"truth": str, "text": str, "partition_id": str, "partition_role": str})
    fits = pd.read_csv(adir / "fits.csv")
    curves = pd.read_csv(adir / "curves.csv", usecols=["method", "x"])
    est = crit["estimate"]
    p = props[props["neg"] == 0]
    chance = p[p["objective_chance"].notna() & p[est].notna()]
    corr = an._corr(chance[est], chance["objective_chance"]) if len(chance) else None
    parts = partition_coherence(props, fits, curves, est)
    parts.to_csv(adir / f"partition_coherence_{est}.csv", index=False)
    if len(parts):
        chi2 = float((parts["z"] ** 2).sum())
        p_value = chi2_sf(chi2, len(parts))
    else:
        chi2, p_value = None, None
    facts = p[(p["tier"] == "fact") & p["truth"].isin(["true", "false"]) & p[est].notna()]
    checks = {
        "gate_passed": {"value": gate["pass"], "pass": gate["pass"]},
        "chance_corr": {"value": None if corr is None or math.isnan(corr) else corr, "min": crit["min_chance_corr"],
                        "n": len(chance)},
        "chance_mae": {"value": float((chance[est] - chance["objective_chance"]).abs().mean()) if len(chance) else None,
                       "max": crit["max_chance_mae"], "n": len(chance)},
        "partition_coherence_p": {"value": p_value, "min": crit["partition_alpha"], "n": len(parts)},
    }
    for name, c in checks.items():
        if name == "gate_passed":
            continue
        v = c["value"]
        # Missing data is not a failure: a check with nothing to test is "not tested" (pass = None).
        c["pass"] = None if v is None or c.get("n") == 0 else (v >= c["min"] if "min" in c else v <= c["max"])
    described = {
        "fact_accuracy": {
            "value": float(((facts[est] > 0.5) == (facts["truth"] == "true")).mean()) if len(facts) else None,
            "n": len(facts), "note": "share of facts where the true version gets the higher credence; describes "
                                     "the model's knowledge, not the measure, so it is not a certification check"},
        "incoherence_sure_loss": {
            "unit": "partitions", "value": float(parts["sure_loss_per_pound"].mean()) if len(parts) else None, "n": len(parts),
            "note": "mean |sum of a partition's cells - 1|: the sure loss per £1 a Dutch book could extract from an "
                    "agent acting on these numbers (degree of incoherence)"},
        "partitions_above_1": {
            "unit": "partitions", "value": sign_test(parts["sum"]) if len(parts) else None, "n": len(parts),
            "note": (f"{int((parts['sum'] > 1).sum())} of {len(parts)} partition sums are above 1; two-sided sign-test "
                     "p-value shown (under coherence, sums scatter evenly either side of 1; no error model needed)")
            if len(parts) else "no partitions"},
        "partitions_within_2sd": {
            "unit": "partitions", "value": float(parts["within_2sd"].mean()) if len(parts) else None, "n": len(parts),
            "note": f"share of partitions whose sum is within 2 measurement-error SDs of 1 (pooled chi-squared "
                    f"{chi2:.1f} on {len(parts)} df)" if len(parts) else "no partitions"},
    }
    results = [c["pass"] for c in checks.values()]
    status = "failed" if False in results else ("incomplete" if None in results else "passed")
    return {"estimate": est, "checks": checks, "described": described, "status": status,
            "passed": status == "passed", "not_tested": [k for k, c in checks.items() if c["pass"] is None]}


def write_report(root: Path, cfg: dict, gate: dict, cert: dict | None, stopped: str | None, log) -> Path:
    run = cfg["run"]
    out = root / "reports" / run
    out.mkdir(parents=True, exist_ok=True)
    if stopped:
        verdict = "NOT CERTIFIED: " + stopped
    elif cert and cert["status"] == "passed":
        verdict = "CERTIFIED"
    elif cert and cert["status"] == "incomplete":
        verdict = ("INCOMPLETE: no tested check failed, but not tested: " + ", ".join(cert["not_tested"])
                   + " (missing data is not a failure, nor a pass)")
    else:
        verdict = "NOT CERTIFIED: see the failed checks"
    lines = [f"# Report card: {cfg['model']}{' (thinking on)' if cfg['thinking'] else ''}", "",
             f"Run `{run}`, {dt.date.today().isoformat()}. **{verdict}**", "", "## Gate (smoke test and Stage 1)", ""]
    for method, s in gate.get("methods", {}).items():
        lines.append(f"- {method}: dominance at the top of the grid {s['share_dominance_at_top']:.0%}, "
                     f"curves crossing 0.5 or confident {s['share_crossing']:.0%} "
                     f"({s.get('confident_beliefs', 0)} confident beliefs), mean order gap {s['mean_abs_order_gap']:.2f}")
    if "stage1_dominance" in gate:
        lines.append(f"- Stage 1 dominance: {gate['stage1_dominance']:.0%} of single-order prompts")
    lines.append(f"- Gate: {'PASS' if gate.get('pass') else 'FAIL'}")
    if gate.get("rule"):
        lines.append(f"- Gate rule {gate['rule']}. Changed on 14 November after the thinking-on run showed rule v1 "
                     "counting certainty as failure; applied to every run.")
    if cert:
        lines += ["", f"## Certification (estimate: {cert['estimate']}, one row per claim)", "",
                  "| Check | Value | Needed | Claims | Pass |", "| --- | --- | --- | --- | --- |"]
        for name, c in cert["checks"].items():
            need = f">= {c['min']}" if "min" in c else (f"<= {c['max']}" if "max" in c else "pass")
            if name == "partition_coherence_p":
                need = f"p >= {c['min']} (consistent with exactly 1)"
            v = c["value"]
            vs = v if isinstance(v, bool) else _num(v)
            ok = "not tested (no data)" if c["pass"] is None else ("yes" if c["pass"] else "no")
            lines.append(f"| {name} | {vs} | {need} | {c.get('n', '')} | {ok} |")
        for name, d in cert.get("described", {}).items():
            v = d["value"]
            lines += ["", f"Described, not tested: {name} = {_num(v)} "
                      f"({d['n']} {d.get('unit', 'claims')}). {d['note'][0].upper() + d['note'][1:]}."]
    if (root / "data" / "analysis" / run / "summary_by_tier.csv").exists():
        long = pd.DataFrame(cr.rows_for(root, run))
        lines += ["", "## All checks", "", cr.table(long, [run])]
    path = out / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    certificate = {"model": cfg["model"], "run": run, "thinking": cfg["thinking"], "date": dt.date.today().isoformat(),
                   "certified": bool(cert and cert["passed"] and not stopped),
                   "status": "stopped" if stopped else (cert["status"] if cert else None), "stopped": stopped,
                   "gate": {k: v for k, v in gate.items() if k != "propositions"}, "certification": cert,
                   "config": {k: v for k, v in cfg.items() if k != "run"}}
    (out / "certificate.json").write_text(json.dumps(certificate, indent=2, default=str))
    log(f"Report card: {path}  ->  {verdict}")
    return path


def check(cfg: dict, root: Path, force: bool = False, log=print) -> int:
    run = cfg["run"]
    log(f"=== check: {cfg['model']} (run {run}, thinking {'on' if cfg['thinking'] else 'off'}) ===")
    problems = preflight(cfg, root, log)
    if problems:
        for p in problems:
            log(f"Pre-flight problem: {p}")
        write_report(root, cfg, {}, None, "pre-flight: " + "; ".join(problems), log)
        return 1
    claims = prepare_prompts(root, log)
    engine = Engine(cfg)

    problem = first_prompt_check(engine, cfg, root, log)
    if problem:
        write_report(root, cfg, {}, None, problem, log)
        return 1

    smoke = score_file(engine, cfg, root, "smoke", log, cut=True)
    stage1 = score_file(engine, cfg, root, "stage1", log, cut=False)
    gate = sg.gate(sg.load(root / "results" / f"raw_calls_smoke_{run}.csv"),
                   sg.load(root / "results" / f"raw_calls_stage1_{run}.csv"))
    log(sg.report(gate, run))
    s1 = sa.analyse(sa.attach_descriptions(sa.load_results(root / "results" / f"raw_calls_stage1_{run}.csv"),
                                           root / "prompts" / "stage1.csv"))
    sa.write_outputs(s1, root / "data" / "analysis" / run, run)
    neutral = cfg["neutral"] if cfg["neutral"] != "auto" else (s1["decisions"]["neutral_event"]["event"] or "die")
    log(f"Neutral event for the utility scale: {neutral}")
    del smoke, stage1
    if not gate["pass"] and not force:
        write_report(root, cfg, gate, None, "failed the gate (see above); the full runs were not done", log)
        return 1

    matching = score_file(engine, cfg, root, "matching", log, cut=True)
    ramsey = score_file(engine, cfg, root, "ramsey", log, cut=True)
    backend = engine.backend_for_run(root / "results" / f"utility_{run}.samples.jsonl")
    out = us.build_scale(backend, cfg["model"], run, neutral, levels=3, log=log)
    us.write_outputs(out, us.summary_for(out, run, cfg["model"], neutral), root / "results", run)

    for df in (matching, ramsey):
        for col in ("p_target", "mass_outside"):
            df[col] = pd.to_numeric(df[col])
        df["grid_value"] = pd.to_numeric(df["grid_value"])
    result = an.analyse(an.load_claims(root / "data" / "claims.csv"), matching, ramsey,
                        an.load_utility(root / "results", run))
    out_dir = root / "data" / "analysis" / run
    for name, table in result.items():
        table.to_csv(out_dir / f"{name}.csv", index=False)
    (out_dir / "report.md").write_text(an.report(run, result), encoding="utf-8")
    del claims
    cert = certify(root, run, gate, cfg["certify"])
    write_report(root, cfg, gate, cert, None, log)
    return 0 if cert["passed"] else 1


def report_existing(cfg: dict, root: Path, results_dir: Path, log=print) -> int:
    """Report card and certificate from results already on disk (no model needed).

    For runs made with the separate scripts (run_prompts.py, run_reasoning.py, utility_scale.py). Needs
    raw_calls_{smoke,stage1,matching,ramsey}_<run>.csv in results_dir; runs the analysis if it is missing.
    """
    run = cfg["run"]
    log(f"=== report: {cfg['model']} (run {run}) from {results_dir} ===")
    need = [results_dir / f"raw_calls_{n}_{run}.csv" for n in ("smoke", "stage1", "matching", "ramsey")]
    if not need[0].exists():  # e.g. a thinking run done without a smoke test: gate on the matching results
        log(f"No smoke results for {run}; the gate uses the matching results instead.")
        need[0] = need[2]
    missing = [str(n) for n in need if not n.exists()]
    if missing:
        raise SystemExit("Missing results: " + ", ".join(missing))
    gate = sg.gate(sg.load(need[0]), sg.load(need[1]))
    log(sg.report(gate, run))
    adir = root / "data" / "analysis" / run
    s1 = sa.analyse(sa.attach_descriptions(sa.load_results(need[1]), root / "prompts" / "stage1.csv"))
    sa.write_outputs(s1, adir, run)
    an.main(["--root", str(root), "--results", str(results_dir), "--run", run])
    cert = certify(root, run, gate, cfg["certify"])
    write_report(root, cfg, gate, cert, None, log)
    return 0 if cert["passed"] else 1


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------


def negate(text: str) -> str:
    """A plain negation for a claim with none supplied."""
    t = text.strip().rstrip(".")
    first, _, rest = t.partition(" ")
    if first in {"The", "A", "An", "This", "That", "These", "Those", "There", "It", "Some", "Most", "All"}:
        t = f"{first.lower()} {rest}"  # lower-case only common opening words, never names
    return f"It is not the case that {t}."


def audit_claims(path: Path) -> pd.DataFrame:
    """Your claims file -> claims in the pipeline's format, each with its negation."""
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "text" not in raw.columns:
        raise SystemExit(f"{path} needs a 'text' column (optional: negation, settles, claim_id)")
    rows = []
    for i, r in raw.iterrows():
        cid = r.get("claim_id") or f"audit_{i:03d}"
        settles = r.get("settles") or JUDGE
        neg_text = r.get("negation") or negate(r["text"])  # check these read well; supply your own if not
        for neg, text in ((0, r["text"].strip()), (1, neg_text.strip())):
            rows.append(dict(prop_id=f"{cid}_{'pn'[neg]}", claim_id=cid, neg=neg, tier="audit", text=text,
                             settles=settles, source="", close_date=""))
    return pd.DataFrame(rows)


def audit(cfg: dict, root: Path, claims_path: Path, force: bool = False, log=print) -> int:
    run = cfg["run"]
    cert_path = root / "reports" / run / "certificate.json"
    certified = cert_path.exists() and json.loads(cert_path.read_text()).get("certified")
    if not certified and not force:
        raise SystemExit(f"{cfg['model']} has no passing certificate ({cert_path}). Run "
                         f"'credence_pipeline.py check' first, or add --force to audit anyway (results won't be vouched for).")
    claims = audit_claims(claims_path)
    name = claims_path.stem
    log(f"=== audit: {len(claims) // 2} claims from {claims_path} with {cfg['model']} ===")
    prompts = {"matching": mp.matching_prompts(claims), "ramsey": mp.ramsey_prompts(claims)}
    engine = Engine(cfg)
    frames = {}
    for method, df in prompts.items():
        if cfg["thinking"]:
            df = rr.subset(df, None, cfg["reasoning"]["every"])
        src = root / "prompts" / f"audit_{name}_{method}.csv"
        src.parent.mkdir(exist_ok=True)
        df.to_csv(src, index=False)
        out = root / "results" / f"raw_calls_audit_{name}_{method}_{run}.csv"
        rp.run(src, out, engine.backend_for_run(out.with_suffix(".samples.jsonl")), f"audit_{name}_{run}",
               cfg["model"], log=log)
        f = pd.read_csv(out, dtype=str, keep_default_na=False)
        for col in ("p_target", "mass_outside"):
            f[col] = pd.to_numeric(f[col])
        f["grid_value"] = pd.to_numeric(f["grid_value"])
        frames[method] = f
    fits = an.fits_table(an.curves_table(pd.concat(frames.values(), ignore_index=True)))
    props = claims.assign(truth="", objective_chance=float("nan"), human_forecast=float("nan"), kuq_category="",
                          partition_id="", partition_role="")
    props = an.propositions_table(fits, props, an.utility_function(an.load_utility(root / "results", run)))
    cols = ["claim_id", "neg", "text", "ramsey_linear_pair", "ramsey_linear", "ramsey_status", "matching_pair",
            "matching", "matching_status"]
    props = props[cols]
    out_dir = root / "reports" / run
    out_dir.mkdir(parents=True, exist_ok=True)
    props.to_csv(out_dir / f"audit_{name}.csv", index=False)
    word = "Credence" if certified else "Betting quotient"
    lines = [f"# {'Credence' if certified else 'Betting-quotient'} audit: {name}", "", f"Model {cfg['model']}, run {run}"
             + ("" if certified else " (NOT certified: --force was used, so these are betting quotients, not "
                "credences)"), "",
             f"{word} = Ramsey, pair-corrected; raw values and curve status alongside.", "",
             f"| Claim | {word} | Ramsey raw (status) | Matching, pair-corrected |", "| --- | --- | --- | --- |"]
    for r in props[props["neg"] == 0].itertuples():
        lines.append(f"| {r.text} | {_pct(r.ramsey_linear_pair)} | {_pct(r.ramsey_linear)} ({r.ramsey_status}) | "
                     f"{_pct(r.matching_pair)} |")
    (out_dir / f"audit_{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log("\n".join(lines))
    return 0


def _num(v) -> str:
    """2 decimals, or scientific notation for very small values (p-values)."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    if isinstance(v, float) and v == 0.0:
        return "< 1e-16"
    return f"{v:.1e}" if 0 < abs(v) < 0.01 else f"{v:.2f}"


def _pct(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.0%}"


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["check", "report", "audit"],
                    help="check: run everything on the pod; report: report card from existing results; "
                         "audit: credences for your claims")
    ap.add_argument("config", type=Path, help="model config, e.g. model_configs/qwen3-32b.json")
    ap.add_argument("--claims", type=Path, help="audit: CSV with a 'text' column (optional: negation, settles)")
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--results", type=Path, help="report: results folder (default: <root>/results)")
    ap.add_argument("--force", action="store_true",
                    help="check: run the full files even if the gate fails; audit: run without a certificate")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    log_dir = args.root / "reports" / cfg["run"]
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = (log_dir / "pipeline.log").open("a", encoding="utf-8")

    def log(msg=""):
        print(msg)
        log_file.write(str(msg) + "\n")
        log_file.flush()

    try:
        if args.command == "check":
            return check(cfg, args.root, args.force, log)
        if args.command == "report":
            return report_existing(cfg, args.root, args.results or args.root / "results", log)
        if not args.claims:
            ap.error("audit needs --claims")
        return audit(cfg, args.root, args.claims, args.force, log)
    finally:
        log_file.close()


if __name__ == "__main__":
    sys.exit(main())

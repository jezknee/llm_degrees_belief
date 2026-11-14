"""Run prompts through a model and record choice probabilities.

Reads a prompts CSV made by make_prompts.py (e.g. prompts/smoke.csv) and writes
one row per prompt to results/raw_calls_<run_id>.csv, plus a summary JSON.

For each prompt the model generates one token; we read the top log-probabilities
of that token, sum the variants of "A" (e.g. "A" and " A") and of "B", and
renormalise over the two:

    p_A = P(A) / (P(A) + P(B))       p_target = p_A or p_B, by target_label
    mass_outside = 1 - P(A) - P(B)   (probability on any other token)

The choice probability is only used to locate indifference (p_target = 0.5);
it is not itself a credence.

On the pod (GPU):
    pip install vllm pandas
    python src/run_prompts.py prompts/smoke.csv --model <Hugging Face repo id> --show-first
    # Mistral models: add the settings from the model card, e.g.
    #   --tokenizer-mode mistral --config-format mistral --load-format mistral

On your laptop, to test the pipeline with no GPU and no model:
    python src/run_prompts.py prompts/smoke.csv --backend fake

Runs resume: rerunning the same command skips prompts already in the output file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROMPT_COLUMNS = [
    "prompt_id", "method", "step", "item_id", "prop_id", "claim_id", "neg", "tier",
    "grid_value", "grid_unit", "order", "target_label", "expected_label", "system", "user",
    "template_version",
]

RESULT_COLUMNS = [
    "run_id", "model", "backend", "input_file", "prompt_id", "template_version", "method", "step",
    "item_id", "prop_id", "claim_id", "neg", "tier", "grid_value", "grid_unit", "order",
    "target_label", "expected_label", "lp_A", "lp_B", "a_missing", "b_missing", "p_A", "p_B",
    "p_target", "mass_outside", "top_token", "top_logprob", "timestamp",
]

FLOOR_LOGPROB = -50.0  # used when no top-k list came back at all

# ---------------------------------------------------------------------------
# Turning top-k log-probabilities into a choice probability
# ---------------------------------------------------------------------------


def logsumexp(values) -> float:
    values = list(values)
    m = max(values)
    return m + math.log(sum(math.exp(v - m) for v in values))


def choice_logprobs(top: list[tuple[str, float]]) -> dict:
    """Log-probabilities of answering A and B from a top-k list of (token, logprob).

    Variants that are "A" or "B" once whitespace is stripped (e.g. " A") are summed.
    Lower-case "a" is not counted (it is also the English article). If A or B is
    not in the top-k list, its log-probability is set to the smallest one listed
    (an upper bound) and flagged as missing.
    """
    a = [lp for t, lp in top if t is not None and t.strip() == "A"]
    b = [lp for t, lp in top if t is not None and t.strip() == "B"]
    floor = min((lp for _, lp in top), default=FLOOR_LOGPROB)
    best = max(top, key=lambda x: x[1]) if top else ("", float("nan"))
    return {
        "lp_A": logsumexp(a) if a else floor,
        "lp_B": logsumexp(b) if b else floor,
        "a_missing": int(not a),
        "b_missing": int(not b),
        "top_token": best[0],
        "top_logprob": best[1],
    }


def choice_probs(lp_a: float, lp_b: float, target_label: str) -> dict:
    pa, pb = math.exp(lp_a), math.exp(lp_b)
    total = pa + pb
    p_a = pa / total
    return {
        "p_A": p_a,
        "p_B": 1 - p_a,
        "p_target": p_a if target_label == "A" else 1 - p_a,
        "mass_outside": max(0.0, 1.0 - total),
    }


def build_messages(row: dict) -> list[dict]:
    messages = []
    if str(row.get("system", "")).strip():
        messages.append({"role": "system", "content": row["system"]})
    messages.append({"role": "user", "content": row["user"]})
    return messages


# ---------------------------------------------------------------------------
# Backends: each takes a list of prompt rows and returns one top-k list per row
# ---------------------------------------------------------------------------


class VLLMBackend:
    """Runs a model inside this machine with vLLM (needs a GPU)."""

    name = "vllm"

    def __init__(self, model: str, top_k: int = 20, max_model_len: int = 4096,
                 gpu_memory_utilization: float = 0.9, dtype: str = "auto",
                 tokenizer_mode: str | None = None, config_format: str | None = None,
                 load_format: str | None = None, chat_template_kwargs: dict | None = None, seed: int = 0):
        from vllm import LLM, SamplingParams  # imported here so tests and laptops don't need vLLM

        kwargs = dict(model=model, max_model_len=max_model_len, dtype=dtype,
                      gpu_memory_utilization=gpu_memory_utilization, seed=seed)
        for key, value in (("tokenizer_mode", tokenizer_mode), ("config_format", config_format),
                           ("load_format", load_format)):
            if value:
                kwargs[key] = value
        self.llm = LLM(**kwargs)
        # One token; temperature 1 so the log-probabilities are the model's own.
        self.params = SamplingParams(max_tokens=1, temperature=1.0, logprobs=top_k, seed=seed)
        self.chat_template_kwargs = chat_template_kwargs
        self._tokenizer = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = self.llm.get_tokenizer()
        return self._tokenizer

    def score(self, rows: list[dict]) -> list[list[tuple[str, float]]]:
        conversations = [build_messages(r) for r in rows]
        extra = {"chat_template_kwargs": self.chat_template_kwargs} if self.chat_template_kwargs else {}
        outputs = self.llm.chat(conversations, self.params, use_tqdm=False, **extra)
        return [extract_top(o, self.tokenizer) for o in outputs]


def extract_top(output, tokenizer=None) -> list[tuple[str, float]]:
    """Pull (token text, logprob) pairs for the first generated token from a vLLM output."""
    steps = output.outputs[0].logprobs
    if not steps:
        return []
    top = []
    for token_id, lp in steps[0].items():
        text = getattr(lp, "decoded_token", None)
        if text is None and tokenizer is not None:
            text = tokenizer.decode([token_id])
        top.append((text if text is not None else "", float(lp.logprob)))
    return top


class FakeBackend:
    """Made-up but realistic choice probabilities, for testing without a GPU.

    Each proposition gets a fixed pseudo-credence from its id. The target bet is
    chosen more often the further that credence is above the reference
    (k/100 for matching, c/1000 for Ramsey), with a small bias towards option A
    and 3% of probability on another token.
    """

    name = "fake"

    def __init__(self, position_bias: float = 0.3, sharpness: float = 12.0, outside: float = 0.03):
        self.position_bias = position_bias
        self.sharpness = sharpness
        self.outside = outside
        self.calls = 0
        self.rows_scored = 0

    @staticmethod
    def credence(key: str) -> float:
        h = int(hashlib.md5(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        return 0.05 + 0.9 * h

    def score(self, rows: list[dict]) -> list[list[tuple[str, float]]]:
        self.calls += 1
        self.rows_scored += len(rows)
        out = []
        for r in rows:
            method = r.get("method", "")
            try:
                g = float(r.get("grid_value", ""))
            except ValueError:
                g = None
            if method == "matching" and g is not None:
                ref = g / 100
            elif method == "ramsey" and g is not None:
                ref = g / 1000
            else:
                ref = 0.5
            cred = self.credence(r.get("prop_id") or r.get("item_id") or r.get("prompt_id", ""))
            if method == "stage1":
                cred = 0.5
            logit = self.sharpness * (cred - ref)
            target_is_a = r.get("target_label", "A") == "A"
            logit_a = (logit if target_is_a else -logit) + self.position_bias
            p_a = 1 / (1 + math.exp(-logit_a))
            keep = 1 - self.outside
            pa, pb = keep * p_a, keep * (1 - p_a)
            out.append([("A", math.log(max(pa, 1e-12))), ("B", math.log(max(pb, 1e-12))),
                        ("I", math.log(self.outside * 0.6)), (" The", math.log(self.outside * 0.4))])
        return out


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.split("/")[-1].lower()).strip("-")


def default_run_id(input_path: Path, model: str) -> str:
    return f"{input_path.stem}_{slug(model)}"


def load_prompts(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in PROMPT_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"{path} is missing columns {missing}; was it made by make_prompts.py?")
    if df["prompt_id"].duplicated().any():
        raise SystemExit(f"{path} has duplicate prompt_id values")
    return df


def result_row(prompt: dict, top, run_id: str, model: str, backend: str, input_file: str) -> dict:
    lp = choice_logprobs(top)
    probs = choice_probs(lp["lp_A"], lp["lp_B"], prompt["target_label"])
    row = {k: prompt.get(k, "") for k in RESULT_COLUMNS if k in prompt}
    row.update(run_id=run_id, model=model, backend=backend, input_file=input_file,
               timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"), **lp, **probs)
    return row


def run(input_path: Path, out_path: Path, backend, run_id: str, model: str,
        batch_size: int = 1000, limit: int | None = None, log=print) -> dict:
    """Score every prompt not already in out_path, appending results in batches."""
    prompts = load_prompts(input_path)
    done: set[str] = set()
    if out_path.exists():
        prev = pd.read_csv(out_path, dtype=str, keep_default_na=False, usecols=["prompt_id", "model"])
        if len(prev) and set(prev["model"]) != {model}:
            raise SystemExit(f"{out_path} holds results from {sorted(set(prev['model']))}, not {model}. "
                             "Use a different --run-id.")
        done = set(prev["prompt_id"])
    todo = prompts[~prompts["prompt_id"].isin(done)]
    if limit is not None:
        todo = todo.head(limit)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"{len(prompts)} prompts in {input_path.name}; {len(done)} already done; scoring {len(todo)}.")

    start = time.time()
    for i in range(0, len(todo), batch_size):
        records = todo.iloc[i:i + batch_size].to_dict("records")
        tops = backend.score(records)
        if len(tops) != len(records):
            raise RuntimeError(f"backend returned {len(tops)} results for {len(records)} prompts")
        rows = [result_row(r, t, run_id, model, backend.name, input_path.name) for r, t in zip(records, tops)]
        pd.DataFrame(rows, columns=RESULT_COLUMNS).to_csv(
            out_path, mode="a", header=not out_path.exists(), index=False)
        log(f"  {min(i + batch_size, len(todo))}/{len(todo)} done ({time.time() - start:.0f}s)")
    return {"total": len(prompts), "already_done": len(done), "scored": len(todo)}


# ---------------------------------------------------------------------------
# Summary and checks
# ---------------------------------------------------------------------------


def rank_corr(x: pd.Series, y: pd.Series) -> float:
    xr, yr = x.rank().to_numpy(), y.rank().to_numpy()
    if np.std(xr) == 0 or np.std(yr) == 0:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def summarize(df: pd.DataFrame) -> dict:
    df = df.copy()
    for col in ("p_A", "p_target", "mass_outside", "a_missing", "b_missing"):
        df[col] = pd.to_numeric(df[col])
    df["grid_num"] = pd.to_numeric(df["grid_value"], errors="coerce")
    out = {"rows": int(len(df)), "by_method": {}}
    for method, g in df.groupby("method"):
        s = {
            "rows": int(len(g)),
            "mean_mass_outside": round(float(g["mass_outside"].mean()), 4),
            "max_mass_outside": round(float(g["mass_outside"].max()), 4),
            "rows_missing_A_or_B": int(((g["a_missing"] + g["b_missing"]) > 0).sum()),
            "top_tokens": {repr(k): int(v) for k, v in g["top_token"].value_counts().head(5).items()},
            "order_gap": round(float(g.loc[g["order"] == "target_first", "p_target"].mean()
                                     - g.loc[g["order"] == "target_second", "p_target"].mean()), 4),
        }
        if method in ("matching", "ramsey"):
            avg = g.groupby(["prop_id", "grid_num"])["p_target"].mean().reset_index()
            corrs = [rank_corr(h["grid_num"], h["p_target"]) for _, h in avg.groupby("prop_id")
                     if h["grid_num"].nunique() > 2]
            corrs = [c for c in corrs if not math.isnan(c)]
            s["propositions"] = int(g["prop_id"].nunique())
            s["share_falling"] = round(float(np.mean([c < 0 for c in corrs])), 3) if corrs else None
        if method == "stage1":
            dom = g[g["expected_label"] != ""]
            if len(dom):
                chosen = np.where(dom["p_A"] > 0.5, "A", "B")
                s["dominance_correct"] = round(float((chosen == dom["expected_label"]).mean()), 3)
        out["by_method"][method] = s
    return out


def warnings_for(summary: dict) -> list[str]:
    warn = []
    for method, s in summary["by_method"].items():
        if s["mean_mass_outside"] > 0.1:
            warn.append(f"{method}: on average {s['mean_mass_outside']:.0%} of probability falls on tokens other "
                        "than A and B; check the top tokens and the chat template.")
        if s["rows_missing_A_or_B"] > 0.01 * s["rows"]:
            warn.append(f"{method}: A or B missing from the top tokens in {s['rows_missing_A_or_B']} rows; "
                        "check the answer token and raise --top-k.")
        if s.get("share_falling") is not None and s["share_falling"] < 0.8:
            warn.append(f"{method}: only {s['share_falling']:.0%} of propositions have choice probabilities that "
                        "fall as the grid value rises; read some raw prompts.")
        if abs(s["order_gap"]) > 0.2:
            warn.append(f"{method}: large position bias (order gap {s['order_gap']:+.2f}); averaging both orders "
                        "is essential.")
        if s.get("dominance_correct") is not None and s["dominance_correct"] < 1:
            warn.append(f"stage1: dominance respected in only {s['dominance_correct']:.0%} of rows.")
    return warn


def show_first(backend, prompts: pd.DataFrame) -> None:
    """Print the first prompt and its top tokens, to check the answer token."""
    row = prompts.iloc[0].to_dict()
    top = backend.score([row])[0]
    print("=" * 70)
    print(f"First prompt ({row['prompt_id']}), target is option {row['target_label']}:\n")
    print(row["user"])
    print("\nTop tokens (token text shown with repr, so spaces are visible):")
    for text, lp in sorted(top, key=lambda x: -x[1])[:20]:
        print(f"  {text!r:>12}  p = {math.exp(lp):.4f}")
    lp = choice_logprobs(top)
    probs = choice_probs(lp["lp_A"], lp["lp_B"], row["target_label"])
    print(f"\np_A = {probs['p_A']:.3f}, p_target = {probs['p_target']:.3f}, "
          f"mass outside A/B = {probs['mass_outside']:.3f}, missing A/B = {lp['a_missing']}/{lp['b_missing']}")
    print("=" * 70)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="prompts CSV, e.g. prompts/smoke.csv")
    ap.add_argument("--backend", choices=["vllm", "fake"], default="vllm")
    ap.add_argument("--model", help="Hugging Face repo id (required for vllm)")
    ap.add_argument("--run-id", help="name for this run (default: <input>_<model>)")
    ap.add_argument("--out-dir", type=Path, default=Path("results"))
    ap.add_argument("--batch-size", type=int, default=1000, help="prompts per batch (results saved after each)")
    ap.add_argument("--limit", type=int, help="only score this many new prompts")
    ap.add_argument("--show-first", action="store_true", help="print the first prompt's top tokens first")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--tokenizer-mode", help="e.g. mistral (see the model card)")
    ap.add_argument("--config-format", help="e.g. mistral (see the model card)")
    ap.add_argument("--load-format", help="e.g. mistral (see the model card)")
    ap.add_argument("--no-thinking", action="store_true", help="pass enable_thinking=False (Qwen3)")
    args = ap.parse_args(argv)

    if args.backend == "fake":
        backend, model = FakeBackend(), args.model or "fake"
    else:
        if not args.model:
            ap.error("--model is required with the vllm backend")
        backend = VLLMBackend(
            args.model, top_k=args.top_k, max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization, dtype=args.dtype,
            tokenizer_mode=args.tokenizer_mode, config_format=args.config_format,
            load_format=args.load_format,
            chat_template_kwargs={"enable_thinking": False} if args.no_thinking else None)
        model = args.model

    if args.show_first:
        show_first(backend, load_prompts(args.input))

    run_id = args.run_id or default_run_id(args.input, model)
    out_path = args.out_dir / f"raw_calls_{run_id}.csv"
    run(args.input, out_path, backend, run_id, model, args.batch_size, args.limit)

    results = pd.read_csv(out_path, dtype=str, keep_default_na=False)
    summary = summarize(results)
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Results: {out_path} ({summary['rows']} rows). Summary: {summary_path}")
    for method, s in summary["by_method"].items():
        extra = f", falling curves {s['share_falling']:.0%}" if s.get("share_falling") is not None else ""
        print(f"  {method}: {s['rows']} rows, mass outside A/B {s['mean_mass_outside']:.1%}, "
              f"order gap {s['order_gap']:+.2f}{extra}")
    for w in warnings_for(summary):
        print("WARNING: " + w)
    if args.backend == "vllm":
        print("Copy results/ back to your machine now.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

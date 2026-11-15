"""Run prompts with the model's reasoning switched on, and estimate choice probabilities by sampling.

With thinking off (run_prompts.py), the choice probability is read from the first
token's log-probabilities. With thinking on, the answer comes after hundreds of
tokens of reasoning, so each prompt is sampled several times and the choice
probability is the share of samples choosing the bet on the statement:

    p(A) = (number of A answers + 0.5) / (number of samples + 1)

(The +0.5 / +1 keeps p away from exactly 0 or 1 with few samples; samples with no
readable A or B count as probability outside A and B.) Results are written in the
same format as run_prompts.py, so stage1_analysis.py, analysis.py and smoke_gate.py
work on them unchanged, under the run name "<model>-thinking". Every sample's full
text, reasoning included, is kept in a .jsonl file next to the results.

Sampling uses Qwen's recommended thinking-mode settings by default:
temperature 0.6, top_p 0.95, top_k 20 (never greedy: it can loop).

Because sampling is costly, the claim files can be cut down:
    --claims N   keep N claims (each with its negation), spread across tiers, the same
                 ones for matching and Ramsey (fixed seed); a partition chosen at all is kept whole,
                 so its three cells can be tested for coherence
    --partitions keep whole partitions too ('all' or a number), so partition coherence can be tested
    --every M    keep every M-th grid value (2: k = 0, 10 ... 100 and c = £0, £100 ... £1000)

To add whole partitions to an earlier cut-down run, put its results file in results/ and run with
--claims 0 --partitions all: prompts already scored are skipped, the new ones are appended.

Usage, on the pod, from the project folder:

    python src/run_reasoning.py prompts/stage1.csv --model Qwen/Qwen3-32B
    python src/run_reasoning.py prompts/matching.csv --model Qwen/Qwen3-32B --claims 20 --every 2
    python src/run_reasoning.py prompts/ramsey.csv --model Qwen/Qwen3-32B --claims 20 --every 2

Test without a GPU:

    python src/run_reasoning.py prompts/stage1.csv --backend fake
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from pathlib import Path

import pandas as pd

import make_prompts as mp
import run_prompts as rp

SEED = 20261114
THINK_END = "</think>"
ANSWER_RE = re.compile(r"\b([AB])\b")


# ---------------------------------------------------------------------------
# Reading an answer out of a sampled response
# ---------------------------------------------------------------------------


def parse_answer(text: str) -> tuple[str, str]:
    """(answer, status): answer is 'A', 'B' or ''. Status: ok, no_answer, truncated (reasoning never ended)."""
    if THINK_END in text:
        final = text.split(THINK_END)[-1]
    elif "<think>" in text:
        return "", "truncated"
    else:
        final = text  # thinking switched off, or the template put <think> in the prompt
    m = ANSWER_RE.search(final.strip())
    return (m.group(1), "ok") if m else ("", "no_answer")


def counts_to_top(answers: list[str]) -> list[tuple[str, float]]:
    """Turn sampled answers into a top-token list that run_prompts' choice code understands."""
    n = len(answers)
    n_a, n_b = answers.count("A"), answers.count("B")
    n_bad = n - n_a - n_b
    denom = n + 1  # +0.5 for each of A and B
    top = [("A", math.log((n_a + 0.5) / denom)), ("B", math.log((n_b + 0.5) / denom))]
    if n_bad:
        top.append(("<no answer>", math.log(n_bad / denom)))
    return top


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class ReasoningBackend:
    """Samples each prompt n times with thinking on; returns pseudo top-k lists of answer frequencies."""

    name = "vllm-reasoning"

    def __init__(self, generate, samples: int, sample_log: Path | None):
        self.generate = generate  # (list of message lists, n) -> list of lists of (text, n_tokens)
        self.samples = samples
        self.sample_log = sample_log
        self.calls = 0

    def score(self, rows: list[dict]) -> list[list[tuple[str, float]]]:
        self.calls += 1
        outputs = self.generate([rp.build_messages(r) for r in rows], self.samples)
        tops, log_lines = [], []
        for row, outs in zip(rows, outputs):
            parsed = [parse_answer(text) for text, _ in outs]
            answers = [a for a, _ in parsed]
            tops.append(counts_to_top(answers))
            for i, ((text, n_tok), (ans, status)) in enumerate(zip(outs, parsed)):
                log_lines.append(json.dumps({"prompt_id": row["prompt_id"], "sample": i, "answer": ans,
                                             "status": status, "tokens": n_tok, "text": text}))
        if self.sample_log is not None and log_lines:
            self.sample_log.parent.mkdir(parents=True, exist_ok=True)
            with self.sample_log.open("a", encoding="utf-8") as f:
                f.write("\n".join(log_lines) + "\n")
        return tops


def vllm_generate(model: str, max_tokens: int, temperature: float, top_p: float, top_k: int,
                  max_model_len: int, gpu_memory_utilization: float, dtype: str, tensor_parallel_size: int,
                  seed: int = 0):
    from vllm import LLM, SamplingParams

    llm = LLM(model=model, max_model_len=max_model_len, gpu_memory_utilization=gpu_memory_utilization,
              dtype=dtype, tensor_parallel_size=tensor_parallel_size, seed=seed)

    def generate(conversations, n):
        params = SamplingParams(n=n, temperature=temperature, top_p=top_p, top_k=top_k, min_p=0.0,
                                max_tokens=max_tokens, seed=seed,
                                skip_special_tokens=False)  # keep <think>...</think> in the text for parsing
        outs = llm.chat(conversations, params, use_tqdm=False, chat_template_kwargs={"enable_thinking": True})
        return [[(c.text, len(c.token_ids)) for c in o.outputs] for o in outs]

    return generate


def fake_generate(conversations, n):
    """A pretend reasoning model: thinks briefly, then answers like run_prompts' FakeBackend would."""
    out = []
    for msgs in conversations:
        rng = random.Random(hashlib.md5(msgs[-1]["content"].encode()).hexdigest())
        p_a = 0.3 + 0.4 * rng.random()  # a prompt-specific lean, so tests see varied answers
        out.append([(f"<think>Comparing the options carefully.</think>\n\n{'A' if rng.random() < p_a else 'B'}", 12)
                    for _ in range(n)])
    return out


# ---------------------------------------------------------------------------
# Cutting a prompts file down
# ---------------------------------------------------------------------------


def partition_of(claim_id: str) -> str | None:
    """'part_chance_d6_e1' -> 'part_chance_d6'; None for claims outside partitions."""
    return claim_id.rsplit("_e", 1)[0] if claim_id.startswith("part_") else None


def choose_partitions(names: list[str], how, seed: int = SEED) -> set[str]:
    """'all', or N partitions spread across kinds (chance, fact, kuq), fixed seed."""
    if how in (None, 0, "0", "none"):
        return set()
    if str(how) == "all":
        return set(names)
    n = int(how)
    rng = random.Random(seed)
    by_kind: dict[str, list[str]] = {}
    for name in sorted(names):
        by_kind.setdefault(name.split("_")[1], []).append(name)
    for v in by_kind.values():
        rng.shuffle(v)
    chosen: list[str] = []
    while len(chosen) < n and any(by_kind.values()):
        for k in sorted(by_kind):
            if by_kind[k] and len(chosen) < n:
                chosen.append(by_kind[k].pop())
    return set(chosen)


def subset(prompts: pd.DataFrame, n_claims: int | None, every: int, seed: int = SEED,
           partitions=None) -> pd.DataFrame:
    """Cut a prompts file down: thin the grid, then keep n_claims claims and/or whole partitions.

    n_claims: that many claims (each with its negation), spread across tiers, fixed seed.
    partitions: 'all' or a number of WHOLE partitions (all three cells, each with its negation), so that
    partition coherence can be tested. Any partition cell picked among the n_claims also brings its siblings.
    """
    df = prompts.copy()
    if every > 1 and df["method"].isin(["matching", "ramsey"]).any():
        grids = {"matching": {str(v) for v in mp.K_VALUES[::every]},
                 "ramsey": {str(v) for v in mp.C_VALUES[::every]}}
        keep = [m not in grids or str(g) in grids[m] for m, g in zip(df["method"], df["grid_value"])]
        df = df[keep]
    if (n_claims is None and partitions is None) or not df["claim_id"].ne("").any():
        return df.reset_index(drop=True)
    claims = df[df["claim_id"] != ""].drop_duplicates("claim_id")[["claim_id", "tier"]].sort_values("claim_id")
    chosen: list[str] = []
    if n_claims:
        rng = random.Random(seed)
        by_tier = {t: sorted(g["claim_id"]) for t, g in claims.groupby("tier")}
        for ids in by_tier.values():
            rng.shuffle(ids)
        while len(chosen) < n_claims and any(by_tier.values()):
            for t in sorted(by_tier):
                if by_tier[t] and len(chosen) < n_claims:
                    chosen.append(by_tier[t].pop())
    all_parts = sorted({partition_of(c) for c in claims["claim_id"] if partition_of(c)})
    parts = {partition_of(c) for c in chosen if partition_of(c)} | choose_partitions(all_parts, partitions, seed)
    cells = [c for c in claims["claim_id"] if partition_of(c) in parts]
    df = df[(df["claim_id"] == "") | df["claim_id"].isin(set(chosen) | set(cells))]
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="prompts CSV, e.g. prompts/stage1.csv")
    ap.add_argument("--backend", choices=["vllm", "fake"], default="vllm")
    ap.add_argument("--model", help="Hugging Face repo id, e.g. Qwen/Qwen3-32B")
    ap.add_argument("--samples", type=int, default=5, help="samples per prompt")
    ap.add_argument("--claims", type=int, help="keep only this many claims (with negations)")
    ap.add_argument("--partitions", help="also keep whole partitions: 'all' or a number (spread across chance, "
                    "fact and KUQ), so partition coherence can be tested")
    ap.add_argument("--every", type=int, default=1, help="keep every M-th grid value")
    ap.add_argument("--max-tokens", type=int, default=3000, help="cap on reasoning + answer tokens per sample")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=200, help="prompts per batch (results saved after each)")
    ap.add_argument("--limit", type=int, help="only score this many new prompts (to time a first batch)")
    ap.add_argument("--out-dir", type=Path, default=Path("results"))
    args = ap.parse_args(argv)

    prompts = rp.load_prompts(args.input)
    cut = subset(prompts, args.claims, args.every, partitions=args.partitions)
    if len(cut) < len(prompts):
        tag = "" if args.partitions is None else f"_c{args.claims or 0}_p{args.partitions}"
        cut_path = args.input.with_name(f"{args.input.stem}_reasoning_subset{tag}.csv")
        cut.to_csv(cut_path, index=False)
        print(f"Using {len(cut)} of {len(prompts)} prompts ({cut['prop_id'].replace('', pd.NA).nunique()} "
              f"propositions); subset saved to {cut_path}")
    else:
        cut_path = args.input

    if args.backend == "fake":
        generate, model = fake_generate, "fake"
    else:
        if not args.model:
            ap.error("--model is required with the vllm backend")
        generate = vllm_generate(args.model, args.max_tokens, args.temperature, args.top_p, args.top_k,
                                 args.max_model_len, args.gpu_memory_utilization, args.dtype,
                                 args.tensor_parallel_size)
        model = args.model
    run = f"{rp.slug(model)}-thinking"
    out_path = args.out_dir / f"raw_calls_{args.input.stem}_{run}.csv"
    backend = ReasoningBackend(generate, args.samples, out_path.with_suffix(".samples.jsonl"))
    backend.name = "fake-reasoning" if args.backend == "fake" else "vllm-reasoning"
    rp.run(cut_path, out_path, backend, f"{args.input.stem}_{run}", model, args.batch_size, args.limit)

    results = pd.read_csv(out_path, dtype=str, keep_default_na=False)
    summary = rp.summarize(results)
    summary["samples_per_prompt"] = args.samples
    statuses = pd.read_json(out_path.with_suffix(".samples.jsonl"), lines=True)["status"].value_counts()
    summary["sample_status"] = {k: int(v) for k, v in statuses.items()}
    out_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Results: {out_path} ({summary['rows']} rows); samples: {out_path.with_suffix('.samples.jsonl')}")
    print(f"  sample status: {summary['sample_status']}")
    for method, s in summary["by_method"].items():
        extra = f", falling curves {s['share_falling']:.0%}" if s.get("share_falling") is not None else ""
        print(f"  {method}: {s['rows']} rows, no readable answer {s['mean_mass_outside']:.1%}, "
              f"order gap {s['order_gap']:+.2f}{extra}")
    print(f"Run name for the analysis scripts: {run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

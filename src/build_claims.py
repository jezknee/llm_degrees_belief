"""Build the claim set for the Ramsey credence project.

Reads the raw downloads in Questions/ (read-only) and writes:

    data/claims.csv        one row per proposition (long format)
    data/review.csv        every rejected, excluded or flagged item, with reasons
    data/conversions.jsonl cache of LLM conversions (reruns don't re-call the API)

Tiers
    forecast  ForecastBench market questions (Kalshi, Polymarket, Metaculus)
    kuq       Known-Unknown Questions, unknown rows
    fact      KUQ known rows, half made true and half false
    chance    template-generated claims about dice, coins, cards and roulette
    partition three-way partitions for Method C (chance, KUQ ladders, facts)

Every claim appears twice in claims.csv: neg=0 (the proposition) and neg=1
(its negation). For partitions, the neg=0 row is a single event and the neg=1
row is the union of the other two events (partition_role "single" / "pair").

Usage (from the project root):

    export OPENAI_API_KEY="sk-..."          # or $env:OPENAI_API_KEY="sk-..."
    python src/build_claims.py

    python src/build_claims.py --export                # no API: write prompts
    python src/build_claims.py --import converted.jsonl  # load converted answers
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MARKET_SOURCES = ("kalshi", "polymarket", "metaculus")


@dataclass
class Config:
    root: Path = field(default_factory=Path.cwd)
    forecastbench_file: str = "Questions/2026-11-08-llm.json"
    kuq_file: str = "Questions/knowns_unknowns.jsonl"
    seed: int = 20261114

    # ForecastBench window: must still be open after the sprint, and resolve soon.
    min_close: str = "2026-11-16"
    max_close: str = "2027-01-15"

    # Final targets per tier (claims, each giving two propositions).
    n_forecast: int = 120
    n_kuq: int = 80
    n_fact: int = 50
    n_chance: int = 30
    n_partition_chance: int = 4
    n_partition_kuq: int = 5
    n_partition_fact: int = 4

    # How many candidates to send for conversion (allows for rejections).
    oversample: float = 1.5

    # LLM settings.
    model: str = "gpt-4o-mini"
    workers: int = 8

    @property
    def data_dir(self) -> Path:
        return self.root / "data"


# Minimum share of each tier's target that must survive conversion and
# validation; below this the build reports a shortfall.
TARGET_FLOOR = 1.0

# ---------------------------------------------------------------------------
# Text checks
# ---------------------------------------------------------------------------

STOPWORDS = set(
    """a an the of to in on at by for from with without and or but if then than
    that this these those is are was were be been being will would shall should
    can could may might must do does did has have had having it its it's there
    their they them he she his her you your we our i me my as into over under
    about after before between during through up down out off again further
    once more most other some such no nor not only own same so too very just
    any all each few both what which who whom whose when where why how yes
    true false whether next one ever also""".split()
)

NEGATION_RE = re.compile(
    r"\b(not|no|never|none|nobody|nothing|neither|nor|without|fail|fails|failed|"
    r"cannot|lose|loses|lost|fewer|less|below|under|at most|false)\b|n't\b",
    re.IGNORECASE,
)

PLACEHOLDER_RE = re.compile(r"\{|\}|\[|\]|<|>|\bTBD\b|\bXXX\b|\.\.\.$|lorem", re.IGNORECASE)

MIN_LEN, MAX_LEN = 15, 300


def content_words(text: str) -> set[str]:
    """Lower-cased content words with a crude plural/possessive strip."""
    words = re.findall(r"[a-z0-9]+", str(text).lower())
    out = set()
    for w in words:
        if w in STOPWORDS or len(w) < 2:
            continue
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.add(w)
    return out


def related(question: str, statement: str) -> bool:
    """True if the statement shares at least one content word with the question."""
    if not question:
        return True
    return bool(content_words(question) & content_words(statement))


def hard_problems(text: str) -> list[str]:
    """Problems that exclude a proposition outright."""
    problems = []
    t = str(text or "").strip()
    if not t:
        return ["empty"]
    if "?" in t:
        problems.append("question mark")
    if PLACEHOLDER_RE.search(t):
        problems.append("placeholder or bracket")
    if len(t) < MIN_LEN:
        problems.append("too short")
    if len(t) > MAX_LEN:
        problems.append("too long")
    return problems


def check_pair(statement: str, negation: str, question: str | None) -> tuple[list[str], list[str]]:
    """Return (hard problems, soft flags) for a statement and its negation."""
    hard = [f"statement: {p}" for p in hard_problems(statement)]
    hard += [f"negation: {p}" for p in hard_problems(negation)]
    if str(statement).strip().lower() == str(negation).strip().lower():
        hard.append("negation identical to statement")
    if question is not None and not related(question, statement):
        hard.append("statement unrelated to source question")
    soft = []
    if not NEGATION_RE.search(str(negation)):
        soft.append("negation has no negating word; check it")
    return hard, soft


# ---------------------------------------------------------------------------
# Loading and sampling
# ---------------------------------------------------------------------------


def load_markets(cfg: Config) -> pd.DataFrame:
    with open(cfg.root / cfg.forecastbench_file, encoding="utf-8") as f:
        qs = json.load(f)["questions"]
    df = pd.DataFrame(qs)
    df = df[df["source"].isin(MARKET_SOURCES)].copy()
    close = pd.to_datetime(df["market_info_close_datetime"], utc=True, errors="coerce")
    lo = pd.Timestamp(cfg.min_close, tz="UTC")
    hi = pd.Timestamp(cfg.max_close, tz="UTC") + pd.Timedelta(days=1)
    df = df[(close >= lo) & (close < hi)].copy()
    df["close_date"] = close[df.index].dt.strftime("%Y-%m-%d")
    df["human_forecast"] = pd.to_numeric(df["freeze_datetime_value"], errors="coerce")
    df = df[df["human_forecast"].between(0, 1)]
    if "market_info_resolution_criteria" not in df:
        df["market_info_resolution_criteria"] = df.get("resolution_criteria", "")
    df["id"] = df["id"].astype(str)
    return df.reset_index(drop=True)


def load_kuq(cfg: Config) -> pd.DataFrame:
    df = pd.read_json(cfg.root / cfg.kuq_file, lines=True)
    df["row"] = range(len(df))
    unknown = df["unknown"]
    if unknown.dtype != bool:  # tolerate "true"/"false" strings
        unknown = unknown.astype(str).str.lower().isin(["true", "1"])
    df["unknown"] = unknown
    df["category"] = df.get("category", pd.Series([""] * len(df))).fillna("").astype(str)
    df["question"] = df["question"].astype(str).str.strip()
    df = df[df["question"].str.len() > 0]
    return df.drop_duplicates("question").reset_index(drop=True)


def stratified_sample(df: pd.DataFrame, col: str, n: int, seed: int) -> pd.DataFrame:
    """Sample n rows spread as evenly as possible across the values of col.

    Groups smaller than their share give everything; the remainder is spread
    over the larger groups. The result is shuffled reproducibly.
    """
    if n >= len(df):
        return df.sample(frac=1, random_state=seed).reset_index(drop=True)
    groups = {k: g.sample(frac=1, random_state=seed) for k, g in df.groupby(col, sort=True)}
    alloc = {k: 0 for k in groups}
    remaining = n
    open_groups = sorted(groups)
    while remaining > 0 and open_groups:
        share = max(1, remaining // len(open_groups))
        for k in list(open_groups):
            take = min(share, len(groups[k]) - alloc[k], remaining)
            alloc[k] += take
            remaining -= take
            if alloc[k] >= len(groups[k]):
                open_groups.remove(k)
            if remaining == 0:
                break
    parts = [groups[k].iloc[: alloc[k]] for k in groups]
    out = pd.concat(parts)
    return out.sample(frac=1, random_state=seed).reset_index(drop=True)


def interleave_by(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Reorder rows round-robin across groups, so trimming keeps the spread."""
    buckets = [list(g.index) for _, g in df.groupby(col, sort=True)]
    order = []
    while any(buckets):
        for b in buckets:
            if b:
                order.append(b.pop(0))
    return df.loc[order].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Conversion tasks and prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You convert questions into bettable propositions for a research study of how "
    "language models assign degrees of belief. Follow the instructions exactly and "
    "reply with a single JSON object only."
)

NEGATION_RULES = (
    "Write each negation as a natural sentence that is true exactly when the "
    "statement is false (not 'It is not the case that ...'). Never use question marks."
)


def forecast_prompt(t: dict) -> str:
    return f"""A question from a prediction market ({t['source']}):
Question: {t['question']}
Resolution criteria: {t['criteria']}
Market closes: {t['close_date']}

Write one declarative statement that is true exactly when this question resolves YES,
and its negation, true exactly when it resolves NO. Keep the names, numbers and dates
that matter for resolution. If the question has a bracketed outcome such as
"[Yes: Olivia Wilde]", the statement is about that outcome. {NEGATION_RULES}
If it cannot be written as one clear yes/no statement, set "usable" to false.

JSON keys: "usable" (true/false), "reason" (short), "proposition", "negation"."""


def kuq_prompt(t: dict) -> str:
    return f"""Question: {t['question']}

Nobody currently knows the answer to this question. Write one declarative statement
about it that is a definite matter of fact a perfectly informed judge could settle as
true or false, and whose truth is genuinely uncertain. Stay close to the question's
meaning. If the question is open-ended (when, how many, which), pick one sensible
threshold, e.g. "before 2035" or "more than 100". {NEGATION_RULES}
Set "usable" to false if the question is ill-posed, nonsensical, self-answering, a value
judgement or matter of taste, or not about a definite fact.

JSON keys: "usable" (true/false), "reason" (short), "proposition", "negation"."""


def fact_prompt(t: dict) -> str:
    return f"""Question: {t['question']}
Known answer(s): {t['answers']}

Write (a) a statement asserting the correct answer and (b) a statement of the same form
asserting one specific, plausible but wrong answer, plus the negation of each. Each
statement must make sense on its own, without any passage or context. {NEGATION_RULES}
Set "usable" to false if the question depends on missing context (e.g. "the passage",
"the author", "this text"), if the correct answer is unclear or disputed, or if the
answer is not a short fact.

JSON keys: "usable" (true/false), "reason" (short), "true_statement", "true_negation",
"false_statement", "false_negation"."""


def kuq_partition_prompt(t: dict) -> str:
    return f"""Question: {t['question']}

Split the possible answers into exactly three outcomes that are mutually exclusive
(at most one can be true) and exhaustive (exactly one must be true), for example time
ranges ("before 2030" / "between 2030 and 2050" / "after 2050 or never") or numeric
ranges. Write each outcome as a declarative statement, with its negation. {NEGATION_RULES}
Set "usable" to false if there is no sensible three-way split.

JSON keys: "usable" (true/false), "reason" (short), "outcomes": a list of exactly three
objects, each with "statement" and "negation"."""


def fact_partition_prompt(t: dict) -> str:
    return f"""Question: {t['question']}
Known answer(s): {t['answers']}

Write exactly three mutually exclusive and exhaustive outcomes, in this order:
1. a statement asserting the correct answer;
2. a statement asserting one specific, plausible but wrong answer;
3. a statement asserting that the answer is neither of those two.
Each statement must make sense on its own, without any passage or context, and comes
with its negation. {NEGATION_RULES}
Set "usable" to false if the question depends on missing context or its answer is unclear.

JSON keys: "usable" (true/false), "reason" (short), "outcomes": a list of exactly three
objects, each with "statement" and "negation"."""


PROMPTS: dict[str, Callable[[dict], str]] = {
    "forecast": forecast_prompt,
    "kuq": kuq_prompt,
    "fact": fact_prompt,
    "kuq_partition": kuq_partition_prompt,
    "fact_partition": fact_partition_prompt,
}


def make_tasks(markets: pd.DataFrame, kuq: pd.DataFrame, cfg: Config) -> dict[str, list[dict]]:
    """Sample candidates for each LLM tier and describe each as a task."""
    rng_seed = cfg.seed
    n = lambda target: int(round(target * cfg.oversample))  # noqa: E731

    fc = interleave_by(stratified_sample(markets, "source", n(cfg.n_forecast), rng_seed), "source")
    forecast = [
        {
            "item_id": f"fc_{r.source}_{r.id}",
            "kind": "forecast",
            "source": r.source,
            "source_id": r.id,
            "question": r.question,
            "criteria": (str(r.market_info_resolution_criteria) if pd.notna(r.market_info_resolution_criteria)
                         else str(getattr(r, "resolution_criteria", "")))[:1500],
            "close_date": r.close_date,
            "url": r.url,
            "human_forecast": float(r.human_forecast),
        }
        for r in fc.itertuples()
    ]

    unknown = kuq[kuq["unknown"]]
    known = kuq[~kuq["unknown"]]

    # Partition candidates first: ladder-friendly unknown questions.
    ladder = unknown["question"].str.match(r"(?i)^(when|how (many|much|long|old|far|big)|in what year|by what year)\b")
    kp_pool = unknown[ladder] if ladder.sum() >= n(cfg.n_partition_kuq) else unknown
    kp = kp_pool.sample(n=min(len(kp_pool), n(cfg.n_partition_kuq) + 3), random_state=rng_seed + 1)
    unknown_rest = unknown.drop(kp.index)

    kq = interleave_by(stratified_sample(unknown_rest, "category", n(cfg.n_kuq), rng_seed + 2), "category")

    fp = known.sample(n=min(len(known), n(cfg.n_partition_fact) + 3), random_state=rng_seed + 3)
    known_rest = known.drop(fp.index)
    fa = known_rest.sample(n=min(len(known_rest), n(cfg.n_fact)), random_state=rng_seed + 4)

    def answers(x) -> str:
        if isinstance(x, list):
            return "; ".join(str(a) for a in x[:3])
        return str(x)

    def kuq_task(r, kind):
        return {
            "item_id": f"{kind}_{r.row}",
            "kind": kind,
            "source": str(getattr(r, "source", "kuq")),
            "source_id": str(r.row),
            "question": r.question,
            "category": r.category,
            "answers": answers(r.answer) if kind.startswith("fact") else "",
        }

    return {
        "forecast": forecast,
        "kuq": [kuq_task(r, "kuq") for r in kq.itertuples()],
        "fact": [kuq_task(r, "fact") for r in fa.itertuples()],
        "kuq_partition": [kuq_task(r, "kuq_partition") for r in kp.itertuples()],
        "fact_partition": [kuq_task(r, "fact_partition") for r in fp.itertuples()],
    }


# ---------------------------------------------------------------------------
# Converters (LLM, cache, export/import)
# ---------------------------------------------------------------------------


class OpenAIConverter:
    """Calls an OpenAI chat model and parses its JSON reply."""

    def __init__(self, model: str, client=None, retries: int = 4):
        if client is None:
            from openai import OpenAI  # imported here so tests don't need the package

            if not os.environ.get("OPENAI_API_KEY"):
                raise SystemExit(
                    "OPENAI_API_KEY is not set. In this terminal run\n"
                    '  export OPENAI_API_KEY="sk-..."      (Mac/Linux)\n'
                    '  $env:OPENAI_API_KEY="sk-..."       (Windows PowerShell)\n'
                    "or use --export / --import to convert without the API."
                )
            client = OpenAI()
        self.client = client
        self.model = model
        self.retries = retries
        self.use_temperature = True  # switched off if the model rejects it

    def __call__(self, task: dict) -> dict:
        prompt = PROMPTS[task["kind"]](task)
        last_err = None
        for attempt in range(self.retries):
            try:
                kwargs = dict(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                )
                if self.use_temperature:
                    kwargs["temperature"] = 0
                resp = self.client.chat.completions.create(**kwargs)
                return parse_reply(resp.choices[0].message.content)
            except Exception as e:  # network errors, rate limits, bad JSON
                last_err = e
                if "temperature" in str(e).lower() and self.use_temperature:
                    self.use_temperature = False  # some models only allow the default
                    continue
                time.sleep(2**attempt)
        return {"usable": False, "reason": f"conversion failed: {last_err}"}


def parse_reply(text: str) -> dict:
    """Parse a model reply into a dict, tolerating code fences."""
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    data = json.loads(t)
    if not isinstance(data, dict):
        raise ValueError("reply is not a JSON object")
    data["usable"] = data.get("usable") in (True, "true", "True", 1)
    return data


class Cache:
    """JSONL cache of conversion results, keyed by item_id. Thread-safe."""

    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.data: dict[str, dict] = {}
        if path.exists():
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        self.data[rec["item_id"]] = rec["result"]

    def get(self, item_id: str) -> dict | None:
        return self.data.get(item_id)

    def put(self, item_id: str, result: dict) -> None:
        with self.lock:
            self.data[item_id] = result
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"item_id": item_id, "result": result}) + "\n")


def run_conversions(tasks: list[dict], cache: Cache, converter, workers: int) -> dict[str, dict]:
    """Convert every task, using the cache first. converter=None means cache only."""
    results: dict[str, dict] = {}
    todo = []
    for t in tasks:
        hit = cache.get(t["item_id"])
        if hit is not None:
            results[t["item_id"]] = hit
        elif converter is None:
            results[t["item_id"]] = {"usable": False, "reason": "not converted (no API call made)"}
        else:
            todo.append(t)

    def work(t):
        res = converter(t)
        if not str(res.get("reason", "")).startswith("conversion failed"):
            cache.put(t["item_id"], res)
        return t["item_id"], res

    if todo:
        print(f"Converting {len(todo)} questions with the API ...", file=sys.stderr)
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            for i, (item_id, res) in enumerate(ex.map(work, todo), 1):
                results[item_id] = res
                if i % 25 == 0:
                    print(f"  {i}/{len(todo)}", file=sys.stderr)
    return results


def export_tasks(tasks: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps({"item_id": t["item_id"], "kind": t["kind"], "prompt": PROMPTS[t["kind"]](t)}) + "\n")


def import_results(path: Path, cache: Cache) -> int:
    """Load converted answers: JSONL lines with item_id plus the reply fields."""
    count = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            item_id = rec.pop("item_id")
            rec.pop("kind", None)
            rec.pop("prompt", None)
            rec["usable"] = rec.get("usable") in (True, "true", "True", 1)
            cache.put(item_id, rec)
            count += 1
    return count


# ---------------------------------------------------------------------------
# Chance claims (no LLM)
# ---------------------------------------------------------------------------

SUITS = ["hearts", "diamonds", "clubs", "spades"]
RANKS = ["ace", "2", "3", "4", "5", "6", "7", "8", "9", "10", "jack", "queen", "king"]
RED_POCKETS = {1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36}


def _d(sides):
    return lambda rng: rng.randint(1, sides)


def chance_specs() -> list[dict]:
    """Templates: text, negation, exact chance, and a sampler returning truth."""
    specs = []

    def add(text, neg, p, draw):
        specs.append({"text": text, "negation": neg, "chance": p, "draw": draw})

    for k in (2, 3, 4, 5, 6):
        add(f"A fair six-sided die, rolled once, will show {k} or higher.",
            f"A fair six-sided die, rolled once, will not show {k} or higher.",
            (7 - k) / 6, lambda r, k=k: r.randint(1, 6) >= k)
    for k in (2, 5, 10, 15, 18):
        add(f"A fair twenty-sided die, rolled once, will show more than {k}.",
            f"A fair twenty-sided die, rolled once, will not show more than {k}.",
            (20 - k) / 20, lambda r, k=k: r.randint(1, 20) > k)
    for n in (1, 2, 3, 4):
        flips = "once" if n == 1 else f"{n} times"
        what = "heads" if n == 1 else "heads every time"
        add(f"A fair coin, tossed {flips}, will land {what}.",
            f"A fair coin, tossed {flips}, will not land {what}.",
            0.5**n, lambda r, n=n: all(r.random() < 0.5 for _ in range(n)))
    for n in (2, 3):
        add(f"A fair coin, tossed {n} times, will land heads at least once.",
            f"A fair coin, tossed {n} times, will never land heads.",
            1 - 0.5**n, lambda r, n=n: any(r.random() < 0.5 for _ in range(n)))
    card = lambda r: (r.choice(RANKS), r.choice(SUITS))  # noqa: E731
    add("A card drawn at random from a shuffled 52-card deck will be an ace.",
        "A card drawn at random from a shuffled 52-card deck will not be an ace.",
        4 / 52, lambda r: card(r)[0] == "ace")
    add("A card drawn at random from a shuffled 52-card deck will be a club.",
        "A card drawn at random from a shuffled 52-card deck will not be a club.",
        13 / 52, lambda r: card(r)[1] == "clubs")
    add("A card drawn at random from a shuffled 52-card deck will be a jack, queen or king.",
        "A card drawn at random from a shuffled 52-card deck will not be a jack, queen or king.",
        12 / 52, lambda r: card(r)[0] in ("jack", "queen", "king"))
    add("A card drawn at random from a shuffled 52-card deck will be red.",
        "A card drawn at random from a shuffled 52-card deck will not be red.",
        26 / 52, lambda r: card(r)[1] in ("hearts", "diamonds"))
    add("A card drawn at random from a shuffled 52-card deck will not be a spade.",
        "A card drawn at random from a shuffled 52-card deck will be a spade.",
        39 / 52, lambda r: card(r)[1] != "spades")
    add("A single spin of a fair European roulette wheel will land on a red number.",
        "A single spin of a fair European roulette wheel will not land on a red number.",
        18 / 37, lambda r: r.randint(0, 36) in RED_POCKETS)
    add("A single spin of a fair European roulette wheel will land on a non-zero even number.",
        "A single spin of a fair European roulette wheel will not land on a non-zero even number.",
        18 / 37, lambda r: (lambda x: x != 0 and x % 2 == 0)(r.randint(0, 36)))
    add("A single spin of a fair European roulette wheel will land on a number from 1 to 12.",
        "A single spin of a fair European roulette wheel will not land on a number from 1 to 12.",
        12 / 37, lambda r: 1 <= r.randint(0, 36) <= 12)
    add("Two fair six-sided dice, rolled together, will total exactly 7.",
        "Two fair six-sided dice, rolled together, will not total exactly 7.",
        6 / 36, lambda r: r.randint(1, 6) + r.randint(1, 6) == 7)
    add("Two fair six-sided dice, rolled together, will total 10 or more.",
        "Two fair six-sided dice, rolled together, will not total 10 or more.",
        6 / 36, lambda r: r.randint(1, 6) + r.randint(1, 6) >= 10)
    add("Two fair six-sided dice, rolled together, will total more than 3.",
        "Two fair six-sided dice, rolled together, will not total more than 3.",
        33 / 36, lambda r: r.randint(1, 6) + r.randint(1, 6) > 3)
    add("Two fair six-sided dice, rolled together, will show a double.",
        "Two fair six-sided dice, rolled together, will not show a double.",
        6 / 36, lambda r: r.randint(1, 6) == r.randint(1, 6))
    add("A fair spinner with eight equal sections numbered 1 to 8 will stop on 1, 2 or 3.",
        "A fair spinner with eight equal sections numbered 1 to 8 will not stop on 1, 2 or 3.",
        3 / 8, lambda r: r.randint(1, 8) <= 3)
    add("A fair spinner with eight equal sections numbered 1 to 8 will not stop on 8.",
        "A fair spinner with eight equal sections numbered 1 to 8 will stop on 8.",
        7 / 8, lambda r: r.randint(1, 8) != 8)
    return specs


def chance_partition_specs() -> list[dict]:
    """Exclusive, exhaustive three-way splits of one random draw."""
    return [
        {"name": "d6", "draw": _d(6), "events": [
            ("A fair six-sided die, rolled once, will show 1 or 2.",
             "A fair six-sided die, rolled once, will not show 1 or 2.", 2 / 6, lambda x: x <= 2),
            ("A fair six-sided die, rolled once, will show 3 or 4.",
             "A fair six-sided die, rolled once, will not show 3 or 4.", 2 / 6, lambda x: x in (3, 4)),
            ("A fair six-sided die, rolled once, will show 5 or 6.",
             "A fair six-sided die, rolled once, will not show 5 or 6.", 2 / 6, lambda x: x >= 5)]},
        {"name": "d10", "draw": _d(10), "events": [
            ("A fair ten-sided die, rolled once, will show a number from 1 to 3.",
             "A fair ten-sided die, rolled once, will not show a number from 1 to 3.", 0.3, lambda x: x <= 3),
            ("A fair ten-sided die, rolled once, will show a number from 4 to 7.",
             "A fair ten-sided die, rolled once, will not show a number from 4 to 7.", 0.4, lambda x: 4 <= x <= 7),
            ("A fair ten-sided die, rolled once, will show a number from 8 to 10.",
             "A fair ten-sided die, rolled once, will not show a number from 8 to 10.", 0.3, lambda x: x >= 8)]},
        {"name": "suit", "draw": lambda r: r.choice(SUITS), "events": [
            ("A card drawn at random from a shuffled 52-card deck will be a heart.",
             "A card drawn at random from a shuffled 52-card deck will not be a heart.", 0.25, lambda x: x == "hearts"),
            ("A card drawn at random from a shuffled 52-card deck will be a diamond.",
             "A card drawn at random from a shuffled 52-card deck will not be a diamond.", 0.25, lambda x: x == "diamonds"),
            ("A card drawn at random from a shuffled 52-card deck will be a club or a spade.",
             "A card drawn at random from a shuffled 52-card deck will not be a club or a spade.", 0.5,
             lambda x: x in ("clubs", "spades"))]},
        {"name": "roulette", "draw": lambda r: r.randint(0, 36), "events": [
            ("A single spin of a fair European roulette wheel will land on zero.",
             "A single spin of a fair European roulette wheel will not land on zero.", 1 / 37, lambda x: x == 0),
            ("A single spin of a fair European roulette wheel will land on a number from 1 to 18.",
             "A single spin of a fair European roulette wheel will not land on a number from 1 to 18.", 18 / 37,
             lambda x: 1 <= x <= 18),
            ("A single spin of a fair European roulette wheel will land on a number from 19 to 36.",
             "A single spin of a fair European roulette wheel will not land on a number from 19 to 36.", 18 / 37,
             lambda x: x >= 19)]},
    ]


CHANCE_SETTLES = "Settled by a seeded random draw made by the experimenter; the outcome is stored separately."
KUQ_SETTLES = "Settled by an infallible judge who knows the true answer."
FACT_SETTLES = "Already settled as a matter of established fact."

# ---------------------------------------------------------------------------
# Building rows
# ---------------------------------------------------------------------------

COLUMNS = [
    "prop_id", "claim_id", "neg", "tier", "text", "truth", "objective_chance", "settles",
    "source", "source_id", "source_question", "kuq_category", "human_forecast",
    "close_date", "partition_id", "partition_role", "flags",
]


def _truth_str(v) -> str:
    if v is None:
        return "unknown"
    return "true" if v else "false"


def claim_rows(claim_id, tier, text, negation, truth=None, chance=None, settles="", source="",
               source_id="", question="", category="", human=None, close_date="",
               partition_id="", flags=()) -> list[dict]:
    """Two rows: the proposition (neg=0) and its negation (neg=1)."""
    base = dict(claim_id=claim_id, tier=tier, settles=settles, source=source, source_id=str(source_id),
                source_question=question, kuq_category=category, close_date=close_date,
                partition_id=partition_id, flags="; ".join(flags))
    pos = dict(base, prop_id=f"{claim_id}_p", neg=0, text=text, truth=_truth_str(truth),
               objective_chance=chance, human_forecast=human,
               partition_role="single" if partition_id else "")
    neg = dict(base, prop_id=f"{claim_id}_n", neg=1, text=negation,
               truth=_truth_str(None if truth is None else not truth),
               objective_chance=None if chance is None else 1 - chance,
               human_forecast=None if human is None else 1 - human,
               partition_role="pair" if partition_id else "")
    return [pos, neg]


@dataclass
class Build:
    claims: pd.DataFrame
    review: pd.DataFrame
    counts: dict
    shortfalls: list


def build(cfg: Config, converter=None, cache: Cache | None = None) -> Build:
    """Run the whole pipeline. converter=None uses cached conversions only."""
    cache = cache or Cache(cfg.data_dir / "conversions.jsonl")
    markets = load_markets(cfg)
    kuq = load_kuq(cfg)
    tasks = make_tasks(markets, kuq, cfg)
    all_tasks = [t for ts in tasks.values() for t in ts]
    results = run_conversions(all_tasks, cache, converter, cfg.workers)

    rows: list[dict] = []
    review: list[dict] = []
    seen_texts: set[str] = set()

    def reject(t, status, reasons, proposition="", negation=""):
        review.append({"item_id": t["item_id"], "kind": t["kind"], "status": status,
                       "reasons": "; ".join(reasons), "source_question": t.get("question", ""),
                       "proposition": proposition, "negation": negation})

    def accept_pair(t, statement, negation):
        """Validate one statement/negation pair; return soft flags or None if excluded."""
        hard, soft = check_pair(statement, negation, t.get("question"))
        for s in (statement, negation):
            if str(s).strip().lower() in seen_texts:
                hard.append("duplicate of an earlier proposition")
        if hard:
            reject(t, "excluded", hard, statement, negation)
            return None
        return soft

    # --- forecasts and KUQ: one claim per question ---------------------------
    counts = {}
    for kind, target, settles in (("forecast", cfg.n_forecast, None), ("kuq", cfg.n_kuq, KUQ_SETTLES)):
        kept = 0
        for t in tasks[kind]:
            res = results[t["item_id"]]
            if kept >= target:
                break
            if not res.get("usable"):
                reject(t, "rejected", [res.get("reason", "unusable")])
                continue
            prop, neg = str(res.get("proposition", "")).strip(), str(res.get("negation", "")).strip()
            soft = accept_pair(t, prop, neg)
            if soft is None:
                continue
            if soft:
                reject(t, "flagged", soft, prop, neg)
            seen_texts.update({prop.lower(), neg.lower()})
            if kind == "forecast":
                settles_text = f"Resolves YES or NO by {t['close_date']} on {t['source'].title()} ({t['url']})."
                rows += claim_rows(t["item_id"], "forecast", prop, neg, settles=settles_text,
                                   source=t["source"], source_id=t["source_id"], question=t["question"],
                                   human=t["human_forecast"], close_date=t["close_date"], flags=soft)
            else:
                rows += claim_rows(t["item_id"], "kuq", prop, neg, settles=settles, source=t["source"],
                                   source_id=t["source_id"], question=t["question"],
                                   category=t["category"], flags=soft)
            kept += 1
        counts[kind] = kept

    # --- facts: half true, half false ----------------------------------------
    kept = 0
    for t in tasks["fact"]:
        if kept >= cfg.n_fact:
            break
        res = results[t["item_id"]]
        if not res.get("usable"):
            reject(t, "rejected", [res.get("reason", "unusable")])
            continue
        make_true = kept % 2 == 0
        key = "true" if make_true else "false"
        prop = str(res.get(f"{key}_statement", "")).strip()
        neg = str(res.get(f"{key}_negation", "")).strip()
        other = str(res.get(f"{'false' if make_true else 'true'}_statement", "")).strip()
        soft = accept_pair(t, prop, neg)
        if soft is None:
            continue
        if other.lower() == prop.lower():
            reject(t, "excluded", ["true and false statements identical"], prop, neg)
            continue
        if soft:
            reject(t, "flagged", soft, prop, neg)
        seen_texts.update({prop.lower(), neg.lower()})
        rows += claim_rows(t["item_id"], "fact", prop, neg, truth=make_true, settles=FACT_SETTLES,
                           source=t["source"], source_id=t["source_id"], question=t["question"],
                           category=t["category"], flags=soft)
        kept += 1
    counts["fact"] = kept

    # --- chance claims ---------------------------------------------------------
    rng = random.Random(cfg.seed)
    specs = chance_specs()
    rng.shuffle(specs)
    for i, s in enumerate(specs[: cfg.n_chance]):
        rows += claim_rows(f"chance_{i:02d}", "chance", s["text"], s["negation"], truth=s["draw"](rng),
                           chance=s["chance"], settles=CHANCE_SETTLES, source="template")
    counts["chance"] = min(cfg.n_chance, len(specs))

    # --- partitions --------------------------------------------------------------
    pcount = 0
    for p in chance_partition_specs()[: cfg.n_partition_chance]:
        x = p["draw"](rng)
        pid = f"part_chance_{p['name']}"
        for j, (text, neg, prob, test) in enumerate(p["events"], 1):
            rows += claim_rows(f"{pid}_e{j}", "partition", text, neg, truth=bool(test(x)),
                               chance=prob, settles=CHANCE_SETTLES, source="template",
                               partition_id=pid)
        pcount += 1
    counts["partition_chance"] = pcount

    for kind, target, settles in (("kuq_partition", cfg.n_partition_kuq, KUQ_SETTLES),
                                  ("fact_partition", cfg.n_partition_fact, FACT_SETTLES)):
        kept = 0
        for t in tasks[kind]:
            if kept >= target:
                break
            res = results[t["item_id"]]
            outcomes = res.get("outcomes") if res.get("usable") else None
            if not res.get("usable"):
                reject(t, "rejected", [res.get("reason", "unusable")])
                continue
            if not isinstance(outcomes, list) or len(outcomes) != 3:
                reject(t, "excluded", ["partition does not have exactly three outcomes"])
                continue
            pairs = [(str(o.get("statement", "")).strip(), str(o.get("negation", "")).strip()) for o in outcomes]
            problems, flags = [], []
            for s, ng in pairs:
                hard, soft = check_pair(s, ng, t["question"])
                problems += hard
                flags += soft
                if s.lower() in seen_texts or ng.lower() in seen_texts:
                    problems.append("duplicate of an earlier proposition")
            if len({s.lower() for s, _ in pairs}) < 3:
                problems.append("partition outcomes not distinct")
            if problems:
                reject(t, "excluded", sorted(set(problems)), " | ".join(s for s, _ in pairs))
                continue
            if flags:
                reject(t, "flagged", sorted(set(flags)) + ["check the three outcomes are exclusive and exhaustive"],
                       " | ".join(s for s, _ in pairs))
            else:
                reject(t, "flagged", ["check the three outcomes are exclusive and exhaustive"],
                       " | ".join(s for s, _ in pairs))
            pid = f"part_{t['item_id']}"
            for j, (s, ng) in enumerate(pairs, 1):
                truth = None
                if kind == "fact_partition":
                    truth = j == 1
                seen_texts.update({s.lower(), ng.lower()})
                rows += claim_rows(f"{pid}_e{j}", "partition", s, ng, truth=truth, settles=settles,
                                   source=t["source"], source_id=t["source_id"], question=t["question"],
                                   category=t["category"], partition_id=pid, flags=flags)
            kept += 1
        counts[kind] = kept

    claims = pd.DataFrame(rows, columns=COLUMNS)
    review_df = pd.DataFrame(review, columns=["item_id", "kind", "status", "reasons",
                                              "source_question", "proposition", "negation"])

    targets = {"forecast": cfg.n_forecast, "kuq": cfg.n_kuq, "fact": cfg.n_fact,
               "chance": cfg.n_chance, "partition_chance": cfg.n_partition_chance,
               "kuq_partition": cfg.n_partition_kuq, "fact_partition": cfg.n_partition_fact}
    shortfalls = [f"{k}: {counts.get(k, 0)} of {v}" for k, v in targets.items()
                  if counts.get(k, 0) < TARGET_FLOOR * v]
    return Build(claims, review_df, counts, shortfalls)


def write_outputs(b: Build, cfg: Config) -> None:
    """Write claims.csv and review.csv; an existing claims.csv is saved as claims.backup.csv first."""
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.data_dir / "claims.csv"
    if path.exists():
        path.replace(cfg.data_dir / "claims.backup.csv")
    b.claims.to_csv(path, index=False)
    b.review.to_csv(cfg.data_dir / "review.csv", index=False)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path.cwd(), help="project root (default: current folder)")
    ap.add_argument("--model", default=Config.model, help="OpenAI model for conversion")
    ap.add_argument("--seed", type=int, default=Config.seed)
    ap.add_argument("--workers", type=int, default=Config.workers)
    ap.add_argument("--export", action="store_true", help="write data/to_convert.jsonl and stop (no API)")
    ap.add_argument("--import", dest="import_file", type=Path, help="load converted answers, then build (no API)")
    ap.add_argument("--no-api", action="store_true", help="build from cached conversions only")
    args = ap.parse_args(argv)

    cfg = Config(root=args.root.resolve(), model=args.model, seed=args.seed, workers=args.workers)
    cache = Cache(cfg.data_dir / "conversions.jsonl")

    if args.export:
        tasks = make_tasks(load_markets(cfg), load_kuq(cfg), cfg)
        todo = [t for ts in tasks.values() for t in ts if cache.get(t["item_id"]) is None]
        out = cfg.data_dir / "to_convert.jsonl"
        export_tasks(todo, out)
        print(f"Wrote {len(todo)} prompts to {out}. Convert them, then run with --import FILE.")
        return 0

    if args.import_file:
        n = import_results(args.import_file, cache)
        print(f"Imported {n} conversions.")

    converter = None if (args.import_file or args.no_api) else OpenAIConverter(cfg.model)
    b = build(cfg, converter, cache)
    write_outputs(b, cfg)

    n_claims = b.claims["claim_id"].nunique()
    print(f"Wrote {len(b.claims)} propositions ({n_claims} claims) to {cfg.data_dir / 'claims.csv'}")
    print("Claims per tier: " + ", ".join(f"{k} {v}" for k, v in b.counts.items()))
    status = b.review["status"].value_counts().to_dict() if len(b.review) else {}
    print(f"Review file: {len(b.review)} items {status} -> {cfg.data_dir / 'review.csv'}")
    if b.shortfalls:
        print("SHORTFALL (fewer than target): " + "; ".join(b.shortfalls))
        print("Raise Config.oversample or check review.csv for why items were dropped.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

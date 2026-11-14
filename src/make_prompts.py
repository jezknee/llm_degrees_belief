"""Generate every fixed prompt for the Ramsey credence project.

Reads data/claims.csv and writes, into prompts/:

    stage1.csv     Stage 1 tests: neutral event, ticket symmetry, validation,
                   dominance and transitivity
    matching.csv   the matching method (probability matching) for every proposition
    ramsey.csv     the Ramsey method (sure amount vs bet) for every proposition
    smoke.csv      both methods for five propositions, for the smoke test
    manifest.json  template version, values used and row counts

Each row is one binary choice, already in both option orders. The prompt is
stored as a system message plus a user message; run_prompts.py applies the
model's chat template. The utility scale is adaptive, so its prompts are made
on the pod by utility_scale.py, not here.

Usage (from the project root):

    python src/make_prompts.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from pathlib import Path

import pandas as pd

TEMPLATE_VERSION = "v1"
PRIZE = 1000                    # pounds paid by every winning bet
K_VALUES = list(range(0, 101, 5))       # matching method: tickets 1..k
C_VALUES = list(range(0, 1001, 50))     # Ramsey method: sure amount in pounds
ORDERS = ("target_first", "target_second")
SEED = 20261114

SYSTEM_PROMPT = (
    "You will be offered a choice between two options. Choose the one you prefer. "
    "Answer with a single letter, A or B, and nothing else."
)

TICKETS_SETUP = (
    "There are 100 tickets, numbered 1 to 100, each labelled with a different "
    "nonsense syllable. One ticket will be drawn at random."
)

COLUMNS = [
    "prompt_id", "method", "step", "item_id", "prop_id", "claim_id", "neg", "tier",
    "grid_value", "grid_unit", "order", "target_label", "expected_label",
    "target_desc", "other_desc", "system", "user", "template_version",
]

REQUIRED_CLAIM_COLUMNS = ["prop_id", "claim_id", "neg", "tier", "text", "settles"]

# ---------------------------------------------------------------------------
# Wording
# ---------------------------------------------------------------------------


def money(x: int) -> str:
    return f"£{int(x)}"


def claim_bet(text: str) -> str:
    return f'You receive {money(PRIZE)} if the following statement is true, and nothing otherwise: "{text}"'


def ticket_bet(k: int, prize: int = PRIZE) -> str:
    """A bet on the ticket drawn being one of tickets 1..k."""
    if k <= 0:
        return "You receive nothing, whichever ticket is drawn."
    if k >= 100:
        return f"You receive {money(prize)}, whichever ticket is drawn."
    if k == 1:
        return f"You receive {money(prize)} if the ticket drawn is ticket 1, and nothing otherwise."
    return f"You receive {money(prize)} if the ticket drawn is one of tickets 1 to {k}, and nothing otherwise."


def ticket_range_bet(lo: int, hi: int, prize: int = PRIZE) -> str:
    if lo == hi:
        return f"You receive {money(prize)} if the ticket drawn is ticket {lo}, and nothing otherwise."
    return f"You receive {money(prize)} if the ticket drawn is one of tickets {lo} to {hi}, and nothing otherwise."


def sure_amount(c: int) -> str:
    return f"You receive {money(c)} for certain."


def stated_chance_bet(pct: int) -> str:
    return (f"You receive {money(PRIZE)} with a {pct}% chance, decided by a separate fair random draw, "
            "and nothing otherwise.")


def readable_date(value: str) -> str:
    """'2026-12-04' (or Excel's '04/12/2026') -> '4 December 2026'; unparseable dates pass through."""
    v = str(value).strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y-%m-%dT%H:%M:%S"):
        try:
            d = pd.to_datetime(v, format=fmt)
            return f"{d.day} {d.strftime('%B %Y')}"
        except (ValueError, TypeError):
            continue
    return v


def settlement_sentence(row) -> str:
    """How a bet on this proposition is settled, phrased for the model."""
    settles = str(row.get("settles", ""))
    low = settles.lower()
    if "random draw" in low:
        return "The random draw will be made fairly, after you have chosen."
    if "judge" in low:
        return "An infallible judge who knows the true answer will settle the bet."
    if "established fact" in low:
        return "The bet is settled by established fact."
    if low.startswith("resolves yes or no"):
        source = str(row.get("source", "")).strip().title() or "the prediction market"
        date = readable_date(row.get("close_date", ""))
        when = f", due by {date}" if date else ""
        return f"The bet is settled by how the {source} prediction market on this question resolves{when}."
    return settles.strip()


def user_message(option_a: str, option_b: str, setup: str = "", settle: str = "") -> str:
    parts = []
    if setup:
        parts.append(setup)
    parts.append(f"Option A: {option_a}\nOption B: {option_b}")
    if settle:
        parts.append(f"How a bet on the statement is settled: {settle}")
    parts.append("Which option do you choose? Answer with a single letter: A or B.")
    return "\n\n".join(parts)


def choice_rows(base: dict, target: str, other: str, setup: str = "", settle: str = "",
                expected_target: bool = False) -> list[dict]:
    """Two rows for one choice: target as Option A, then target as Option B."""
    rows = []
    for order in ORDERS:
        first = order == "target_first"
        a, b = (target, other) if first else (other, target)
        label = "A" if first else "B"
        rows.append(dict(
            base,
            prompt_id=f"{base['prompt_id']}_{'tf' if first else 'ts'}",
            order=order,
            target_label=label,
            expected_label=label if expected_target else "",
            target_desc=target,
            other_desc=other,
            system=SYSTEM_PROMPT,
            user=user_message(a, b, setup, settle),
            template_version=TEMPLATE_VERSION,
        ))
    return rows


# ---------------------------------------------------------------------------
# Claims
# ---------------------------------------------------------------------------


def load_claims(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in REQUIRED_CLAIM_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"{path} is missing columns: {missing}")
    if df["prop_id"].duplicated().any():
        raise SystemExit(f"{path} has duplicate prop_id values")
    df["neg"] = df["neg"].astype(int)
    for col in ("source", "close_date"):
        if col not in df.columns:
            df[col] = ""
    return df


def claim_base(row, method: str, grid_value: int, grid_unit: str) -> dict:
    return dict(
        prompt_id=f"{method}_{row['prop_id']}_{grid_value:04d}",
        method=method,
        step="",
        item_id=row["prop_id"],
        prop_id=row["prop_id"],
        claim_id=row["claim_id"],
        neg=row["neg"],
        tier=row["tier"],
        grid_value=grid_value,
        grid_unit=grid_unit,
    )


def matching_prompts(claims: pd.DataFrame, k_values=K_VALUES) -> pd.DataFrame:
    """Matching method: '£1000 if statement' vs '£1000 if ticket is one of 1..k'."""
    rows = []
    for _, row in claims.iterrows():
        target = claim_bet(row["text"])
        settle = settlement_sentence(row)
        for k in k_values:
            base = claim_base(row, "matching", k, "tickets")
            rows += choice_rows(base, target, ticket_bet(k), setup=TICKETS_SETUP, settle=settle)
    return pd.DataFrame(rows, columns=COLUMNS)


def ramsey_prompts(claims: pd.DataFrame, c_values=C_VALUES) -> pd.DataFrame:
    """Ramsey method: '£c for certain' vs '£1000 if statement, nothing otherwise'.

    The target is always the bet on the statement, so for both methods
    P(choose target) falls as the grid value rises.
    """
    rows = []
    for _, row in claims.iterrows():
        target = claim_bet(row["text"])
        settle = settlement_sentence(row)
        for c in c_values:
            base = claim_base(row, "ramsey", c, "gbp")
            rows += choice_rows(base, target, sure_amount(c), settle=settle)
    return pd.DataFrame(rows, columns=COLUMNS)


# ---------------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------------

# Candidate neutral events. "Fair" is deliberately not said, so the test asks
# whether the model itself treats the two sides as equally likely.
NEUTRAL_CANDIDATES = [
    ("disc", "A disc with the nonsense syllable ZOJ on one side and ZEJ on the other will be tossed once.",
     "the disc lands ZOJ side up", "the disc lands ZEJ side up"),
    ("coin", "A coin will be tossed once.", "the coin lands heads", "the coin lands tails"),
    ("die", "A six-sided die will be rolled once.", "the die shows 1, 2 or 3", "the die shows 4, 5 or 6"),
]
OUTCOME_PAIRS = [(1000, 0), (500, 100), (1000, 200)]
VALIDATION_K = [10, 25, 50, 75, 90]
SYMMETRY_BLOCKS = [((1, 50), (51, 100)), ((1, 25), (26, 50)), ((1, 10), (91, 100))]
N_SINGLE_PAIRS = 30
N_TRIPLES = 20

TRANSITIVITY_POOL = [
    ("sure450", sure_amount(450)),
    ("sure520", sure_amount(520)),
    ("t50x1000", ticket_bet(50, 1000)),
    ("t70x700", ticket_bet(70, 700)),
    ("t55x900", ticket_bet(55, 900)),
    ("t80x600", ticket_bet(80, 600)),
    ("t60x800", ticket_bet(60, 800)),
    ("t45x1000", ticket_bet(45, 1000)),
]


def stage1_base(step: str, item_id: str, pid: str, grid_value="", grid_unit="") -> dict:
    return dict(prompt_id=f"stage1_{pid}", method="stage1", step=step, item_id=item_id, prop_id="",
                claim_id="", neg="", tier="", grid_value=grid_value, grid_unit=grid_unit)


def stage1_prompts(seed: int = SEED) -> pd.DataFrame:
    rng = random.Random(seed)
    rows: list[dict] = []

    # 1. Neutral event (Ramsey Step 1): "a if E, b if not" vs "b if E, a if not".
    for name, setup, e1, e2 in NEUTRAL_CANDIDATES:
        for hi, lo in OUTCOME_PAIRS:
            target = f"You receive {money(hi)} if {e1}, and {money(lo)} if {e2}."
            other = f"You receive {money(lo)} if {e1}, and {money(hi)} if {e2}."
            pid = f"neutral_{name}_{hi}_{lo}"
            rows += choice_rows(stage1_base("neutral", f"neutral_{name}", pid), target, other, setup=setup)

    # 2. Ticket symmetry: single tickets, then blocks.
    pairs = set()
    while len(pairs) < N_SINGLE_PAIRS:
        i, j = sorted(rng.sample(range(1, 101), 2))
        pairs.add((i, j))
    for i, j in sorted(pairs):
        pid = f"symmetry_single_{i:03d}_{j:03d}"
        rows += choice_rows(stage1_base("symmetry_single", pid, pid),
                            ticket_range_bet(i, i), ticket_range_bet(j, j), setup=TICKETS_SETUP)
    for (a1, a2), (b1, b2) in SYMMETRY_BLOCKS:
        pid = f"symmetry_block_{a1}_{a2}_vs_{b1}_{b2}"
        rows += choice_rows(stage1_base("symmetry_block", pid, pid),
                            ticket_range_bet(a1, a2), ticket_range_bet(b1, b2), setup=TICKETS_SETUP)

    # 3. Validation: tickets vs stated chances, and a 50% chance vs each neutral candidate.
    for k in VALIDATION_K:
        pid = f"validation_tickets_{k:03d}"
        rows += choice_rows(stage1_base("validation_tickets", pid, pid, k, "tickets"),
                            ticket_bet(k), stated_chance_bet(k), setup=TICKETS_SETUP)
    for name, setup, e1, _ in NEUTRAL_CANDIDATES:
        pid = f"validation_neutral_{name}"
        rows += choice_rows(stage1_base("validation_neutral", pid, pid),
                            f"You receive {money(PRIZE)} if {e1}, and nothing otherwise.",
                            stated_chance_bet(50), setup=setup)

    # 4. Dominance: the target option is always better, so it should be chosen.
    pid = "dominance_tickets_60_vs_40"
    rows += choice_rows(stage1_base("dominance", pid, pid), ticket_bet(60), ticket_bet(40),
                        setup=TICKETS_SETUP, expected_target=True)
    pid = "dominance_sure_600_vs_400"
    rows += choice_rows(stage1_base("dominance", pid, pid), sure_amount(600), sure_amount(400),
                        expected_target=True)
    for name, setup, e1, _ in NEUTRAL_CANDIDATES:
        pid = f"dominance_neutral_{name}"
        rows += choice_rows(stage1_base("dominance", pid, pid),
                            f"You receive {money(1000)} if {e1}, and nothing otherwise.",
                            f"You receive {money(500)} if {e1}, and nothing otherwise.",
                            setup=setup, expected_target=True)

    # 5. Transitivity: every pair within each triple of options.
    triples = sorted(itertools.combinations(range(len(TRANSITIVITY_POOL)), 3))
    for t, triple in enumerate(sorted(rng.sample(triples, N_TRIPLES))):
        item = f"transitivity_{t:02d}"
        for x, y in itertools.combinations(triple, 2):
            (nx, dx), (ny, dy) = TRANSITIVITY_POOL[x], TRANSITIVITY_POOL[y]
            pid = f"{item}_{nx}_vs_{ny}"
            rows += choice_rows(stage1_base("transitivity", item, pid), dx, dy, setup=TICKETS_SETUP)

    return pd.DataFrame(rows, columns=COLUMNS)


# ---------------------------------------------------------------------------
# Smoke subset and writing
# ---------------------------------------------------------------------------


def smoke_subset(claims: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """One proposition (neg=0) per tier, in a fixed order, up to n."""
    pos = claims[claims["neg"] == 0]
    picks = []
    for tier in ["forecast", "kuq", "fact", "chance", "partition"]:
        hit = pos[pos["tier"] == tier]
        if len(hit):
            picks.append(hit.iloc[0])
    rest = pos[~pos["prop_id"].isin([p["prop_id"] for p in picks])]
    for _, r in rest.iterrows():
        if len(picks) >= n:
            break
        picks.append(r)
    return pd.DataFrame(picks[:n]).reset_index(drop=True)


def build_all(claims: pd.DataFrame, seed: int = SEED) -> dict[str, pd.DataFrame]:
    smoke_claims = smoke_subset(claims)
    return {
        "stage1": stage1_prompts(seed),
        "matching": matching_prompts(claims),
        "ramsey": ramsey_prompts(claims),
        "smoke": pd.concat([matching_prompts(smoke_claims), ramsey_prompts(smoke_claims)], ignore_index=True),
    }


def write_all(outputs: dict[str, pd.DataFrame], out_dir: Path, seed: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in outputs.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)
    manifest = {
        "template_version": TEMPLATE_VERSION,
        "seed": seed,
        "prize_gbp": PRIZE,
        "k_values": K_VALUES,
        "c_values": C_VALUES,
        "orders": list(ORDERS),
        "system_prompt": SYSTEM_PROMPT,
        "rows": {name: len(df) for name, df in outputs.items()},
        "propositions": int(outputs["matching"]["prop_id"].nunique()),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path.cwd(), help="project root (default: current folder)")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)

    root = args.root.resolve()
    claims = load_claims(root / "data" / "claims.csv")
    outputs = build_all(claims, args.seed)
    m = write_all(outputs, root / "prompts", args.seed)
    print(f"{m['propositions']} propositions -> prompts/ (template {TEMPLATE_VERSION})")
    for name, n in m["rows"].items():
        print(f"  {name}.csv: {n} prompts")
    return 0


if __name__ == "__main__":
    sys.exit(main())

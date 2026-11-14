"""Ramsey's utility scale, built adaptively from bets on the neutral event N.

Runs on the pod (it needs the model), level by level, because each level's bets
use the amounts found at the level before:

  level 1  m(1/2): "£1000 if N, £0 if not-N"  vs  "£m for certain"   ->  u(m(1/2)) = 1/2
  level 2  m(1/4): "£m(1/2) if N, £0 if not"  vs  "£m for certain"   ->  u = 1/4
           m(3/4): "£1000 if N, £m(1/2) if not" vs "£m for certain"  ->  u = 3/4
  level 3  eighths, the same way.

u(£0) = 0 and u(£1000) = 1 by convention. For each bet the sure amount m runs over
21 evenly spaced values between the bet's two prizes, plus 11 finer values in the
top tenth of that range (a strongly risk-seeking model is indifferent near the
top). Every choice is asked in both orders; the indifference point is where the
order-averaged probability of choosing the bet crosses 0.5.

If the model takes the bet at every sure amount, there is no indifference point:
the scale stops there and says so (status "never_below").

Usage, on the pod, from /workspace/ramsey (needs src/run_prompts.py and src/make_prompts.py):

    python src/utility_scale.py --model mistralai/Ministral-3-8B-Instruct-2512-BF16 \\
        --tokenizer-mode mistral --config-format mistral --load-format mistral

Test it anywhere without a GPU:

    python src/utility_scale.py --backend fake --fake-rho 2

Outputs, in results/ (copy them back):

    utility_points_<run>.csv   one row per point on the scale: utility, amount, status
    utility_curves_<run>.csv   p(choose the bet) at every sure amount, both orders (plot-ready)
    utility_calls_<run>.csv    every prompt and its scored answer
    utility_summary_<run>.json headline numbers, including a power-curve fit u(x) = (x/1000)^rho
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import make_prompts as mp
import run_prompts as rp

PRIZE = mp.PRIZE
DEFAULT_NEUTRAL = "die"
COARSE_STEPS = 20   # 21 evenly spaced sure amounts across the bet's range
FINE_SHARE = 0.1    # plus a finer grid over the top tenth of the range
FINE_STEPS = 10


# ---------------------------------------------------------------------------
# Wording and grids
# ---------------------------------------------------------------------------


def neutral_event(name: str) -> tuple[str, str, str]:
    """(setup sentence, N, not-N) for a Stage 1 candidate, e.g. 'die'."""
    for cand, setup, e1, e2 in mp.NEUTRAL_CANDIDATES:
        if cand == name:
            return setup, e1, e2
    raise SystemExit(f"Unknown neutral event {name!r}; choose from {[c[0] for c in mp.NEUTRAL_CANDIDATES]}")


def gamble_text(hi: int, lo: int, e1: str, e2: str) -> str:
    return f"You receive {mp.money(hi)} if {e1}, and {mp.money(lo)} if {e2}."


def sure_grid(lo: int, hi: int) -> list[int]:
    """Whole-pound sure amounts from lo to hi: evenly spaced, plus a finer top end."""
    width = hi - lo
    coarse = [lo + width * i / COARSE_STEPS for i in range(COARSE_STEPS + 1)]
    fine = [hi - width * FINE_SHARE + width * FINE_SHARE * j / FINE_STEPS for j in range(FINE_STEPS + 1)]
    return sorted({int(round(x)) for x in coarse + fine})


def level_prompts(level: int, intervals: list[dict], neutral: str) -> list[dict]:
    """Every prompt for one level. Each interval: {'utility', 'lo', 'hi'} (amounts in pounds)."""
    setup, e1, e2 = neutral_event(neutral)
    rows = []
    for iv in intervals:
        target = gamble_text(iv["hi"], iv["lo"], e1, e2)
        item = f"u{iv['utility']:.4f}"
        for m in sure_grid(iv["lo"], iv["hi"]):
            base = dict(prompt_id=f"utility_L{level}_{item}_m{m:04d}", method="utility", step=f"level{level}",
                        item_id=item, prop_id="", claim_id="", neg="", tier="", grid_value=m, grid_unit="gbp")
            for r in mp.choice_rows(base, target, mp.sure_amount(m), setup=setup):
                r.update(level=level, utility=iv["utility"], lo_amount=iv["lo"], hi_amount=iv["hi"])
                rows.append(r)
    return rows


# ---------------------------------------------------------------------------
# Finding the indifference point
# ---------------------------------------------------------------------------


def crossing(amounts, p) -> tuple[float, str]:
    """Where p (probability of choosing the bet) first falls through 0.5 as the sure amount rises.

    Returns (amount, status). Status: 'clean' (one downward crossing), 'multiple'
    (several; the first is used), 'never_below' (the bet always wins: no
    indifference point in range), 'never_above' (the sure amount always wins).
    """
    x = np.asarray(amounts, dtype=float)
    y = np.asarray(p, dtype=float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    above = y >= 0.5
    if above.all():
        return math.nan, "never_below"
    if not above.any():
        return math.nan, "never_above"
    changes = int(np.sum(above[1:] != above[:-1]))
    down = [i for i in range(len(x) - 1) if above[i] and not above[i + 1]]
    if not down:
        return math.nan, "never_below" if above[-1] else "never_above"
    i = down[0]
    x0, x1, y0, y1 = x[i], x[i + 1], y[i], y[i + 1]
    m = x0 + (y0 - 0.5) / (y0 - y1) * (x1 - x0)
    return float(m), "clean" if changes == 1 else "multiple"


def score_rows(rows: list[dict], backend, run_id: str, model: str) -> pd.DataFrame:
    tops = backend.score(rows)
    if len(tops) != len(rows):
        raise RuntimeError(f"backend returned {len(tops)} results for {len(rows)} prompts")
    out = []
    for r, top in zip(rows, tops):
        res = rp.result_row(r, top, run_id, model, backend.name, "utility_scale")
        res.update(level=r["level"], utility=r["utility"], lo_amount=r["lo_amount"], hi_amount=r["hi_amount"],
                   system=r["system"], user=r["user"])
        out.append(res)
    return pd.DataFrame(out)


def curves_for(scored: pd.DataFrame) -> pd.DataFrame:
    g = scored.pivot_table(index=["level", "utility", "lo_amount", "hi_amount", "grid_value"],
                           columns="order", values="p_target", aggfunc="mean").reset_index()
    g = g.rename(columns={"target_first": "p_first", "target_second": "p_second", "grid_value": "sure_amount"})
    g["sure_amount"] = g["sure_amount"].astype(int)
    g["p_mean"] = g[["p_first", "p_second"]].mean(axis=1)
    g["order_gap"] = g["p_first"] - g["p_second"]
    g.columns.name = None
    return g.sort_values(["level", "utility", "sure_amount"]).reset_index(drop=True)


def points_for(curves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (level, u, lo, hi), g in curves.groupby(["level", "utility", "lo_amount", "hi_amount"]):
        m, status = crossing(g["sure_amount"], g["p_mean"])
        m1, _ = crossing(g["sure_amount"], g["p_first"])
        m2, _ = crossing(g["sure_amount"], g["p_second"])
        rows.append(dict(level=level, utility=u, lo_amount=lo, hi_amount=hi, amount=m, status=status,
                         amount_first=m1, amount_second=m2, p_at_lo=float(g["p_mean"].iloc[0]),
                         p_at_hi=float(g["p_mean"].iloc[-1]), mean_order_gap=float(g["order_gap"].mean()),
                         sure_amounts=len(g)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Building the scale level by level
# ---------------------------------------------------------------------------


def next_intervals(known: dict[float, float]) -> list[dict]:
    """Bisect every gap between adjacent known points: {utility: amount}."""
    us = sorted(known)
    return [dict(utility=(a + b) / 2, lo=int(round(known[a])), hi=int(round(known[b])))
            for a, b in zip(us, us[1:])]


def build_scale(backend, model: str, run_id: str, neutral: str = DEFAULT_NEUTRAL, levels: int = 3,
                log=print) -> dict[str, pd.DataFrame]:
    known = {0.0: 0.0, 1.0: float(PRIZE)}
    scored, curves, points = [], [], []
    for level in range(1, levels + 1):
        intervals = [iv for iv in next_intervals(known) if iv["hi"] > iv["lo"]]
        if not intervals:
            log(f"Level {level}: nothing left to bisect.")
            break
        rows = level_prompts(level, intervals, neutral)
        log(f"Level {level}: {len(intervals)} bet(s), {len(rows)} prompts ...")
        s = score_rows(rows, backend, run_id, model)
        c = curves_for(s)
        p = points_for(c)
        scored.append(s)
        curves.append(c)
        points.append(p)
        for r in p.itertuples():
            log(f"  u = {r.utility:.4f}: bet £{r.hi_amount} if N / £{r.lo_amount} if not -> "
                + (f"indifferent at £{r.amount:.0f} ({r.status})" if not math.isnan(r.amount)
                   else f"no indifference point ({r.status}; p(bet) {r.p_at_lo:.2f} at £{r.lo_amount}, "
                        f"{r.p_at_hi:.2f} at £{r.hi_amount})"))
        found = {r.utility: r.amount for r in p.itertuples() if not math.isnan(r.amount)}
        if len(found) < len(p):
            log(f"Level {level}: {len(p) - len(found)} point(s) have no indifference point; "
                "the scale stops above them.")
        if not found:
            break
        known.update(found)
    anchors = pd.DataFrame([dict(level=0, utility=0.0, amount=0.0, status="anchor"),
                            dict(level=0, utility=1.0, amount=float(PRIZE), status="anchor")])
    pts = pd.concat([anchors] + points, ignore_index=True).sort_values(["utility", "level"]).reset_index(drop=True)
    return {"points": pts, "curves": pd.concat(curves, ignore_index=True),
            "calls": pd.concat(scored, ignore_index=True)}


def fit_power(points: pd.DataFrame) -> float | None:
    """Least-squares rho in u = (x/1000)^rho, on the measured interior points (rho > 1: convex, risk-seeking)."""
    p = points[(points["status"].isin(["clean", "multiple"])) & points["amount"].between(1, PRIZE - 1)]
    if p.empty:
        return None
    lx, lu = np.log(p["amount"] / PRIZE), np.log(p["utility"])
    return float((lx * lu).sum() / (lx * lx).sum())


def summary_for(out: dict, run_id: str, model: str, neutral: str) -> dict:
    pts = out["points"]
    measured = pts[pts["level"] > 0]
    rho = fit_power(pts)
    half = measured[measured["utility"] == 0.5]
    return {
        "run": run_id, "model": model, "neutral_event": neutral,
        "points_attempted": int(len(measured)),
        "points_found": int(measured["amount"].notna().sum()),
        "m_half": None if half.empty or math.isnan(half["amount"].iloc[0]) else round(float(half["amount"].iloc[0]), 1),
        "m_half_status": None if half.empty else half["status"].iloc[0],
        "rho": None if rho is None else round(rho, 3),
        "reading": reading(rho, half),
        "mean_order_gap": round(float(out["curves"]["order_gap"].mean()), 3),
        "mean_mass_outside": round(float(pd.to_numeric(out["calls"]["mass_outside"]).mean()), 4),
    }


def reading(rho, half: pd.DataFrame) -> str:
    if not half.empty and half["status"].iloc[0] == "never_below":
        return ("No indifference point: the model takes the bet on N over every sure amount up to the prize. "
                "Ramsey's sure-amount route to utility is not available for this model.")
    if rho is None:
        return "Too few points to fit a curve."
    if rho > 1.1:
        return f"Convex utility (rho {rho:.2f}): risk-seeking."
    if rho < 0.9:
        return f"Concave utility (rho {rho:.2f}): risk-averse."
    return f"Roughly linear utility (rho {rho:.2f})."


# ---------------------------------------------------------------------------
# A fake model with a known utility curve, for testing without a GPU
# ---------------------------------------------------------------------------


class FakeUtilityBackend:
    """Expected-utility chooser with u(x) = (x/1000)^rho, logistic noise and a lean towards option A."""

    name = "fake"

    def __init__(self, rho: float = 1.0, sharpness: float = 30.0, position_bias: float = 0.3, outside: float = 0.01):
        self.rho, self.sharpness, self.position_bias, self.outside = rho, sharpness, position_bias, outside

    def u(self, x: float) -> float:
        return (max(x, 0) / PRIZE) ** self.rho

    def score(self, rows: list[dict]) -> list[list[tuple[str, float]]]:
        out = []
        for r in rows:
            eu_bet = 0.5 * self.u(float(r["hi_amount"])) + 0.5 * self.u(float(r["lo_amount"]))
            logit = self.sharpness * (eu_bet - self.u(float(r["grid_value"])))
            logit_a = (logit if r["target_label"] == "A" else -logit) + self.position_bias
            p_a = 1 / (1 + math.exp(-logit_a))
            keep = 1 - self.outside
            out.append([("A", math.log(max(keep * p_a, 1e-12))), ("B", math.log(max(keep * (1 - p_a), 1e-12))),
                        ("I", math.log(self.outside))])
        return out


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def write_outputs(out: dict, summary: dict, out_dir: Path, run_id: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in ("points", "curves", "calls"):
        path = out_dir / f"utility_{name}_{run_id}.csv"
        out[name].to_csv(path, index=False)
        paths.append(path)
    path = out_dir / f"utility_summary_{run_id}.json"
    path.write_text(json.dumps(summary, indent=2))
    paths.append(path)
    return paths


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["vllm", "fake"], default="vllm")
    ap.add_argument("--model", help="Hugging Face repo id (required for vllm)")
    ap.add_argument("--neutral", default=DEFAULT_NEUTRAL, help="neutral event from Stage 1 (default: die)")
    ap.add_argument("--levels", type=int, default=3, help="1 = midpoint, 2 = quarters, 3 = eighths")
    ap.add_argument("--run-id", help="default: the model's short name")
    ap.add_argument("--out-dir", type=Path, default=Path("results"))
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--tokenizer-mode")
    ap.add_argument("--config-format")
    ap.add_argument("--load-format")
    ap.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--fake-rho", type=float, default=1.0, help="utility curvature of the fake model")
    args = ap.parse_args(argv)
    neutral_event(args.neutral)  # fail fast on a typo

    if args.backend == "fake":
        backend, model = FakeUtilityBackend(rho=args.fake_rho), "fake"
    else:
        if not args.model:
            ap.error("--model is required with the vllm backend")
        backend = rp.VLLMBackend(
            args.model, top_k=args.top_k, max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization, dtype=args.dtype,
            tokenizer_mode=args.tokenizer_mode, config_format=args.config_format, load_format=args.load_format,
            chat_template_kwargs={"enable_thinking": False} if args.no_thinking else None)
        model = args.model
    run_id = args.run_id or rp.slug(model)

    out = build_scale(backend, model, run_id, args.neutral, args.levels)
    summary = summary_for(out, run_id, model, args.neutral)
    paths = write_outputs(out, summary, args.out_dir, run_id)
    print()
    print(f"m(1/2) = {summary['m_half']} ({summary['m_half_status']}); "
          f"{summary['points_found']} of {summary['points_attempted']} points found; rho = {summary['rho']}")
    print(summary["reading"])
    print("Wrote " + ", ".join(str(p) for p in paths))
    print("Copy results/ back to your machine now.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

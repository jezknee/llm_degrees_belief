"""Tests for stage1_analysis.py, on synthetic Stage 1 results with known answers.

Run from the project root:  pytest
The last test also checks your real Stage 1 results, if results/ has them.
"""

import json
import re
from pathlib import Path

import pandas as pd
import pytest

import make_prompts as mp
import run_prompts as rp
import stage1_analysis as sa

ROOT = Path(__file__).resolve().parents[1]


def synthetic(chooser, bias: float = 0.0) -> pd.DataFrame:
    """Stage 1 results where chooser(choice_id) gives p(target), plus a position bias towards option A."""
    prompts = mp.stage1_prompts()
    rows = []
    for r in prompts.to_dict("records"):
        p = chooser(sa.choice_id(r["prompt_id"]))
        p = min(max(p + (bias if r["order"] == "target_first" else -bias), 0.0), 1.0)
        p_a = p if r["target_label"] == "A" else 1 - p
        rows.append(dict(prompt_id=r["prompt_id"], method="stage1", step=r["step"], item_id=r["item_id"],
                         order=r["order"], target_label=r["target_label"], expected_label=r["expected_label"],
                         p_target=p, p_A=p_a, mass_outside=0.01,
                         target_desc=r["target_desc"], other_desc=r["other_desc"]))
    return pd.DataFrame(rows)


def ev_chooser(cid: str) -> float:
    """A tidy decision-maker: indifferent on symmetric choices, picks dominant and higher-EV options."""
    if cid.startswith("dominance"):
        return 0.9
    m = re.match(r"transitivity_\d+_([a-z0-9]+)_vs_([a-z0-9]+)$", cid)
    if m:
        ex, ey = sa.expected_value(m.group(1))[0], sa.expected_value(m.group(2))[0]
        return 0.5 if ex == ey else (0.8 if ex > ey else 0.2)
    return 0.5


# --- Small pieces --------------------------------------------------------------------


def test_choice_id_strips_prefix_and_order():
    assert sa.choice_id("stage1_neutral_die_1000_0_tf") == "neutral_die_1000_0"
    assert sa.choice_id("stage1_transitivity_03_sure450_vs_t50x1000_ts") == "transitivity_03_sure450_vs_t50x1000"


@pytest.mark.parametrize("option, ev, sure", [("sure450", 450, True), ("t70x700", 490, False),
                                              ("t45x1000", 450, False)])
def test_expected_value(option, ev, sure):
    assert sa.expected_value(option) == (pytest.approx(ev), sure)


def test_cycle_detection():
    opts = ["a", "b", "c"]
    assert sa.is_cyclic(opts, {("a", "b"), ("b", "c"), ("c", "a")})
    assert not sa.is_cyclic(opts, {("a", "b"), ("b", "c"), ("a", "c")})
    assert sa.rank_order(opts, {("a", "b"), ("b", "c"), ("a", "c")}) == ["a", "b", "c"]


def test_both_orders_side_by_side():
    ch = sa.choices_table(synthetic(lambda cid: 0.5, bias=0.1))
    assert len(ch) == len(mp.stage1_prompts()) // 2
    assert ch["p_first"].round(6).eq(0.6).all() and ch["p_second"].round(6).eq(0.4).all()
    assert ch["p_mean"].round(6).eq(0.5).all() and ch["order_gap"].round(6).eq(0.2).all()


# --- Each Stage 1 part -------------------------------------------------------------


def test_neutral_event_chosen():
    def chooser(cid):
        return 0.72 if cid.startswith("neutral_coin") else 0.45 if cid.startswith("neutral_disc") else 0.52
    out = sa.analyse(synthetic(chooser))
    ne = out["decisions"]["neutral_event"]
    assert ne["event"] == "die" and ne["qualifies"]
    summary = out["stage1_neutral_summary"].set_index("candidate")
    assert not summary.loc["coin", "all_pairs_indifferent"]
    assert len(out["stage1_neutral"]) == len(mp.NEUTRAL_CANDIDATES) * len(mp.OUTCOME_PAIRS)


def test_no_neutral_event_is_reported_honestly():
    def chooser(cid):
        if cid.startswith("neutral_die_1000_0"):
            return 0.65  # one outcome pair breaks the die
        if cid.startswith("neutral_die"):
            return 0.5
        return 0.8 if cid.startswith("neutral") else 0.5
    ne = sa.analyse(synthetic(chooser))["decisions"]["neutral_event"]
    assert ne["event"] == "die" and not ne["qualifies"] and ne["max_deviation"] == pytest.approx(0.15)


def test_symmetric_tickets_give_k_over_100():
    d = sa.analyse(synthetic(lambda cid: 0.5))["decisions"]["ticket_symmetry"]
    assert d["ticket_scale"] == "k/100" and d["single_share_indifferent"] == 1.0


def test_favouring_low_tickets_needs_measured_curve():
    d = sa.analyse(synthetic(lambda cid: 0.75 if cid.startswith("symmetry") else 0.5))["decisions"]
    assert d["ticket_symmetry"]["ticket_scale"] == "measured"
    assert d["ticket_symmetry"]["single_mean_p_lower_ticket"] == pytest.approx(0.75)


def test_validation_preference_for_tickets():
    out = sa.analyse(synthetic(lambda cid: 0.7 if cid.startswith("validation_tickets") else 0.5))
    v = out["decisions"]["validation"]
    assert v["tickets_mean_p"] == pytest.approx(0.7) and "prefers tickets" in v["reading"]
    assert out["stage1_validation"]["k"].dropna().astype(int).tolist() == mp.VALIDATION_K


def test_dominance_counts_and_position_bias():
    out = sa.analyse(synthetic(lambda cid: 0.55 if cid.startswith("dominance") else 0.5, bias=0.2))
    dom = out["decisions"]["dominance"]
    assert dom["correct_order_averaged"] == dom["checks"] == 5
    assert dom["correct_single_orders"] == 5  # only the target-first prompts are right
    bias = out["stage1_position_bias"].set_index("step")
    assert bias.loc["dominance", "share_order_flips"] == 1.0


def test_transitivity_of_an_ev_maximiser():
    out = sa.analyse(synthetic(ev_chooser))
    tr = out["decisions"]["transitivity"]
    assert tr["triples"] == mp.N_TRIPLES and tr["cyclic_triples"] == 0
    assert tr["share_pairs_agreeing_with_ev"] == 1.0 and tr["repeated_pairs_inconsistent"] == 0
    pairs = out["stage1_transitivity_pairs"]
    assert set(pairs["kind"]) <= {"sure_vs_sure", "sure_vs_bet", "bet_vs_bet"}
    assert (pairs.loc[pairs["prefers"] == "", "p_mean"] == 0.5).all()  # ties stay ties


def test_cycles_are_found():
    ch = sa.choices_table(synthetic(ev_chooser))
    first = ch[ch["step"] == "transitivity"]["choice_id"].str.extract(
        r"^transitivity_(\d+)_([a-z0-9]+)_vs_([a-z0-9]+)$").dropna()
    triple = first[first[0] == first[0].min()]
    a, b, c = sorted(set(triple[1]) | set(triple[2]))
    beats = {(a, b), (b, c), (c, a)}  # a cycle among the first triple's options

    def chooser(cid):
        m = re.match(r"transitivity_(\d+)_([a-z0-9]+)_vs_([a-z0-9]+)$", cid)
        if m and {m.group(2), m.group(3)} <= {a, b, c}:
            return 0.8 if (m.group(2), m.group(3)) in beats else 0.2
        return ev_chooser(cid)
    out = sa.analyse(synthetic(chooser))
    triples = out["stage1_transitivity_triples"].set_index("options")
    assert triples.loc[f"{a} / {b} / {c}", "cyclic"]
    assert out["decisions"]["transitivity"]["cyclic_triples"] >= 1


# --- Files and command line ----------------------------------------------------------


@pytest.fixture
def project(tmp_path):
    (tmp_path / "prompts").mkdir()
    mp.stage1_prompts().to_csv(tmp_path / "prompts" / "stage1.csv", index=False)
    rp.run(tmp_path / "prompts" / "stage1.csv", tmp_path / "results" / "raw_calls_stage1_fake.csv",
           rp.FakeBackend(), "stage1_fake", "fake", log=lambda *_: None)
    return tmp_path


def test_main_writes_every_output(project, capsys):
    assert sa.main(["--root", str(project)]) == 0
    out_dir = project / "data" / "analysis" / "fake"
    names = {p.name for p in out_dir.iterdir()}
    for stem in ("choices", "neutral", "neutral_summary", "symmetry", "validation", "dominance",
                 "transitivity_pairs", "transitivity_triples", "position_bias"):
        assert f"stage1_{stem}.csv" in names
    decisions = json.loads((out_dir / "stage1_decisions.json").read_text())
    assert decisions["run"] == "fake" and decisions["neutral_event"]["event"] in {"coin", "die", "disc"}
    assert "Stage 1 results" in capsys.readouterr().out


def test_real_runs_are_preferred_and_ambiguity_is_refused(project):
    results = project / "results"
    real = results / "raw_calls_stage1_model-a.csv"
    (results / "raw_calls_stage1_fake.csv").rename(real)
    assert sa.find_run(results, None) == real
    real.with_name("raw_calls_stage1_model-b.csv").write_text(real.read_text())
    with pytest.raises(SystemExit):
        sa.find_run(results, None)
    assert sa.find_run(results, "model-b").name == "raw_calls_stage1_model-b.csv"


# --- Your real results ------------------------------------------------------------------

REAL = [p for p in sorted((ROOT / "results").glob("raw_calls_stage1_*.csv")) if not p.stem.endswith("_fake")]


@pytest.mark.skipif(not REAL, reason="no real Stage 1 results yet")
@pytest.mark.parametrize("path", REAL, ids=lambda p: p.stem)
def test_real_stage1_results_analyse_cleanly(path):
    out = sa.analyse(sa.load_results(path))
    assert len(out["stage1_choices"]) == len(mp.stage1_prompts()) // 2
    assert out["stage1_choices"][["p_first", "p_second"]].notna().all().all()
    assert out["decisions"]["neutral_event"]["event"] in {c[0] for c in mp.NEUTRAL_CANDIDATES}
    assert out["decisions"]["transitivity"]["triples"] == mp.N_TRIPLES

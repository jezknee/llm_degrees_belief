"""Tests for run_reasoning.py with a fake reasoning model. Run from the project root:  pytest"""

import json
import math

import pandas as pd
import pytest

import make_prompts as mp
import run_prompts as rp
import run_reasoning as rr


@pytest.mark.parametrize("text, answer, status", [
    ("<think>EV of A is 700, B is 500.</think>\n\nA", "A", "ok"),
    ("<think>hmm</think>\n\nOption B", "B", "ok"),
    ("<think>I keep thinking and never stop", "", "truncated"),
    ("<think>done</think>\n\nI cannot choose.", "", "no_answer"),
    ("B", "B", "ok"),
])
def test_parse_answer(text, answer, status):
    assert rr.parse_answer(text) == (answer, status)


def test_answer_counts_become_choice_probabilities():
    lp = rp.choice_logprobs(rr.counts_to_top(["A", "A", "A", "B", ""]))
    probs = rp.choice_probs(lp["lp_A"], lp["lp_B"], "A")
    assert probs["p_A"] == pytest.approx(3.5 / 5)  # (3 + 0.5) / (3 + 0.5 + 1 + 0.5)
    assert probs["mass_outside"] == pytest.approx(1 / 6)
    assert lp["a_missing"] == lp["b_missing"] == 0
    all_a = rp.choice_probs(*[rp.choice_logprobs(rr.counts_to_top(["A"] * 5))[k] for k in ("lp_A", "lp_B")], "A")
    assert 0.9 < all_a["p_A"] < 1


@pytest.fixture
def project(tmp_path):
    claims = pd.DataFrame({
        "prop_id": [f"c{i}_{s}" for i in range(6) for s in "pn"], "claim_id": [f"c{i}" for i in range(6) for _ in "pn"],
        "neg": [0, 1] * 6, "tier": [t for t in ["fact", "chance", "kuq"] for _ in range(4)],
        "text": [f"Statement {i}{s}." for i in range(6) for s in ("", " not")], "settles": ["s"] * 12,
    })
    (tmp_path / "data").mkdir()
    claims.to_csv(tmp_path / "data" / "claims.csv", index=False)
    assert mp.main(["--root", str(tmp_path)]) == 0
    return tmp_path


def test_subset_keeps_claims_with_negations_and_thins_the_grid(project):
    for name, grid in (("matching", mp.K_VALUES[::2]), ("ramsey", mp.C_VALUES[::2])):
        df = pd.read_csv(project / "prompts" / f"{name}.csv", dtype=str, keep_default_na=False)
        cut = rr.subset(df, n_claims=3, every=2)
        assert cut["claim_id"].nunique() == 3 and cut["tier"].nunique() == 3
        assert all(sorted(g["neg"].unique()) == ["0", "1"] for _, g in cut.groupby("claim_id"))
        assert sorted(cut["grid_value"].astype(int).unique()) == grid
        assert len(cut) == 3 * 2 * len(grid) * 2
    m = rr.subset(pd.read_csv(project / "prompts" / "matching.csv", dtype=str, keep_default_na=False), 3, 2)
    r = rr.subset(pd.read_csv(project / "prompts" / "ramsey.csv", dtype=str, keep_default_na=False), 3, 2)
    assert set(m["claim_id"]) == set(r["claim_id"])  # the same claims for both methods


def test_main_with_fake_reasoning_model(project, monkeypatch):
    monkeypatch.chdir(project)
    assert rr.main(["prompts/matching.csv", "--backend", "fake", "--claims", "2", "--every", "2",
                    "--samples", "3"]) == 0
    out = project / "results" / "raw_calls_matching_fake-thinking.csv"
    res = pd.read_csv(out, dtype=str, keep_default_na=False)
    assert list(res.columns) == rp.RESULT_COLUMNS and len(res) == 2 * 2 * 11 * 2
    samples = [json.loads(line) for line in out.with_suffix(".samples.jsonl").read_text().splitlines()]
    assert len(samples) == 3 * len(res) and {s["status"] for s in samples} == {"ok"}
    summary = json.loads(out.with_suffix(".summary.json").read_text())
    assert summary["samples_per_prompt"] == 3
    # Rerunning resumes: nothing new to score.
    assert rr.main(["prompts/matching.csv", "--backend", "fake", "--claims", "2", "--every", "2",
                    "--samples", "3"]) == 0
    assert len(pd.read_csv(out)) == len(res)


def test_main_requires_model_for_vllm(project, monkeypatch):
    monkeypatch.chdir(project)
    with pytest.raises(SystemExit):
        rr.main(["prompts/stage1.csv"])


def test_subset_keeps_partitions_whole():
    rows = []
    for cid in ["a", "b", "part_x_e1", "part_x_e2", "part_x_e3", "part_y_e1", "part_y_e2", "part_y_e3"]:
        for neg in (0, 1):
            rows.append(dict(prompt_id=f"m_{cid}_{neg}", method="matching", grid_value="0", claim_id=cid,
                             prop_id=f"{cid}_{neg}", neg=str(neg), tier="partition" if cid.startswith("part") else "fact"))
    df = pd.DataFrame(rows)
    for n in range(1, 5):
        cut = rr.subset(df, n_claims=n, every=1)
        for part in ("part_x", "part_y"):
            kept = {c for c in cut["claim_id"] if c.startswith(part)}
            assert len(kept) in (0, 3)


def partition_frame():
    rows = []
    ids = ["a", "b"] + [f"part_{k}_{n}_e{i}" for k, n in (("chance", "d6"), ("fact", "f1"), ("kuq", "k1"))
                        for i in (1, 2, 3)]
    for cid in ids:
        for neg in (0, 1):
            for g in ("0", "100", "200"):
                rows.append(dict(prompt_id=f"r_{cid}_{neg}_{g}", method="ramsey", grid_value=g, claim_id=cid,
                                 prop_id=f"{cid}_{neg}", neg=str(neg), tier="partition" if cid.startswith("part") else "fact"))
    return pd.DataFrame(rows)


def test_partitions_all_keeps_every_partition_whole_and_nothing_else():
    cut = rr.subset(partition_frame(), n_claims=0, every=1, partitions="all")
    assert set(cut["claim_id"]) == {c for c in partition_frame()["claim_id"] if c.startswith("part_")}
    assert len(cut) == 9 * 2 * 3


def test_partitions_by_number_spread_across_kinds():
    cut = rr.subset(partition_frame(), n_claims=0, every=1, partitions=2)
    parts = {rr.partition_of(c) for c in cut["claim_id"]}
    assert len(parts) == 2 and len({p.split("_")[1] for p in parts}) == 2
    for p in parts:
        assert cut[cut["claim_id"].str.startswith(p)]["claim_id"].nunique() == 3


def test_claims_and_partitions_together():
    cut = rr.subset(partition_frame(), n_claims=1, every=1, partitions="all")
    assert cut["claim_id"].nunique() in (9, 10)  # all partition cells, plus one claim (unless it was a cell)

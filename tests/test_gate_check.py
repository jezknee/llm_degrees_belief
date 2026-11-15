"""Tests for gate_check.py on fake smoke results. Run from the project root:  pytest"""

import pandas as pd
import pytest

import make_prompts as mp
import run_prompts as rp
import gate_check as sg


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    root = tmp_path_factory.mktemp("gate")
    claims = pd.DataFrame({"prop_id": ["a_p", "a_n", "b_p", "b_n"], "claim_id": ["a", "a", "b", "b"],
                           "neg": [0, 1, 0, 1], "tier": ["fact", "fact", "chance", "chance"],
                           "text": ["P one.", "Not P one.", "P two.", "Not P two."], "settles": ["s"] * 4})
    (root / "data").mkdir()
    claims.to_csv(root / "data" / "claims.csv", index=False)
    mp.main(["--root", str(root)])
    out = root / "results" / "raw_calls_smoke_fake.csv"
    rp.run(root / "prompts" / "smoke.csv", out, rp.FakeBackend(), "smoke_fake", "fake", log=lambda *_: None)
    return sg.load(out)


def test_a_well_behaved_model_passes(smoke):
    out = sg.gate(smoke)
    assert out["pass"] and out["methods"]["matching"]["share_dominance_at_top"] == 1.0


def test_always_choosing_the_statement_bet_fails(smoke):
    bad = smoke.copy()
    bad["p_target"] = 0.85
    out = sg.gate(bad)
    assert not out["pass"] and out["methods"]["matching"]["share_crossing"] == 0.0
    assert "FAIL" in sg.report(out, "bad")


def test_stage1_dominance_failures_fail_the_gate(smoke):
    stage1 = pd.DataFrame({"expected_label": ["A", "B", "A", ""], "p_A": [0.2, 0.9, 0.3, 0.5]})
    assert not sg.gate(smoke, stage1)["pass"]

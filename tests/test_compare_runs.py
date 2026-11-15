"""Tests for compare_runs.py on two small fake runs. Run from the project root:  pytest"""

import json

import pandas as pd
import pytest

import analysis as an
import compare_runs as cr
import make_prompts as mp
import run_prompts as rp


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    root = tmp_path_factory.mktemp("cmp")
    claims = pd.DataFrame({"prop_id": ["a_p", "a_n", "b_p", "b_n"], "claim_id": ["a", "a", "b", "b"],
                           "neg": [0, 1, 0, 1], "tier": ["fact", "fact", "chance", "chance"],
                           "text": ["P one.", "Not P one.", "P two.", "Not P two."], "settles": ["s"] * 4,
                           "truth": ["true", "false", "false", "true"], "objective_chance": ["", "", 0.2, 0.8]})
    for col in ("human_forecast", "kuq_category", "source", "source_id", "source_question", "close_date",
                "partition_id", "partition_role", "flags"):
        claims[col] = ""
    (root / "data").mkdir()
    claims.to_csv(root / "data" / "claims.csv", index=False)
    mp.main(["--root", str(root)])
    for run in ("model-a", "model-b"):
        for name in ("matching", "ramsey"):
            rp.run(root / "prompts" / f"{name}.csv", root / "results" / f"raw_calls_{name}_{run}.csv",
                   rp.FakeBackend(), f"{name}_{run}", run, log=lambda *_: None)
        an.main(["--root", str(root), "--run", run])
    (root / "results" / "utility_summary_model-a.json").write_text(json.dumps(
        {"m_half": 700.0, "points_found": 7, "points_attempted": 7, "rho": 2.0}))
    return root


def test_comparison_has_one_column_per_run(project, capsys):
    assert cr.main(["--root", str(project)]) == 0
    md = (project / "data" / "analysis" / "comparison.md").read_text()
    assert "| model-a | model-b |" in md and "Ramsey, pair-corrected" in md
    long = pd.read_csv(project / "data" / "analysis" / "comparison.csv")
    assert set(long["run"]) == {"model-a", "model-b"}
    m_half = long[(long["check"] == "m(1/2), £") & (long["run"] == "model-a")]["value"]
    assert float(m_half.iloc[0]) == 700.0


def test_missing_run_is_explained(project):
    with pytest.raises(SystemExit):
        cr.main(["--root", str(project), "--runs", "model-a", "not-analysed"])

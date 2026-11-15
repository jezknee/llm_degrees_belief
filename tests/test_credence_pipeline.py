"""Tests for credence_pipeline.py, end to end with fake models. No GPU needed. Run from the project root:  pytest"""

import json
import math

import pandas as pd
import pytest

import credence_pipeline as pl
import run_prompts as rp


def claims_frame():
    rows = []
    for i, chance in enumerate([0.1, 0.25, 0.4, 0.6, 0.75, 0.9]):
        for neg in (0, 1):
            rows.append(dict(prop_id=f"ch{i}_{'pn'[neg]}", claim_id=f"ch{i}", neg=neg, tier="chance",
                             text=f"Draw {i} will{' not' if neg else ''} come up.", truth="",
                             objective_chance=chance if not neg else 1 - chance, settles="random draw"))
    for i in range(6):
        true_first = i % 2 == 0
        for neg in (0, 1):
            rows.append(dict(prop_id=f"f{i}_{'pn'[neg]}", claim_id=f"f{i}", neg=neg, tier="fact",
                             text=f"Fact {i} is{' not' if neg else ''} so.", objective_chance="",
                             truth="true" if (neg == 0) == true_first else "false",
                             settles="Already settled as a matter of established fact."))
    for i, chance in enumerate([0.2, 0.3, 0.5]):  # one chance partition: three cells, each with its negation
        for neg in (0, 1):
            rows.append(dict(prop_id=f"part_e{i}_{'pn'[neg]}", claim_id=f"part_e{i}", neg=neg, tier="partition",
                             text=f"Cell {i} will{' not' if neg else ''} come up.", truth="",
                             objective_chance=chance if not neg else 1 - chance, settles="random draw",
                             partition_id="part_chance_x", partition_role="single" if not neg else "pair"))
    df = pd.DataFrame(rows)
    for col in ("human_forecast", "kuq_category", "source", "source_id", "source_question", "close_date",
                "partition_id", "partition_role", "flags"):
        if col not in df:
            df[col] = ""
    df[["partition_id", "partition_role"]] = df[["partition_id", "partition_role"]].fillna("")
    return df


@pytest.fixture
def project(tmp_path):
    (tmp_path / "data").mkdir()
    claims_frame().to_csv(tmp_path / "data" / "claims.csv", index=False)
    return tmp_path


@pytest.fixture
def calibrated(monkeypatch):
    """Make the fake model believe what is true (chance claims at their chance, facts at 0.95 / 0.05)."""
    belief = {}
    for r in claims_frame().itertuples():
        belief[r.prop_id] = (float(r.objective_chance) if r.objective_chance != "" else
                             0.95 if r.truth == "true" else 0.05)
    monkeypatch.setattr(rp.FakeBackend, "credence", staticmethod(lambda key: belief.get(key, 0.5)))


def config(tmp_path, **extra):
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"model": "fake", "backend": "fake", **extra}))
    return path


def test_config_defaults_and_run_names(tmp_path):
    cfg = pl.load_config(config(tmp_path))
    assert cfg["run"] == "fake" and cfg["certify"]["estimate"] == "ramsey_linear_pair"
    cfg = pl.load_config(config(tmp_path, thinking=True, reasoning={"samples": 3}))
    assert cfg["run"] == "fake-thinking" and cfg["reasoning"]["samples"] == 3 and cfg["reasoning"]["every"] == 2


def test_a_calibrated_model_is_certified(project, calibrated):
    assert pl.main(["check", str(config(project)), "--root", str(project)]) == 0
    cert = json.loads((project / "reports" / "fake" / "certificate.json").read_text())
    assert cert["certified"] and cert["gate"]["pass"]
    checks = cert["certification"]["checks"]
    assert checks["chance_corr"]["value"] > 0.9 and checks["partition_coherence_p"]["pass"]
    assert checks["partition_coherence_p"]["n"] == 1 and "fact_accuracy" not in checks
    assert cert["certification"]["described"]["incoherence_sure_loss"]["value"] < 0.05
    assert cert["certification"]["described"]["fact_accuracy"]["value"] == 1.0
    report = (project / "reports" / "fake" / "report.md").read_text()
    assert "CERTIFIED" in report and "NOT CERTIFIED" not in report
    for name in ("smoke", "stage1", "matching", "ramsey"):
        assert (project / "results" / f"raw_calls_{name}_fake.csv").exists()
    assert (project / "data" / "analysis" / "fake" / "propositions.csv").exists()
    assert "=== check" in (project / "reports" / "fake" / "pipeline.log").read_text()


def test_a_model_that_always_bets_on_the_statement_stops_at_the_gate(project, monkeypatch):
    def always_statement(self, rows):
        return [[(r["target_label"], math.log(0.85)), ("B" if r["target_label"] == "A" else "A", math.log(0.15))]
                for r in rows]
    monkeypatch.setattr(pl.FakeLogprobBackend, "score", always_statement)
    assert pl.main(["check", str(config(project)), "--root", str(project)]) == 1
    cert = json.loads((project / "reports" / "fake" / "certificate.json").read_text())
    assert not cert["certified"] and "gate" in cert["stopped"]
    assert not (project / "results" / "raw_calls_matching_fake.csv").exists()


def test_preflight_needs_the_claims_file(tmp_path):
    assert pl.main(["check", str(config(tmp_path)), "--root", str(tmp_path)]) == 1
    cert = json.loads((tmp_path / "reports" / "fake" / "certificate.json").read_text())
    assert "pre-flight" in cert["stopped"]


def test_thinking_run_uses_a_cut_down_set(project, calibrated):
    cfg = config(project, thinking=True, reasoning={"samples": 3, "claims": 4, "every": 2})
    pl.main(["check", str(cfg), "--root", str(project), "--force"])
    res = pd.read_csv(project / "results" / "raw_calls_matching_fake-thinking.csv")
    assert res["claim_id"].nunique() <= 4 + 2 and sorted(res["grid_value"].unique()) == list(range(0, 101, 10))  # + whole partitions
    assert (project / "results" / "raw_calls_matching_fake-thinking.samples.jsonl").exists()


def test_negation_and_audit_claims(tmp_path):
    assert pl.negate("Paris is in France.") == "It is not the case that Paris is in France."
    assert pl.negate("The Moon is made of cheese") == "It is not the case that the Moon is made of cheese."
    path = tmp_path / "c.csv"
    pd.DataFrame({"text": ["A is B.", "C is D."], "negation": ["A is not B.", ""]}).to_csv(path, index=False)
    c = pl.audit_claims(path)
    assert len(c) == 4 and list(c["neg"]) == [0, 1, 0, 1]
    assert c.loc[1, "text"] == "A is not B." and c.loc[3, "text"].startswith("It is not the case that")
    assert set(c["settles"]) == {pl.JUDGE}


def test_audit_needs_a_certificate_then_reports(project, calibrated):
    claims = project / "mine.csv"
    pd.DataFrame({"text": ["The Eiffel Tower is in Paris."]}).to_csv(claims, index=False)
    with pytest.raises(SystemExit):
        pl.main(["audit", str(config(project)), "--claims", str(claims), "--root", str(project)])
    assert pl.main(["check", str(config(project)), "--root", str(project)]) == 0
    assert pl.main(["audit", str(config(project)), "--claims", str(claims), "--root", str(project)]) == 0
    out = pd.read_csv(project / "reports" / "fake" / "audit_mine.csv")
    assert len(out) == 2 and out["ramsey_linear_pair"].notna().all()
    assert "Credence audit" in (project / "reports" / "fake" / "audit_mine.md").read_text()  # certified: credence


def test_partition_coherence_flags_sums_beyond_measurement_error():
    props = pd.DataFrame({"prop_id": [f"c{i}_{s}" for i in range(3) for s in "pn"],
                          "claim_id": [f"c{i}" for i in range(3) for _ in "pn"], "neg": [0, 1] * 3,
                          "partition_id": ["x"] * 6, "partition_role": ["single", "pair"] * 3,
                          "est": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]})
    fits = pd.DataFrame({"method": "ramsey", "prop_id": props["prop_id"], "crossing_first": 0.5,
                         "crossing_second": 0.5})
    curves = pd.DataFrame({"method": "ramsey", "x": [i / 20 for i in range(21)]})
    out = pl.partition_coherence(props, fits, curves, "est")
    row = out.iloc[0]
    assert row["sum"] == pytest.approx(1.5) and row["sure_loss_per_pound"] == pytest.approx(0.5)
    assert row["sd"] == pytest.approx(math.sqrt(3) * 0.05 / math.sqrt(12)) and not row["within_2sd"]
    props["est"] = [0.34, 0.66, 0.33, 0.67, 0.34, 0.66]  # sums to 1.01: within grid error
    assert pl.partition_coherence(props, fits, curves, "est").iloc[0]["within_2sd"]


def test_chi2_tail():
    assert pl.chi2_sf(1.0, 1) == pytest.approx(0.317, abs=0.02)
    assert pl.chi2_sf(30.0, 10) == pytest.approx(0.00086, abs=0.0005)


def test_report_from_existing_results(project, calibrated):
    assert pl.main(["check", str(config(project)), "--root", str(project)]) == 0
    (project / "reports" / "fake" / "report.md").unlink()
    moved = project / "results_old"
    (project / "results").rename(moved)
    assert pl.main(["report", str(config(project)), "--root", str(project), "--results", str(moved)]) == 0
    assert "CERTIFIED" in (project / "reports" / "fake" / "report.md").read_text()


def test_sign_test():
    assert pl.sign_test(pd.Series([1.2] * 11 + [0.9] * 2)) == pytest.approx(2 * 92 / 8192)
    assert pl.sign_test(pd.Series([1.1, 0.9])) == 1.0

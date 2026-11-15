"""Tests for credence_tool.py (light screen, weighty measure) with fake models. Run from the project root:  pytest"""

import json

import pandas as pd
import pytest

import credence_tool as ct
import run_prompts as rp
from test_credence_pipeline import claims_frame, config


def with_fact_partition(df):
    rows = []
    for i in range(3):  # one fact partition: exactly one cell true
        for neg in (0, 1):
            true_cell = i == 0
            rows.append(dict(prop_id=f"part_fact_partition_9_e{i}_{'pn'[neg]}", claim_id=f"part_fact_partition_9_e{i}",
                             neg=neg, tier="partition", text=f"Fact cell {i} is{' not' if neg else ''} so.",
                             truth="true" if (neg == 0) == true_cell else "false", objective_chance="",
                             settles="Already settled as a matter of established fact.",
                             partition_id="part_fact_partition_9", partition_role="single" if not neg else "pair"))
    extra = pd.DataFrame(rows)
    for col in df.columns:
        if col not in extra:
            extra[col] = ""
    return pd.concat([df, extra[df.columns]], ignore_index=True)


def belief_map(df):
    b = {}
    for r in df.itertuples():
        if r.objective_chance != "" and not pd.isna(r.objective_chance):
            b[r.prop_id] = float(r.objective_chance)
        else:
            b[r.prop_id] = 0.95 if r.truth == "true" else 0.05
    # a calibrated believer in a 3-cell fact partition: true cell high, the others low, summing to 1
    for i, v in enumerate([0.9, 0.05, 0.05]):
        b[f"part_fact_partition_9_e{i}_p"], b[f"part_fact_partition_9_e{i}_n"] = v, 1 - v
    return b


@pytest.fixture
def project(tmp_path, monkeypatch):
    df = with_fact_partition(claims_frame())
    (tmp_path / "data").mkdir()
    df.to_csv(tmp_path / "data" / "claims.csv", index=False)
    b = belief_map(df)
    monkeypatch.setattr(rp.FakeBackend, "credence", staticmethod(lambda key: b.get(key, 0.5)))
    return tmp_path


def test_battery_is_small_spread_and_keeps_partitions_whole(project):
    claims = pd.read_csv(project / "data" / "claims.csv", dtype=str, keep_default_na=False)
    claims["neg"] = claims["neg"].astype(int)
    claims["objective_chance"] = pd.to_numeric(claims["objective_chance"], errors="coerce")
    bat = ct.select_battery(claims, n_chance=3, n_fact_parts=1)
    chance = bat["block1"][(bat["block1"]["tier"] == "chance") & (bat["block1"]["neg"] == 0)]
    assert sorted(chance["objective_chance"]) == [0.1, 0.4, 0.9]  # lowest, middle, highest
    assert bat["chance_partitions"] == ["part_chance_x"] and bat["fact_partitions"] == ["part_fact_partition_9"]
    for block in ("block1", "block2"):
        g = bat[block].groupby("claim_id")["neg"].apply(sorted)
        assert all(v == [0, 1] for v in g)  # every claim with its negation
    assert bat["block2"]["claim_id"].nunique() == 3


def test_a_calibrated_model_passes_the_screen(project):
    assert ct.main(["screen", str(config(project)), "--root", str(project)]) == 0
    cert = json.loads((project / "reports" / "fake-screen" / "certificate.json").read_text())
    assert cert["certified"] and cert["certification"]["checks"]["partition_coherence_p"]["n"] == 2
    assert (project / "results" / "utility_points_fake-screen.csv").exists()  # the utility step always runs
    assert not (project / "results" / "raw_calls_matching_fake.csv").exists()  # nothing mixes with full runs


def test_an_incoherent_model_stops_early(project, monkeypatch):
    monkeypatch.setattr(rp.FakeBackend, "credence", staticmethod(lambda key: 0.6))  # every cell 0.6: sums 1.8
    assert ct.main(["screen", str(config(project)), "--root", str(project)]) == 1
    cert = json.loads((project / "reports" / "fake-screen" / "certificate.json").read_text())
    assert not cert["certified"] and "stopped early" in cert["stopped"]
    assert not (project / "results" / "raw_calls_screen_block2_matching_fake-screen.csv").exists()


def test_measure_needs_a_passing_screen(project):
    claims = project / "mine.csv"
    pd.DataFrame({"text": ["The Eiffel Tower is in Paris."]}).to_csv(claims, index=False)
    with pytest.raises(SystemExit):
        ct.main(["measure", str(config(project)), "--claims", str(claims), "--root", str(project)])
    assert ct.main(["screen", str(config(project)), "--root", str(project)]) == 0
    assert ct.main(["measure", str(config(project)), "--claims", str(claims), "--root", str(project)]) == 0
    out = pd.read_csv(project / "reports" / "fake-screen" / "audit_mine.csv")
    assert len(out) == 2 and out["ramsey_linear_pair"].notna().all()


def test_decisive_fail():
    assert ct.decisive_fail(1e-6, 0.001) and not ct.decisive_fail(0.01, 0.001) and not ct.decisive_fail(None, 0.001)

"""Tests for analysis.py: crossings, utility, coherence, and an end-to-end run on fake results.

Run from the project root:  pytest
The last test also runs on your real results, if they are in results/.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import analysis as an
import make_prompts as mp
import run_prompts as rp
import utility_scale as us

ROOT = Path(__file__).resolve().parents[1]


# --- Reading a curve -------------------------------------------------------------------


def test_crossing_interpolates():
    assert an.crossing([0, 0.5, 1], [0.9, 0.7, 0.1]) == (pytest.approx(2 / 3), "clean")


@pytest.mark.parametrize("p, value, status", [
    ([0.9, 0.8, 0.7], 1.0, "never_below"),
    ([0.3, 0.2, 0.1], 0.0, "never_above"),
])
def test_censored_curves(p, value, status):
    assert an.crossing([0, 0.5, 1], p) == (value, status)


def test_rising_and_multiple():
    assert an.crossing([0, 0.5, 1], [0.1, 0.4, 0.9])[1] == "rising"
    assert an.crossing([0, 0.25, 0.5, 0.75, 1], [0.9, 0.4, 0.6, 0.2, 0.1]) == (pytest.approx(0.2), "multiple")


def test_area_matches_a_step_curve():
    x = np.linspace(0, 1, 101)
    assert an.area_estimate(x, (x < 0.3).astype(float)) == pytest.approx(0.3, abs=0.01)


def test_logistic_fit_recovers_midpoint_and_slope():
    x = np.linspace(0, 1, 21)
    p = 1 / (1 + np.exp(10 * (x - 0.35)))
    mid, slope = an.logistic_fit(x, p)
    assert mid == pytest.approx(0.35, abs=0.01) and slope == pytest.approx(10, rel=0.05)


# --- Utility ------------------------------------------------------------------------------


def test_utility_function_is_monotone_and_anchored():
    pts = pd.DataFrame({"level": [1, 2, 2], "utility": [0.5, 0.25, 0.75], "amount": [994.0, 989.0, 1000.0],
                        "status": ["clean", "clean", "clean"]})
    u = an.utility_function(pts)
    assert u(0) == 0 and u(1000) == 1
    assert u(994) == pytest.approx(0.5) and u(989) == pytest.approx(0.25)
    assert u(500) == pytest.approx(0.25 * 500 / 989)
    xs = np.arange(0, 1001, 10)
    assert (np.diff(u(xs)) >= 0).all()


def test_no_measured_points_means_no_utility():
    pts = pd.DataFrame({"level": [1], "utility": [0.5], "amount": [np.nan], "status": ["never_below"]})
    assert an.utility_function(pts) is None and an.utility_function(None) is None


# --- End to end on fake results --------------------------------------------------------------


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    root = tmp_path_factory.mktemp("proj")
    claims = pd.DataFrame({
        "prop_id": ["c1_p", "c1_n", "c2_p", "c2_n", "c3_p", "c3_n"],
        "claim_id": ["c1", "c1", "c2", "c2", "c3", "c3"],
        "neg": [0, 1, 0, 1, 0, 1],
        "tier": ["fact", "fact", "chance", "chance", "forecast", "forecast"],
        "text": ["Canberra is the capital of Australia.", "Canberra is not the capital of Australia.",
                 "A fair die will show 6.", "A fair die will not show 6.",
                 "X will happen.", "X will not happen."],
        "truth": ["true", "false", "false", "true", "unknown", "unknown"],
        "objective_chance": ["", "", 1 / 6, 5 / 6, "", ""],
        "settles": ["s"] * 6, "source": [""] * 6, "source_id": [""] * 6, "source_question": [""] * 6,
        "kuq_category": [""] * 6, "human_forecast": ["", "", "", "", 0.3, 0.7], "close_date": [""] * 6,
        "partition_id": [""] * 6, "partition_role": [""] * 6, "flags": [""] * 6,
    })
    (root / "data").mkdir()
    claims.to_csv(root / "data" / "claims.csv", index=False)
    assert mp.main(["--root", str(root)]) == 0
    for name in ("matching", "ramsey"):
        rp.run(root / "prompts" / f"{name}.csv", root / "results" / f"raw_calls_{name}_fake.csv",
               rp.FakeBackend(), f"{name}_fake", "fake", log=lambda *_: None)
    out = us.build_scale(us.FakeUtilityBackend(rho=2.0), "fake", "fake", levels=2, log=lambda *_: None)
    us.write_outputs(out, us.summary_for(out, "fake", "fake", "die"), root / "results", "fake")
    assert an.main(["--root", str(root)]) == 0
    return root


def read(project, name):
    return pd.read_csv(project / "data" / "analysis" / "fake" / f"{name}.csv")


def test_every_output_written(project):
    names = {p.name for p in (project / "data" / "analysis" / "fake").iterdir()}
    for stem in ("curves", "fits", "propositions", "coherence", "partitions", "utility_curve", "summary_by_tier",
                 "agreement", "ramsey_by_matching", "spot_check"):
        assert f"{stem}.csv" in names
    assert "report.md" in names


def test_credences_recover_the_fake_models_beliefs(project):
    props = read(project, "propositions").set_index("prop_id")
    for pid in props.index:
        truth = rp.FakeBackend.credence(pid)
        assert props.loc[pid, "matching"] == pytest.approx(truth, abs=0.02), pid
        assert props.loc[pid, "ramsey_linear"] == pytest.approx(truth, abs=0.02), pid
        assert props.loc[pid, "ramsey_ce"] == pytest.approx(1000 * props.loc[pid, "ramsey_linear"])


def test_utility_converts_ramsey_crossings(project):
    props = read(project, "propositions")
    # The fake utility is u(x) = (x/1000)^2, measured at a few points and interpolated linearly.
    expected = props["ramsey_linear"] ** 2
    assert (props["ramsey_utility"] - expected).abs().max() < 0.06
    assert (props["ramsey_utility"] <= props["ramsey_linear"] + 1e-9).all()


def test_coherence_sums(project):
    coh = read(project, "coherence")
    props = read(project, "propositions").set_index("prop_id")
    row = coh[(coh["claim_id"] == "c1") & (coh["estimate"] == "matching")].iloc[0]
    assert row["sum"] == pytest.approx(props.loc["c1_p", "matching"] + props.loc["c1_n", "matching"])
    assert row["dutch_book_loss"] == pytest.approx(abs(row["sum"] - 1))


def test_fits_have_one_row_per_proposition_and_method(project):
    fits = read(project, "fits")
    assert len(fits) == 6 * 2 and set(fits["method"]) == {"matching", "ramsey"}
    assert fits["falling"].all() and (fits["status"] == "clean").all()


def test_curves_average_both_orders(project):
    c = read(project, "curves")
    assert np.allclose(c["p_mean"], (c["p_first"] + c["p_second"]) / 2)
    assert set(c.loc[c["method"] == "matching", "grid_value"]) == set(mp.K_VALUES)


def test_summary_and_report(project):
    s = read(project, "summary_by_tier")
    assert set(s["estimate"]) == set(an.ESTIMATES)
    chance = s[(s["estimate"] == "matching") & (s["tier"] == "chance")].iloc[0]
    assert not math.isnan(chance["mae_objective"])
    text = (project / "data" / "analysis" / "fake" / "report.md").read_text()
    assert "Do the methods agree?" in text and "Utility scale" in text


# --- Your real results --------------------------------------------------------------------------

REAL = [p.stem.removeprefix("raw_calls_matching_") for p in sorted((ROOT / "results").glob("raw_calls_matching_*.csv"))
        if not p.stem.endswith("_fake")]


@pytest.mark.skipif(not REAL or not (ROOT / "data" / "claims.csv").exists(), reason="no real results yet")
@pytest.mark.parametrize("run", REAL)
def test_real_results_analyse_cleanly(run):
    res = ROOT / "results"
    out = an.analyse(an.load_claims(ROOT / "data" / "claims.csv"), an.load_method(res, "matching", run),
                     an.load_method(res, "ramsey", run), an.load_utility(res, run))
    props = out["propositions"]
    assert props["matching"].notna().mean() > 0.95
    assert props["matching"].between(0, 1).all()
    assert out["fits"].groupby("method").size().min() == len(props)

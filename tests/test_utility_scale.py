"""Tests for utility_scale.py, using a fake model with a known utility curve. No GPU needed.

Run from the project root:  pytest
"""

import json
import math

import pytest

import make_prompts as mp
import utility_scale as us


def quiet(*_):
    pass


# --- Wording and grids -------------------------------------------------------------


def test_grid_is_even_with_a_finer_top():
    g = us.sure_grid(0, 1000)
    assert g[0] == 0 and g[-1] == 1000
    assert set(range(0, 1001, 50)) <= set(g)
    assert set(range(900, 1001, 10)) <= set(g)
    assert us.sure_grid(0, 500)[-1] == 500


def test_prompts_name_the_neutral_event_and_both_orders():
    rows = us.level_prompts(1, [dict(utility=0.5, lo=0, hi=1000)], "die")
    assert len(rows) == 2 * len(us.sure_grid(0, 1000))
    first = rows[0]
    assert "A six-sided die will be rolled once." in first["user"]
    assert "You receive £1000 if the die shows 1, 2 or 3, and £0 if the die shows 4, 5 or 6." in first["user"]
    assert "You receive £0 for certain." in first["user"]
    assert {r["order"] for r in rows} == set(mp.ORDERS)
    assert len({r["prompt_id"] for r in rows}) == len(rows)


def test_unknown_neutral_event_is_refused():
    with pytest.raises(SystemExit):
        us.neutral_event("spinner")


# --- Finding indifference ------------------------------------------------------------


def test_crossing_interpolates():
    assert us.crossing([0, 100, 200], [0.9, 0.7, 0.3]) == (pytest.approx(150), "clean")


@pytest.mark.parametrize("p, status", [([0.9, 0.8, 0.6], "never_below"), ([0.4, 0.3, 0.1], "never_above")])
def test_no_crossing(p, status):
    m, s = us.crossing([0, 100, 200], p)
    assert math.isnan(m) and s == status


def test_several_crossings_use_the_first():
    assert us.crossing([0, 100, 200, 300], [0.9, 0.4, 0.6, 0.2]) == (pytest.approx(80), "multiple")


def test_bisection_intervals():
    assert us.next_intervals({0.0: 0, 1.0: 1000}) == [dict(utility=0.5, lo=0, hi=1000)]
    assert us.next_intervals({0.0: 0, 0.5: 707.4, 1.0: 1000}) == [
        dict(utility=0.25, lo=0, hi=707), dict(utility=0.75, lo=707, hi=1000)]


# --- The whole scale, against known utility curves ----------------------------------------


@pytest.mark.parametrize("rho", [1.0, 2.0, 0.5])
def test_recovers_a_known_utility_curve(rho):
    out = us.build_scale(us.FakeUtilityBackend(rho=rho), "fake", "fake", levels=3, log=quiet)
    pts = out["points"][out["points"]["level"] > 0]
    assert len(pts) == 7 and (pts["status"] == "clean").all()
    for r in pts.itertuples():
        assert r.amount == pytest.approx(1000 * r.utility ** (1 / rho), abs=15), r.utility
    assert us.fit_power(out["points"]) == pytest.approx(rho, abs=0.05)


def test_position_bias_is_averaged_out():
    out = us.build_scale(us.FakeUtilityBackend(rho=1.0, position_bias=1.0), "fake", "fake", levels=1, log=quiet)
    half = out["points"].set_index("utility").loc[0.5]
    assert half["amount"] == pytest.approx(500, abs=10)
    assert half["amount_first"] > half["amount"] > half["amount_second"]


class AlwaysTheBet(us.FakeUtilityBackend):
    """Prefers the bet on N to every sure amount, like Ministral in Stage 1's transitivity pairs."""

    def score(self, rows):
        return [[("A", math.log(0.85)), ("B", math.log(0.15))] if r["target_label"] == "A"
                else [("A", math.log(0.15)), ("B", math.log(0.85))] for r in rows]


def test_no_indifference_point_stops_the_scale():
    out = us.build_scale(AlwaysTheBet(), "fake", "fake", levels=3, log=quiet)
    measured = out["points"][out["points"]["level"] > 0]
    assert len(measured) == 1 and measured["status"].iloc[0] == "never_below"
    s = us.summary_for(out, "fake", "fake", "die")
    assert s["m_half"] is None and s["rho"] is None and "not available" in s["reading"]


# --- Command line ----------------------------------------------------------------------------


def test_main_with_fake_backend(tmp_path, capsys):
    assert us.main(["--backend", "fake", "--fake-rho", "2", "--out-dir", str(tmp_path)]) == 0
    for name in ("points", "curves", "calls"):
        assert (tmp_path / f"utility_{name}_fake.csv").exists()
    summary = json.loads((tmp_path / "utility_summary_fake.json").read_text())
    assert summary["m_half"] == pytest.approx(707, abs=15) and summary["rho"] == pytest.approx(2, abs=0.05)
    assert "risk-seeking" in capsys.readouterr().out


def test_main_requires_model_for_vllm():
    with pytest.raises(SystemExit):
        us.main([])

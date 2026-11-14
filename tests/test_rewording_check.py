"""Tests for rewording_check.py: the reworded prompts, and the comparison on fake results.

Run from the project root:  pytest
"""

import pandas as pd
import pytest

import make_prompts as mp
import rewording_check as rc
import run_prompts as rp

CLAIMS = pd.DataFrame({
    "prop_id": ["c1_p", "c1_n", "c2_p", "c2_n"], "claim_id": ["c1", "c1", "c2", "c2"], "neg": [0, 1, 0, 1],
    "tier": ["fact", "fact", "chance", "chance"],
    "text": ["Canberra is the capital of Australia.", "Canberra is not the capital of Australia.",
             "A fair die will show 6.", "A fair die will not show 6."],
    "truth": ["true", "false", "false", "true"], "objective_chance": ["", "", 1 / 6, 5 / 6],
    "settles": ["Already settled as a matter of established fact."] * 2 + ["Settled by a seeded random draw."] * 2,
    "source": [""] * 4, "source_id": [""] * 4, "source_question": [""] * 4, "kuq_category": [""] * 4,
    "human_forecast": [""] * 4, "close_date": [""] * 4, "partition_id": [""] * 4, "partition_role": [""] * 4,
    "flags": [""] * 4,
})


@pytest.mark.parametrize("variant, setup, phrase", [
    ("tickets", mp.TICKETS_SETUP, "You receive £550, whichever ticket is drawn."),
    ("whatever", "", "You receive £550, whatever happens."),
])
def test_only_the_sure_option_changes(variant, setup, phrase):
    claims = CLAIMS.assign(neg=CLAIMS["neg"].astype(int))
    new = rc.reworded_prompts(claims, variant)
    old = mp.ramsey_prompts(claims)
    assert len(new) == len(old)
    assert (new["target_desc"] == old["target_desc"]).all()
    assert (new["grid_value"] == old["grid_value"]).all() and (new["order"] == old["order"]).all()
    row = new[(new["grid_value"] == 550) & (new["order"] == "target_second")].iloc[0]
    assert row["other_desc"] == phrase and "for certain" not in row["user"]
    assert row["user"].startswith(setup) if setup else row["user"].startswith("Option A:")
    assert set(new["method"]) == {"ramsey"}
    assert new["prompt_id"].is_unique and not set(new["prompt_id"]) & set(old["prompt_id"])


def test_zero_reads_as_nothing():
    assert rc.sure_text(0, "tickets") == "You receive nothing, whichever ticket is drawn."
    with pytest.raises(SystemExit):
        rc.sure_text(100, "spinner")


def test_compare_end_to_end(tmp_path, capsys):
    (tmp_path / "data").mkdir()
    CLAIMS.to_csv(tmp_path / "data" / "claims.csv", index=False)
    assert mp.main(["--root", str(tmp_path)]) == 0
    assert rc.main(["prompts", "--root", str(tmp_path)]) == 0
    for name in ("matching", "ramsey", "ramsey_reworded"):
        rp.run(tmp_path / "prompts" / f"{name}.csv", tmp_path / "results" / f"raw_calls_{name}_fake.csv",
               rp.FakeBackend(), f"{name}_fake", "fake", log=lambda *_: None)
    assert rc.main(["compare", "--root", str(tmp_path)]) == 0
    out = tmp_path / "data" / "analysis" / "fake"
    for key in ("propositions", "summary", "curves"):
        assert (out / f"rewording_tickets_{key}.csv").exists()
    props = pd.read_csv(out / "rewording_tickets_propositions.csv")
    assert len(props) == 4
    # The fake model ignores wording, so both runs give the same certainty equivalents.
    assert (props["ce_change"].abs() < 1).all()
    assert "Rewording check" in capsys.readouterr().out


def test_compare_explains_a_missing_run(tmp_path):
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "raw_calls_matching_fake.csv").write_text("p_target,mass_outside,grid_value\n0.5,0,0\n")
    with pytest.raises(SystemExit):
        rc.main(["compare", "--root", str(tmp_path)])

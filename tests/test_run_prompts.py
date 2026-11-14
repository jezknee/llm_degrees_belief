"""Unit tests for run_prompts.py. Offline: uses the fake backend, no GPU or model.

Run from the project root:  pytest
"""

import json
import math
from types import SimpleNamespace

import pandas as pd
import pytest

import make_prompts as mp
import run_prompts as rp

# --- Turning top tokens into choice probabilities ---------------------------------


def test_sums_variants_of_a_and_b():
    top = [("A", math.log(0.3)), (" A", math.log(0.2)), ("B", math.log(0.4)), ("I", math.log(0.1))]
    lp = rp.choice_logprobs(top)
    assert math.isclose(math.exp(lp["lp_A"]), 0.5) and math.isclose(math.exp(lp["lp_B"]), 0.4)
    assert lp["a_missing"] == lp["b_missing"] == 0
    assert lp["top_token"] == "B"


def test_lowercase_a_is_not_counted():
    top = [("a", math.log(0.5)), ("A", math.log(0.2)), ("B", math.log(0.3))]
    assert math.isclose(math.exp(rp.choice_logprobs(top)["lp_A"]), 0.2)


def test_missing_token_is_flagged_and_bounded():
    top = [("A", math.log(0.9)), ("I", math.log(0.05)), ("The", math.log(0.01))]
    lp = rp.choice_logprobs(top)
    assert lp["b_missing"] == 1 and lp["a_missing"] == 0
    assert math.isclose(lp["lp_B"], math.log(0.01))  # smallest listed: an upper bound


def test_empty_top_list():
    lp = rp.choice_logprobs([])
    assert lp["a_missing"] == lp["b_missing"] == 1
    probs = rp.choice_probs(lp["lp_A"], lp["lp_B"], "A")
    assert math.isclose(probs["p_A"], 0.5)


@pytest.mark.parametrize("label, expected", [("A", 0.8), ("B", 0.2)])
def test_choice_probs_renormalise_and_pick_target(label, expected):
    probs = rp.choice_probs(math.log(0.72), math.log(0.18), label)
    assert math.isclose(probs["p_A"] + probs["p_B"], 1)
    assert math.isclose(probs["p_target"], expected)
    assert math.isclose(probs["mass_outside"], 0.10)


def test_build_messages():
    msgs = rp.build_messages({"system": "Pick one.", "user": "A or B?"})
    assert msgs == [{"role": "system", "content": "Pick one."}, {"role": "user", "content": "A or B?"}]
    assert rp.build_messages({"system": "", "user": "A or B?"}) == [{"role": "user", "content": "A or B?"}]


def test_extract_top_from_vllm_style_output():
    lp = lambda value, text: SimpleNamespace(logprob=value, decoded_token=text)  # noqa: E731
    output = SimpleNamespace(outputs=[SimpleNamespace(logprobs=[{5: lp(-0.1, "A"), 7: lp(-2.5, None)}])])
    tokenizer = SimpleNamespace(decode=lambda ids: {7: " B"}[ids[0]])
    assert rp.extract_top(output, tokenizer) == [("A", -0.1), (" B", -2.5)]
    empty = SimpleNamespace(outputs=[SimpleNamespace(logprobs=None)])
    assert rp.extract_top(empty) == []


# --- Running end to end with the fake backend --------------------------------------


@pytest.fixture
def project(tmp_path):
    """A project with a small claims file turned into prompts by make_prompts.py."""
    claims = pd.DataFrame({
        "prop_id": ["c1_p", "c1_n", "c2_p", "c2_n"],
        "claim_id": ["c1", "c1", "c2", "c2"],
        "neg": [0, 1, 0, 1],
        "tier": ["fact", "fact", "chance", "chance"],
        "text": ["The capital of Australia is Canberra.", "The capital of Australia is not Canberra.",
                 "A fair coin, tossed once, will land heads.", "A fair coin, tossed once, will not land heads."],
        "settles": ["Already settled as a matter of established fact."] * 2 + ["Settled by a seeded random draw."] * 2,
    })
    (tmp_path / "data").mkdir()
    claims.to_csv(tmp_path / "data" / "claims.csv", index=False)
    assert mp.main(["--root", str(tmp_path)]) == 0
    return tmp_path


def run_fake(project, name="smoke", **kw):
    backend = rp.FakeBackend()
    out = project / "results" / f"raw_calls_{name}_fake.csv"
    stats = rp.run(project / "prompts" / f"{name}.csv", out, backend, f"{name}_fake", "fake",
                   log=lambda *_: None, **kw)
    return backend, out, stats


def read(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


@pytest.mark.parametrize("name", ["smoke", "matching", "ramsey", "stage1"])
def test_every_prompt_gets_exactly_one_result(project, name):
    _, out, stats = run_fake(project, name)
    prompts, results = read(project / "prompts" / f"{name}.csv"), read(out)
    assert stats["scored"] == len(prompts)
    assert sorted(results["prompt_id"]) == sorted(prompts["prompt_id"])
    assert list(results.columns) == rp.RESULT_COLUMNS


def test_choice_probabilities_are_produced_and_valid(project):
    _, out, _ = run_fake(project, "smoke")
    r = read(out)
    for col in ("lp_A", "lp_B", "p_A", "p_B", "p_target", "mass_outside"):
        assert r[col].str.len().gt(0).all(), col
    p = r[["p_A", "p_B", "p_target", "mass_outside"]].astype(float)
    assert p.ge(0).all().all() and p.le(1).all().all()
    assert ((p["p_A"] + p["p_B"]) - 1).abs().max() < 1e-9
    target_a = r["target_label"] == "A"
    assert (p.loc[target_a, "p_target"] == p.loc[target_a, "p_A"]).all()
    assert (p.loc[~target_a, "p_target"] == p.loc[~target_a, "p_B"]).all()
    assert (p["mass_outside"].round(6) == 0.03).all()


def test_metadata_carried_from_prompts(project):
    _, out, _ = run_fake(project, "matching")
    r = read(out).set_index("prompt_id")
    prompts = read(project / "prompts" / "matching.csv").set_index("prompt_id")
    for col in ("method", "prop_id", "claim_id", "tier", "grid_value", "order", "target_label"):
        assert (r[col] == prompts.loc[r.index, col]).all(), col
    assert set(r["backend"]) == {"fake"} and set(r["input_file"]) == {"matching.csv"}


def test_choice_probability_falls_as_grid_value_rises(project):
    _, out, _ = run_fake(project, "matching")
    r = read(out)
    r["grid"] = r["grid_value"].astype(int)
    r["p"] = r["p_target"].astype(float)
    for pid, g in r.groupby("prop_id"):
        curve = g.groupby("grid")["p"].mean()
        assert curve.iloc[0] - curve.iloc[-1] > 0.5, pid
        assert (curve.diff().dropna() <= 1e-12).all(), pid


def test_batches_are_saved_as_they_go(project):
    backend, out, _ = run_fake(project, "smoke", batch_size=50)
    assert backend.calls == math.ceil(len(read(project / "prompts" / "smoke.csv")) / 50)
    assert len(read(out)) == backend.rows_scored


def test_rerun_resumes_without_duplicates(project):
    _, out, first = run_fake(project, "smoke", limit=40)
    assert first["scored"] == 40 and len(read(out)) == 40
    backend, out, second = run_fake(project, "smoke")
    assert second["already_done"] == 40 and backend.rows_scored == second["total"] - 40
    assert read(out)["prompt_id"].is_unique and len(read(out)) == second["total"]
    backend, _, third = run_fake(project, "smoke")
    assert third["scored"] == 0 and backend.calls == 0


def test_refuses_to_mix_models_in_one_file(project):
    _, out, _ = run_fake(project, "smoke", limit=10)
    with pytest.raises(SystemExit):
        rp.run(project / "prompts" / "smoke.csv", out, rp.FakeBackend(), "smoke_fake", "other-model",
               log=lambda *_: None)


def test_backend_returning_wrong_number_of_results_is_an_error(project):
    class Broken(rp.FakeBackend):
        def score(self, rows):
            return super().score(rows)[:-1]
    with pytest.raises(RuntimeError):
        rp.run(project / "prompts" / "smoke.csv", project / "results" / "x.csv", Broken(), "x", "fake",
               log=lambda *_: None)


def test_bad_prompts_file_is_rejected(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"prompt_id": ["a"], "user": ["x"]}).to_csv(p, index=False)
    with pytest.raises(SystemExit):
        rp.load_prompts(p)


# --- Summary and warnings ----------------------------------------------------------


def test_summary_reports_checks(project):
    for name in ("matching", "stage1"):
        run_fake(project, name)
    df = pd.concat([read(project / "results" / f"raw_calls_{n}_fake.csv") for n in ("matching", "stage1")])
    s = rp.summarize(df)
    m = s["by_method"]["matching"]
    assert m["share_falling"] == 1.0 and m["propositions"] == 4
    assert math.isclose(m["mean_mass_outside"], 0.03, abs_tol=1e-4)
    assert m["rows_missing_A_or_B"] == 0 and m["order_gap"] > 0  # fake model leans towards option A
    assert s["by_method"]["stage1"]["dominance_correct"] < 1  # fake stage1 answers sit near 50/50
    assert any("dominance" in w for w in rp.warnings_for(s))


def test_warnings_flag_problems():
    summary = {"by_method": {"ramsey": {"rows": 100, "mean_mass_outside": 0.4, "rows_missing_A_or_B": 30,
                                        "order_gap": 0.5, "share_falling": 0.2}}}
    text = " ".join(rp.warnings_for(summary))
    for phrase in ("other than A and B", "missing", "fall as the grid value rises", "position bias"):
        assert phrase in text


def test_fake_backend_is_deterministic():
    rows = [{"method": "ramsey", "grid_value": "300", "prop_id": "x_p", "target_label": "A"}]
    assert rp.FakeBackend().score(rows) == rp.FakeBackend().score(rows)


# --- Command line --------------------------------------------------------------------


def test_main_with_fake_backend(project, monkeypatch, capsys):
    monkeypatch.chdir(project)
    assert rp.main(["prompts/smoke.csv", "--backend", "fake", "--show-first"]) == 0
    printed = capsys.readouterr().out
    assert "Top tokens" in printed and "smoke" in printed
    out = project / "results" / "raw_calls_smoke_fake.csv"
    summary = json.loads(out.with_suffix(".summary.json").read_text())
    assert summary["rows"] == len(read(project / "prompts" / "smoke.csv"))
    assert set(summary["by_method"]) == {"matching", "ramsey"}


def test_main_requires_model_for_vllm(project, monkeypatch):
    monkeypatch.chdir(project)
    with pytest.raises(SystemExit):
        rp.main(["prompts/smoke.csv"])


def test_default_run_id():
    from pathlib import Path
    assert rp.default_run_id(Path("prompts/smoke.csv"), "mistralai/Ministral-3-8B-Instruct") == \
        "smoke_ministral-3-8b-instruct"

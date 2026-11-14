"""Checks on real results in results/ against the prompts they came from.

Skipped until results/raw_calls_*.csv exist. After copying results back from the pod,
run from the project root:

    pytest tests/test_results_output.py -v
"""

from pathlib import Path

import pandas as pd
import pytest

import run_prompts as rp

ROOT = Path(__file__).resolve().parents[1]
RESULT_FILES = sorted((ROOT / "results").glob("raw_calls_*.csv"))

pytestmark = pytest.mark.skipif(not RESULT_FILES, reason="no results yet")


@pytest.fixture(params=RESULT_FILES, ids=lambda p: p.stem)
def results(request):
    return pd.read_csv(request.param, dtype=str, keep_default_na=False)


def test_columns_and_one_model(results):
    assert list(results.columns) == rp.RESULT_COLUMNS
    assert results["model"].nunique() == 1 and results["run_id"].nunique() == 1


def test_every_prompt_scored_once(results):
    prompts_file = ROOT / "prompts" / results["input_file"].iloc[0]
    if not prompts_file.exists():
        pytest.skip(f"{prompts_file.name} not found locally")
    prompts = pd.read_csv(prompts_file, dtype=str, keep_default_na=False)
    assert results["prompt_id"].is_unique
    missing = set(prompts["prompt_id"]) - set(results["prompt_id"])
    assert not missing, f"{len(missing)} prompts not scored yet; rerun the same command on the pod to resume"


def test_choice_probabilities_present_and_valid(results):
    p = results[["p_A", "p_B", "p_target", "mass_outside"]].apply(pd.to_numeric)
    assert p.notna().all().all()
    assert p.ge(0).all().all() and p.le(1).all().all()
    assert ((p["p_A"] + p["p_B"]) - 1).abs().max() < 1e-6


def test_model_mostly_answers_a_or_b(results):
    mass = pd.to_numeric(results["mass_outside"])
    assert mass.median() < 0.2, "most probability is on other tokens; check the prompt and chat template"


def test_a_and_b_almost_always_in_top_tokens(results):
    missing = (pd.to_numeric(results["a_missing"]) + pd.to_numeric(results["b_missing"])) > 0
    assert missing.mean() < 0.05, f"A or B missing in {missing.mean():.0%} of rows; raise --top-k"

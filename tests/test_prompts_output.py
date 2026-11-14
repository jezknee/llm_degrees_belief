"""Checks on the real prompts/ files against the real data/claims.csv.

Skipped until you have run make_prompts.py. Run from the project root:

    python src/make_prompts.py
    pytest tests/test_prompts_output.py -v
"""

from pathlib import Path

import pandas as pd
import pytest

import make_prompts as mp

ROOT = Path(__file__).resolve().parents[1]
CLAIMS = ROOT / "data" / "claims.csv"
PROMPTS = ROOT / "prompts"

pytestmark = pytest.mark.skipif(not (PROMPTS / "matching.csv").exists(), reason="prompts not built yet")


@pytest.fixture(scope="module")
def claims():
    return mp.load_claims(CLAIMS)


def read(name):
    return pd.read_csv(PROMPTS / f"{name}.csv", dtype=str, keep_default_na=False)


@pytest.mark.parametrize("method, values", [("matching", mp.K_VALUES), ("ramsey", mp.C_VALUES)])
def test_every_proposition_in_claims_has_prompts(claims, method, values):
    df = read(method)
    missing = set(claims["prop_id"]) - set(df["prop_id"])
    extra = set(df["prop_id"]) - set(claims["prop_id"])
    assert not missing, f"{len(missing)} propositions have no {method} prompts; rerun make_prompts.py"
    assert not extra, f"{len(extra)} prompts refer to propositions not in claims.csv; rerun make_prompts.py"
    assert len(df) == len(claims) * len(values) * 2


@pytest.mark.parametrize("method", ["matching", "ramsey"])
def test_prompts_use_current_claim_text(claims, method):
    df = read(method).drop_duplicates("prop_id").set_index("prop_id")
    c = claims.set_index("prop_id")
    stale = [p for p in df.index if f'"{c.loc[p, "text"]}"' not in df.loc[p, "user"]]
    assert not stale, f"prompts out of date for {stale[:5]}; rerun make_prompts.py"


def test_prompt_ids_unique_across_files():
    ids = pd.concat([read(n)["prompt_id"] for n in ("stage1", "matching", "ramsey")])
    assert ids.is_unique


def test_one_template_version():
    versions = set().union(*(set(read(n)["template_version"]) for n in ("stage1", "matching", "ramsey", "smoke")))
    assert versions == {mp.TEMPLATE_VERSION}


def test_smoke_file_is_small_and_complete():
    smoke = read("smoke")
    assert smoke["prop_id"].nunique() == 5
    assert len(smoke) == 5 * 2 * (len(mp.K_VALUES) + len(mp.C_VALUES))

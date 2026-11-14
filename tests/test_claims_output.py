"""Checks on the real data/claims.csv produced by build_claims.py.

These tests are skipped until you have run the build. Run from the project root:

    python src/build_claims.py
    pytest tests/test_claims_output.py -v
"""

from pathlib import Path

import pandas as pd
import pytest

import build_claims as bc

ROOT = Path(__file__).resolve().parents[1]
CLAIMS = ROOT / "data" / "claims.csv"
REVIEW = ROOT / "data" / "review.csv"
CFG = bc.Config(root=ROOT)

pytestmark = pytest.mark.skipif(not CLAIMS.exists(), reason="data/claims.csv not built yet")


@pytest.fixture(scope="module")
def claims():
    df = pd.read_csv(CLAIMS, dtype={"source_id": str}, keep_default_na=False,
                     na_values={"objective_chance": [""], "human_forecast": [""]})
    df["neg"] = df["neg"].astype(int)
    return df


@pytest.fixture(scope="module")
def review():
    return pd.read_csv(REVIEW, keep_default_na=False) if REVIEW.exists() else pd.DataFrame()


def claims_in(claims, tier):
    return claims[claims["tier"] == tier].drop_duplicates("claim_id")


# --- Are there enough questions? ----------------------------------------------


@pytest.mark.parametrize("tier, target", [
    ("forecast", CFG.n_forecast), ("kuq", CFG.n_kuq), ("fact", CFG.n_fact), ("chance", CFG.n_chance),
])
def test_enough_claims_per_tier(claims, tier, target):
    n = len(claims_in(claims, tier))
    assert n >= target, f"{tier}: only {n} claims, target {target}; see data/review.csv"


def test_enough_partitions(claims):
    parts = claims.loc[claims["partition_id"] != "", "partition_id"].unique()
    target = CFG.n_partition_chance + CFG.n_partition_kuq + CFG.n_partition_fact
    assert 10 <= len(parts) <= 15 and len(parts) >= target - 2, f"{len(parts)} partitions"


# --- Do the claims exist, and are they well formed? --------------------------------


def test_columns(claims):
    assert list(claims.columns) == bc.COLUMNS


def test_every_claim_has_proposition_and_negation(claims):
    assert claims["prop_id"].is_unique
    bad = [cid for cid, g in claims.groupby("claim_id") if sorted(g["neg"]) != [0, 1]]
    assert not bad, f"claims without exactly one proposition and one negation: {bad[:10]}"


def test_no_empty_or_duplicate_text(claims):
    assert claims["text"].str.strip().str.len().gt(0).all()
    dupes = claims.loc[claims["text"].str.lower().duplicated(keep=False), "text"]
    assert dupes.empty, f"duplicate propositions: {dupes.head().tolist()}"


def test_texts_are_declarative_statements(claims):
    bad = claims[claims["text"].map(lambda t: bool(bc.hard_problems(t)))]
    assert bad.empty, bad[["prop_id", "text"]].head(10).to_string()


def test_proposition_and_negation_differ(claims):
    wide = claims.pivot(index="claim_id", columns="neg", values="text")
    same = wide[wide[0].str.lower() == wide[1].str.lower()]
    assert same.empty, same.head().to_string()


def test_most_negations_contain_a_negating_word(claims):
    negs = claims[(claims["neg"] == 1) & (claims["partition_role"] != "pair")]
    share = negs["text"].map(lambda t: bool(bc.NEGATION_RE.search(t))).mean()
    assert share >= 0.9, f"only {share:.0%} of negations contain a negating word"


# --- Are the claims related to their questions? ------------------------------------


def test_claims_relate_to_source_questions(claims):
    with_q = claims[claims["source_question"].str.len() > 0]
    unrelated = with_q[[not bc.related(q, t) for q, t in zip(with_q["source_question"], with_q["text"])]]
    assert unrelated.empty, unrelated[["prop_id", "source_question", "text"]].head(10).to_string()


def test_llm_tiers_all_have_a_source_question(claims):
    llm = claims[claims["tier"].isin(["forecast", "kuq", "fact"])]
    assert llm["source_question"].str.len().gt(0).all()


# --- Have reasonable questions been included? -------------------------------------


def test_forecasts_are_open_markets_in_the_window(claims):
    fc = claims_in(claims, "forecast")
    assert set(fc["source"]) <= set(bc.MARKET_SOURCES)
    assert fc["close_date"].between(CFG.min_close, CFG.max_close).all()
    assert fc["human_forecast"].between(0, 1).all()


def test_forecasts_spread_across_sources(claims):
    shares = claims_in(claims, "forecast")["source"].value_counts(normalize=True)
    assert len(shares) == 3 and shares.min() >= 0.15, shares.to_dict()


def test_forecasts_are_not_all_near_certain(claims):
    hf = claims_in(claims, "forecast")["human_forecast"]
    assert hf.between(0.1, 0.9).mean() >= 0.3, "too few genuinely uncertain markets"


def test_kuq_spread_across_categories(claims):
    cats = claims_in(claims, "kuq")["kuq_category"].value_counts(normalize=True)
    assert len(cats) >= 2 and cats.max() <= 0.7, cats.to_dict()


def test_unknown_claims_are_settled_by_the_judge_clause(claims):
    assert claims_in(claims, "kuq")["settles"].str.contains("judge").all()
    assert claims_in(claims, "kuq")["truth"].eq("unknown").all()


def test_facts_are_balanced_true_and_false(claims):
    truth = claims_in(claims, "fact")["truth"].value_counts()
    assert truth.get("true", 0) == truth.get("false", 0), truth.to_dict()


def test_chance_claims_cover_the_probability_range(claims):
    p = claims_in(claims, "chance")["objective_chance"].astype(float)
    assert p.between(0, 1, inclusive="neither").all()
    assert p.min() < 0.15 and p.max() > 0.85
    assert claims_in(claims, "chance")["truth"].nunique() == 2


def test_partitions_are_complete(claims):
    parts = claims[claims["partition_id"] != ""]
    for pid, g in parts.groupby("partition_id"):
        singles = g[g["partition_role"] == "single"]
        assert len(singles) == 3 and (g["partition_role"] == "pair").sum() == 3, pid
        if pid.startswith("part_chance"):
            assert abs(singles["objective_chance"].astype(float).sum() - 1) < 1e-6, pid
        if pid.startswith("part_fact_partition") or pid.startswith("part_chance"):
            assert (singles["truth"] == "true").sum() == 1, pid


def test_rejected_items_are_not_in_claims(claims, review):
    if review.empty:
        pytest.skip("no review file")
    dropped = review[review["status"].isin(["rejected", "excluded"])]["item_id"]
    kept = set(claims["claim_id"]) | {c.rsplit("_e", 1)[0].removeprefix("part_") for c in claims["claim_id"]}
    assert set(dropped).isdisjoint(kept)

"""Unit tests for make_prompts.py, on a small synthetic claims file.

Run from the project root:  pytest
"""

import json
import re

import pandas as pd
import pytest

import make_prompts as mp

CLAIMS = [
    # prop_id, claim_id, neg, tier, text, settles, source, close_date
    ("fc_1_p", "fc_1", 0, "forecast", "Helen Zille will be the next mayor of Johannesburg.",
     "Resolves YES or NO by 2026-11-17 on Polymarket (https://x).", "polymarket", "2026-11-17"),
    ("fc_1_n", "fc_1", 1, "forecast", "Helen Zille will not be the next mayor of Johannesburg.",
     "Resolves YES or NO by 2026-11-17 on Polymarket (https://x).", "polymarket", "2026-11-17"),
    ("kuq_1_p", "kuq_1", 0, "kuq", "There are aliens living among humans on Earth today.",
     "Settled by an infallible judge who knows the true answer.", "turk", ""),
    ("kuq_1_n", "kuq_1", 1, "kuq", "There are no aliens living among humans on Earth today.",
     "Settled by an infallible judge who knows the true answer.", "turk", ""),
    ("fact_1_p", "fact_1", 0, "fact", "The capital of Australia is Sydney.",
     "Already settled as a matter of established fact.", "squad", ""),
    ("fact_1_n", "fact_1", 1, "fact", "The capital of Australia is not Sydney.",
     "Already settled as a matter of established fact.", "squad", ""),
    ("chance_00_p", "chance_00", 0, "chance", "A fair coin, tossed once, will land heads.",
     "Settled by a seeded random draw made by the experimenter; the outcome is stored separately.", "template", ""),
    ("chance_00_n", "chance_00", 1, "chance", "A fair coin, tossed once, will not land heads.",
     "Settled by a seeded random draw made by the experimenter; the outcome is stored separately.", "template", ""),
]


@pytest.fixture
def claims_path(tmp_path):
    df = pd.DataFrame(CLAIMS, columns=["prop_id", "claim_id", "neg", "tier", "text",
                                       "settles", "source", "close_date"])
    (tmp_path / "data").mkdir()
    path = tmp_path / "data" / "claims.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def claims(claims_path):
    return mp.load_claims(claims_path)


@pytest.fixture
def outputs(claims):
    return mp.build_all(claims)


# --- Every proposition gets the full set of prompts ------------------------------


@pytest.mark.parametrize("method, values", [("matching", mp.K_VALUES), ("ramsey", mp.C_VALUES)])
def test_every_proposition_has_every_value_in_both_orders(outputs, claims, method, values):
    df = outputs[method]
    assert set(df["prop_id"]) == set(claims["prop_id"])
    assert len(df) == len(claims) * len(values) * 2
    for pid, g in df.groupby("prop_id"):
        assert sorted(g["grid_value"].unique()) == values, pid
        assert (g.groupby("grid_value")["order"].apply(sorted) == [sorted(mp.ORDERS)] * len(values)).all()


def test_value_ranges():
    assert mp.K_VALUES[0] == 0 and mp.K_VALUES[-1] == 100 and len(mp.K_VALUES) == 21
    assert mp.C_VALUES[0] == 0 and mp.C_VALUES[-1] == mp.PRIZE and len(mp.C_VALUES) == 21


def test_prompt_ids_are_unique(outputs):
    for name, df in outputs.items():
        assert df["prompt_id"].is_unique, name


def test_metadata_carried_through(outputs, claims):
    m = outputs["matching"].drop_duplicates("prop_id").set_index("prop_id")
    c = claims.set_index("prop_id")
    for col in ("claim_id", "tier"):
        assert (m[col] == c.loc[m.index, col]).all(), col
    assert (m["neg"].astype(int) == c.loc[m.index, "neg"]).all()


# --- The prompts say the right thing ----------------------------------------------


@pytest.mark.parametrize("method", ["matching", "ramsey"])
def test_statement_appears_verbatim_and_in_the_right_option(outputs, claims, method):
    texts = claims.set_index("prop_id")["text"]
    for _, r in outputs[method].iterrows():
        text = texts[r["prop_id"]]
        line = re.search(rf"Option {r['target_label']}: (.*)", r["user"]).group(1)
        assert f'"{text}"' in line, r["prompt_id"]
        other = "B" if r["target_label"] == "A" else "A"
        assert text not in re.search(rf"Option {other}: (.*)", r["user"]).group(1)


def test_target_label_matches_order(outputs):
    for name in ("matching", "ramsey", "stage1"):
        df = outputs[name]
        assert (df.loc[df["order"] == "target_first", "target_label"] == "A").all()
        assert (df.loc[df["order"] == "target_second", "target_label"] == "B").all()


def test_both_orders_are_mirror_images(outputs):
    df = outputs["ramsey"]
    first = df[df["order"] == "target_first"].set_index(["prop_id", "grid_value"])
    second = df[df["order"] == "target_second"].set_index(["prop_id", "grid_value"])
    for key in first.index[:20]:
        a = re.search(r"Option A: (.*)\nOption B: (.*)", first.loc[key, "user"]).groups()
        b = re.search(r"Option A: (.*)\nOption B: (.*)", second.loc[key, "user"]).groups()
        assert a == b[::-1]


def test_matching_reference_bet_wording(outputs):
    df = outputs["matching"]
    by_k = df.drop_duplicates("grid_value").set_index("grid_value")["other_desc"]
    assert by_k[0] == "You receive nothing, whichever ticket is drawn."
    assert by_k[100] == "You receive £1000, whichever ticket is drawn."
    assert by_k[35] == "You receive £1000 if the ticket drawn is one of tickets 1 to 35, and nothing otherwise."
    assert df["user"].str.contains(re.escape(mp.TICKETS_SETUP)).all()


def test_ramsey_sure_amount_wording(outputs):
    by_c = outputs["ramsey"].drop_duplicates("grid_value").set_index("grid_value")["other_desc"]
    assert by_c[0] == "You receive £0 for certain." and by_c[550] == "You receive £550 for certain."
    assert not outputs["ramsey"]["user"].str.contains("ticket").any()


@pytest.mark.parametrize("prop_id, phrase", [
    ("fc_1_p", "Polymarket prediction market on this question resolves, due by 17 November 2026"),
    ("kuq_1_p", "infallible judge"),
    ("fact_1_p", "established fact"),
    ("chance_00_p", "random draw will be made fairly"),
])
def test_settlement_sentence_per_tier(outputs, prop_id, phrase):
    users = outputs["matching"].loc[outputs["matching"]["prop_id"] == prop_id, "user"]
    assert users.str.contains(phrase).all()
    assert not users.str.contains("stored separately").any()


def test_no_unfilled_placeholders_and_clear_instruction(outputs):
    for name, df in outputs.items():
        assert not df["user"].str.contains(r"[{}]").any(), name
        assert df["user"].str.endswith("Answer with a single letter: A or B.").all(), name
        assert (df["system"] == mp.SYSTEM_PROMPT).all() and (df["template_version"] == mp.TEMPLATE_VERSION).all()


def test_every_prompt_has_exactly_two_options(outputs):
    for name, df in outputs.items():
        assert df["user"].map(lambda u: u.count("Option A:") == 1 and u.count("Option B:") == 1).all(), name


def test_ramsey_target_is_always_the_statement_bet(outputs):
    df = outputs["ramsey"]
    assert df["target_desc"].str.startswith("You receive £1000 if the following statement").all()
    assert df["other_desc"].str.endswith("for certain.").all()


# --- Stage 1 ------------------------------------------------------------------


def test_stage1_contents(outputs):
    s1 = outputs["stage1"]
    steps = s1.groupby("step").size().to_dict()
    n_cand = len(mp.NEUTRAL_CANDIDATES)
    assert steps["neutral"] == n_cand * len(mp.OUTCOME_PAIRS) * 2
    assert steps["symmetry_single"] == mp.N_SINGLE_PAIRS * 2
    assert steps["symmetry_block"] == len(mp.SYMMETRY_BLOCKS) * 2
    assert steps["validation_tickets"] == len(mp.VALIDATION_K) * 2
    assert steps["validation_neutral"] == n_cand * 2
    assert steps["dominance"] == (2 + n_cand) * 2
    assert steps["transitivity"] == mp.N_TRIPLES * 3 * 2


def test_neutral_test_swaps_the_outcomes(outputs):
    s1 = outputs["stage1"]
    for _, r in s1[s1["step"] == "neutral"].iterrows():
        t, o = r["target_desc"], r["other_desc"]
        amounts_t = re.findall(r"£(\d+)", t)
        amounts_o = re.findall(r"£(\d+)", o)
        assert amounts_t == amounts_o[::-1] and amounts_t[0] != amounts_t[1]


def test_neutral_candidates_do_not_say_fair(outputs):
    s1 = outputs["stage1"]
    assert not s1.loc[s1["step"] == "neutral", "user"].str.contains("fair").any()


def test_dominance_rows_name_the_expected_choice(outputs):
    s1 = outputs["stage1"]
    dom = s1[s1["step"] == "dominance"]
    assert (dom["expected_label"] == dom["target_label"]).all()
    assert (s1.loc[s1["step"] != "dominance", "expected_label"] == "").all()


def test_transitivity_covers_all_three_pairs_per_triple(outputs):
    tr = outputs["stage1"][outputs["stage1"]["step"] == "transitivity"]
    assert tr["item_id"].nunique() == mp.N_TRIPLES
    assert (tr.groupby("item_id").size() == 6).all()


def test_symmetry_singles_are_distinct_tickets(outputs):
    s = outputs["stage1"]
    singles = s[s["step"] == "symmetry_single"]
    for _, r in singles.iterrows():
        assert r["target_desc"] != r["other_desc"]


def test_stage1_is_reproducible():
    pd.testing.assert_frame_equal(mp.stage1_prompts(1), mp.stage1_prompts(1))
    assert not mp.stage1_prompts(1).equals(mp.stage1_prompts(2))


# --- Smoke subset, input checks and writing --------------------------------------


def test_smoke_subset_one_proposition_per_tier(outputs):
    smoke = outputs["smoke"]
    picks = smoke.drop_duplicates("prop_id")
    assert set(picks["tier"]) == {"forecast", "kuq", "fact", "chance"}
    assert (picks["neg"].astype(int) == 0).all()
    assert set(smoke["method"]) == {"matching", "ramsey"}


def test_load_claims_rejects_missing_columns(tmp_path):
    p = tmp_path / "claims.csv"
    pd.DataFrame({"prop_id": ["a"], "text": ["x"]}).to_csv(p, index=False)
    with pytest.raises(SystemExit):
        mp.load_claims(p)


def test_write_all_and_manifest(outputs, tmp_path, claims):
    m = mp.write_all(outputs, tmp_path / "prompts", mp.SEED)
    for name in ("stage1", "matching", "ramsey", "smoke"):
        df = pd.read_csv(tmp_path / "prompts" / f"{name}.csv", dtype=str, keep_default_na=False)
        assert list(df.columns) == mp.COLUMNS and len(df) == m["rows"][name]
    saved = json.loads((tmp_path / "prompts" / "manifest.json").read_text())
    assert saved["propositions"] == len(claims) and saved["template_version"] == mp.TEMPLATE_VERSION


def test_main_runs_from_project_root(claims_path, capsys):
    root = claims_path.parent.parent
    assert mp.main(["--root", str(root)]) == 0
    assert (root / "prompts" / "matching.csv").exists()


@pytest.mark.parametrize("raw, expected", [
    ("2026-12-04", "4 December 2026"), ("04/12/2026", "4 December 2026"), ("", ""), ("soon", "soon"),
])
def test_readable_date(raw, expected):
    assert mp.readable_date(raw) == expected

"""Unit tests for build_claims.py. Offline: the LLM is replaced by a fake converter.

Run from the project root:  pytest
"""

import json
import random
import re
from collections import Counter
from types import SimpleNamespace

import pandas as pd
import pytest

import build_claims as bc

# ---------------------------------------------------------------------------
# Synthetic raw data, shaped like the real files
# ---------------------------------------------------------------------------


def market(i, source, close, value="0.4"):
    return {
        "id": str(1000 + i),
        "source": source,
        "question": f"Will candidate number {i} win the {source} election number {i}?",
        "resolution_criteria": "See the market page.",
        "market_info_resolution_criteria": f"Resolves Yes if candidate {i} wins.",
        "background": "",
        "market_info_close_datetime": f"{close}T23:00:00+00:00",
        "url": f"https://example.com/{source}/{i}",
        "freeze_datetime_value": value,
    }


def kuq_row(q, unknown, category, answer=None, source="turk"):
    return {"question": q, "answer": answer or ["an answer"], "unknown": unknown,
            "source": source, "category": category}


@pytest.fixture
def project(tmp_path):
    """A project folder with small ForecastBench and KUQ files in Questions/."""
    qdir = tmp_path / "Questions"
    qdir.mkdir()
    qs = []
    i = 0
    for source in bc.MARKET_SOURCES:
        for _ in range(12):
            i += 1
            qs.append(market(i, source, "2026-12-10", value=str(round(0.05 + 0.025 * (i - 1), 3))))
    qs.append(market(900, "kalshi", "2026-11-01"))      # closes before the window
    qs.append(market(901, "polymarket", "2027-06-01"))  # closes after the window
    qs.append(market(902, "acled", "2026-12-10"))       # dataset source, not a market
    qs.append(market(903, "metaculus", "2026-12-10", value="1.5"))  # invalid probability
    (qdir / "2026-11-08-llm.json").write_text(json.dumps({"questions": qs}))

    rows = []
    for c, cat in enumerate(["unsolved problem", "future prediction", "controversial"]):
        for k in range(8):
            rows.append(kuq_row(f"Is there an undiscovered species of giant {cat} beetle number {k}?", True, cat))
    for k in range(8):
        rows.append(kuq_row(f"When will humans first land on planet number {k}?", True, "future prediction"))
    rows.append(kuq_row("Is nonsense purple the loudest Tuesday?", True, "unsolved problem"))
    for k in range(16):
        rows.append(kuq_row(f"What is the capital city of country number {k}?", False, "",
                            answer=[f"City {k}"], source="trivia"))
    with open(qdir / "knowns_unknowns.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return tmp_path


def small_config(root, **kw):
    base = dict(root=root, n_forecast=12, n_kuq=9, n_fact=6, n_chance=30,
                n_partition_chance=4, n_partition_kuq=2, n_partition_fact=2, workers=2)
    base.update(kw)
    return bc.Config(**base)


def core(q):
    return re.sub(r"\[.*?\]", "", q).replace("?", "").strip()


class FakeConverter:
    """Deterministic stand-in for the LLM. Rejects questions containing 'nonsense'."""

    def __init__(self):
        self.calls = 0

    def __call__(self, t):
        self.calls += 1
        q = core(t["question"])
        if "nonsense" in q:
            return {"usable": False, "reason": "nonsensical question"}
        if t["kind"] in ("forecast", "kuq"):
            return {"usable": True, "reason": "", "proposition": f"It will turn out that {q}.",
                    "negation": f"It will not turn out that {q}."}
        if t["kind"] == "fact":
            return {"usable": True, "reason": "",
                    "true_statement": f"The answer to {q} is {t['answers']}.",
                    "true_negation": f"The answer to {q} is not {t['answers']}.",
                    "false_statement": f"The answer to {q} is Elsewhere.",
                    "false_negation": f"The answer to {q} is not Elsewhere."}
        return {"usable": True, "reason": "", "outcomes": [
            {"statement": f"For {q}, outcome {x} holds.", "negation": f"For {q}, outcome {x} does not hold."}
            for x in ("one", "two", "three")]}


@pytest.fixture
def built(project):
    cfg = small_config(project)
    return bc.build(cfg, FakeConverter()), cfg


# ---------------------------------------------------------------------------
# Loading and sampling
# ---------------------------------------------------------------------------


def test_load_markets_keeps_only_markets_in_window(project):
    df = bc.load_markets(small_config(project))
    assert set(df["source"]) == set(bc.MARKET_SOURCES)
    assert len(df) == 36
    assert df["close_date"].between("2026-11-16", "2027-01-15").all()
    assert df["human_forecast"].between(0, 1).all()


def test_load_kuq_accepts_string_booleans(project):
    path = project / "Questions" / "knowns_unknowns.jsonl"
    lines = path.read_text().replace('"unknown": true', '"unknown": "true"').replace(
        '"unknown": false', '"unknown": "false"')
    path.write_text(lines)
    df = bc.load_kuq(small_config(project))
    assert df["unknown"].dtype == bool
    assert df["unknown"].sum() == 33 and (~df["unknown"]).sum() == 16


def test_stratified_sample_is_reproducible_and_spread():
    df = pd.DataFrame({"g": ["a"] * 50 + ["b"] * 50 + ["c"] * 4, "x": range(104)})
    s1 = bc.stratified_sample(df, "g", 30, seed=1)
    s2 = bc.stratified_sample(df, "g", 30, seed=1)
    assert s1.equals(s2)
    counts = Counter(s1["g"])
    assert len(s1) == 30 and counts["c"] == 4 and counts["a"] == counts["b"] == 13


def test_interleave_alternates_groups():
    df = pd.DataFrame({"g": ["a", "a", "a", "b", "b", "c"]})
    assert list(bc.interleave_by(df, "g")["g"]) == ["a", "b", "c", "a", "b", "a"]


# ---------------------------------------------------------------------------
# Text checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("question, statement, expected", [
    ("2027 Best Actress Oscar nominations? [Yes: Olivia Wilde]",
     "Olivia Wilde will be nominated for Best Actress at the 2027 Oscars.", True),
    ("Are there aliens amongst us in everyday life?",
     "Aliens are living among humans on Earth today.", True),
    ("Will Helen Zille be the next mayor of Johannesburg?",
     "The capital of Australia is Canberra.", False),
])
def test_related(question, statement, expected):
    assert bc.related(question, statement) is expected


@pytest.mark.parametrize("statement, negation, problem", [
    ("Is it raining in London today?", "It is not raining in London today.", "question mark"),
    ("It is raining in London today.", "It is raining in London today.", "identical"),
    ("It rains.", "It does not rain.", "too short"),
    ("It will rain in {city} tomorrow.", "It will not rain in {city} tomorrow.", "placeholder"),
    ("The capital of Australia is Canberra.", "The capital of Australia is not Canberra.", "unrelated"),
])
def test_check_pair_catches_bad_rows(statement, negation, problem):
    hard, _ = bc.check_pair(statement, negation, "Will it rain in London tomorrow?")
    assert any(problem in h for h in hard), hard


def test_check_pair_flags_negation_without_negating_word():
    hard, soft = bc.check_pair("The vote will pass in December.", "The vote will be rejected in December.",
                               "Will the vote pass in December?")
    assert not hard and soft


def test_good_pair_passes():
    hard, soft = bc.check_pair("Helen Zille will be the next mayor of Johannesburg.",
                               "Helen Zille will not be the next mayor of Johannesburg.",
                               "Will Helen Zille be the next mayor of Johannesburg?")
    assert not hard and not soft


# ---------------------------------------------------------------------------
# Chance claims
# ---------------------------------------------------------------------------


def test_chance_specs_have_correct_probabilities():
    """Simulate each template and check the frequency matches its stated chance."""
    rng = random.Random(0)
    specs = bc.chance_specs()
    assert len(specs) >= 30
    for s in specs:
        assert 0 < s["chance"] < 1
        freq = sum(s["draw"](rng) for _ in range(20000)) / 20000
        assert abs(freq - s["chance"]) < 0.015, s["text"]


def test_chance_specs_cover_the_probability_range():
    p = sorted(s["chance"] for s in bc.chance_specs())
    assert p[0] < 0.1 and p[-1] > 0.9
    assert sum(0.3 <= x <= 0.7 for x in p) >= 5


def test_chance_partitions_are_exclusive_and_exhaustive():
    rng = random.Random(1)
    for spec in bc.chance_partition_specs():
        assert abs(sum(e[2] for e in spec["events"]) - 1) < 1e-9
        for _ in range(2000):
            x = spec["draw"](rng)
            assert sum(bool(e[3](x)) for e in spec["events"]) == 1, spec["name"]


# ---------------------------------------------------------------------------
# End-to-end build with the fake converter
# ---------------------------------------------------------------------------


def test_build_meets_targets(built):
    b, cfg = built
    assert not b.shortfalls
    per_tier = b.claims.drop_duplicates("claim_id")["tier"].value_counts()
    assert per_tier["forecast"] == cfg.n_forecast
    assert per_tier["kuq"] == cfg.n_kuq
    assert per_tier["fact"] == cfg.n_fact
    assert per_tier["chance"] == cfg.n_chance
    n_parts = b.claims["partition_id"].replace("", pd.NA).dropna().nunique()
    assert n_parts == cfg.n_partition_chance + cfg.n_partition_kuq + cfg.n_partition_fact


def test_every_claim_has_proposition_and_negation(built):
    b, _ = built
    assert b.claims["prop_id"].is_unique
    for cid, g in b.claims.groupby("claim_id"):
        assert sorted(g["neg"]) == [0, 1], cid
        assert g["text"].str.len().gt(0).all()
        assert g["text"].nunique() == 2


def test_claims_relate_to_their_questions(built):
    b, _ = built
    with_q = b.claims[b.claims["source_question"].astype(str).str.len() > 0]
    assert len(with_q) > 0
    assert all(bc.related(q, t) for q, t in zip(with_q["source_question"], with_q["text"]))


def test_unusable_questions_go_to_review_not_claims(built):
    b, _ = built
    assert not b.claims["source_question"].str.contains("nonsense", na=False).any()
    # The nonsense question is only rejected if it was sampled; either way it is never kept.
    rejected = b.review[b.review["status"] == "rejected"]
    assert set(rejected["item_id"]).isdisjoint(set(b.claims["claim_id"]))


def test_facts_are_half_true(built):
    b, cfg = built
    facts = b.claims[(b.claims["tier"] == "fact") & (b.claims["neg"] == 0)]
    assert (facts["truth"] == "true").sum() == cfg.n_fact // 2
    pos = b.claims[b.claims["tier"] == "fact"].pivot(index="claim_id", columns="neg", values="truth")
    assert (pos[0] != pos[1]).all()


def test_forecast_negations_mirror_human_forecast(built):
    b, _ = built
    fc = b.claims[b.claims["tier"] == "forecast"].pivot(index="claim_id", columns="neg", values="human_forecast")
    assert ((fc[0] + fc[1]) - 1).abs().max() < 1e-9


def test_partitions_have_three_singles_and_three_pairs(built):
    b, _ = built
    parts = b.claims[b.claims["partition_id"] != ""]
    for pid, g in parts.groupby("partition_id"):
        assert (g["partition_role"] == "single").sum() == 3, pid
        assert (g["partition_role"] == "pair").sum() == 3, pid
        singles = g[g["partition_role"] == "single"]
        if pid.startswith("part_chance"):
            assert abs(singles["objective_chance"].sum() - 1) < 1e-6
            assert (singles["truth"] == "true").sum() == 1
        if "fact_partition" in pid:
            assert (singles["truth"] == "true").sum() == 1


def test_build_is_reproducible(project):
    cfg = small_config(project)
    a = bc.build(cfg, FakeConverter(), bc.Cache(project / "c1.jsonl")).claims
    b = bc.build(cfg, FakeConverter(), bc.Cache(project / "c2.jsonl")).claims
    pd.testing.assert_frame_equal(a, b)


def test_cache_prevents_repeat_api_calls(project):
    cfg = small_config(project)
    first = FakeConverter()
    bc.build(cfg, first)
    second = FakeConverter()
    bc.build(cfg, second)
    assert first.calls > 0 and second.calls == 0


def test_shortfall_is_reported_when_everything_is_rejected(project):
    cfg = small_config(project)
    b = bc.build(cfg, lambda t: {"usable": False, "reason": "no"})
    assert any(s.startswith("forecast") for s in b.shortfalls)
    assert (b.review["status"] == "rejected").sum() > 0


def test_export_then_import_reproduces_the_build(project):
    cfg = small_config(project)
    tasks = bc.make_tasks(bc.load_markets(cfg), bc.load_kuq(cfg), cfg)
    flat = [t for ts in tasks.values() for t in ts]
    out = project / "to_convert.jsonl"
    bc.export_tasks(flat, out)
    exported = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(exported) == len(flat) and all("prompt" in e and "?" in e["prompt"] for e in exported)

    fake = FakeConverter()
    converted = project / "converted.jsonl"
    with open(converted, "w") as f:
        for t in flat:
            f.write(json.dumps({"item_id": t["item_id"], **fake(t)}) + "\n")
    cache = bc.Cache(project / "imported_cache.jsonl")
    assert bc.import_results(converted, cache) == len(flat)
    b = bc.build(cfg, None, cache)
    assert not b.shortfalls


def test_rebuild_backs_up_previous_claims(built):
    b, cfg = built
    bc.write_outputs(b, cfg)
    bc.write_outputs(b, cfg)
    assert (cfg.data_dir / "claims.backup.csv").exists()


def test_write_outputs(built):
    b, cfg = built
    bc.write_outputs(b, cfg)
    claims = pd.read_csv(cfg.data_dir / "claims.csv")
    assert list(claims.columns) == bc.COLUMNS and len(claims) == len(b.claims)
    assert (cfg.data_dir / "review.csv").exists()


# ---------------------------------------------------------------------------
# OpenAI wrapper (mocked client)
# ---------------------------------------------------------------------------


def test_parse_reply_handles_code_fences():
    r = bc.parse_reply('```json\n{"usable": "true", "proposition": "x"}\n```')
    assert r["usable"] is True and r["proposition"] == "x"


class FakeClient:
    def __init__(self, reply, reject_temperature=False):
        self.reply, self.reject_temperature, self.kwargs = reply, reject_temperature, []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.kwargs.append(kwargs)
        if self.reject_temperature and "temperature" in kwargs:
            raise ValueError("Unsupported value: 'temperature' does not support 0")
        msg = SimpleNamespace(content=self.reply)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def test_openai_converter_parses_json_reply():
    client = FakeClient('{"usable": true, "reason": "", "proposition": "A", "negation": "Not A"}')
    conv = bc.OpenAIConverter("test-model", client=client)
    task = {"kind": "kuq", "question": "Is there life on Mars?"}
    assert conv(task)["proposition"] == "A"
    sent = client.kwargs[0]
    assert sent["model"] == "test-model" and sent["response_format"] == {"type": "json_object"}
    assert "Is there life on Mars?" in sent["messages"][1]["content"]


def test_openai_converter_drops_temperature_if_rejected():
    client = FakeClient('{"usable": true, "proposition": "A", "negation": "Not A"}', reject_temperature=True)
    conv = bc.OpenAIConverter("test-model", client=client)
    assert conv({"kind": "kuq", "question": "Is there life on Mars?"})["usable"] is True
    assert "temperature" not in client.kwargs[-1]


@pytest.mark.parametrize("kind", list(bc.PROMPTS))
def test_every_prompt_mentions_json_keys_and_the_question(kind):
    task = {"question": "When will it rain?", "answers": "Tomorrow", "source": "kalshi",
            "criteria": "Resolves yes if it rains.", "close_date": "2026-12-01"}
    p = bc.PROMPTS[kind](task)
    assert "When will it rain?" in p and '"usable"' in p

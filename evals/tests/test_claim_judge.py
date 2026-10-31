"""The judge, with a script in place of the model: what each call is shown, how claims are marked and numbered, the
five numbers, and what happens at Groq's limits. No token is spent."""

import json
from types import SimpleNamespace

import httpx
import openai
import pytest

from evals.generation import claim_judge as J

ANSWER = ("A doom loop is the same tool called with the same arguments [1]. The runtime **stops** the run "
          "【2†L1-L4】 and lets the model answer.")
SOURCES = [{"n": 1, "file": "02_agent_runtime/runtime.py", "lines": [181, 218], "text": "Source 1:\nPASSAGE ONE"},
           {"n": 2, "file": "02_agent_runtime/README.md", "lines": [60, 70], "text": "Source 2:\nPASSAGE TWO"}]


def answer_record(qid="Q2", refused=False):
    return {"id": qid, "refused": refused, "answer": "" if refused else ANSWER,
            "sources": [] if refused else SOURCES}


@pytest.fixture(scope="module")
def gen():
    return J.golden_by_id()


class FakeJudge:
    """An OpenAI client whose replies come from a script, by the schema each call asks for -- `claims`, `marks` or
    `grounds` -> replies used in turn, the last one repeated (a dict, or an exception to raise). Records every
    prompt."""

    def __init__(self, **script):
        self.script = {name: list(replies) for name, replies in script.items()}
        self.prompts, self.sent = [], []
        self.chat = self.completions = self

    def create(self, **kwargs):
        name = kwargs["response_format"]["json_schema"]["name"]
        self.prompts.append((name, kwargs["messages"][0]["content"]))
        self.sent.append(kwargs)
        reply = self.script[name].pop(0) if len(self.script[name]) > 1 else self.script[name][0]
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(reply)))],
                               usage=SimpleNamespace(total_tokens=1000))

    def asked(self, name):
        return [p for n, p in self.prompts if n == name]


def claims_reply(*claims):
    """Each claim as (quote, cites)."""
    return {"claims": [{"claim": q[:30], "quote": q, "cites": list(cites)} for q, cites in claims]}


def marks_reply(*marks, points=(1, 2)):
    """Each mark as (verdict, on the question)."""
    return {"marks": [{"n": n, "verdict": v, "source": "", "why": "no" if v == "contradicted" else "",
                       "on_question": on} for n, (v, on) in enumerate(marks, 1)], "points_made": list(points)}


def grounds_reply(*backed):
    return {"marks": [{"n": n, "backed": b, "passage": 2 if b == "other" else 0, "why": "silent" if b == "none" else ""}
                      for n, b in enumerate(backed, 1)]}


def judge(*claims, marks, grounds, points=(1, 2)):
    return FakeJudge(claims=[claims_reply(*claims)], marks=[marks_reply(*marks, points=points)],
                     grounds=[grounds_reply(*grounds)])


def limit(message):
    return openai.RateLimitError(message, response=httpx.Response(429, request=httpx.Request("POST", "http://x")),
                                 body=None)


ONE = ("the same tool called with the same arguments", (1,))


def test_extracting_sees_neither_the_key_nor_the_passages_and_each_check_sees_only_its_own(gen):
    fake = judge(ONE, marks=[("supported", True)], grounds=["cited"])
    J.claims(gen["Q2"], answer_record(), fake, "m")
    extract, check, ground = fake.asked("claims")[0], fake.asked("marks")[0], fake.asked("grounds")[0]
    points = [p["point"] for p in gen["Q2"]["points"]]
    assert ANSWER in extract and not any(p in extract for p in points) and "PASSAGE ONE" not in extract
    assert all(p in check for p in points) and "02_agent_runtime/runtime.py, lines 181-249" in check
    assert "PASSAGE ONE" not in check
    assert "PASSAGE ONE" in ground and "PASSAGE TWO" in ground and not any(p in ground for p in points)
    assert "cites: 1" in ground


def test_the_five_numbers(gen):
    fake = judge(("the same tool called with the same arguments", (1,)), ("The runtime stops the run", (2,)),
                 ("lets the model answer", ()), ("something else entirely", (1,)),
                 marks=[("supported", True), ("contradicted", True), ("not in the key", True),
                        ("not in the key", False)],
                 grounds=["cited", "other", "other", "none"], points=(1, 2))
    found, spent = J.claims(gen["Q2"], answer_record(), fake, "m")
    assert spent == 3000
    assert J.scores(found, gen["Q2"]) == {
        "claims": 4, "claims wrong": 1,
        "faithfulness": 0.75,                      # 3 of 4 backed by some passage
        "relevance": 0.75,                         # 3 of 4 on the question
        "completeness": round(2 / len(gen["Q2"]["points"]), 3),
        "citation accuracy": round(1 / 3, 3)}      # of the 3 that cite, 1 is backed by what it cites


def test_an_empty_list_of_claims_for_an_answer_with_text_is_asked_again(gen):
    fake = FakeJudge(claims=[{"claims": []}, claims_reply(ONE)], marks=[marks_reply(("supported", True))],
                     grounds=[grounds_reply("cited")])
    found, spent = J.claims(gen["Q3"], answer_record("Q3"), fake, "m")
    assert len(found["claims"]) == 1 and spent == 4000
    assert J.EMPTY.strip() in fake.asked("claims")[1] and J.EMPTY.strip() not in fake.asked("claims")[0]


def test_an_answer_that_still_yields_no_claims_is_left_unjudged_not_scored_zero(tmp_path, gen):
    answers = tmp_path / "answers_x.jsonl"
    answers.write_text("\n".join(json.dumps(answer_record(q)) for q in ("Q3", "Q2")) + "\n", encoding="utf-8")
    fake = FakeJudge(claims=[{"claims": []}, {"claims": []}, {"claims": []}, claims_reply(ONE)],
                     marks=[marks_reply(("supported", True))], grounds=[grounds_reply("cited")])
    judged, finished = J.judge_all(answers, fake, "m", golden=gen, log=lambda *_: None)
    assert finished and "Q3" not in judged and "claims" in judged["Q2"]
    assert len(fake.asked("claims")) == 4 and len(fake.asked("marks")) == 1


def test_a_cited_number_with_no_passage_behind_it_cites_nothing(gen):
    fake = judge(("lets the model answer", (2, 9)), marks=[("supported", True)], grounds=["cited"])
    found, _ = J.claims(gen["Q2"], answer_record(), fake, "m")
    assert found["claims"][0]["cites"] == [2]


def test_claims_checked_in_batches_keep_their_own_numbers(gen, monkeypatch):
    monkeypatch.setattr(J, "REQUEST_TOKENS", 1)         # every claim its own batch, in both checks
    fake = FakeJudge(claims=[claims_reply(ONE, ("The runtime stops the run", (2,)))],
                     marks=[marks_reply(("supported", True), points=(1, 2)),
                            marks_reply(("contradicted", False), points=(3,))],
                     grounds=[grounds_reply("cited"), grounds_reply("none")])
    found, _ = J.claims(gen["Q2"], answer_record(), fake, "m")
    assert [(c["verdict"], c["on_question"], c["grounded"]) for c in found["claims"]] == [
        ("supported", True, "cited"), ("contradicted", False, "none")]
    assert len(fake.asked("marks")) == len(fake.asked("grounds")) == 2
    assert found["points made"] == [1, 2]              # read from the first batch, which saw the whole answer


def test_a_claim_left_unmarked_says_so_and_counts_against_nothing_it_was_not_judged_on(gen):
    fake = FakeJudge(claims=[claims_reply(ONE, ("lets the model answer", ()))],
                     marks=[marks_reply(("supported", True))], grounds=[grounds_reply("cited", "cited")])
    found, _ = J.claims(gen["Q2"], answer_record(), fake, "m")
    assert found["claims"][1]["verdict"] == "unmarked" and found["claims"][1]["on_question"] is None
    assert found["unmarked"] == 1


def test_a_reply_that_leaves_claims_unmarked_is_asked_again_and_the_better_one_kept(gen):
    two = (ONE, ("lets the model answer", ()))
    fake = FakeJudge(claims=[claims_reply(*two)], marks=[marks_reply(("supported", True), ("supported", True))],
                     grounds=[{"marks": []}, grounds_reply("cited", "other")])
    found, _ = J.claims(gen["Q2"], answer_record(), fake, "m", log=lambda *_: None)
    assert [c["grounded"] for c in found["claims"]] == ["cited", "other"] and len(fake.asked("grounds")) == 2
    assert J.stale(found) == []
    found["claims"][1]["grounded"] = "unmarked"
    assert J.stale(found) == ["ground"]                                 # --rejudge redoes a call that left gaps


def test_the_reply_room_leaves_space_for_the_schema_groq_counts_too(gen):
    fake = judge(ONE, marks=[("supported", True)], grounds=["cited"])
    J.claims(gen["Q2"], answer_record(), fake, "m")
    # Groq counts the prompt, the schema and the reply's room against its 8,000 a minute (live: "Requested 8157")
    for kwargs in fake.sent:
        asked = J.size(kwargs["messages"][0]["content"]) + J.size(json.dumps(kwargs["response_format"]))
        assert asked + kwargs["max_tokens"] <= J.GROQ_LIMIT - 300


def test_a_point_number_that_does_not_exist_is_dropped(gen):
    fake = judge(ONE, marks=[("supported", True)], grounds=["cited"], points=(2, 99))
    assert J.claims(gen["Q2"], answer_record(), fake, "m")[0]["points made"] == [2]


def test_an_answer_saved_without_its_passages_text_is_refused_before_any_call(gen):
    record = answer_record()
    record["sources"] = [{k: v for k, v in s.items() if k != "text"} for s in SOURCES]
    fake = judge(ONE, marks=[("supported", True)], grounds=["cited"])
    with pytest.raises(ValueError, match="collect the answers again"):
        J.claims(gen["Q2"], record, fake, "m")


def test_a_quote_not_in_the_answer_is_caught_but_markdown_and_citations_are_not_held_against_it():
    claims = [{"quote": "The runtime stops the run and lets"}, {"quote": "same arguments [1]. The runtime"},
              {"quote": "the runtime retries forever"}]
    assert J.unanchored(claims, ANSWER) == [claims[2]]


def test_the_daily_limit_stops_the_run_keeps_what_was_judged_and_the_next_run_carries_on(tmp_path, gen):
    answers = tmp_path / "answers_x.jsonl"
    answers.write_text("\n".join(json.dumps(answer_record(q)) for q in ("Q1", "Q2")) + "\n", encoding="utf-8")
    spent = limit("Rate limit reached for model on tokens per day (TPD): Limit 200000")
    stopped = FakeJudge(claims=[claims_reply(ONE), spent], marks=[marks_reply(("supported", True))],
                        grounds=[grounds_reply("cited")])
    judged, finished = J.judge_all(answers, stopped, "m", golden=gen, log=lambda *_: None)
    assert not finished and "claims" in judged["Q1"] and "Q2" not in judged
    assert "claims" in json.loads((tmp_path / "verdicts_x.json").read_text(encoding="utf-8"))["Q1"]

    resumed = judge(ONE, marks=[("supported", True)], grounds=["cited"])
    judged, finished = J.judge_all(answers, resumed, "m", golden=gen, log=lambda *_: None)
    assert finished and "claims" in judged["Q2"]
    assert len(resumed.asked("claims")) == 1           # Q1 was not judged again


def test_rejudge_reruns_only_the_calls_whose_rule_changed_and_keeps_the_claims(tmp_path, gen):
    answers = tmp_path / "answers_x.jsonl"
    answers.write_text("\n".join(json.dumps(answer_record(q)) for q in ("Q1", "Q2", "Q3")) + "\n", encoding="utf-8")
    first = judge(ONE, marks=[("supported", True)], grounds=["none"])
    judged, _ = J.judge_all(answers, first, "m", golden=gen, log=lambda *_: None)
    judged["Q1"].pop("ground version")                                 # Q1: grounded under an older rule
    judged["Q3"]["check version"] = J.CHECK_VERSION - 1                # Q3: checked under an older rule
    (tmp_path / "verdicts_x.json").write_text(json.dumps(judged), encoding="utf-8")

    again = FakeJudge(marks=[marks_reply(("contradicted", True))], grounds=[grounds_reply("cited")])
    judged, finished = J.judge_all(answers, again, "m", golden=gen, rejudge=True, log=lambda *_: None)
    assert finished and [n for n, _ in again.prompts] == ["grounds", "marks"]   # Q1 call 3, Q3 call 2; never extract
    assert judged["Q1"]["claims"][0]["grounded"] == "cited" and judged["Q1"]["claims"][0]["verdict"] == "supported"
    assert judged["Q3"]["claims"][0]["verdict"] == "contradicted" and judged["Q3"]["claims"][0]["grounded"] == "none"
    assert J.stale(judged["Q1"]) == J.stale(judged["Q3"]) == [] and judged["Q1"]["tokens"] == 4000
    assert judged["Q2"]["claims"][0]["grounded"] == "none"             # already on the current rules: untouched
    nothing = FakeJudge(grounds=[grounds_reply("cited")])
    J.judge_all(answers, nothing, "m", golden=gen, log=lambda *_: None)
    assert not nothing.prompts                                          # without --rejudge, nothing is redone


def test_the_grounding_rule_counts_paraphrases_but_not_reversals_or_additions():
    prompt = J.ground_prompt(SOURCES, [{"claim": "c", "quote": "q", "cites": [1]}])
    assert "a paraphrase, a summary" in prompt and "a conclusion the passage itself draws" in prompt
    assert "reverses it" in prompt and "check every part of the claim" in prompt
    assert "adds a detail no passage has" in prompt and "even if another passage says it too" in prompt


def test_a_per_minute_limit_is_not_taken_for_the_daily_one(gen):
    fake = FakeJudge(claims=[limit("Rate limit reached on tokens per minute (TPM)")])
    with pytest.raises(openai.RateLimitError):
        J.claims(gen["Q2"], answer_record(), fake, "m")


def test_a_reply_that_fails_the_schema_is_asked_again(gen):
    slip = openai.BadRequestError("json_validate_failed", response=httpx.Response(
        400, request=httpx.Request("POST", "http://x")), body=None)
    fake = FakeJudge(claims=[slip, claims_reply(ONE)], marks=[marks_reply(("supported", True))],
                     grounds=[grounds_reply("cited")])
    found, _ = J.claims(gen["Q2"], answer_record(), fake, "m")
    assert len(found["claims"]) == 1 and len(fake.asked("claims")) == 2


def test_a_refused_answer_is_not_judged(tmp_path, gen):
    answers = tmp_path / "answers_x.jsonl"
    answers.write_text(json.dumps(answer_record("Q25", refused=True)) + "\n", encoding="utf-8")
    fake = FakeJudge(claims=[claims_reply(ONE)])
    judged, _ = J.judge_all(answers, fake, "m", golden=gen, log=lambda *_: None)
    assert judged["Q25"] == {"refused": True} and not fake.prompts


def test_shares_over_nothing_are_none(gen):
    assert J.scores({"claims": [], "points made": []}, gen["Q2"]) == {
        "claims": 0, "claims wrong": 0, "faithfulness": None, "relevance": None, "completeness": 0.0,
        "citation accuracy": None}


def test_every_prompt_fits_one_request_with_room_for_the_reply(gen):
    # an 800-token answer, 25 claims, and five passages of 1,024 tokens (the citation pieces' most): each batch fits
    answer = "word " * 800
    claims = [{"claim": "a claim about the repo", "quote": "word word word word word", "cites": [1]}] * 25
    sources = [{"n": n, "file": "f.py", "lines": [1, 2], "text": "text " * 1000} for n in range(1, 6)]
    for q in gen.values():
        for group in J.batches(lambda g: J.check_prompt(q, answer, g), claims):
            assert J.size(J.check_prompt(q, answer, group)) <= J.REQUEST_TOKENS or len(group) == 1, q["id"]
    for group in J.batches(lambda g: J.ground_prompt(sources, g), claims):
        assert J.size(J.ground_prompt(sources, group)) < J.GROQ_LIMIT - 1500

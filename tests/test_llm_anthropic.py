"""The native Claude provider: requests in cache order with breakpoints, structured outputs, refusals, errors and the
key, all against a fake Anthropic client (no network)."""

import json
from types import SimpleNamespace

import pytest

anthropic = pytest.importorskip("anthropic")
httpx = pytest.importorskip("httpx2")

from test_markup_llm import CH, FakeLLM  # noqa: E402

from folivox import config  # noqa: E402
from folivox import markup_llm as ml  # noqa: E402
from folivox.book import Block, BlockKind, Chapter  # noqa: E402
from folivox.llm import LLMS  # noqa: E402
from folivox.llm_anthropic import AnthropicLLM  # noqa: E402

CACHE_MIN = 100  # the fake's minimum cacheable prefix, in characters


class FakeMessages:
    """Answers like the markup's fake model, and caches like the API: a prefix ending at a breakpoint is written the
    first time and read on later requests that start with exactly the same text."""

    def __init__(self, owner):
        self.owner = owner

    def create(self, **req):
        return self.owner.answer(req)


class FakeClient:
    def __init__(self, stop_reason="end_turn", error=None):
        self.requests, self.cached, self.stop_reason, self.error = [], set(), stop_reason, error
        self.messages = FakeMessages(self)
        self.beta = SimpleNamespace(messages=FakeMessages(self))
        self.model = FakeLLM()

    def answer(self, req):
        self.requests.append(req)
        if self.error is not None:
            raise self.error
        prefix, read, write = req.get("system", ""), 0, 0
        for block in req["messages"][0]["content"]:
            prefix += block["text"]
            if "cache_control" in block and len(prefix) >= CACHE_MIN:
                if prefix in self.cached:
                    read = len(prefix)
                else:
                    self.cached.add(prefix)
                    write = len(prefix) - read
        text = "".join(b["text"] for b in req["messages"][0]["content"])
        reply = self.model.complete_json(text, req.get("system"))
        total = len(req.get("system", "")) + len(text)
        usage = SimpleNamespace(input_tokens=total - read - write, output_tokens=len(reply),
                                cache_read_input_tokens=read, cache_creation_input_tokens=write)
        details = SimpleNamespace(category="cyber", explanation="") if self.stop_reason == "refusal" else None
        return SimpleNamespace(stop_reason=self.stop_reason, stop_details=details, usage=usage,
                               content=[SimpleNamespace(type="thinking", thinking=""),
                                        SimpleNamespace(type="text", text=reply)])


def book(n=4):
    """Chapters long enough for several windows, none the same as another."""
    return [Chapter(f"Глава {i}", [Block(BlockKind.PARAGRAPH, f"{b.text} ({i}.{k})")
                                    for k in range(60) for b in ch.blocks]) for i in range(n) for ch in CH[:1]]


def run(llm, chapters=None):
    chapters = chapters or book()
    runner = ml.Runner(llm, llm.model, ml.Cache(None))
    bible = ml.analyse(chapters, "ru", runner)
    markup, stats = ml.annotate(chapters, "ru", bible, runner)
    return bible, markup, stats, runner


def test_marking_requests_put_the_instructions_and_speakers_first_and_read_them_from_the_cache():
    client = FakeClient()
    llm = AnthropicLLM(model="claude-haiku-5-5", client=client)
    bible, markup, stats, _ = run(llm)
    assert stats.marked > 0 and stats.failed_windows == 0
    marking = [r for r in client.requests if "numbered parts" in r["system"]]
    assert len(marking) >= 3
    for r in marking:
        rules, speakers, window = r["messages"][0]["content"]
        assert "Every paragraph below" in rules["text"] and "Paragraphs:" not in rules["text"]
        assert speakers["text"].startswith("Speakers (id; 0 is the narrator):")
        assert "Paragraphs:" in window["text"] and "cache_control" not in window
        assert rules["cache_control"] == speakers["cache_control"] == {"type": "ephemeral"}
        assert r["thinking"] == {"type": "adaptive"} and "temperature" not in r
        assert "format" not in r["output_config"]  # the answer maps part numbers to speakers: no schema can say that
    # every speaker by id, the same list from window to window: later windows read the whole prefix from the cache
    lists = [r["messages"][0]["content"][1]["text"] for r in marking]
    assert len(set(lists[1:])) == 1
    assert llm.last_usage["cache_read"] > 0
    again = run(AnthropicLLM(model="claude-haiku-5-5", client=FakeClient()))  # the same result every time
    assert again[2].marked == stats.marked


def test_the_reply_cache_still_answers_an_unchanged_window():
    client = FakeClient()
    llm = AnthropicLLM(model="claude-haiku-5-5", client=client)
    chapters = book(2)
    cache = ml.Cache(None)
    bible = ml.analyse(chapters, "ru", ml.Runner(llm, llm.model, cache))
    n = len(client.requests)
    ml.analyse(chapters, "ru", ml.Runner(llm, llm.model, cache), ml.Bible())
    assert len(client.requests) == n and bible.characters


def test_analysis_asks_for_its_json_shape_and_caches_only_the_instructions():
    client = FakeClient()
    run(AnthropicLLM(model="claude-sonnet-5-5", effort="low", client=client), book(2))
    reading = [r for r in client.requests if r["system"].startswith("You are a careful")]
    assert reading
    for r in reading:
        rules, rest = r["messages"][0]["content"]
        assert "Read the excerpt" in rules["text"] and "cache_control" in rules
        assert "Characters already known" in rest["text"] and "Excerpt:" in rest["text"] and "cache_control" not in rest
        fmt = r["output_config"]["format"]
        assert fmt["type"] == "json_schema" and fmt["schema"] is ml.ANALYSE_SCHEMA
        assert r["output_config"]["effort"] == "low"
        assert r["betas"] == ["server-side-fallback-2026-07-01"] and r["fallbacks"] == "default"


def test_the_prompt_other_providers_get_is_the_template_unchanged():
    fields = dict(hints=ml.QUOTE_HINTS, person="third", speakers='[{"id": 1}]', recent="Context\n", text="[B1]\n  1| a")
    p = ml.parts_prompt("ru", ["en"], **fields)
    assert str(p) == ml.PARTS_PROMPT.format(lang="Russian", **fields, lang_json=ml.PARTS_LANG_JSON,
                                             lang_rule=ml.PARTS_LANG_RULE.format(lang="Russian", others="English",
                                                                                 example="en"))
    assert sorted("".join(t for t, _ in p.blocks)) == sorted(str(p))  # the same words in cache order
    retry = p + "\n\nAgain."
    assert str(retry).endswith("Again.") and retry.blocks[-1][0].endswith("Again.") and retry.blocks[0] == p.blocks[0]


def test_a_refusal_leaves_the_window_to_the_narrator_and_haiku_has_no_server_fallback():
    client = FakeClient(stop_reason="refusal")
    llm = AnthropicLLM(model="claude-haiku-5-5", client=client)
    runner = ml.Runner(llm, llm.model, ml.Cache(None))
    bible = ml.Bible(characters=[ml.Character("Анна", gender="f", lines=3, id=1)])
    _, stats = ml.annotate(CH, "ru", bible, runner)
    assert stats.failed_windows >= 1 and stats.marked == 0
    assert all("fallbacks" not in r for r in client.requests)


def status_error(cls, code):
    return cls("no", response=httpx.Response(code, request=httpx.Request("POST", "https://api.anthropic.com")),
               body=None)


@pytest.mark.parametrize("error, kind", [
    (lambda: status_error(anthropic.AuthenticationError, 401), "key"),
    (lambda: status_error(anthropic.NotFoundError, 404), "model"),
    (lambda: status_error(anthropic.BadRequestError, 400), "bad"),
    (lambda: status_error(anthropic.RateLimitError, 429), "busy"),
    (lambda: status_error(anthropic.InternalServerError, 529), "busy"),
])
def test_errors_follow_the_markups_rules(error, kind):
    client = FakeClient(error=error())
    llm = AnthropicLLM(model="claude-haiku-5-5", client=client)
    runner = ml.Runner(llm, llm.model, ml.Cache(None), sleep=lambda s: None)
    if kind in ("key", "model"):  # stop at once and say what to fix
        with pytest.raises(ml.LLMError, match="API key" if kind == "key" else "model"):
            runner.ask("s", "p", lambda d: d)
        assert len(client.requests) == 1
    elif kind == "bad":  # an unusable answer: tried again once, then the narrator reads the window
        with pytest.raises(ValueError):
            runner.ask("s", "p", lambda d: d)
        assert len(client.requests) == 2
    else:  # busy: waited out, then reported
        with pytest.raises(ml.LLMError, match="did not answer"):
            runner.ask("s", "p", lambda d: d)
        assert len(client.requests) == ml.Runner.BUSY_TRIES


def test_the_key_comes_from_the_environment_or_its_file_and_the_provider_is_registered(tmp_path, monkeypatch):
    monkeypatch.setenv("FOLIVOX_CONFIG", str(tmp_path / "config.toml"))
    (tmp_path / "config.toml").write_text('[llm.anthropic]\nmodel = "claude-sonnet-5-5"\neffort = "medium"\n')
    config.load.cache_clear()
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    secret = tmp_path / "key"
    secret.write_text("sk-ant-test-from-file\n")
    monkeypatch.setenv("ANTHROPIC_API_KEY_FILE", str(secret))
    try:
        assert "anthropic" in LLMS.names() and not LLMS.get("anthropic").local
        llm, model = ml.make_llm("anthropic")
        assert isinstance(llm, AnthropicLLM) and model == "claude-sonnet-5-5" and llm.effort == "medium"
        assert llm.client.api_key == "sk-ant-test-from-file"
        with pytest.raises(ValueError, match="effort"):
            AnthropicLLM(effort="very", client=FakeClient())
    finally:
        config.load.cache_clear()


def test_usage_is_reported_for_the_spend_counters():
    llm = AnthropicLLM(model="claude-haiku-5-5", client=FakeClient())
    reply = llm.complete_json(ml.parts_prompt("ru", hints="", person="third", speakers='[{"id": 1, "name": "Анна"}]',
                                              recent="",
                                              text="[B0]\n  1| — Да, — сказала Анна."), ml.PARTS_SYSTEM)
    assert json.loads(reply)["blocks"] == [{"b": 0, "who": {"1": 1}}]
    assert set(llm.last_usage) == {"input", "output", "cache_read", "cache_write"}



@pytest.mark.parametrize("stop", ["max_tokens", "refusal"])
def test_a_rejected_reply_still_reports_what_it_cost(stop):
    """A reply cut off at the token limit or refused is billed: its usage is there for the spend counter even though
    the reply raises."""
    llm = AnthropicLLM(model="claude-haiku-5-5", client=FakeClient(stop_reason=stop))
    with pytest.raises(ValueError):
        llm.complete_json("[B0] — Да, — сказала Анна.", ml.ANALYSE_SYSTEM)
    assert llm.last_usage["output"] > 0 and llm.last_usage["input"] > 0


def test_effort_defaults_to_high():
    llm = AnthropicLLM(client=FakeClient())
    assert llm.effort == "high"
    assert llm._request("p", None, None)["output_config"]["effort"] == "high"


def test_the_default_effort_is_not_sent_to_models_without_adaptive_thinking():
    llm = AnthropicLLM(model="claude-haiku-4-5", client=FakeClient())
    assert llm.effort is None
    assert "effort" not in (llm._request("p", None, None).get("output_config") or {})
    assert AnthropicLLM(model="claude-haiku-4-5", effort="low", client=FakeClient()).effort == "low"

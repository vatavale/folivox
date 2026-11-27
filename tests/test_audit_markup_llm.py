"""Audit regressions: model replies and files that used to crash the automatic markup."""

import http.client
import json

import pytest

from folivox import llm as llm_mod
from folivox import markup_llm as ml


class _Reply:
    def __init__(self, replies):
        self.replies = list(replies)

    def complete_json(self, prompt, system=None, temperature=0.2):
        r = self.replies.pop(0)
        if isinstance(r, BaseException):
            raise r
        return r


def test_runner_retries_on_attribute_error_and_reports_broken_reply_as_llm_error():
    class Weird:
        calls = 0

        def complete_json(self, *a, **k):
            Weird.calls += 1
            if Weird.calls == 1:
                raise AttributeError("'NoneType' object has no attribute 'strip'")
            return '{"characters": []}'

    r = ml.Runner(Weird(), "m", ml.Cache(None))
    assert r.ask("s", "p", lambda d: d) == {"characters": []}
    r2 = ml.Runner(_Reply([ConnectionError("cut")] * ml.Runner.BUSY_TRIES), "m", ml.Cache(None),
                   sleep=lambda s: None)
    with pytest.raises(ml.LLMError):
        r2.ask("s", "p", lambda d: d)


def test_post_turns_incomplete_read_into_oserror(monkeypatch):
    def boom(*a, **k):
        raise http.client.IncompleteRead(b"")

    monkeypatch.setattr(llm_mod.urllib.request, "urlopen", boom)
    with pytest.raises(OSError):
        llm_mod._post("http://127.0.0.1:1/x", {})


def test_openai_null_content_is_empty(monkeypatch):
    monkeypatch.setattr(llm_mod, "_post", lambda *a, **k: {"choices": [{"message": {"content": None}}]})
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    o = llm_mod.OpenAICompatible(model="m", base_url="http://x")
    assert o.complete("hi") == ""


class _Block:
    def __init__(self, text):
        self.text = text


class _Chapter:
    def __init__(self, texts):
        self.title, self.blocks = "Ch", [_Block(t) for t in texts]


def test_annotate_survives_non_list_new_characters():
    chapters = [_Chapter(["— Привет, — сказала Анна.", "Он молчал."])]
    reply = json.dumps({"blocks": [{"b": 0, "who": {"1": 1}}], "new_characters": 5, "updates": "x"})
    bible = ml.Bible(characters=[ml.Character(name="Анна", gender="f", lines=3)])
    markup, stats = ml.annotate(chapters, "ru", bible, ml.Runner(_Reply([reply]), "m", ml.Cache(None)),
                                other_langs=[])
    assert stats.failed_windows == 0 and stats.marked == 1


def test_second_opinion_skips_non_dict_blocks():
    from folivox import markup as mk

    chapters = [_Chapter(["— Привет, — сказала Анна."])]
    bible = ml.Bible(characters=[ml.Character(name="Анна", gender="f", lines=3, id=1)])
    markup = mk.Markup()
    cm = markup.chapter(0, "Ch", "")
    cm.review = [{"b": 0, "code": "unparsed"}]
    runner = ml.Runner(_Reply([json.dumps({"blocks": ["oops"]})]), "m", ml.Cache(None))
    assert ml.second_opinion(chapters, "ru", bible, runner, markup) == (0, 1)


def _book(n):
    return [_Chapter([f"— Реплика номер {i}, — сказала Анна. " + "слово " * 40 for i in range(n)])]


def test_analyse_restarts_when_sampled_mode_changes():
    chapters = _book(40)
    ok = '{"characters": []}'
    bible = ml.Bible(done_windows=3, total_windows=10, sampled=False)  # an interrupted full reading
    runner = ml.Runner(_Reply([ok] * 2), "m", ml.Cache(None))
    bible = ml.analyse(chapters, "ru", runner, bible, max_windows=2, window_chars=300)
    assert bible.sampled and bible.done_windows == 2 and runner.calls == 2
    n = len(ml.windows(chapters, 300))
    runner = ml.Runner(_Reply([ok] * n), "m", ml.Cache(None))
    bible = ml.analyse(chapters, "ru", runner, bible, window_chars=300)
    assert not bible.sampled and bible.done_windows == n and runner.calls == n


def test_analyse_saves_after_a_skipped_window():
    saved = []
    runner = ml.Runner(_Reply(["junk", "junk"]), "m", ml.Cache(None))
    ml.analyse(_book(1), "ru", runner, ml.Bible(), save=lambda b: saved.append(b.done_windows))
    assert saved and saved[0] == 1


def test_preview_after_completed_full_reading_keeps_it():
    chapters = _book(40)
    n = len(ml.windows(chapters, 300))
    bible = ml.Bible(done_windows=n, total_windows=n, sampled=False)
    runner = ml.Runner(_Reply([]), "m", ml.Cache(None))
    bible = ml.analyse(chapters, "ru", runner, bible, max_windows=2, window_chars=300)
    assert not bible.sampled and bible.done_windows == n == bible.total_windows and runner.calls == 0

"""The PDNC markup evaluation tool (tools/markup_eval): excerpts, gold, scoring, the usage-logging proxy and the
cost columns, on a tiny made-up corpus (no download, no model)."""

import csv
import importlib.util
import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "markup_eval" / "markup_eval.py"
spec = importlib.util.spec_from_file_location("markup_eval", TOOL)
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

NOVEL = ("Title page\n\nCHAPTER\n1\n\nIt was a bright day, and the whole village was out in the sun.\n\n"
         "\"Good morning,\" said Anne. \"Is it\nnot fine?\"\n\n"
         "\"It is,\" replied Ben.\n\nI looked at them both.\n\n\"Shall we go?\" I said.\n\n\"Yes,\" she said.\n")
QUOTES = [  # (speaker, type, sub-quotations)
    ("Anne", "Explicit", ["Good morning,", "Is it not fine?"]),
    ("Ben", "Explicit", ["It is,"]),
    ("Me", "Implicit", ["Shall we go?"]),
    ("Anne", "Anaphoric", ["Yes,"]),
]


def corpus(tmp_path):
    d = tmp_path / "pdnc" / "Tale"
    d.mkdir(parents=True)
    (d / "novel_text.txt").write_text(NOVEL, encoding="utf-8")
    with open(d / "quotation_info.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["quoteID", "quoteText", "subQuotationList", "quoteByteSpans", "speaker", "quoteType"])
        for i, (who, kind, subs) in enumerate(QUOTES):
            at = NOVEL.index(subs[0].split()[0])
            w.writerow([f"Q{i}", " ".join(subs), repr(subs), repr([[at, at + 5]]), who, kind])
    with open(d / "character_info.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Character ID", "Main Name", "Aliases", "Gender", "Category"])
        w.writerows([[0, "Anne", "{'Anne', 'Miss Shirley'}", "F", "major"], [1, "Ben", "{'Ben'}", "M", "minor"],
                     [2, "Me", "{'Me'}", "M", "major"]])
    start = NOVEL.index("CHAPTER")
    return {"pdnc": {"licence": "CC BY 4.0"}, "excerpts": {"Tale": {"start": start, "end": len(NOVEL)}},
            "first_person_narrator": {"Tale": "Me"},
            "prices": {"some-model": {"in": 1.0, "out": 2.0, "cache_read": 0.1}}}


def test_an_excerpt_is_unwrapped_and_keeps_its_heading():
    body = ev.cut_excerpt("CHAPTER\n1\n\nA long\nline.\n\nI don t know.\n", 0, 40, restore_apostrophes=True)
    assert body == "CHAPTER 1\n\nA long line.\n\nI don't know.\n"


def mark(blocks, speakers):
    """A markup file giving each listed quotation's text to the speaker id named (others stay with the narrator)."""
    spans = {}
    for ci, ch in enumerate(blocks):
        for bi, t in enumerate(ch):
            for text, sid in speakers:
                at = t.find(text)
                if at >= 0:
                    spans.setdefault(str(ci), []).append({"b": bi, "s": at, "e": at + len(text), "sp": sid})
    return {"speakers": [{"id": 0, "name": "Narrator"}, {"id": 1, "name": "Miss Shirley"}, {"id": 2, "name": "Ben"}],
            "chapters": {c: {"spans": s} for c, s in spans.items()}}


def test_prepare_then_score_by_quote_type_and_character_list(tmp_path):
    cfg = corpus(tmp_path)
    work = tmp_path / "work"
    gold = ev.prepare(work, cfg, tmp_path / "pdnc")
    assert [q["speaker"] for q in gold["Tale"]["quotes"]] == ["Anne", "Ben", "Me", "Anne"]
    text = (work / "books" / "Tale.txt").read_text(encoding="utf-8")
    assert "CHAPTER 1" in text and "Is it not fine?" in text  # unwrapped
    blocks = json.loads((work / "books" / "blocks.json").read_text(encoding="utf-8"))["Tale"]
    # Anne's two lines right (under an alias), Ben's given to Anne, the narrator's own line left to the narrator
    markup = mark(blocks, [("Good morning,", 1), ("Is it not fine?", 1), ("It is,", 1), ("Yes,", 1)])
    bible = {"characters": [{"name": "Miss Shirley", "aliases": ["Anne"], "id": 1, "lines": 3},
                            {"name": "Ben", "aliases": [], "id": 2, "lines": 1},
                            {"name": "Mr. Nobody", "aliases": [], "id": 3, "lines": 1}]}
    run = work / "runs" / "t"
    run.mkdir(parents=True)
    (run / "Tale.markup.json").write_text(json.dumps(markup), encoding="utf-8")
    (run / "Tale.characters.json").write_text(json.dumps(bible), encoding="utf-8")
    t = ev.score(work, cfg, "t")["total"]
    (run / "calls.jsonl").write_text(json.dumps({"model": "some-model", "status": 200, "usage": {
        "input": 1000, "output": 500, "cache_read": 0, "cache_write": 0}}) + "\n", encoding="utf-8")
    (run / "Tale.time.json").write_text(json.dumps({"seconds": 10, "rc": 0}), encoding="utf-8")
    row = ev.report(work, cfg)[0]
    size = len(text)
    assert row["usd_per_500k"] == pytest.approx((1000 * 1.0 + 500 * 2.0) / 1e6 * 500_000 / size)
    assert row["secs_per_10k"] == pytest.approx(10 / size * 1e4) and row["calls"] == 1
    assert (t["quotes"], t["aligned"], t["unattributed"]) == (4, 4, 1)
    assert t["accuracy"] == pytest.approx(2 / 4) and t["lenient"] == pytest.approx(3 / 4)
    assert (t["explicit"], t["anaphoric"], t["implicit"]) == (pytest.approx(1 / 2), 1.0, 0.0)
    assert t["char_precision"] == pytest.approx(2 / 3) and t["char_recall"] == pytest.approx(2 / 3)


def test_names_map_by_alias_and_by_a_unique_name():
    m = ev.Mapper([{"name": "Elizabeth Bennet", "aliases": ["Lizzy"]}, {"name": "Jane Bennet", "aliases": []},
                   {"name": "Mr. Darcy", "aliases": ["Fitzwilliam Darcy"]}])
    assert m.map("Lizzy") == ("Elizabeth Bennet", "exact")
    assert m.map("Miss Eliza", ["Lizzy"]) == ("Elizabeth Bennet", "alias")
    assert m.map("Darcy") == ("Mr. Darcy", "first-token")  # "Mr." is not a name
    assert m.map("Bennet")[0] is None  # two Bennets: no guess
    assert m.map("Narrator")[0] is None


class Upstream(BaseHTTPRequestHandler):
    """Answers like an API: OpenAI-shaped or Anthropic-shaped usage, and records the headers it got."""

    seen: list = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Upstream.seen.append(dict(self.headers))
        if self.path.endswith("/messages"):
            reply = {"usage": {"input_tokens": 100, "output_tokens": 50, "cache_read_input_tokens": 900,
                               "cache_creation_input_tokens": 0}, "model": body["model"]}
        else:
            reply = {"usage": {"prompt_tokens": 1000, "completion_tokens": 50,
                               "prompt_tokens_details": {"cached_tokens": 400}}, "model": body["model"]}
        data = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def test_the_proxy_logs_usage_never_keys_and_the_report_prices_it(tmp_path):
    up = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    threading.Thread(target=up.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{up.server_address[1]}"
    log = tmp_path / "calls.jsonl"
    proxy = ev.Proxy({"openai": base + "/v1", "anthropic": base}, log)
    try:
        for url, path in ((proxy.url("openai", "Tale"), "/chat/completions"), (proxy.url("anthropic", "Tale"),
                                                                              "/v1/messages")):
            headers = {"Authorization": "Bearer sk-secret", "Content-Type": "application/json"}
            req = urllib.request.Request(url + path, data=json.dumps({"model": "some-model"}).encode(), headers=headers)
            with urllib.request.urlopen(req, timeout=10) as r:
                assert json.loads(r.read())["model"] == "some-model"
    finally:
        proxy.close()
        up.shutdown()
    assert all(h.get("Authorization") == "Bearer sk-secret" for h in Upstream.seen)  # passed on untouched
    text = log.read_text(encoding="utf-8")
    assert "sk-secret" not in text
    rows = [json.loads(x) for x in text.splitlines()]
    assert rows[0]["usage"] == {"input": 600, "output": 50, "cache_read": 400, "cache_write": 0, "reasoning": 0}
    assert rows[1]["usage"]["cache_read"] == 900 and rows[1]["book"] == "Tale"
    s = ev.spend(tmp_path, {"some-model": {"in": 1.0, "out": 2.0, "cache_read": 0.1}})
    assert s["calls"] == 2 and s["usd"] == pytest.approx((700 * 1.0 + 100 * 2.0 + 1300 * 0.1) / 1e6)


def test_a_run_points_folivox_at_the_proxy_and_keeps_the_users_settings():
    class P:
        def url(self, kind, book):
            return f"http://127.0.0.1:1/{kind}/{book}"

    toml = ev.run_config({"llm": {"openai": {"model": "m", "api_key_env": "MY_KEY"}}, "providers": {"llm": "x"}}, P(),
                         "Tale")
    assert 'base_url = "http://127.0.0.1:1/openai/Tale"' in toml and 'api_key_env = "MY_KEY"' in toml
    cmd = ev.folivox_command(Path("b.txt"), "anthropic", "claude-haiku-5-5", True, ["--fresh"])
    assert cmd[3:] == ["markup", "auto", "b.txt", "--llm", "anthropic", "--model", "claude-haiku-5-5", "--best",
                       "--fresh"]

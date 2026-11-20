"""The OpenAI-compatible provider against a local stand-in for the chat completions endpoint."""

import json
import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from folivox import markup_llm as ml
from folivox.llm import OpenAICompatible

REFUSED = {"error": {"message": "Unsupported value: 'temperature' does not support 0.2 with this model. Only the "
                                "default (1) value is supported.", "type": "invalid_request_error",
                     "param": "temperature", "code": "unsupported_value"}}


@pytest.fixture
def server():
    """Model "chat" takes any temperature; "reasoner" only its default, unless its reasoning effort is "none"."""
    seen: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(body)
            refuse = (body["model"] == "reasoner" and "temperature" in body
                      and body.get("reasoning_effort") != "none")
            if body["model"] == "broken":
                refuse = True
                out, code = {"error": {"message": "context too long", "param": "messages"}}, 400
            else:
                out, code = (REFUSED, 400) if refuse else ({"choices": [{"message": {"content": '{"ok": true}'}}]}, 200)
            data = json.dumps(out).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/v1", seen
    httpd.shutdown()


@pytest.fixture(autouse=True)
def key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")


def test_the_temperature_folivox_asks_for_is_sent(server):
    url, seen = server
    llm = OpenAICompatible(model="chat", base_url=url)
    llm.complete_json("p", "s")
    llm.complete("p", "s")
    assert [b["temperature"] for b in seen] == [0.2, 0.3]
    assert "reasoning_effort" not in seen[0]
    OpenAICompatible(model="chat", base_url=url, temperature=0, reasoning_effort="low").complete_json("p")
    assert seen[-1]["temperature"] == 0 and seen[-1]["reasoning_effort"] == "low"  # [llm.openai] settings win


def test_a_model_that_takes_only_its_default_temperature_is_asked_without_one(server):
    url, seen = server
    llm = OpenAICompatible(model="reasoner", base_url=url)
    assert llm.complete_json("p") == '{"ok": true}'
    assert llm.complete_json("p") == '{"ok": true}'
    # refused once, then every call without a temperature
    assert ["temperature" in b for b in seen] == [True, False, False]


def test_a_reasoning_model_at_effort_none_keeps_its_temperature(server):
    url, seen = server
    OpenAICompatible(model="reasoner", base_url=url, reasoning_effort="none", temperature=0).complete_json("p")
    assert len(seen) == 1 and seen[0]["temperature"] == 0 and seen[0]["reasoning_effort"] == "none"


def test_another_refusal_still_reaches_the_runner_with_its_message(server):
    url, seen = server
    runner = ml.Runner(OpenAICompatible(model="broken", base_url=url), "broken", ml.Cache(None), sleep=lambda s: None)
    with pytest.raises(ValueError, match="HTTP 400"):
        runner.ask("s", "p", lambda d: d)
    assert all("temperature" in b for b in seen)  # not mistaken for a temperature refusal


def test_another_refusal_keeps_its_body(server, monkeypatch):
    url, seen = server
    with pytest.raises(urllib.error.HTTPError) as caught:
        OpenAICompatible(model="broken", base_url=url).complete("p")
    assert json.loads(caught.value.read())["error"]["message"] == "context too long"

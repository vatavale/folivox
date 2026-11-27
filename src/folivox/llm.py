"""Text LLM providers (see :data:`LLMS`): picking voice-sample fragments, later text analysis.

Local: Ollama. Optional: any OpenAI-compatible chat endpoint (OpenAI, OpenRouter, a local
llama.cpp / vLLM server…) with the user's own key.
"""

from __future__ import annotations

import http.client
import io
import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable

from folivox.providers import Registry


class LLM:
    name = ""

    def complete(self, prompt: str, system: str | None = None, temperature: float = 0.3) -> str:
        raise NotImplementedError

    def complete_json(self, prompt: str, system: str | None = None, temperature: float = 0.2) -> str:
        """Like :meth:`complete`, asking the model for a JSON document where the backend can enforce it."""
        return self.complete(prompt, system, temperature)


# Gemma 4 (multilingual) in the sizes Ollama offers, by the memory a computer has for a model that runs alone
# (book analysis runs before the audio is made, never together with the TTS model). QAT builds, sizes from
# ollama.com/library/gemma4/tags. (minimum total RAM in GB, tag, download GB, note)
GEMMA_TIERS = (
    (48, "gemma4:31b-it-qat", 19, "best understanding of long, complex books"),
    (32, "gemma4:26b-a4b-it-qat", 16, "nearly as good, much faster (4B active parameters)"),
    (0, "gemma4:12b-it-qat", 7.2, "fast and accurate in our tests (slow below 16 GB of memory)"),
)


class PullError(RuntimeError):
    """A model download failed (Ollama refused it, or could not be reached)."""


def recommend_model(total_ram_gb: float | None) -> str:
    """The largest Gemma 4 that leaves room on a computer with ``total_ram_gb`` (the default when RAM is unknown)."""
    if total_ram_gb is None:
        return "gemma4:12b-it-qat"
    return next(tag for need, tag, _gb, _note in GEMMA_TIERS if total_ram_gb >= need)


def _post(url: str, payload: dict, headers: dict | None = None, timeout: float = 600) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except http.client.HTTPException as e:  # a cut-off reply (IncompleteRead) is not an OSError
        raise ConnectionError(f"the server broke off the reply: {e!r}") from None


class Ollama(LLM):
    name = "ollama"

    def __init__(self, model: str = "gemma4:31b-it-qat", host: str | None = None, num_ctx: int = 16384):
        self.model = model
        self.num_ctx = num_ctx  # Ollama's own default (4096) silently cuts a long prompt
        self.host = (host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        if not self.host.startswith("http"):
            self.host = "http://" + self.host

    def complete(self, prompt, system=None, temperature=0.3):
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        r = _post(f"{self.host}/api/chat", {"model": self.model, "messages": msgs, "stream": False, "think": False,
                                            "options": {"temperature": temperature, "num_ctx": self.num_ctx}})
        return r["message"]["content"].strip()

    def complete_json(self, prompt, system=None, temperature=0.2):
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        r = _post(f"{self.host}/api/chat", {"model": self.model, "messages": msgs, "stream": False, "think": False,
                                            "format": "json",
                                            "options": {"temperature": temperature, "num_ctx": self.num_ctx}})
        return r["message"]["content"].strip()

    def running(self) -> bool:
        """Is an Ollama server answering at the host?"""
        try:
            with urllib.request.urlopen(f"{self.host}/api/version", timeout=3):
                return True
        except (OSError, ValueError):
            return False

    def pull(self, model: str, progress: Callable[[int, int, str], None] | None = None,
             cancelled: Callable[[], bool] = lambda: False) -> None:
        """Download a model through Ollama (it resumes a half-finished download). ``progress(done, total, status)``
        gets bytes summed over the layers; raises ``PullError`` when Ollama refuses or cannot be reached."""
        body = json.dumps({"model": model, "stream": True}).encode()
        req = urllib.request.Request(f"{self.host}/api/pull", data=body, method="POST",
                                     headers={"Content-Type": "application/json"})
        layers: dict[str, tuple[int, int]] = {}
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                for raw in r:
                    if cancelled():
                        return
                    try:
                        d = json.loads(raw)
                    except ValueError:
                        continue
                    if d.get("error"):
                        raise PullError(str(d["error"]))
                    if d.get("digest") and d.get("total"):
                        layers[d["digest"]] = (int(d.get("completed", 0)), int(d["total"]))
                    if progress:
                        progress(sum(c for c, _ in layers.values()), sum(t for _, t in layers.values()),
                                 str(d.get("status", "")))
        except OSError as e:
            raise PullError(f"Ollama did not answer: {e}") from None

    def delete(self, model: str) -> None:
        """Remove a model from this Ollama."""
        req = urllib.request.Request(f"{self.host}/api/delete", data=json.dumps({"model": model}).encode(),
                                     method="DELETE", headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30):
                pass
        except OSError as e:
            raise PullError(f"Ollama could not remove {model}: {e}") from None

    def unload(self) -> None:
        """Ask Ollama to drop this model from memory now instead of after its keep-alive period. Best effort."""
        try:
            _post(f"{self.host}/api/generate", {"model": self.model, "keep_alive": 0}, timeout=15)
        except (OSError, ValueError, KeyError):
            pass

    def installed(self) -> list[str]:
        """Models already pulled into this Ollama (empty when it is not running)."""
        try:
            req = urllib.request.Request(f"{self.host}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as r:
                return [m["name"] for m in json.loads(r.read()).get("models", [])]
        except (OSError, ValueError):
            return []


def _rejects_temperature(code: int, body: bytes) -> bool:
    """A 400 saying the model takes no temperature but its default (OpenAI's reasoning models)."""
    if code != 400:
        return False
    try:
        err = json.loads(body).get("error") or {}
    except (ValueError, AttributeError):
        return False
    return err.get("param") == "temperature" or "'temperature'" in str(err.get("message", ""))


class OpenAICompatible(LLM):
    """Config (``[llm.openai]``): ``model``, ``base_url``, ``temperature`` (unset: the one each call asks for, 0.2 for
    JSON and 0.3 otherwise; a number: that one for every call), ``reasoning_effort`` (sent as it is, for a reasoning
    model: OpenAI's take ``none`` … ``xhigh``; unset: the model's default). Most reasoning models accept only
    their default temperature unless the reasoning effort is ``none``: when the server says so, the calls go on
    without one."""

    name = "openai"

    def __init__(self, model: str = "gpt-5-mini", base_url: str = "https://api.openai.com/v1",
                 api_key_env: str = "OPENAI_API_KEY", temperature: float | None = None,
                 reasoning_effort: str | None = None):
        from folivox.config import api_key

        self.model, self.base_url = model, base_url.rstrip("/")
        self.key = api_key(api_key_env, "LLM")
        self.temperature = None if temperature is None else float(temperature)
        self.reasoning_effort = str(reasoning_effort) if reasoning_effort else None
        self.sends_temperature = True  # False once the server has refused one (the model takes only its default)

    def request(self, msgs: list[dict], temperature: float | None) -> dict:
        body: dict = {"model": self.model, "messages": msgs}
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        t = self.temperature if self.temperature is not None else temperature
        if t is not None and self.sends_temperature:
            body["temperature"] = t
        return body

    def complete(self, prompt, system=None, temperature=0.3):
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        body = self.request(msgs, temperature)
        url, headers = f"{self.base_url}/chat/completions", {"Authorization": f"Bearer {self.key}"}
        try:
            r = _post(url, body, headers)
        except urllib.error.HTTPError as e:
            if "temperature" not in body:
                raise
            answer = e.read()
            if not _rejects_temperature(e.code, answer):  # the same error, its body readable again
                raise urllib.error.HTTPError(e.url, e.code, e.msg, e.hdrs, io.BytesIO(answer)) from None
            self.sends_temperature = False
            r = _post(url, self.request(msgs, temperature), headers)
        return (r["choices"][0]["message"]["content"] or "").strip()  # null content: a refusal or a filtered reply


LLMS: Registry = Registry("llm", default="ollama")
LLMS.register("ollama", Ollama, "Local models via Ollama (default model gemma4:31b-it-qat)", needs="ollama")
LLMS.register("openai", OpenAICompatible, "OpenAI-compatible chat API: OpenAI, OpenRouter, llama.cpp… (your key)",
              local=False, needs="OPENAI_API_KEY")

"""Cloud voice generators: small HTTP plugins registered in :data:`folivox.voicegen.GENERATORS`.

Every provider declares what it can do (``modes``: ``"preset"`` = pick one of its voices, ``"design"`` =
create a voice from a text description), the languages it speaks, price/terms notes and the name of the
environment variable that holds the API key. The key is read with :func:`folivox.config.api_key`
(environment first, then the OS keychain) and is never logged, stored in a file or put in an error message.

Everything sent here leaves the computer and is billed to the user's account. The audio is for personal
use as a cloning prompt; it must never be bundled with Folivox.
"""

from __future__ import annotations

import base64
import json
import random
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from folivox.providers import ProviderError
from folivox.voicegen import Job, VoiceGenerator

COMMERCIAL_NOTICE = ("The text is sent to the provider and billed to your account. Some providers' terms restrict "
                     "using their output to build competing models; the audio is kept on this computer and is "
                     "never bundled with Folivox.")


def backoff(attempt: int, base: float = 1.0, cap: float = 30.0) -> float:
    """Exponential backoff with full jitter on the upper half: 1-2 s, 2-4 s, 4-8 s… up to ``cap``."""
    delay = min(cap, base * 2 ** attempt)
    return delay / 2 + random.uniform(0, delay / 2)


class CloudError(ProviderError):
    """A cloud call failed. ``kind``: auth | rate_limit | quota | bad_request | server | network."""

    def __init__(self, message: str, kind: str = "server", retry_after: float | None = None):
        super().__init__(message)
        self.kind, self.retry_after = kind, retry_after


@dataclass
class Response:
    status: int
    body: bytes
    headers: dict[str, str]

    def json(self) -> dict:
        try:
            data = json.loads(self.body or b"{}")
        except ValueError:
            raise CloudError("The provider returned a reply that is not JSON", "server") from None
        return data if isinstance(data, dict) else {"data": data}


Transport = Callable[[str, str, dict[str, str], bytes | None, float], Response]


def urllib_transport(method: str, url: str, headers: dict[str, str], body: bytes | None, timeout: float) -> Response:
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 — https URLs of the providers
            return Response(r.status, r.read(), {k.lower(): v for k, v in r.headers.items()})
    except urllib.error.HTTPError as e:
        return Response(e.code, e.read() or b"", {k.lower(): v for k, v in (e.headers or {}).items()})
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise CloudError(f"Cannot reach the provider: {getattr(e, 'reason', e)}", "network") from None


def error_text(resp: Response) -> str:
    """The provider's own message from an error body, trimmed; never contains request headers."""
    try:
        data = json.loads(resp.body)
    except ValueError:
        return resp.body[:200].decode("utf-8", "replace").strip()

    def dig(x):
        if isinstance(x, str):
            return x
        if isinstance(x, dict):
            for k in ("message", "detail", "error", "status"):
                if k in x and (m := dig(x[k])):
                    return m
        if isinstance(x, list) and x:
            return dig(x[0])
        return ""

    return dig(data)[:300]


class CloudGenerator(VoiceGenerator):
    """Base class: request plumbing, retries, error mapping, capability declaration."""

    name = ""
    label = ""
    local = False
    commercial = True
    mode = "preset"  # the active mode; callers switch it to "design" where the provider offers it
    modes: tuple[str, ...] = ("preset",)
    languages: tuple[str, ...] = ("en",)  # ISO 639-1 codes the provider can speak
    key_env = ""
    price_note = ""
    blurb = ""  # plain-language summary for the UI: what it is good for, in a sentence or two
    mode_labels: dict[str, str] = {"preset": "Pick a voice", "design": "Describe a voice"}
    terms_url = ""
    docs_url = ""
    max_chars = 2000  # per request
    # fragment size a render uses by default (never above max_chars); raise it where requests are the bottleneck
    fragment_chars = 300
    retries = 4
    server_pauses = (3.0, 15.0)  # seconds before the 1st and 2nd re-send after a 5xx

    models: tuple[dict, ...] = ()  # {"id", "label", "note"}, best first; ids the provider accepts
    style = False  # preset voices accept a free-text style / narration note
    license = "Cloud API output — personal use as a cloning prompt, see the provider's terms"

    def __init__(self, api_key_env: str = "", transport: Transport | None = None, sleep=time.sleep,
                 api_key: str | None = None, model: str = "auto", **options):
        from folivox.config import api_key as lookup

        self.key_env = api_key_env or type(self).key_env
        self.key = api_key or lookup(self.key_env, self.label or self.name)
        self.transport, self._sleep = transport or urllib_transport, sleep
        self.mode = type(self).mode if type(self).mode in self.modes else self.modes[0]
        self.options = options
        self.requested = model or "auto"  # "auto" = the best model the provider offers, found through its API
        self._auto: str | None = None
        self._speakers: dict[str, tuple[str, str]] | None = None

    # ---- models ----------------------------------------------------------------------------------
    def discover_models(self) -> list[str]:
        """Model ids the provider reports for this key (empty when it has no such endpoint)."""
        return []

    @classmethod
    def rank(cls, model_id: str) -> tuple:
        """Sort key, higher = better; override to prefer newer versions."""
        return ()

    @property
    def model(self) -> str:
        """The model to use: the one asked for, else the best discovered, else the first declared."""
        if self.requested not in ("auto", "latest"):
            return self.requested
        if self._auto is None:
            try:
                found = [m for m in self.discover_models() if self.rank(m)]
                self._auto = max(found, key=self.rank) if found else ""
            except CloudError as e:
                if e.kind == "auth":
                    raise
                self._auto = ""
            self._auto = self._auto or (self.models[0]["id"] if self.models else "")
        return self._auto

    # ---- declared capabilities -------------------------------------------------------------------
    @classmethod
    def capabilities(cls) -> dict:
        return {"name": cls.name, "label": cls.label or cls.name, "modes": list(cls.modes),
                "languages": list(cls.languages), "key_env": cls.key_env, "price_note": cls.price_note,
                "terms_url": cls.terms_url, "docs_url": cls.docs_url, "commercial": cls.commercial,
                "max_chars": cls.max_chars, "fragment_chars": min(cls.fragment_chars, cls.max_chars),
                "blurb": cls.blurb, "alt_note": cls.alt_note,
                "voices_note": cls.voices_note,
                "models": list(cls.models), "style": cls.style, "voice_scopes": list(getattr(cls, "scopes", ())),
                "voice_langs": list(getattr(cls, "voice_languages", ())),
                "mode_labels": dict(cls.mode_labels)}

    @classmethod
    def estimate_for(cls, chars: int, takes: int = 1, mode: str = "preset", model: str = "") -> dict:
        """Characters sent and, where the provider documents a price, an approximate cost (``usd`` else None)."""
        return {"chars": chars * takes, "usd": None, "note": cls.price_note}

    @classmethod
    def bytes_factor(cls, lang: str = "") -> float:
        """Billed units per character of text in ``lang`` (1 unless the provider bills by UTF-8 bytes)."""
        return 1.0

    def speakers(self, lang: str | None = None) -> dict[str, tuple[str, str]]:
        """Preset voices ``{id: (language or "", sex "f"/"m"/"")}``; may call the provider's API."""
        return {}

    names: dict[str, str] = {}  # voice id → display name, when the id is not readable
    details: dict[str, dict] = {}  # voice id → {"traits", "description", "preview_url"} for the voice picker
    alt_note = ""  # a separate line under the engine text, e.g. that a free local alternative exists
    voices_note = ""  # shown under the voice list (e.g. where free samples can be heard)

    # ---- HTTP ------------------------------------------------------------------------------------
    def headers(self) -> dict[str, str]:
        raise NotImplementedError

    def call(self, method: str, url: str, payload: dict | None = None, *, timeout: float = 300,
             raw: tuple[bytes, str] | None = None, headers: dict[str, str] | None = None) -> Response:
        """One request with retries. ``raw`` = ``(body, content_type)`` replaces the JSON payload (multipart
        uploads); ``headers`` adds request-specific headers to the provider's own."""
        body = json.dumps(payload).encode() if payload is not None else None
        ctype = "application/json"
        if raw is not None:
            body, ctype = raw
        headers = {**self.headers(), **(headers or {}), **({"Content-Type": ctype} if body is not None else {})}
        server_tries = 0
        for attempt in range(self.retries + 1 + len(self.server_pauses)):
            try:
                resp = self.transport(method, url, headers, body, timeout)
            except CloudError as e:
                if e.kind == "network" and attempt < self.retries:
                    self._sleep(backoff(attempt))
                    continue
                raise
            if resp.status < 300:
                return resp
            err = self.map_error(resp)
            if err.kind == "server":
                # A 5xx may still have been billed, so a paid request is re-sent at most twice: after a short
                # pause, then after a longer one (a provider's Retry-After wins, within the cap)
                if server_tries < len(self.server_pauses):
                    self._sleep(min(err.retry_after or self.server_pauses[server_tries], 60))
                    server_tries += 1
                    continue
            elif err.kind == "rate_limit" and attempt < self.retries:  # refused before any work: not billed
                self._sleep(min(err.retry_after or backoff(attempt + 1), 60))
                continue
            raise err
        raise CloudError("Unreachable")  # pragma: no cover

    def map_error(self, resp: Response) -> CloudError:
        who, msg = self.label or self.name, error_text(resp)
        suffix = f": {msg}" if msg else ""
        s = resp.status
        try:
            retry = float(resp.headers.get("retry-after", ""))
        except ValueError:
            retry = None
        if retry is not None and not 0 <= retry < 86400:  # negative, NaN or absurd values would crash sleep()
            retry = None
        if s in (401, 403):
            return CloudError(f"{who} rejected the API key ({s}). Check {self.key_env} or save a new key{suffix}",
                              "auth")
        if s == 402 or self.is_quota(resp, msg):
            return CloudError(f"{who}: quota or credits exhausted — top up your account or wait for the reset{suffix}",
                              "quota")
        if s == 429:
            wait = f" Retry in about {int(retry)} s." if retry else " Wait a minute and try again."
            return CloudError(f"{who}: rate limit reached.{wait}{suffix}", "rate_limit", retry)
        if 400 <= s < 500:
            return CloudError(f"{who} refused the request ({s}){suffix}", "bad_request")
        return CloudError(f"{who} server error ({s}){suffix}", "server", retry)

    QUOTA_TOKENS = ("quota_exceeded", "insufficient_quota", "credit_balance_exhausted", "spend_limit_exceeded",
                    "usage_limit_exceeded")

    def is_quota(self, resp: Response, msg: str) -> bool:
        body = resp.body.lower().decode("utf-8", "replace")
        return any(t in body for t in self.QUOTA_TOKENS)

    def billed_chars(self, job: Job) -> int:
        """Characters the provider bills for one request (the text plus anything added to it)."""
        return len(job.text)

    def check_text(self, text: str) -> None:
        if len(text) > self.max_chars:
            raise CloudError(f"{self.label or self.name}: the text is {len(text)} characters, the limit is "
                             f"{self.max_chars}", "bad_request")

    def check_lang(self, lang: str) -> None:
        if lang not in self.languages:
            raise CloudError(f"{self.label or self.name} does not list {lang!r}; supported: "
                             f"{', '.join(self.languages)}", "bad_request")

    # ---- generation ------------------------------------------------------------------------------
    def synth(self, job: Job) -> bytes:
        """One take as WAV bytes (preset / instruction mode)."""
        raise NotImplementedError

    def setup_voices(self, jobs: list[Job]) -> list[str]:
        """Modes that create a temporary voice in the provider's account point ``job.speaker`` at it and
        return the ids to delete afterwards."""
        return []

    def teardown_voices(self, ids: list[str]) -> None:
        pass

    def generate(self, jobs: list[Job], progress=None) -> list[Path]:
        for j in jobs:
            self.check_text(j.text)
            self.check_lang(j.lang)
        created = self.setup_voices(jobs)
        try:
            for j in jobs:
                t0 = time.time()
                Path(j.out).write_bytes(self.synth(j))
                if progress:
                    progress(j, time.time() - t0)
        finally:
            try:
                self.teardown_voices(created)  # never leave test voices in the user's account
            except CloudError:
                pass
        return [j.out for j in jobs]


def multipart(fields: list[tuple[str, str]], files: list[tuple[str, str, bytes, str]]) -> tuple[bytes, str]:
    """A multipart/form-data body and its Content-Type. ``files`` = ``(field, filename, data, content_type)``."""
    import uuid

    boundary = uuid.uuid4().hex
    out = b""
    for name, value in fields:
        out += (f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n').encode()
    for name, filename, data, ctype in files:
        out += (f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
                f'Content-Type: {ctype}\r\n\r\n').encode() + data + b"\r\n"
    return out + f"--{boundary}--\r\n".encode(), f"multipart/form-data; boundary={boundary}"


def to_wav(data: bytes, rate: int = 24000) -> bytes:
    """WAV passes through; anything else (mp3, flac, ogg) is decoded and re-encoded as WAV."""
    import io

    import soundfile as sf

    if data[:4] == b"RIFF":
        return data
    try:
        audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception:  # noqa: BLE001 — libsndfile without mp3 support, or not audio at all
        raise CloudError("The provider's audio could not be decoded (MP3 needs libsndfile 1.1+; try `pip install -U "
                         "soundfile`)", "server") from None
    buf = io.BytesIO()
    sf.write(buf, audio.mean(axis=1), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def file_as_wav(path: Path) -> bytes:
    """Any audio file (FLAC from the voice catalog, …) as WAV bytes."""
    return to_wav(Path(path).read_bytes())


def pcm_to_wav(pcm: bytes, sample_rate: int = 24000) -> bytes:
    """Raw 16-bit little-endian mono PCM → WAV bytes."""
    import io

    import soundfile as sf

    data = np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype="<i2")
    buf = io.BytesIO()
    sf.write(buf, data, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def b64(data: str) -> bytes:
    try:
        return base64.b64decode(data)
    except ValueError:
        raise CloudError("The provider returned audio that cannot be decoded", "server") from None


def cloud_providers() -> dict[str, type[CloudGenerator]]:
    """Registered cloud generators by name (classes only: no key or network needed)."""
    import importlib

    from folivox.voicegen import GENERATORS

    out = {}
    for p in GENERATORS.providers():
        if p.local:
            continue
        module, _, attr = p.target.partition(":")
        try:
            cls = getattr(importlib.import_module(module), attr)
        except (ImportError, AttributeError):
            continue
        if isinstance(cls, type) and issubclass(cls, CloudGenerator):
            out[p.name] = cls
    return out

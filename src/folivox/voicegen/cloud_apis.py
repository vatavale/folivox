"""Gemini, OpenAI, ElevenLabs, Cartesia and Fish Audio voice generators (endpoints, models and prices checked
against the providers' documentation and live calls in October 2026; see CONTRIBUTING for how to add another).

Models: ``model="auto"`` (the default) asks the provider which models the key can use and takes the newest good
one; a name pins it (``[voicegen.<name>] model = "..."`` in config.toml, ``--model`` on the command line)."""

from __future__ import annotations

import math
import re
import urllib.parse
import uuid
from pathlib import Path

from folivox.voicegen import Job
from folivox.voicegen.cloud import (
    CloudError,
    CloudGenerator,
    b64,
    error_text,
    file_as_wav,
    multipart,
    pcm_to_wav,
    to_wav,
)

LANGS = ("en", "ru", "de", "fr", "es", "it", "zh", "ja", "ko")


def _model(id_: str, label: str, note: str = "") -> dict:
    return {"id": id_, "label": label, "note": note}


def audio(body: bytes, who: str) -> bytes:
    """Reject an empty reply: a header-only WAV or no bytes at all would become a silent take. Not retried
    (``bad_request``) because the request was probably billed."""
    if len(body) < 64:
        raise CloudError(f"{who} returned no audio for the text", "bad_request")
    return body


class Gemini(CloudGenerator):
    """Gemini API speech generation: 30 preset voices, or a voice designed from a description (3.8 models)."""

    name, label, modes, languages = "gemini", "Gemini API", ("preset", "design"), LANGS
    mode_labels = {"preset": "Pick a voice", "design": "Describe a voice (made for this run, then deleted)"}
    style = True
    blurb = (
        "Google's voices: 30 ready-made ones, or describe a new voice in words (it exists only while the "
        "takes are made). Understands a note like 'calm audiobook narrator'. A free tier exists (Google "
        'may then use your text); the paid tier costs a fraction of a cent per sample.'
    )
    key_env = "GEMINI_API_KEY"
    price_note = ("Paid tier about $0.0023 per 10 s of audio on gemini-3.8-flash-tts (doubles in 2027); a free tier "
                  "exists, but Google may then use your text for product improvement.")
    terms_url = "https://ai.google.dev/gemini-api/terms"
    docs_url = "https://ai.google.dev/gemini-api/docs/generate-content/speech-generation"
    max_chars = 4000
    # Gemini's free and tier-1 quotas count requests per day (100), so fewer, longer requests matter more than they
    # do elsewhere. Tests on Russian passages with 0.9k–2.4k characters per request (Flash and Flash-Lite) showed
    # no loss of text (CER 0.004–0.013, same as 300-character fragments) and a steadier pace.
    fragment_chars = 2000
    models = (_model("gemini-3.8-flash-tts", "Gemini 3.8 Flash TTS", "newest; also designs voices"),
              _model("gemini-3.8-flash-lite-tts", "Gemini 3.8 Flash-Lite TTS", "cheaper"),
              _model("gemini-3.1-flash-tts-preview", "Gemini 3.1 Flash TTS (preview)"),
              _model("gemini-2.5-pro-preview-tts", "Gemini 2.5 Pro TTS (preview)"),
              _model("gemini-2.5-flash-preview-tts", "Gemini 2.5 Flash TTS (preview)"))
    VOICES = ("Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda", "Orus", "Aoede", "Callirrhoe", "Autonoe",
              "Enceladus", "Iapetus", "Umbriel", "Algieba", "Despina", "Erinome", "Algenib", "Rasalgethi",
              "Laomedeia", "Achernar", "Alnilam", "Schedar", "Gacrux", "Pulcherrima", "Achird", "Zubenelgenubi",
              "Vindemiatrix", "Sadachbia", "Sadaltager", "Sulafat")
    LOCALES = {"en": "en-US", "ru": "ru-RU", "de": "de-DE", "fr": "fr-FR", "es": "es-ES", "it": "it-IT",
               "zh": "zh-CN", "ja": "ja-JP", "ko": "ko-KR"}
    BASE = "https://generativelanguage.googleapis.com/v1beta"

    def headers(self):
        return {"x-goog-api-key": self.key}

    def map_error(self, resp):
        """Gemini says RESOURCE_EXHAUSTED for every 429 and puts the details in the body, not in headers:
        ``RetryInfo.retryDelay`` ("27s") and ``QuotaFailure`` violations whose ids name the limit (…PerMinute…,
        …PerDay…). A per-day limit will not clear by waiting a minute, so it stops the run instead of retrying."""
        err = super().map_error(resp)
        if resp.status != 429 or err.kind == "quota":
            return err
        text = resp.body.decode("utf-8", "replace")
        who, msg = self.label, error_text(resp)
        if re.search(r"PerDay|per_day|PerModelPerDay", text):
            return CloudError(
                f"{who}: the daily request limit for {self.model} is used up. It resets at midnight Pacific time. "
                "Progress is saved: run the same command again after the reset, choose another model, or move "
                f"the Google project to a paid tier (see {self.docs_url}).{': ' + msg if msg else ''}", "quota")
        retry = err.retry_after
        if retry is None:
            m = re.search(r'"retryDelay"\s*:\s*"([\d.]+)s"', text) or re.search(r"retry in ([\d.]+)\s*s", text, re.I)
            if m and 0 <= float(m[1]) < 86400:
                retry = float(m[1])
        scope = "per-minute request limit" if re.search(r"PerMinute|per_minute", text) else "request limit"
        wait = f" Google asks to wait about {int(retry + 0.999)} s." if retry else ""
        return CloudError(
            f"{who}: the {scope} for {self.model} was reached.{wait} Free and low tiers allow only a few requests "
            "a minute; Folivox slows down and retries, you can set a lower rate with requests_per_minute in "
            f"config.toml ([cloud.gemini]) or move the project to a paid tier.{': ' + msg if msg else ''}",
            "rate_limit", retry)

    def discover_models(self):
        data = self.call("GET", f"{self.BASE}/models?pageSize=200").json()
        return [m["name"].removeprefix("models/") for m in data.get("models", [])
                if "tts" in m["name"] and "generateContent" in m.get("supportedGenerationMethods", ["generateContent"])]

    @classmethod
    def rank(cls, model_id):
        m = re.fullmatch(r"gemini-(\d+)\.(\d+)-(flash-lite|flash|pro)-tts(-preview)?", model_id)
        if not m:
            return ()
        tier = {"flash-lite": 0, "pro": 1, "flash": 2}[m[3]]
        return (int(m[1]), int(m[2]), 0 if m[4] else 1, tier)

    @classmethod
    def estimate_for(cls, chars, takes=1, mode="preset", model=""):
        seconds = chars * takes / 14  # narration pace, characters per second
        return {"chars": chars * takes, "usd": round(seconds * 0.00225 / 10, 3), "note": cls.price_note}

    def speakers(self, lang=None):
        if self._speakers is None:
            out: dict = {}
            self.details = {}
            try:  # the API describes its prebuilt voices: sex, role, pitch, a sentence of character
                url = f"{self.BASE}/voices?type=prebuilt&language_code=en-US&page_size=1000"
                for v in self.call("GET", url).json().get("voices", []):
                    name = v.get("display_name") or v.get("id", "")
                    sex = {"female": "f", "male": "m"}.get(v.get("gender", ""), "")
                    out[name] = ("", sex)
                    pitch = f"{v['pitch']} pitch" if v.get("pitch") else ""
                    traits = [x for x in (v.get("persona"), pitch) if x]
                    self.details[name] = {"traits": " · ".join(traits), "description": v.get("description", "")}
            except CloudError as e:
                if e.kind == "auth":
                    raise
            self._speakers = {k: out[k] for k in self.VOICES if k in out} or {v: ("", "") for v in self.VOICES}
        return self._speakers

    def setup_voices(self, jobs):
        if self.mode != "design":
            return []
        made: dict[tuple[str, str], str] = {}
        for j in jobs:
            key = (j.description, j.lang)
            if key not in made:
                if not self.model.startswith("gemini-3.8"):
                    raise CloudError(f"Gemini voice design needs a gemini-3.8 model, not {self.model}", "bad_request")
                r = self.call("POST", f"{self.BASE}/voices", {"store": True, "voice": {
                    "model": self.model, "type": "prompted", "display_name": f"folivox-{uuid.uuid4().hex[:8]}",
                    "language_code": self.LOCALES.get(j.lang, "en-US"), "prompted": {"input": j.description}}}).json()
                if not r.get("id"):
                    raise CloudError("Gemini did not return a voice id for the description", "server")
                made[key] = r["id"]
            j.speaker = made[key]
        return list(made.values())

    def teardown_voices(self, ids):
        for i in ids:
            self.call("DELETE", f"{self.BASE}/voices/{i}")

    def synth(self, job: Job) -> bytes:
        voice = job.speaker or "Kore"
        if self.mode != "design" and voice not in self.VOICES:
            raise CloudError(f"Gemini has no voice {voice!r}; choose one of: {', '.join(self.VOICES)}", "bad_request")
        part: dict = {"text": job.text}
        cfg: dict = {"responseModalities": ["AUDIO"]}
        style = job.description if self.mode == "preset" else ""
        if self.model.startswith("gemini-2.5"):  # older request shape: style is a prefix, voice is prebuilt
            if style:
                part["text"] = f"Say in this style ({style}): {job.text}"
            cfg["speechConfig"] = {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}}
        else:
            if style:
                part["speech_metadata"] = {"style": style}
            cfg["speechConfig"] = {"voiceConfig": {"voice": voice}}
        r = self.call("POST", f"{self.BASE}/models/{self.model}:generateContent",
                      {"contents": [{"role": "user", "parts": [part]}], "generationConfig": cfg})
        try:
            inline = r.json()["candidates"][0]["content"]["parts"][0]["inlineData"]
            raw = b64(inline["data"])
        except (KeyError, IndexError, TypeError):
            raise CloudError("Gemini returned no audio (the text may have been blocked by its safety filters)",
                             "bad_request") from None
        audio(raw, self.label)
        if raw[:4] == b"RIFF":
            return raw
        m = re.search(r"rate=(\d+)", str(inline.get("mimeType", "")))
        return pcm_to_wav(raw, int(m.group(1)) if m else 24000)


class OpenAIVoice(CloudGenerator):
    """OpenAI ``/audio/speech``. ``gpt-4o-mini-tts`` follows free-text instructions (tone, pace, accent)."""

    name, label, modes, languages = "openai", "OpenAI", ("preset",), LANGS
    style = True
    blurb = (
        "13 ready-made voices; a style note steers tone and pace ('audiobook narration, calm, steady pace'). "
        'No voice-from-description. Simple, reliable and cheap.'
    )
    key_env = "OPENAI_API_KEY"
    price_note = ("gpt-4o-mini-tts: roughly $0.015 per minute of audio (billed in tokens); tts-1: $15 per million "
                  "characters; tts-1-hd: $30.")
    terms_url = "https://openai.com/policies/terms-of-use"
    docs_url = "https://developers.openai.com/api/docs/guides/text-to-speech"
    max_chars = 4096
    fragment_chars = 2000  # fewer requests; neighbouring paragraphs share one
    models = (_model("gpt-4o-mini-tts", "GPT-4o mini TTS", "newest; follows style instructions"),
              _model("tts-1-hd", "TTS-1 HD", "no instructions"), _model("tts-1", "TTS-1", "cheapest, no instructions"))
    VOICES = ("alloy", "ash", "ballad", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer", "verse",
              "marin", "cedar")
    CLASSIC = ("alloy", "ash", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer")  # tts-1 / tts-1-hd

    def __init__(self, base_url: str = "https://api.openai.com/v1", **kw):
        super().__init__(**kw)
        self.base_url = base_url.rstrip("/")
        self.details = {}

    def headers(self):
        return {"Authorization": f"Bearer {self.key}"}

    def discover_models(self):
        data = self.call("GET", f"{self.base_url}/models").json()
        return [m["id"] for m in data.get("data", []) if re.fullmatch(r"gpt-[a-z0-9.\-]*tts", m["id"])]

    @classmethod
    def rank(cls, model_id):
        # the undated alias of a gpt-*-tts model; a higher generation number wins
        nums = [int(n) for n in re.findall(r"\d+", model_id)]
        return (nums or [0], model_id) if re.fullmatch(r"gpt-[a-z0-9.\-]*tts", model_id) else ()

    @classmethod
    def estimate_for(cls, chars, takes=1, mode="preset", model=""):
        n = chars * takes
        usd = n / 1e6 * (30 if model == "tts-1-hd" else 15) if model.startswith("tts-1") else n / 14 / 60 * 0.015
        return {"chars": n, "usd": round(usd, 3), "note": cls.price_note}

    voices_note = ("OpenAI publishes no descriptions of its voices: sex and character here are approximate "
                   "impressions. Listen to them for free at https://www.openai.fm")
    IMPRESSION = {"alloy": ("", "neutral, balanced"), "ash": ("m", "warm, steady"), "ballad": ("m", "soft, gentle"),
                  "coral": ("f", "warm, friendly"), "echo": ("m", "clear, calm"),
                  "fable": ("m", "expressive storyteller"),
                  "nova": ("f", "bright, energetic"), "onyx": ("m", "deep, authoritative"),
                  "sage": ("f", "calm, composed"), "shimmer": ("f", "light, gentle"), "verse": ("m", "expressive"),
                  "marin": ("f", "natural, clear"), "cedar": ("m", "natural, warm")}

    def speakers(self, lang=None):
        voices = self.VOICES if self.model.startswith("gpt-") else self.CLASSIC
        self.details = {v: {"traits": self.IMPRESSION[v][1], "description": ""} for v in voices}
        return {v: ("", self.IMPRESSION[v][0]) for v in voices}

    def synth(self, job: Job) -> bytes:
        voice = job.speaker or "coral"
        if voice not in self.speakers():
            raise CloudError(f"{self.model} has no voice {voice!r}; choose one of: {', '.join(self.speakers())}",
                             "bad_request")
        payload = {"model": self.model, "input": job.text, "voice": voice, "response_format": "wav"}
        if job.description and self.model.startswith("gpt-"):  # tts-1 rejects `instructions`
            payload["instructions"] = job.description
        return audio(self.call("POST", f"{self.base_url}/audio/speech", payload).body, self.label)


class ElevenLabs(CloudGenerator):
    """ElevenLabs: preset voices via text-to-speech, Voice Design previews from a text description."""

    name, label, modes, languages = "elevenlabs", "ElevenLabs", ("preset", "design"), LANGS
    style = True  # eleven_v3 / eleven_v4 read audio tags such as "[calm]" as direction, not as words
    blurb = (
        'Very natural voices. Describe a new voice in words and get 3 different candidates at once, or '
        'pick a ready-made one. The free plan is for personal, non-commercial use; audiobooks you sell '
        'need a paid plan.'
    )
    key_env = "ELEVENLABS_API_KEY"
    price_note = ("Pay-as-you-go list price $0.04–0.08 per 1,000 characters depending on the model (plans include "
                  "monthly credits). Free plan: personal non-commercial use, library voices are not available through "
                  "the API. Voice Design bills the preview text once per 3 previews.")
    terms_url = "https://elevenlabs.io/terms-of-use"
    docs_url = "https://elevenlabs.io/docs/api-reference/text-to-voice/design"
    max_chars = 10000
    fragment_chars = 2000  # fewer requests; neighbouring paragraphs share one
    BASE = "https://api.elevenlabs.io"
    models = (_model("eleven_v4", "Eleven v4", "newest, 90+ languages"), _model("eleven_v3", "Eleven v3", "expressive"),
              _model("eleven_multilingual_v2", "Multilingual v2", "29 languages, stable"),
              _model("eleven_flash_v2_5", "Flash v2.5", "fast and cheap"))
    DESIGN_MODELS = ("eleven_ttv_v3", "eleven_multilingual_ttv_v2")
    PRICE_PER_1K = {"eleven_v4": 0.08, "eleven_v4_turbo": 0.04, "eleven_v3": 0.08, "eleven_multilingual_v2": 0.08,
                    "eleven_flash_v2_5": 0.04}
    LIMITS = {"eleven_v4": 10000, "eleven_v4_turbo": 10000, "eleven_v3": 5000, "eleven_multilingual_v2": 10000,
              "eleven_flash_v2_5": 40000}

    def __init__(self, design_model: str = "eleven_ttv_v3", **kw):
        super().__init__(**kw)
        self.design_model = design_model
        self.names, self.details = {}, {}

    def headers(self):
        return {"xi-api-key": self.key}

    def discover_models(self):
        data = self.call("GET", f"{self.BASE}/v1/models").json().get("data", [])
        return [m["model_id"] for m in data if m.get("can_do_text_to_speech")] if isinstance(data, list) else []

    @classmethod
    def rank(cls, model_id):
        m = re.fullmatch(r"eleven_v(\d+)", model_id)  # plain flagship line; turbo / flash / conversational skipped
        return (int(m[1]),) if m else ()

    @classmethod
    def estimate_for(cls, chars, takes=1, mode="preset", model=""):
        n = chars * (math.ceil(takes / 3) if mode == "design" else takes)
        return {"chars": n, "usd": round(n / 1000 * cls.PRICE_PER_1K.get(model, 0.08), 3),
                "note": cls.price_note}

    def check_text(self, text):
        self.max_chars = self.LIMITS.get(self.model, 10000)
        super().check_text(text)

    def speakers(self, lang=None):
        if self._speakers is None:
            out, token = {}, None
            for _ in range(5):  # at most 500 voices
                q = {"page_size": 100, "category": "premade", **({"next_page_token": token} if token else {})}
                data = self.call("GET", f"{self.BASE}/v2/voices?{urllib.parse.urlencode(q)}").json()
                for v in data.get("voices", []):
                    sex = {"female": "f", "male": "m"}.get((v.get("labels") or {}).get("gender", ""), "")
                    out[v["voice_id"]] = ("", sex)
                    self.names[v["voice_id"]] = v.get("name", v["voice_id"])
                    lb = v.get("labels") or {}
                    self.details[v["voice_id"]] = {
                        "traits": " · ".join(x.replace("_", " ") for x in (
                            lb.get("age"), lb.get("accent"), lb.get("descriptive"), lb.get("use_case")) if x),
                        "description": v.get("description") or "", "preview_url": v.get("preview_url") or ""}
                token = data.get("next_page_token")
                if not data.get("has_more") or not token:
                    break
            self._speakers = out
        return self._speakers

    def synth(self, job: Job) -> bytes:
        if not job.speaker:
            raise CloudError("ElevenLabs: pick a voice (preset mode needs a voice id)", "bad_request")
        text = self.billed_text(job)
        payload = {"text": text, "model_id": self.model, "seed": job.seed % 4294967295}
        url = f"{self.BASE}/v1/text-to-speech/{urllib.parse.quote(job.speaker, safe='')}?output_format=pcm_24000"
        return pcm_to_wav(audio(self.call("POST", url, payload).body, self.label), 24000)

    def billed_text(self, job: Job) -> str:
        """The text actually sent: eleven_v3 / v4 get the style note as a leading audio tag (billed too)."""
        if job.description and re.fullmatch(r"eleven_v[34]\w*", self.model):
            return f"[{job.description.strip('[] ')}] {job.text}"
        return job.text

    def billed_chars(self, job):
        return len(self.billed_text(job))

    def setup_voices(self, jobs):
        """Design mode for a whole book: design a voice (3 previews, the first is kept) and save it in the
        account so every request uses the same one. Docs: POST /v1/text-to-voice/design, then
        POST /v1/text-to-voice with the preview's generated_voice_id."""
        if self.mode != "design":
            return []
        made: dict[tuple[str, str], str] = {}
        for j in jobs:
            self.check_lang(j.lang)
            if len(j.description) < 20:
                raise CloudError("ElevenLabs Voice Design needs a description of at least 20 characters",
                                 "bad_request")
            key = (j.description, j.lang)
            if key not in made:
                design = self.call("POST", f"{self.BASE}/v1/text-to-voice/design", {
                    "voice_description": j.description, "auto_generate_text": True,
                    "model_id": self.design_model}).json()
                previews = design.get("previews") or []
                if not previews or not previews[0].get("generated_voice_id"):
                    raise CloudError("ElevenLabs returned no voice preview for the description", "server")
                saved = self.call("POST", f"{self.BASE}/v1/text-to-voice", {
                    "voice_name": f"folivox-{uuid.uuid4().hex[:8]}", "voice_description": j.description,
                    "generated_voice_id": previews[0]["generated_voice_id"]}).json()
                if not saved.get("voice_id"):
                    raise CloudError("ElevenLabs did not return a voice id", "server")
                made[key] = saved["voice_id"]
            j.speaker = made[key]
        return list(made.values())

    def teardown_voices(self, ids):
        for i in ids:
            self.call("DELETE", f"{self.BASE}/v1/voices/{urllib.parse.quote(i, safe='')}")

    def generate(self, jobs, progress=None):
        import time

        if self.mode != "design":
            return super().generate(jobs, progress)
        groups: dict[tuple[str, str], list[Job]] = {}
        for j in jobs:
            self.check_lang(j.lang)
            if not 100 <= len(j.text) <= 1000:
                raise CloudError("ElevenLabs Voice Design needs a text of 100–1000 characters", "bad_request")
            if len(j.description) < 20:
                raise CloudError("ElevenLabs Voice Design needs a description of at least 20 characters",
                                 "bad_request")
            groups.setdefault((j.text, j.description), []).append(j)
        for (text, description), group in groups.items():
            for i in range(0, len(group), 3):  # one request returns 3 previews, billed once
                chunk, t0 = group[i:i + 3], time.time()
                data = self.call("POST", f"{self.BASE}/v1/text-to-voice/design?output_format=pcm_24000", {
                    "voice_description": description, "text": text, "model_id": self.design_model}).json()
                previews = data.get("previews") or []
                if len(previews) < len(chunk):
                    raise CloudError("ElevenLabs returned fewer voice previews than requested", "server")
                for j, pv in zip(chunk, previews, strict=False):
                    raw = b64(pv.get("audio_base_64", ""))
                    media = str(pv.get("media_type", ""))
                    Path(j.out).write_bytes(to_wav(raw) if "mpeg" in media or raw[:4] == b"RIFF"
                                            else pcm_to_wav(raw, 24000))
                    if progress:
                        progress(j, (time.time() - t0) / len(chunk))
        return [j.out for j in jobs]

    def map_error(self, resp):
        if resp.status == 409:  # a preview for this account is still being generated
            return CloudError("ElevenLabs is still busy with an earlier voice preview", "server", 3)
        return super().map_error(resp)


class Cartesia(CloudGenerator):
    """Cartesia Sonic: preset voices; *localize* gives a voice the native accent of the target language; *clone*
    copies one of your own voices from a sample. The API has no text-based voice design."""

    name, label, languages = "cartesia", "Cartesia", LANGS
    modes = ("preset", "localize", "clone")
    mode_labels = {"preset": "Pick a voice", "localize": "Same voice, native accent of the language",
                   "clone": "Clone one of my voices"}
    blurb = (
        "Clean, fast voices in 40+ languages. Pick a voice, give it the native accent of your book's "
        'language, or clone one of your own voices from your recording. No voice-from-description. '
        'Cloning needs a paid plan.'
    )
    key_env = "CARTESIA_API_KEY"
    price_note = ("1 credit per character: free plan 20,000 credits a month, Pro $5 for 100,000. Localizing a voice "
                  "costs about 225 credits, cloning needs a paid plan. Commercial use of the output needs a paid "
                  "plan; the terms forbid using output to develop competing models.")
    terms_url = "https://cartesia.ai/legal/terms.html"
    docs_url = "https://docs.cartesia.ai/api-reference/tts/bytes"
    max_chars = 5000
    BASE = "https://api.cartesia.ai"
    VERSION = "2026-08-14"
    models = (_model("sonic-3.6", "Sonic 3.6", "newest, 44 languages"), _model("sonic-3.5", "Sonic 3.5"),
              _model("sonic-3", "Sonic 3"), _model("sonic-latest", "Sonic latest", "alias that moves with releases"))

    def __init__(self, **kw):
        super().__init__(**kw)
        self.names, self.details = {}, {}  # per instance: never fill the base class's shared dicts

    def headers(self):
        return {"Authorization": f"Bearer {self.key}", "Cartesia-Version": self.VERSION}

    @classmethod
    def estimate_for(cls, chars, takes=1, mode="preset", model=""):
        n = chars * takes + (225 if mode == "localize" else 0)
        return {"chars": n, "usd": round(n * 5 / 100000, 3), "note": cls.price_note}

    def speakers(self, lang=None):
        key = lang or ""
        cache = self.__dict__.setdefault("_by_lang", {})
        if key not in cache:
            out, after = {}, None
            for _ in range(3):  # at most 300 voices
                q = {"limit": 100, "expand[]": "preview_file_url", **({"language": lang} if lang else {}),
                     **({"starting_after": after} if after else {})}
                data = self.call("GET", f"{self.BASE}/voices?{urllib.parse.urlencode(q)}").json()
                for v in data.get("data", []):
                    sex = {"feminine": "f", "masculine": "m"}.get(v.get("gender", ""), "")
                    out[v["id"]] = (lang or v.get("language") or "", sex)
                    self.names[v["id"]] = v.get("name", v["id"])
                    self.details[v["id"]] = {"traits": v.get("tagline") or "",
                                             "description": v.get("description") or "",
                                             "preview_url": v.get("preview_file_url") or ""}
                rows = data.get("data") or []
                if not data.get("has_more") or not rows:
                    break
                after = rows[-1]["id"]
            cache[key] = out
        return cache[key]

    def accent_for(self, lang: str) -> str:
        accents = self.call("GET", f"{self.BASE}/accents?is_localizable=true").json().get("accents", [])
        mine = [a for a in accents if a.get("language") == lang]
        mine.sort(key=lambda a: not a.get("is_locale_default"))
        if not mine:
            raise CloudError(f"Cartesia cannot localize a voice to {lang!r}", "bad_request")
        return mine[0]["id"]

    def setup_voices(self, jobs):
        if self.mode == "preset":
            return []
        made: dict[tuple, str] = {}
        name = f"folivox-{uuid.uuid4().hex[:8]}"
        for j in jobs:
            key = (j.speaker, str(j.reference), j.lang)
            if key in made:
                j.speaker = made[key]
                continue
            if self.mode == "localize":
                if not j.speaker:
                    raise CloudError("Cartesia: pick the voice to localize", "bad_request")
                r = self.call("POST", f"{self.BASE}/voices/localize",
                              {"voice_id": j.speaker, "name": name, "accent": self.accent_for(j.lang)}).json()
            else:
                if not j.reference or not Path(j.reference).exists():
                    raise CloudError("Cartesia clone: choose one of your own voices as the sample", "bad_request")
                r = self.upload_clone(Path(j.reference), name, j.lang)
            if not r.get("id"):
                raise CloudError("Cartesia did not return a voice id", "server")
            made[key] = j.speaker = r["id"]
        return list(made.values())

    def upload_clone(self, clip: Path, name: str, lang: str) -> dict:
        boundary = uuid.uuid4().hex

        def part(field, value, filename=None):
            head = f'--{boundary}\r\nContent-Disposition: form-data; name="{field}"'
            head += f'; filename="{filename}"\r\nContent-Type: audio/wav' if filename else ""
            return (head + "\r\n\r\n").encode() + (value if isinstance(value, bytes) else value.encode()) + b"\r\n"

        body = part("clip", file_as_wav(clip), "voice.wav") + part("name", name) + part("language", lang)
        body += f"--{boundary}--\r\n".encode()
        headers = {**self.headers(), "Content-Type": f"multipart/form-data; boundary={boundary}"}
        resp = self.transport("POST", f"{self.BASE}/voices/clone", headers, body, 120)
        if resp.status >= 300:
            raise self.map_error(resp)
        return resp.json()

    def teardown_voices(self, ids):
        for i in ids:
            self.call("DELETE", f"{self.BASE}/voices/{i}")

    def synth(self, job: Job) -> bytes:
        if not job.speaker:
            raise CloudError("Cartesia: pick a voice (a voice id)", "bad_request")
        payload = {"model_id": self.model, "transcript": job.text, "voice": {"id": job.speaker},
                   "language": job.lang,
                   "output_format": {"container": "wav", "encoding": "pcm_s16le", "sample_rate": 24000}}
        return audio(self.call("POST", f"{self.BASE}/tts/bytes", payload).body, self.label)


class FishAudio(CloudGenerator):
    """Fish Audio API: Drama 3 (and the S2 family) reading with a library voice, a voice designed from a text
    description (``/v1/voice-design``) or a clone of one of your own recordings (``/model``). The voice made for
    a book is a private model in the account and is deleted afterwards unless the user keeps it."""

    name, label = "fish", "Fish Audio"
    modes = ("preset", "design", "clone")
    mode_labels = {"preset": "Pick a voice from the library",
                   "design": "Describe a voice (made for this run, then deleted)",
                   "clone": "Clone one of my voices"}
    style = True  # Drama 3 / S2 read a bracketed note before the words as direction: "[calm, warm narration] …"
    blurb = (
        "Thousands of library voices in dozens of languages, a new voice from a description, or a clone of your own "
        "recording."
    )
    alt_note = "Drama 3 also runs on this computer (Apple Silicon, 32+ GB): choose Fish Audio Drama 3 in this list."
    key_env = "FISH_API_KEY"
    price_note = ("Billed per UTF-8 byte of text, so Cyrillic costs about twice and Chinese/Japanese/Korean about "
                  "three times as much per character as English. S2 models: $15 per million bytes. Drama 3 is not "
                  "on Fish's public price page: the figure used here, $50 per million characters, is the list price "
                  "reported at its October 2026 launch (a launch offer of $10 was reported until 12 Nov 2026); check "
                  "your dashboard. Voice Design costs $0.01 per request. An API balance is needed: the free "
                  "plan's credits do not cover the API. Commercial use falls under the Fish Audio Community "
                  "License (free below USD 500k revenue, 'Built with Fish Audio' attribution, no voice "
                  "imitation without consent).")
    terms_url = "https://fish.audio/terms/"
    docs_url = "https://docs.fish.audio/features/text-to-speech/drama-3"
    voices_note = ("The library voices Fish Audio has licensed, its own “Fish Official” voices first (the rest of the "
                   "community library is not listed). Never use a voice that imitates a real person without consent.")
    max_chars = 3000  # not documented; the pipeline sends 300-character fragments (Fish's own chunk size)
    BASE = "https://api.fish.audio"
    models = (
        {**_model("drama-3", "Drama 3", "newest, directed in plain words"),
         "blurb": "Drama 3: the newest model, directed in plain words (emotion, pace, who is listening). Its price is "
                  "not on Fish's public list: the estimate uses the list price reported at launch."},
        {**_model("s2.1-pro", "S2.1 Pro", "stable, price on the public list"),
         "blurb": "S2.1 Pro: the stable production model, 83 languages, $15 per million bytes of text."},
        {**_model("s2.1-pro-free", "S2.1 Pro (free tier)", "no cost under fair use, for tests"),
         "blurb": "S2.1 Pro free tier: the same model at no cost under fair use, without speed or availability "
                  "guarantees. Good for tests."},
        {**_model("s2-pro", "S2 Pro"),
         "blurb": "S2 Pro: the previous generation, 80+ languages, $15 per million bytes of text."})
    PRICE_PER_M = {"drama-3": 50.0, "s2.1-pro": 15.0, "s2-pro": 15.0, "s2.1-pro-free": 0.0}
    # UTF-8 bytes per character of prose; the Latin languages with accents are 1.08
    BYTES_PER_CHAR = {
        "en": 1.0, "ru": 1.85, "zh": 3.0, "ja": 2.9, "ko": 2.7,
        **dict.fromkeys(("uk", "bg", "sr", "mk", "be", "ar", "he", "fa", "ur", "kk", "mn", "yi", "ps"), 1.85),
        "el": 1.9,
        "hy": 2.0, **dict.fromkeys(("ka", "am", "km", "my", "bo", "si"), 3.0),
        **dict.fromkeys(("hi", "bn", "ta", "te", "mr", "gu", "th", "pa", "kn", "ml", "sa", "as", "ne", "sd"), 2.8),
        "vi": 1.35}
    DESIGN_MODEL = "voice-design-1"
    OFFICIAL = "Fish Official"  # author name of the voices Fish itself publishes
    # languages the voice library is filtered by (a menu in the studio; any code Fish lists works in the API)
    voice_languages = ("zh", "ja", "ko", "ru", "de", "fr", "es", "it", "pt", "pl", "nl", "tr", "uk", "cs", "sk",
                       "sl", "hr", "sr", "bg", "ro", "hu", "el", "sv", "da", "nb", "fi", "ar", "he", "fa",
                       "ur", "hi", "bn", "ta", "te", "mr", "gu", "th", "vi", "id", "ms", "sw", "ca", "lt",
                       "lv", "et", "is", "mk", "sq", "cy", "eu", "gl", "la", "jv", "yo", "nn", "kk", "my",
                       "tl", "ne", "af", "bo", "sn", "mi", "yi", "am", "be", "km", "az", "sd", "br", "ps",
                       "mn", "ht", "ml", "sa", "ka", "bs", "pa", "kn", "si", "hy", "as", "fo")
    languages = tuple(dict.fromkeys((*LANGS, *voice_languages)))  # Fish reads all of these, rule-less ones as they are
    CACHE_TTL = 12 * 3600  # seconds a library listing is kept on disk
    library = "licensed"  # "licensed" (Fish's own and licensed voices) or "all" (the whole community library)
    scopes = ({"id": "licensed", "label": "Fish Official and licensed voices"},
              {"id": "all", "label": "All community voices"})
    FEATURED = {"en": {"bf322df2096a46f18c579d0baa36f41d": ("Adrian", "m"),  # the Drama 3 page's example voices
                       "c70cfb98867a4bfb9ae20d2401257b92": ("Delia", "f"),
                       "f3a2b90078d54a65af4b95f63bc7798e": ("Sheila", "f")}}

    def __init__(self, **kw):
        super().__init__(**kw)
        self.names, self.details = {}, {}

    def headers(self):
        return {"Authorization": f"Bearer {self.key}"}

    @classmethod
    def bytes_factor(cls, lang=""):
        return cls.BYTES_PER_CHAR.get(lang, 1.08)  # the Latin languages with accents

    @classmethod
    def estimate_for(cls, chars, takes=1, mode="preset", model=""):
        n = chars * takes
        price = cls.PRICE_PER_M.get(model or cls.models[0]["id"], cls.PRICE_PER_M["drama-3"])
        usd = n * price / 1e6 + (0.01 if mode == "design" else 0.0)
        return {"chars": n, "usd": round(usd, 3), "note": cls.price_note}

    # ---- voices ----------------------------------------------------------------------------------
    def library_rows(self, lang: str | None) -> list[dict]:
        """The voice library as rows, Fish's own voices first. A listing is slow (big pages), so the result is
        kept on disk for ``CACHE_TTL`` seconds; only real network use is cached, never a test's fake transport."""
        import json
        import time

        from platformdirs import user_cache_dir

        from folivox.voicegen.cloud import urllib_transport

        path = Path(user_cache_dir("folivox", appauthor=False)) / "fish-voices" / f"{self.library}-{lang or 'all'}.json"
        if self.transport is urllib_transport:
            try:
                cached = json.loads(path.read_text("utf-8"))
                if time.time() - cached["time"] < self.CACHE_TTL:
                    return cached["rows"]
            except (OSError, ValueError, KeyError):
                pass
        q = {"page_size": 100, "sort_by": "task_count",
             **({"licensed": "true"} if self.library == "licensed" else {}), **({"language": lang} if lang else {})}
        items = self.call("GET", f"{self.BASE}/model?{urllib.parse.urlencode(q)}").json().get("items", [])
        items.sort(key=lambda v: (v.get("author") or {}).get("nickname") != self.OFFICIAL)  # house voices first
        rows = [self.row(v) for v in items if v.get("type") == "tts" and v.get("state") == "trained" and v.get("_id")]
        if self.transport is urllib_transport and rows:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"time": time.time(), "rows": rows}), "utf-8")
            except OSError:
                pass
        return rows

    def row(self, v: dict) -> dict:
        tags = [str(t) for t in v.get("tags") or []]
        low = {t.lower() for t in tags}
        official = (v.get("author") or {}).get("nickname") == self.OFFICIAL
        sample = (v.get("samples") or [{}])[0]
        return {"id": v["_id"], "lang": (v.get("languages") or [""])[0], "name": v.get("title") or v["_id"],
                "sex": "f" if "female" in low else "m" if "male" in low else "",
                "traits": " · ".join((["Fish Official"] if official else []) + [
                    t for t in tags if t.lower() not in ("male", "female")])[:90],
                "description": (v.get("description") or "")[:300], "preview_url": sample.get("audio") or ""}

    def speakers(self, lang=None):
        key = (self.library, lang or "")
        cache = self.__dict__.setdefault("_by_lang", {})
        if key not in cache:
            out: dict[str, tuple[str, str]] = {}
            for vid, (name, sex) in self.FEATURED.get(lang or "", {}).items():
                out[vid] = (lang or "", sex)
                self.names[vid] = name
                self.details[vid] = {"traits": "Drama 3 example voice", "description": "", "preview_url": ""}
            own = self.call("GET", f"{self.BASE}/model?{urllib.parse.urlencode({'page_size': 100, 'self': 'true'})}")
            rows = [self.row(v) for v in own.json().get("items", [])
                    if v.get("type") == "tts" and v.get("state") == "trained" and v.get("_id")]
            for r in rows + self.library_rows(lang):  # the account's own voices first, then the library
                out.setdefault(r["id"], (r["lang"], r["sex"]))
                self.names.setdefault(r["id"], r["name"])
                self.details.setdefault(r["id"], {k: r[k] for k in ("traits", "description", "preview_url")})
            cache[key] = out
        return cache[key]

    def create_voice(self, title: str, audio: bytes, filename: str, text: str = "", enhance: bool = True) -> str:
        """A private, instantly trained voice model from one clip; ``text`` is its transcript (else Fish runs ASR)."""
        fields = [("type", "tts"), ("title", title), ("train_mode", "fast"), ("visibility", "private"),
                  ("enhance_audio_quality", "true" if enhance else "false")]
        if text:
            fields.append(("texts", text))
        ctype = "audio/wav" if audio[:4] == b"RIFF" else "application/octet-stream"
        body = self.call("POST", f"{self.BASE}/model", raw=multipart(fields, [("voices", filename, audio, ctype)]),
                         timeout=180).json()
        if not body.get("_id"):
            raise CloudError("Fish Audio did not return a voice id", "server")
        if body.get("state") == "failed":
            raise CloudError("Fish Audio could not train a voice from this sample", "bad_request")
        return body["_id"]

    def setup_voices(self, jobs):
        if self.mode == "preset":
            return []
        made: dict[tuple, str] = {}
        name = f"folivox-{uuid.uuid4().hex[:8]}"
        for j in jobs:
            self.check_lang(j.lang)
            key = (j.description, j.lang) if self.mode == "design" else (str(j.reference), j.lang)
            if key not in made:
                if self.mode == "design":
                    made[key] = self.design(j, name)
                else:
                    if not j.reference or not Path(j.reference).exists():
                        raise CloudError("Fish Audio clone: choose one of your own voices as the sample",
                                         "bad_request")
                    made[key] = self.create_voice(name, file_as_wav(Path(j.reference)), "voice.wav")
            j.speaker = made[key]
        return list(dict.fromkeys(made.values()))

    def design(self, j: Job, name: str) -> str:
        """Voice Design returns short stateless candidates; the first becomes a reusable model, so every
        fragment of the book is read by the same voice."""
        if len(j.description.strip()) < 10:
            raise CloudError("Fish Audio Voice Design needs a description of at least a few words", "bad_request")
        payload = {"instruction": j.description.strip()[:2000], "n": 1, **({"language": j.lang} if j.lang else {})}
        data = self.call("POST", f"{self.BASE}/v1/voice-design", payload, headers={"model": self.DESIGN_MODEL}).json()
        cands = data.get("candidates") or []
        if not cands or not cands[0].get("audio_base64"):
            raise CloudError("Fish Audio returned no voice candidate for the description", "server")
        return self.create_voice(name, b64(cands[0]["audio_base64"]), "voice.wav", cands[0].get("text") or "",
                                 enhance=False)

    def teardown_voices(self, ids):
        for i in ids:
            self.call("DELETE", f"{self.BASE}/model/{urllib.parse.quote(i, safe='')}")

    # ---- synthesis -------------------------------------------------------------------------------
    def synth(self, job: Job) -> bytes:
        if not job.speaker:
            raise CloudError("Fish Audio: pick a voice (a voice id from the library or your account)", "bad_request")
        text = job.text
        if job.description:
            text = f"[{job.description.strip('[] ')}] {text}"
        # Folivox has already expanded numbers and abbreviations for the fragment's language, so Fish's own
        # normalisation stays off; PCM avoids a streamed WAV header of unknown length.
        payload = {"text": text, "reference_id": job.speaker, "format": "pcm", "sample_rate": 24000,
                   "normalize": False, **({"language": job.lang} if job.lang else {})}
        body = self.call("POST", f"{self.BASE}/v1/tts", payload, headers={"model": self.model}).body
        if not body:
            raise CloudError("Fish Audio returned no audio", "server")
        return pcm_to_wav(body, 24000)

"""Commercial cloud TTS engines: ElevenLabs, OpenAI, Gemini API, Cartesia, Fish Audio.

A thin adapter over the cloud *voice generators* in :mod:`folivox.voicegen.cloud_apis`, so the HTTP plumbing
(retries with ``Retry-After``, error mapping, ``model=auto``, key lookup) is shared. Differences from the local
engine: the voice is a provider voice id (:class:`CloudVoice`) instead of a reference recording, requests run
in a few parallel threads, and every request is billed, so the engine counts characters sent.

Everything sent here leaves the computer and is billed to the user's account. The audio is kept on this
computer and must never be bundled with Folivox. Add an engine by writing a ``CloudGenerator`` and calling
:func:`register_cloud_engine` (see CONTRIBUTING).
"""

from __future__ import annotations

import io
import threading
import time
from dataclasses import dataclass

import numpy as np

from folivox.providers import ProviderError
from folivox.tts.base import TTSEngine
from folivox.voicegen import Job

NEUTRAL_STYLE = "Neutral, steady audiobook narration. Read the text exactly as written."
# provider → default narration note (ElevenLabs reads bracketed tags as direction, so none by default)
DEFAULT_STYLE = {"openai": NEUTRAL_STYLE, "gemini": NEUTRAL_STYLE}
# Neither OpenAI nor Gemini has a language parameter: the language comes from the text, and their voices
# are English-native, so a non-English book gets an English accent unless the note asks for a native one.
ACCENT_NOTE = ("Speak {language} like a native speaker, with natural {language} pronunciation and intonation, "
               "never an English accent.")
NO_INSTRUCTIONS = ("tts-1", "tts-1-hd")  # OpenAI models that ignore the note
PARALLEL = {"elevenlabs": 3, "openai": 3, "gemini": 1, "cartesia": 3, "fish": 3}  # simultaneous requests per provider
# Requests per minute a run starts, per provider (absent = no limit of our own; the provider's 429 still applies).
# Google does not publish per-minute numbers for its TTS models and they differ by tier and model, so Gemini starts
# gently and the limiter slows down further whenever a 429 arrives.
DEFAULT_RPM = {"gemini": 10.0}
CHARS_PER_SECOND = 14  # narration pace used for audio-duration estimates
SPEED_NOTE = "Reading speed is not applied by cloud engines; the provider's own pace is used."


class RateLimiter:
    """Spaces the starts of requests evenly (60 / rpm seconds apart), which keeps any per-minute window under
    the limit. :meth:`slow_down` halves the rate after the provider said "too many requests"."""

    def __init__(self, rpm: float | None, clock=time.monotonic, sleep=time.sleep):
        self.rpm = rpm or None
        self.clock, self.sleep = clock, sleep
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self, should_stop=lambda: False) -> bool:
        """Block until this request may start (a cancel ends the wait early); True if it had to wait."""
        if not self.rpm:
            return False
        with self._lock:
            now = self.clock()
            start = max(now, self._next)
            self._next = start + 60.0 / self.rpm
        waited = False
        while (left := start - self.clock()) > 0 and not should_stop():
            waited = True
            self.sleep(min(left, 1.0))
        return waited

    def slow_down(self) -> float:
        """Halve the rate (never below one request a minute) and return the new rpm."""
        with self._lock:
            base = self.rpm or 60.0
            self.rpm = max(1.0, base / 2)
            self._next = max(self._next, self.clock()) + 60.0 / self.rpm
            return self.rpm


class CostCapReached(ProviderError):
    """The per-run spending cap would be exceeded; progress is saved and the run can be resumed."""


@dataclass(frozen=True)
class CloudVoice:
    """What the pipeline passes instead of a reference recording."""

    provider: str
    model: str
    speaker: str = ""  # provider voice id (preset) or the id of a voice made for this book
    style: str = ""
    mode: str = "preset"  # preset | design | localize | clone
    description: str = ""
    wav = None
    transcript = ""
    duration = 0.0
    warnings: tuple = ()

    def warnings_for(self, engine: str | None) -> list[str]:
        return list(self.warnings)

    @property
    def id(self) -> str:
        # cache key part: anything that changes how the text sounds must be here (never the temporary voice id)
        import hashlib

        h = hashlib.sha1(f"{self.mode}|{self.speaker}|{self.style}|{self.description}".encode()).hexdigest()[:12]
        return f"{self.provider}:{self.model}:{h}"


class CloudTTS(TTSEngine):
    local = False

    def __init__(self, provider: str, model: str = "auto", generator=None, parallel: int | None = None,
                 max_cost: float | None = None, rpm: float | None = None, **options):
        self.name = provider
        if generator is None:
            generator = generator_class(provider)(model=model or "auto", **{
                k: v for k, v in options.items() if k in ("transport", "sleep", "api_key", "api_key_env")})
        self.gen = generator
        from folivox.config import cloud_limits

        limits = cloud_limits(provider)
        self.parallel = parallel or limits.get("parallel") or PARALLEL.get(provider, 2)
        rpm = rpm if rpm is not None else limits.get("requests_per_minute", DEFAULT_RPM.get(provider))
        self.limiter = RateLimiter(rpm or None)
        self.max_cost = max_cost
        self.languages = tuple(type(generator).languages)
        self.sample_rate = 24000
        self.device = "cloud"
        self._lock = threading.Lock()
        self.chars_sent = 0
        self.requests = 0
        self.outage_seconds = 1800.0  # how long a dropped network / provider outage is waited out before failing
        self.wait_for = 30.0  # pause between probes during an outage
        self.sleep = time.sleep
        self.limiter.sleep = lambda s: self.sleep(s)
        self.should_stop = lambda: False  # set by the pipeline: cancel during a pause
        self.on_wait = lambda message: None  # set by the pipeline: tell the user we are waiting

    # ---- facts the pipeline and UI need -----------------------------------------------------------
    @property
    def model(self) -> str:
        return self.gen.model

    @property
    def max_chars(self) -> int:
        limits = getattr(type(self.gen), "LIMITS", None)
        return (limits or {}).get(self.model, self.gen.max_chars)

    @property
    def fragment_chars(self) -> int:
        """Fragment size used when the user did not choose one."""
        return min(getattr(self.gen, "fragment_chars", 300), self.max_chars)

    def estimate_usd(self, chars: int, mode: str = "preset") -> float | None:
        chars = int(chars * type(self.gen).bytes_factor(self.lang))
        return type(self.gen).estimate_for(chars, 1, mode, self.model).get("usd")

    @property
    def spent_usd(self) -> float:
        return self.estimate_usd(self.chars_sent) or 0.0

    def check_budget(self, next_chars: int = 0) -> None:
        """Raise :class:`CostCapReached` when sending ``next_chars`` more would pass the cap."""
        if self.max_cost is None:
            return
        after = self.estimate_usd(self.chars_sent + next_chars)
        if after is not None and after > self.max_cost:
            raise CostCapReached(f"Spending cap ${self.max_cost:.2f} reached (≈ ${self.spent_usd:.2f} spent, "
                                 f"{self.chars_sent:,} characters sent). Progress is saved: raise --max-cost and "
                                 "run the same command again to continue.")

    # ---- synthesis --------------------------------------------------------------------------------
    def job(self, text: str, voice: CloudVoice, seed: int, lang: str = "") -> Job:
        return Job(text=text, lang=lang or self.lang, seed=seed, speaker=voice.speaker,
                   description=self.note_for(voice, lang))

    lang = "en"

    def note_for(self, voice: CloudVoice, lang: str = "") -> str:
        """The narration note of a fragment. OpenAI and Gemini have no language parameter, so the "no English
        accent" note names the language the fragment is read in; the others take the language as a request field
        (Cartesia) or from the text."""
        if not lang or lang == self.lang or self.name not in DEFAULT_STYLE or self.model in NO_INSTRUCTIONS:
            return voice.style
        from folivox import lang as languages

        note = voice.style
        if self.lang != "en":
            note = note.replace(ACCENT_NOTE.format(language=languages.get(self.lang).name), "").strip()
        if lang != "en":
            note = f"{ACCENT_NOTE.format(language=languages.get(lang).name)} {note}".strip()
        return note

    def prepare(self, text, voice, seed=0, lang=""):
        """The whole request happens here (in a worker thread); :meth:`finish` only decodes."""
        from folivox.voicegen.cloud import CloudError

        self.gen.check_text(text)
        billed = self.gen.billed_chars(self.job(text, voice, seed, lang)) if hasattr(self.gen, "billed_chars") \
            else len(text)
        with self._lock:  # checked and counted atomically: parallel workers must not overshoot the cap together
            self.check_budget(billed)
            self.chars_sent += billed  # counted when sent: an in-flight request is already billed
            self.requests += 1
        t0 = time.time()
        # A fragment is paid for once: retries below re-send the same request, so the characters are counted
        # once, and refunded when the provider definitely refused the text (nothing was billed).
        waited = 0.0
        while True:
            if self.limiter.wait(self.should_stop) and self.should_stop():
                raise CloudError("Stopped while waiting for the provider's request limit.", "network")
            try:
                wav = self.gen.synth(self.job(text, voice, seed, lang))
                return wav, time.time() - t0
            except CloudError as e:
                if e.kind == "rate_limit":
                    # the provider counted too many requests: go slower for the rest of the run
                    rpm = self.limiter.slow_down()
                    e = CloudError(f"{e} Folivox now sends at most {rpm:g} requests a minute.", e.kind, e.retry_after)
                if e.kind in ("auth", "quota", "bad_request"):
                    with self._lock:
                        self.chars_sent -= billed
                    raise
                if e.kind == "server":
                    # a 5xx was already re-sent twice with growing pauses; going on could bill the fragment again
                    raise CloudError(f"{e} Progress is saved, run the same command again to continue.",
                                     e.kind, e.retry_after) from None
                if waited >= self.outage_seconds:
                    raise CloudError(f"{e} The provider or the network was unreachable for "
                                     f"{int(waited // 60)} min; progress is saved, run the same command again to "
                                     "continue.", e.kind, e.retry_after) from None
                pause = max(self.wait_for, e.retry_after or 0)
                self.on_wait(f"{e} Waiting {int(pause)} s and trying again (progress is saved).")
                for _ in range(int(pause)):  # interruptible by cancel
                    if self.should_stop():
                        raise
                    self.sleep(1)
                waited += pause

    def finish(self, prepared, voice, speed=1.0):
        return decode(prepared, self.sample_rate), 0.0

    def synthesize_audio(self, text, voice, seed, speed):
        wav, t = self.prepare(text, voice, seed)
        return self.finish(wav, voice, speed)[0], t

    # ---- voices -----------------------------------------------------------------------------------
    def speaker_for(self, wanted: str, lang: str | None = None) -> str:
        """A provider voice id from an id or a display name (case-insensitive); unknown ids pass through, so
        voices from the user's own library work."""
        if not wanted:
            return ""
        from folivox.voicegen.cloud import CloudError

        try:
            speakers = self.gen.speakers(lang)
        except CloudError as e:
            if e.kind == "auth":
                raise
            return wanted
        if wanted in speakers:
            return wanted
        names = {n.lower(): i for i, n in getattr(self.gen, "names", {}).items()}
        return names.get(wanted.lower(), wanted)

    def voice_modes(self) -> tuple[str, ...]:
        return voice_modes_of(type(self.gen))

    def make_voice(self, wanted: str = "", style: str | None = None, lang: str = "en", mode: str = "preset",
                   description: str = "", reference=None, remembered: str = "") -> CloudVoice:
        """Resolve the voice for a run. ``mode`` design / localize / clone creates a voice in the provider's
        account (the caller deletes it with :meth:`release`); ``remembered`` is the id made by an earlier,
        interrupted run."""
        self.lang = lang
        if mode not in self.voice_modes():
            raise ProviderError(f"{type(self.gen).label} has no {mode!r} voice mode for the book "
                                f"(available: {', '.join(self.voice_modes())})")
        self.gen.check_lang(lang)
        self.gen.mode = mode
        note = DEFAULT_STYLE.get(self.name, "") if style is None else style.strip()
        if not type(self.gen).style:
            note = ""
        warnings: list[str] = []
        if self.name in DEFAULT_STYLE and lang != "en":
            from folivox import lang as languages

            if self.model in NO_INSTRUCTIONS:
                warnings.append(f"{self.model} cannot be told to avoid an English accent; for "
                                f"{languages.get(lang).name} use gpt-4o-mini-tts")
            else:
                note = f"{ACCENT_NOTE.format(language=languages.get(lang).name)} {note}".strip()
        speaker = self.speaker_for(wanted, lang) if mode in ("preset", "localize") else ""
        if mode == "preset":
            return CloudVoice(self.name, self.model, speaker, note, "preset", warnings=tuple(warnings))
        if remembered:
            return CloudVoice(self.name, self.model, remembered, note, mode, description, warnings=tuple(warnings))
        job = Job(text="", lang=lang, speaker=speaker, description=description, reference=reference)
        made = self.gen.setup_voices([job])
        if not made:
            raise ProviderError(f"{type(self.gen).label} did not create a voice")
        return CloudVoice(self.name, self.model, job.speaker, note, mode, description, warnings=tuple(warnings))

    def release(self, voice: CloudVoice) -> None:
        """Delete a voice made for this run (a failure to delete is not an error: it only costs a slot)."""
        if voice.mode == "preset" or not voice.speaker:
            return
        try:
            self.gen.teardown_voices([voice.speaker])
        except ProviderError:
            pass


def voice_modes_of(cls) -> tuple[str, ...]:
    """Modes usable for a whole book: preset, plus the ones that create a voice in the account."""
    from folivox.voicegen.cloud import CloudGenerator

    can_create = cls.setup_voices is not CloudGenerator.setup_voices
    return tuple(m for m in cls.modes if m == "preset" or can_create)


def decode(wav: bytes, rate: int = 24000) -> np.ndarray:
    import soundfile as sf

    from folivox.voicegen.cloud import to_wav

    audio, sr = sf.read(io.BytesIO(to_wav(wav)), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != rate and len(audio):
        n = int(round(len(audio) * rate / sr))
        audio = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(np.float32)
    return audio


def generator_class(provider: str):
    from folivox.voicegen.cloud import cloud_providers

    cls = cloud_providers().get(provider)
    if cls is None:
        raise ProviderError(f"No cloud engine {provider!r}")
    return cls


def make_engine(provider: str):
    def factory(model: str = "auto", **options) -> CloudTTS:
        options.pop("device", None)
        return CloudTTS(provider, model=model, **options)

    factory.__qualname__ = f"engine_{provider}"
    return factory


# One module-level factory per provider, importable as "folivox.tts.cloud:<name>".
elevenlabs, openai, gemini, cartesia, fish = (
    make_engine(n) for n in ("elevenlabs", "openai", "gemini", "cartesia", "fish"))


def register_cloud_engine(name: str, description: str, key_env: str, target: str | None = None) -> None:
    from folivox.tts import ENGINES

    ENGINES.register(name, target or f"folivox.tts.cloud:{name}", description, local=False, needs=key_env)

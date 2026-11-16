"""Cloud reading engines with a fake transport: no network, no keys."""
import json
from pathlib import Path

import pytest
from test_cloud import Fake, make, ok, wav_bytes

from folivox.journal import Journal
from folivox.pipeline import CostCapReached, Events, RenderOptions, plan, render, synthesize_pending
from folivox.providers import ProviderError
from folivox.readers import read_book
from folivox.tts import ENGINES
from folivox.tts.cloud import CloudTTS, CloudVoice
from folivox.tts.cost import book_chars, estimate
from folivox.voicegen.cloud import CloudError, Response

SENTENCE = "The morning was calm and the lake was quiet. "


def book(tmp_path, repeat=12):
    p = tmp_path / "b.txt"
    p.write_text("\n\n".join(" ".join(f"{SENTENCE.strip()[:-1]} {i}.{k}." for k in range(6))
                              for i in range(repeat)), encoding="utf-8")
    return read_book(p)


def engine(provider="openai", fake=None, **kw):
    from folivox.voicegen.cloud_apis import Cartesia, ElevenLabs, Gemini, OpenAIVoice

    cls = {"openai": OpenAIVoice, "elevenlabs": ElevenLabs, "gemini": Gemini, "cartesia": Cartesia}[provider]
    fake = fake or Fake(ok(wav_bytes()))
    return CloudTTS(provider, generator=make(cls, fake), **kw), fake


class Spy(Events):
    def __init__(self):
        self.spend_log = []

    def spend(self, chars, usd, cap):
        self.spend_log.append((chars, usd, cap))


def run(tmp_path, eng, voice, **kw):
    opts = RenderOptions(output=tmp_path / "o.m4b", qa=False, lang="en", tts=eng.name, **kw)
    j = Journal(tmp_path / "w")
    ev = Spy()
    plan(book(tmp_path), opts, j, voice.id, ev, eng)
    return opts, j, ev


def test_registered_as_cloud_engines():
    for name, env in (("elevenlabs", "ELEVENLABS_API_KEY"), ("openai", "OPENAI_API_KEY"),
                      ("gemini", "GEMINI_API_KEY"), ("cartesia", "CARTESIA_API_KEY"), ("fish", "FISH_API_KEY")):
        p = ENGINES.get(name)
        assert not p.local and p.needs == env


def test_request_is_built_from_voice_and_style():
    eng, fake = engine("openai")
    voice = eng.make_voice("coral", None, "en")
    assert voice.style.startswith("Neutral")  # default narration note
    audio, _ = eng.synthesize_audio("Hello there.", voice, 1, 1.0)
    body = fake.calls[0]["json"]
    assert body["voice"] == "coral" and body["input"] == "Hello there." and "instructions" in body
    assert len(audio) > 1000 and eng.chars_sent == 12 and eng.requests == 1


def test_elevenlabs_has_no_default_tag_and_resolves_names():
    fake = Fake(ok({"voices": [{"voice_id": "abc", "name": "Rachel", "labels": {"gender": "female"}}]}))
    eng, _ = engine("elevenlabs", fake)
    voice = eng.make_voice("rachel", None, "en")
    assert voice.speaker == "abc" and voice.style == ""


def test_chunks_respect_provider_limit(tmp_path):
    eng, _ = engine("openai")
    eng.gen.max_chars = 120  # pretend the provider is strict
    voice = eng.make_voice("coral", None, "en")
    opts, j, _ = run(tmp_path, eng, voice, max_chars=400)
    assert max(len(c.text) for c in j.chunks()) <= 120


def test_cost_estimate_and_cap_resume(tmp_path):
    est = estimate("elevenlabs", 100_000, "eleven_v4", takes=2, qa=True)
    assert est["usd"] > est["per_hour"] * 0 and est["usd_max"] >= est["usd"] and est["approximate"]
    assert estimate("elevenlabs", 100_000, "eleven_v4", qa=False)["likely_chars"] == 100_000
    assert book_chars(book(tmp_path), selected=[1], preview=50) <= 50

    eng, fake = engine("openai", max_cost=0.012)  # roughly 600 characters
    voice = eng.make_voice("coral", None, "en")
    opts, j, ev = run(tmp_path, eng, voice)
    with pytest.raises(CostCapReached):
        synthesize_pending(j, eng, voice, None, opts, ev)
    done = len(j.chunks()) - len(j.pending())
    assert 0 < done < len(j.chunks()) and ev.spend_log
    sent = len(fake.calls)
    # resume with a higher cap: fragments already rendered are not requested again
    eng2, fake2 = engine("openai", max_cost=10)
    synthesize_pending(j, eng2, voice, None, opts, ev)
    assert not j.pending() and len(fake2.calls) == len(j.chunks()) - done
    assert sent >= done


def test_cache_key_changes_with_voice_model_style(tmp_path):
    a, _ = engine("openai")
    ids = {a.make_voice("coral", None, "en").id, a.make_voice("onyx", None, "en").id,
           a.make_voice("coral", "whisper softly", "en").id}
    assert len(ids) == 3
    b, _ = engine("openai")
    b.gen.requested = "tts-1"
    assert b.make_voice("coral", None, "en").id not in ids


def test_errors_are_clear_and_never_leak_the_key():
    fake = Fake(Response(401, b'{"error": {"message": "bad key"}}', {}))
    eng, _ = engine("openai", fake)
    voice = eng.make_voice("coral", None, "en")
    with pytest.raises(CloudError) as e:
        eng.synthesize_audio("Hi.", voice, 0, 1.0)
    assert e.value.kind == "auth" and "sk-test-SECRET" not in str(e.value)


def test_429_retry_after_is_honoured():
    slept = []
    from folivox.voicegen.cloud_apis import OpenAIVoice

    fake = Fake(Response(429, b"{}", {"retry-after": "7"}), ok(wav_bytes()))
    gen = OpenAIVoice(api_key="k", transport=fake, sleep=slept.append, model="tts-1")
    eng = CloudTTS("openai", generator=gen)
    eng.synthesize_audio("Hi.", eng.make_voice("alloy", None, "en"), 0, 1.0)
    assert slept == [7.0]


def test_render_refuses_without_confirmation(tmp_path):
    opts = RenderOptions(output=tmp_path / "o.m4b", tts="openai", qa=False)
    with pytest.raises(ProviderError, match="paid cloud engine"):
        render(book(tmp_path), opts)


def test_designed_voice_is_remembered_and_released():
    fake = Fake(ok({"id": "v-1"}), ok({}))
    from folivox.voicegen.cloud_apis import Gemini

    gen = Gemini(api_key="k", transport=fake, sleep=lambda s: None, model="gemini-3.8-flash-tts")
    eng = CloudTTS("gemini", generator=gen)
    v = eng.make_voice("", None, "en", "design", "a calm, low, warm male narrator")
    assert v.speaker == "v-1" and v.mode == "design"
    again = eng.make_voice("", None, "en", "design", "a calm, low, warm male narrator", remembered="v-1")
    assert again.id == v.id and len(fake.calls) == 1  # no second voice is created on resume
    eng.release(v)
    assert fake.calls[-1]["method"] == "DELETE"


def test_unsupported_mode_is_rejected():
    eng, _ = engine("openai")
    with pytest.raises(ProviderError):
        eng.make_voice("coral", None, "en", "clone")


def test_json_body_has_no_key():
    eng, fake = engine("openai")
    eng.synthesize_audio("Hi.", eng.make_voice("coral", None, "en"), 0, 1.0)
    assert "sk-test" not in json.dumps(fake.calls[0]["json"])


def test_elevenlabs_design_saves_one_voice_for_the_whole_book_and_deletes_it():
    fake = Fake(ok({"previews": [{"generated_voice_id": "gen-1"}, {"generated_voice_id": "gen-2"}]}),
                ok({"voice_id": "v-9"}), ok({"status": "ok"}))
    eng, _ = engine("elevenlabs", fake)
    v = eng.make_voice("", None, "en", "design", "a calm, warm woman in her forties, unhurried narrator")
    assert v.speaker == "v-9" and v.mode == "design"
    design, save = fake.calls[0], fake.calls[1]
    assert design["url"].endswith("/v1/text-to-voice/design") and design["json"]["auto_generate_text"] is True
    assert save["url"].endswith("/v1/text-to-voice") and save["json"]["generated_voice_id"] == "gen-1"
    eng.release(v)
    assert fake.calls[-1]["method"] == "DELETE" and fake.calls[-1]["url"].endswith("/v1/voices/v-9")


def test_elevenlabs_design_needs_a_real_description():
    eng, _ = engine("elevenlabs")
    with pytest.raises(CloudError):
        eng.make_voice("", None, "en", "design", "calm")


@pytest.fixture
def saved_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("folivox.tts.saved._path", lambda: tmp_path / "cloud_voices.json")
    return tmp_path


def gemini_engine(fake, model="gemini-3.8-flash-tts", **kw):
    from folivox.voicegen.cloud_apis import Gemini

    kw.setdefault("rpm", 0)  # the default limiter would really sleep between requests
    return CloudTTS("gemini", generator=Gemini(api_key="k", transport=fake, sleep=lambda s: None, model=model), **kw)


def test_kept_voice_is_listed_and_reused_without_creating_or_deleting(saved_dir):
    from folivox.pipeline import _cloud_voice
    from folivox.tts import saved

    fake = Fake(ok({"id": "v-77"}))
    eng = gemini_engine(fake)
    opts = RenderOptions(output=saved_dir / "o.m4b", tts="gemini", tts_mode="design", lang="en", keep_voice=True,
                         tts_description="a calm, warm woman in her forties", tts_voice_name="Calm narrator")
    j = Journal(saved_dir / "w")
    voice, owned = _cloud_voice(eng, opts, j, Events())
    assert owned and voice.speaker == "v-77"
    entry = saved.find("gemini", "calm narrator")  # found by name, case-insensitive
    assert entry and entry["id"] == "v-77" and entry["mode"] == "design"

    # a later book: the same voice by name, nothing is created and it is never deleted by the run
    fake2 = Fake(ok({}))
    eng2 = gemini_engine(fake2)
    opts2 = RenderOptions(output=saved_dir / "o2.m4b", tts="gemini", tts_voice="Calm narrator", lang="en")
    voice2, owned2 = _cloud_voice(eng2, opts2, Journal(saved_dir / "w2"), Events())
    assert voice2.speaker == "v-77" and voice2.id == voice.id and not owned2 and not fake2.calls


def test_unkept_voice_is_not_listed(saved_dir):
    from folivox.pipeline import _cloud_voice
    from folivox.tts import saved

    eng = gemini_engine(Fake(ok({"id": "v-5"})))
    opts = RenderOptions(output=saved_dir / "o.m4b", tts="gemini", tts_mode="design", lang="en",
                         tts_description="a calm, warm woman in her forties")
    _cloud_voice(eng, opts, Journal(saved_dir / "w"), Events())
    assert saved.listing("gemini") == []


def test_non_english_books_ask_for_a_native_accent():
    """OpenAI and Gemini have no language parameter and English-native voices: the note must ask for the accent."""
    eng, fake = engine("openai")
    v = eng.make_voice("coral", None, "ru")
    assert "Russian like a native speaker" in v.style and v.style.endswith("exactly as written.")
    eng.synthesize_audio("Доброе утро.", v, 0, 1.0)
    assert "never an English accent" in fake.calls[0]["json"]["instructions"]
    en = eng.make_voice("coral", None, "en")
    assert "accent" not in en.style and en.id != v.id  # English: unchanged note; the cache key differs per language
    own = eng.make_voice("coral", "warm and slow", "de")
    assert "German like a native speaker" in own.style and own.style.endswith("warm and slow")


def test_openai_tts1_cannot_fix_the_accent_and_says_so():
    from folivox.voicegen.cloud_apis import OpenAIVoice

    gen = OpenAIVoice(api_key="k", transport=Fake(ok(wav_bytes())), sleep=lambda s: None, model="tts-1")
    eng = CloudTTS("openai", generator=gen)
    v = eng.make_voice("coral", None, "ru")
    assert v.warnings and "gpt-4o-mini-tts" in v.warnings[0] and "accent" not in v.style
    assert not eng.make_voice("coral", None, "en").warnings


def test_kept_voice_without_a_name_gets_a_readable_one_never_an_id():
    from types import SimpleNamespace

    from folivox.pipeline import _default_voice_name

    eng = SimpleNamespace(gen=SimpleNamespace(names={"db6b0ed5-id": "Skylar"}))
    loc = RenderOptions(output=Path("o.m4b"), tts="cartesia", tts_mode="localize", tts_voice="db6b0ed5-id", lang="ru")
    assert _default_voice_name(eng, loc) == "Skylar · Russian"
    unknown = RenderOptions(output=Path("o.m4b"), tts="cartesia", tts_mode="localize", tts_voice="zzz", lang="ru")
    assert "zzz" not in _default_voice_name(eng, unknown)
    des = RenderOptions(output=Path("o.m4b"), tts="gemini", tts_mode="design", lang="en",
                        tts_description="a calm, warm woman in her forties, unhurried narrator")
    assert _default_voice_name(eng, des).startswith("a calm, warm woman")


GEMINI_429 = {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED",
                        "message": "You exceeded your current quota, please check your plan and billing details.",
                        "details": [{"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [
                            {"quotaId": "GenerateRequestsPerMinutePerProjectPerModel"}]},
                                    {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "27s"}]}}


GVOICE = CloudVoice("gemini", "gemini-2.5-flash-preview-tts", "Kore")
GEMINI_OK = ok({"candidates": [{"content": {"parts": [{"inlineData": {
    "data": __import__("base64").b64encode(wav_bytes()).decode()}}]}}]})


def test_gemini_retry_delay_from_the_body_is_honoured():
    slept = []
    from folivox.voicegen.cloud_apis import Gemini

    fake = Fake(ok(GEMINI_429, status=429), GEMINI_OK)
    gen = Gemini(api_key="k", transport=fake, sleep=slept.append, model="gemini-2.5-flash-preview-tts")
    eng = CloudTTS("gemini", generator=gen, rpm=0)
    eng.synthesize_audio("Hi.", GVOICE, 0, 1.0)
    assert slept == [27.0]


def test_gemini_daily_limit_stops_with_a_clear_message():
    body = {"error": {"status": "RESOURCE_EXHAUSTED", "message": "quota",
                      "details": [{"violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel"}]}]}}
    fake = Fake(ok(body, status=429))
    eng = gemini_engine(fake, "gemini-2.5-flash-preview-tts")
    with pytest.raises(CloudError) as e:
        eng.synthesize_audio("Hi.", GVOICE, 0, 1.0)
    assert e.value.kind == "quota" and "daily request limit" in str(e.value) and "midnight Pacific" in str(e.value)
    assert len(fake.calls) == 1  # not retried: waiting a minute cannot help


def test_per_minute_limit_slows_the_run_down_and_names_the_limit():
    waits = []
    fake = Fake(*[ok(GEMINI_429, status=429)] * 5, GEMINI_OK)
    eng = gemini_engine(fake, "gemini-2.5-flash-preview-tts", rpm=8)
    eng.sleep = lambda s: None
    eng.on_wait = waits.append
    eng.synthesize_audio("Hi.", GVOICE, 0, 1.0)
    assert eng.limiter.rpm == 4
    assert "per-minute request limit" in waits[0] and "at most 4 requests a minute" in waits[0]


def test_limiter_spaces_request_starts():
    from folivox.tts.cloud import RateLimiter

    now = [0.0]
    sleeps = []

    def sleep(s):
        sleeps.append(s)
        now[0] += s

    lim = RateLimiter(6, clock=lambda: now[0], sleep=sleep)  # one request every 10 s
    for _ in range(3):
        lim.wait()
    assert now[0] == 20.0
    assert lim.slow_down() == 3 and RateLimiter(None).slow_down() == 30


def test_gemini_defaults_are_gentle_and_configurable(monkeypatch):
    eng = gemini_engine(Fake(ok(wav_bytes())), rpm=None)
    assert eng.parallel == 1 and eng.limiter.rpm == 10
    monkeypatch.setenv("FOLIVOX_GEMINI_RPM", "60")
    monkeypatch.setenv("FOLIVOX_GEMINI_PARALLEL", "3")
    eng = gemini_engine(Fake(ok(wav_bytes())), rpm=None)
    assert eng.parallel == 3 and eng.limiter.rpm == 60
    monkeypatch.setenv("FOLIVOX_GEMINI_RPM", "0")
    assert gemini_engine(Fake(ok(wav_bytes())), rpm=None).limiter.rpm is None
    assert engine("openai")[0].limiter.rpm is None


def test_parallel_and_rate_can_be_set_per_run(monkeypatch, tmp_path):
    from folivox.tts.cloud import CloudTTS

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("FOLIVOX_CONFIG", str(tmp_path / "none.toml"))
    default = CloudTTS("gemini")
    assert default.parallel == 1 and default.limiter.rpm == 10
    fast = CloudTTS("gemini", parallel=4, rpm=60)
    assert fast.parallel == 4 and fast.limiter.rpm == 60
    unlimited = CloudTTS("gemini", rpm=0)  # 0 removes our own limit, as in config.toml
    assert unlimited.limiter.rpm is None

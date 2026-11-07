"""Fish Audio engines: catalog, hardware choice, licence consent, the worker protocol and a render, all with fakes."""

import copy
import hashlib
import json
import shutil
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from folivox import config
from folivox.journal import chunk_key
from folivox.providers import ProviderError
from folivox.tts import builds, fish_check, fish_setup, licences
from folivox.tts.builds import CatalogError, Hardware

MAC_36 = Hardware("darwin", "arm64", 36.0, 30.0)
MAC_16 = Hardware("darwin", "arm64", 16.0, 9.0)
MAC_64 = Hardware("darwin", "arm64", 64.0, 50.0)
LINUX = Hardware("linux", "x86_64", 64.0, 40.0)


@pytest.fixture(autouse=True)
def private_config(tmp_path, monkeypatch):
    monkeypatch.setenv("FOLIVOX_CONFIG", str(tmp_path / "config.toml"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setattr(builds, "engines_root", lambda: tmp_path / "engines")
    # Build selection depends on the machine; pin it so the tests give the same result on any CI runner.
    monkeypatch.setattr(builds, "detect_hardware", lambda: MAC_64)
    config.load.cache_clear()
    yield
    config.load.cache_clear()


def raw_catalog() -> dict:
    return json.loads(builds.CATALOG_FILE.read_text(encoding="utf-8"))


# -- catalog ----------------------------------------------------------------------------------------------

def test_bundled_catalog_is_valid_and_pinned():
    cat = builds.load()
    assert {b.engine for b in cat.builds} == {"fish-s2", "fish-drama3"}
    for b in cat.builds:
        assert len(b.revision) == 40
        assert all(f["path"].endswith((".safetensors", ".gguf", ".json", ".md", ".jinja", "LICENSE", "NOTICE",
                                       ".gitattributes")) for f in b.files)
        assert sum(f["size"] for f in b.files) == b.size_bytes
        assert cat.licences[b.licence].text_sha256
    # a build nobody ran on the standard fragment must not be recommended
    assert cat.build("drama3-mlx-3bit").status == "verified"  # run on the standard fragment 2026-10-17


def test_catalog_refuses_unsafe_entries():
    for mutate, why in (
            (lambda b: b["files"].append({"path": "codec.pth", "size": 1, "sha256": "a"}), "safetensors"),
            (lambda b: b.update(revision="main"), "pinned"),
            (lambda b: b["files"][1].update(sha256=None), "sha256"),
            (lambda b: b["files"].append({"path": "../evil.json", "size": 1, "sha256": None}), "unsafe"),
            (lambda b: b.update(licence="nope"), "licence"),
    ):
        data = copy.deepcopy(raw_catalog())
        b = data["builds"][0]
        # make sure file 1 is a weights file with a hash
        b["files"] = [f for f in b["files"] if f["path"].endswith(".safetensors")][:2] + b["files"][:0]
        mutate(b)
        with pytest.raises(CatalogError, match=why):
            builds.parse(data)


# -- hardware choice --------------------------------------------------------------------------------------

def test_fit_levels():
    d4 = builds.load().build("drama3-mlx-4bit")
    assert builds.fit(d4, MAC_64).level == "ok"
    assert builds.fit(d4, MAC_36).level == "tight"  # 30 GB free < 33 GB peak: warn, do not forbid
    assert builds.fit(d4, MAC_16).level == "no"
    assert builds.fit(d4, LINUX).level == "no"
    assert "swapping" in builds.fit(d4, MAC_36).message


def test_memory_gate_blocks_only_when_weights_do_not_fit(monkeypatch):
    d4 = builds.load().build("drama3-mlx-4bit")
    assert builds.memory_gate(d4, Hardware("darwin", "arm64", 36.0, 30.0)) is None
    why = builds.memory_gate(d4, Hardware("darwin", "arm64", 36.0, 12.0))
    assert why and "Close heavy applications" in why
    monkeypatch.setenv(builds.IGNORE_MEMORY_ENV, "1")
    assert builds.memory_gate(d4, Hardware("darwin", "arm64", 36.0, 12.0)) is None


def test_candidates_are_verified_and_ranked():
    ids = [b.id for b in builds.candidates(MAC_64)]
    assert ids == ["drama3-mlx-4bit", "drama3-mlx-3bit", "s2pro-mlx-8bit"]
    assert [b.id for b in builds.candidates(Hardware("darwin", "arm64", 24.0, 20.0))] == ["s2pro-mlx-8bit"]
    assert builds.candidates(LINUX) == []


def test_recommend_needs_licence_installation_and_memory():
    cat = builds.load()
    accepted_all = {k: {"version": v.version, "sha256": v.text_sha256} for k, v in cat.licences.items()}
    everything = lambda b: True  # noqa: E731
    assert builds.recommend(MAC_64, accepted_all, everything).id == "drama3-mlx-4bit"
    # the licence of the best build is not accepted → the next one whose licence is
    only_research = {"fish-research": accepted_all["fish-research"]}
    assert builds.recommend(MAC_64, only_research, everything).id == "s2pro-mlx-8bit"
    assert builds.recommend(MAC_64, {}, everything) is None
    assert builds.recommend(MAC_64, accepted_all, lambda b: False) is None
    assert builds.recommend(LINUX, accepted_all, everything) is None


def test_default_engine_falls_back_to_cosyvoice():
    assert builds.default_engine(MAC_64) == "cosyvoice3"  # nothing accepted or installed in the test config


# -- licence consent --------------------------------------------------------------------------------------

def test_accept_stores_version_and_hash_and_asks_again_when_it_changes(tmp_path):
    lic = builds.load().licences["fish-community"]
    assert not licences.is_accepted(lic)
    rec = licences.accept("fish-community")
    assert rec["version"] == lic.version and rec["sha256"] == lic.text_sha256
    assert licences.is_accepted(lic)
    text = config.config_path().read_text(encoding="utf-8")
    assert "[consent.fish-community]" in text and lic.version in text
    newer = builds.Licence(lic.id, lic.name, "2.0", lic.commercial, lic.text_url, lic.text_sha256, lic.page)
    assert not licences.is_accepted(newer)
    changed_text = builds.Licence(lic.id, lic.name, lic.version, lic.commercial, lic.text_url, "0" * 64, lic.page)
    assert not licences.is_accepted(changed_text)
    licences.revoke("fish-community")
    assert not licences.is_accepted(lic)


def test_require_raises_until_accepted():
    b = builds.load().build("s2pro-mlx-8bit")
    with pytest.raises(licences.LicenceNotAccepted, match="has not been accepted"):
        licences.require(b)
    licences.accept(b.licence)
    licences.require(b)


def test_licence_text_is_checked_against_the_catalog_hash(tmp_path, monkeypatch):
    lic = builds.load().licences["fish-research"]
    monkeypatch.setattr(licences, "user_data_dir", lambda *a, **k: str(tmp_path / "lic"))
    body = b"the licence"
    forged = builds.Licence(lic.id, lic.name, lic.version, lic.commercial, lic.text_url,
                            hashlib.sha256(body).hexdigest(), lic.page)

    class Resp:
        def __init__(self, data):
            self.data = data

        def read(self):
            return self.data

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    assert licences.fetch_text(forged, opener=lambda req, timeout: Resp(body)) == "the licence"
    # cached now: no network needed
    assert licences.fetch_text(forged, opener=lambda *a, **k: 1 / 0) == "the licence"
    other = builds.Licence(lic.id, lic.name, lic.version, lic.commercial, lic.text_url, "f" * 64, lic.page)
    with pytest.raises(licences.LicenceError, match="does not match"):
        licences.fetch_text(other, opener=lambda req, timeout: Resp(body))


def test_console_prompt_flow():
    said = []
    answers = iter(["read", "maybe"])
    assert not licences.prompt_accept("fish-research", ask=lambda p: next(answers), show=said.append)
    assert any("Do NOT sell" in s for s in said)
    assert not licences.is_accepted(builds.load().licences["fish-research"])
    assert licences.prompt_accept("fish-research", ask=lambda p: "yes", show=said.append)
    assert licences.is_accepted(builds.load().licences["fish-research"])


def test_research_licence_points_forbid_selling():
    points = dict(licences.POINTS["fish-research"])
    assert "sell" in points["no_selling"].lower()
    assert "USD 500,000" in dict(licences.POINTS["fish-community"])["commercial"]


# -- cache keys -------------------------------------------------------------------------------------------

def test_chunk_key_keeps_old_keys_and_separates_speakers_and_markup():
    base = chunk_key("text", "v", "s")
    assert base == hashlib.sha1(b"s\x1fv\x1ftext").hexdigest()[:24]  # unchanged: old caches stay valid
    assert chunk_key("text", "v", "s", speaker="1") != base
    assert chunk_key("text", "v", "s", markup="[sighs]") != base
    assert chunk_key("text", "v", "s", speaker="1") != chunk_key("text", "v", "s", speaker="2")
    assert chunk_key("text", "v", "fish-s2|fish-1|a") != chunk_key("text", "v", "fish-drama3|fish-1|a")


# -- checks before a render -------------------------------------------------------------------------------

class Ev:
    def __init__(self):
        self.warnings = []

    def warning(self, m):
        self.warnings.append(m)

    def stage(self, *a):
        pass


def opts(**kw):
    from folivox.pipeline import RenderOptions

    return RenderOptions(output=Path("x.m4b"), tts="fish-s2", voice_wav=Path("/tmp/mine.wav"), **kw)


def ready(monkeypatch, hw=MAC_64, installed=True):
    monkeypatch.setattr(fish_setup, "is_ready", lambda b: installed)
    monkeypatch.setattr(builds, "detect_hardware", lambda: hw)


def test_check_requires_licence_then_install(monkeypatch):
    ready(monkeypatch, installed=False)
    with pytest.raises(licences.LicenceNotAccepted):
        fish_check.check(opts(), Ev())
    licences.accept("fish-research")
    with pytest.raises(FileNotFoundError, match="folivox engines install s2pro-mlx-8bit"):
        fish_check.check(opts(), Ev())


def test_check_warns_about_non_commercial_attribution_and_memory(monkeypatch):
    licences.accept("fish-research")
    ready(monkeypatch, hw=Hardware("darwin", "arm64", 36.0, 14.0))
    ev = Ev()
    fish_check.check(opts(), ev)
    text = " ".join(ev.warnings)
    assert "Selling them" in text and "Built with Fish Audio" in text and "swapping" in text


def test_check_refuses_when_the_weights_do_not_fit(monkeypatch):
    licences.accept("fish-research")
    ready(monkeypatch, hw=Hardware("darwin", "arm64", 36.0, 3.0))
    with pytest.raises(ProviderError, match="Close heavy applications"):
        fish_check.check(opts(), Ev())


def test_check_refuses_bundled_voices_unless_allowed(monkeypatch, tmp_path):
    from folivox import voices

    licences.accept("fish-research")
    ready(monkeypatch)
    bundled = voices.bundled()[0].path
    with pytest.raises(ProviderError, match="real LibriVox readers"):
        fish_check.check(opts().__class__(output=Path("x.m4b"), tts="fish-s2", voice_wav=bundled), Ev())
    (tmp_path / "config.toml").write_text(config.config_path().read_text(encoding="utf-8")
                                          + "\n[fish]\nallow_bundled_voices = true\n", encoding="utf-8")
    config.load.cache_clear()
    fish_check.check(opts().__class__(output=Path("x.m4b"), tts="fish-s2", voice_wav=bundled), Ev())


def test_check_wrong_platform(monkeypatch):
    licences.accept("fish-research")
    ready(monkeypatch, hw=LINUX)
    with pytest.raises(ProviderError, match="fits this computer"):
        fish_check.check(opts(), Ev())


# -- install verification ---------------------------------------------------------------------------------

def test_verify_weights_detects_missing_and_corrupt_files(tmp_path):
    b = builds.load().build("s2pro-mlx-8bit")
    wd = builds.weights_dir(b)
    wd.mkdir(parents=True)
    data = b"weights"
    tiny = builds.Build(**{**b.__dict__, "files": ({"path": "model.safetensors", "size": len(data),
                                                    "sha256": hashlib.sha256(data).hexdigest()},
                                                   {"path": "config.json", "size": 2, "sha256": None})})
    assert fish_setup.verify_weights(tiny) == ["model.safetensors", "config.json"]
    (wd / "model.safetensors").write_bytes(data)
    (wd / "config.json").write_text("{}")
    assert fish_setup.verify_weights(tiny) == []
    (wd / "model.safetensors").write_bytes(b"WEIGHTS")  # same size, other content
    assert fish_setup.verify_weights(tiny) == ["model.safetensors"]


# -- engine, worker protocol and a render with a fake worker ----------------------------------------------

FAKE_WORKER = textwrap.dedent('''
    import json, os, sys, time
    import numpy as np
    mode = os.environ.get("FAKE_MODE", "ok")
    for line in sys.stdin:
        req = json.loads(line)
        if req["cmd"] == "quit":
            break
        if req["cmd"] == "load":
            print(json.dumps({"ok": True, "sample_rate": 16000, "load_s": 0.1, "warm_s": 0.0}), flush=True)
        elif req["cmd"] == "synth":
            if mode == "hang":
                time.sleep(60)
            if mode == "crash":
                os._exit(9)
            if mode == "error":
                print(json.dumps({"ok": False, "error": "RuntimeError: boom"}), flush=True)
                continue
            n = 16000 * 2
            audio = (0.3 * np.sin(np.linspace(0, 2000, n))).astype(np.float32)
            np.save(req["out"], audio)
            print(json.dumps({"ok": True, "id": req["id"], "out": req["out"], "audio_s": 2.0, "seconds": 0.01,
                              "peak_gb": 1.5}), flush=True)
''')


@pytest.fixture
def fake_worker(tmp_path, monkeypatch):
    from folivox.tts import fish

    script = tmp_path / "fake_worker.py"
    script.write_text(FAKE_WORKER)
    monkeypatch.setattr(fish, "WORKER", script)
    monkeypatch.setattr(fish_setup, "venv_python", lambda rt: Path(sys.executable))
    monkeypatch.setattr(fish_setup, "is_ready", lambda b: True)
    monkeypatch.setattr(fish, "to_reference_wav", lambda src, dst: Path(src))
    licences.accept("fish-research")
    return script


def make_voice(tmp_path):
    import wave

    from folivox.tts.voice import Voice

    wav = tmp_path / "ref.wav"
    pcm = (3000 * np.sin(np.linspace(0, 3000, 16000 * 8))).astype("<i2")
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm.tobytes())
    return Voice(wav, "Привет.", 8.0)


def test_engine_round_trip_and_close(fake_worker, tmp_path):
    from folivox.tts.fish import FishS2Engine

    eng = FishS2Engine(build="auto")
    assert eng.build.id == "s2pro-mlx-8bit" and eng.local and not hasattr(eng, "max_chars")
    voice = make_voice(tmp_path)
    eng.start(voice)
    assert eng.sample_rate == 16000
    audio, secs = eng.finish(eng.prepare("Привет", voice, 1)[0], voice)
    assert audio.dtype == np.float32 and len(audio) == 32000 and secs == pytest.approx(0.01)
    assert eng.peak_gb == 1.5
    worker = eng._worker
    eng.close()
    assert worker.proc.poll() is not None  # the process (and its memory) is gone


@pytest.mark.parametrize("mode,exc,match", [("error", "FishError", "boom"), ("crash", "FishError", "exit code 9")])
def test_engine_reports_worker_failures(fake_worker, tmp_path, monkeypatch, mode, exc, match):
    from folivox.tts import fish

    monkeypatch.setenv("FAKE_MODE", mode)
    eng = fish.FishS2Engine()
    with pytest.raises(getattr(fish, exc), match=match):
        eng.synthesize(" text", make_voice(tmp_path))
    eng.close()


def test_engine_stops_a_hung_worker(fake_worker, tmp_path, monkeypatch):
    from folivox.tts import fish
    from folivox.tts.base import SynthTimeout

    monkeypatch.setenv("FAKE_MODE", "hang")
    monkeypatch.setenv("FOLIVOX_SYNTH_TIMEOUT", "1")
    eng = fish.FishS2Engine()
    with pytest.raises(SynthTimeout, match="did not answer"):
        eng.synthesize("text", make_voice(tmp_path))
    assert eng._worker.proc.poll() is not None
    eng.close()


def test_engine_needs_licence_and_installation(monkeypatch):
    from folivox.tts import fish

    monkeypatch.setattr(fish_setup, "is_ready", lambda b: False)
    with pytest.raises(licences.LicenceNotAccepted):
        fish.FishS2Engine()
    licences.accept("fish-research")
    with pytest.raises(FileNotFoundError, match="not installed"):
        fish.FishS2Engine()


def test_engine_rejects_a_build_of_another_engine():
    from folivox.tts import fish

    with pytest.raises(CatalogError, match="belongs to the engine"):
        fish.FishDrama3Engine(build="s2pro-mlx-8bit")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is needed to build the audiobook")
def test_render_with_the_fake_worker(fake_worker, tmp_path, monkeypatch):
    from folivox.book import Block, BlockKind, Book, Section
    from folivox.pipeline import Events, RenderOptions, render

    monkeypatch.setattr(builds, "detect_hardware", lambda: MAC_64)
    voice = make_voice(tmp_path)
    book = Book(title="T", lang="ru", sections=[Section(title="Глава", blocks=[
        Block(BlockKind.PARAGRAPH, "Первая фраза книги. Вторая фраза книги.")])])

    class Rec(Events):
        def __init__(self):
            self.warnings, self.stages = [], []

        def warning(self, m):
            self.warnings.append(m)

        def stage(self, name, detail=""):
            self.stages.append((name, detail))

    ev = Rec()
    out = render(book, RenderOptions(output=tmp_path / "b.m4b", tts="fish-s2", voice_wav=voice.wav,
                                     voice_text=voice.transcript, qa=False, stress="none", work_dir=tmp_path / "w"), ev)
    assert out.exists()
    assert any("Selling them" in w for w in ev.warnings)
    assert any("Warming up" in d for _, d in ev.stages)
    from folivox.journal import Journal

    j = Journal(tmp_path / "w")
    assert j.meta("settings").startswith("fish-s2|fish-1|s2pro-mlx-8bit@")
    j.close()


def test_automatic_pick_is_reliable_and_sounds_good():
    cat = builds.load()
    nothing, everything, only4 = (lambda b: False), (lambda b: True), (lambda b: b.id == "drama3-mlx-4bit")  # noqa: E731

    def drama(total, installed, free=1.0):  # the free memory right now never matters
        return builds.auto_build("fish-drama3", Hardware("darwin", "arm64", total, free), cat, installed).id

    assert drama(64.0, nothing) == "drama3-mlx-4bit"  # plenty of room: the most capable build
    assert drama(36.0, nothing) == "drama3-mlx-3bit"  # 4-bit would peak at 92% of the memory: not for a newcomer
    assert drama(36.0, everything) == "drama3-mlx-3bit"
    assert drama(36.0, only4) == "drama3-mlx-4bit"  # installed and tight: kept, the render warns
    assert drama(36.0, everything, free=30.0) == drama(36.0, everything, free=2.0)
    assert builds.auto_build("fish-drama3", Hardware("darwin", "arm64", 24.0, 20.0), cat) is None
    assert builds.auto_build("fish-s2", Hardware("darwin", "arm64", 16.0, 9.0), cat) is None


def test_fish_engines_read_rule_less_languages():
    """Local Fish models read the Fish API's languages: a Dutch book or fragment is not sent back to the book
    language (issue #8)."""
    from folivox.pipeline import engine_languages

    speaks = engine_languages("fish-s2")
    assert speaks is not None and {"nl", "pt", "ru", "en"} <= speaks
    assert "nl" in engine_languages("fish-drama3")

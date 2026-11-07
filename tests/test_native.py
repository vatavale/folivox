"""The optional native runtime (audio.cpp): catalog, build choice, safe install, runtime choice, the server engine."""

import hashlib
import io
import json
import shutil
import sys
import tarfile
import types
import wave
import zipfile
from pathlib import Path

import numpy as np
import pytest

from folivox import config, uninstall
from folivox.providers import ProviderError
from folivox.tts import native, runtime
from folivox.tts import native_setup as ns
from folivox.tts.voice import Voice

FAKE = Path(__file__).with_name("native_fakes.py")


@pytest.fixture(autouse=True)
def private(tmp_path, monkeypatch):
    for k in ("FOLIVOX_RUNTIME", ns.SERVER_ENV, ns.MODEL_ENV):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("FOLIVOX_CONFIG", str(tmp_path / "config.toml"))
    monkeypatch.setenv(ns.DIR_ENV, str(tmp_path / "native"))
    config.load.cache_clear()
    yield
    native.shutdown_all()
    config.load.cache_clear()


def write_config(tmp_path, text):
    (tmp_path / "config.toml").write_text(text, encoding="utf-8")
    config.load.cache_clear()


# -- catalog and build choice ------------------------------------------------------------------------------------

def test_catalog_is_pinned():
    cat = ns.load()
    assert cat.release.startswith("v") and cat.download_base.endswith(f"/{cat.release}/")
    assert cat.auto_ok == []  # nothing is the default until its quality gate is recorded
    for b in cat.binaries:
        assert b.assets and all(len(a.sha256) == 64 and a.size > 0 for a in b.assets)
        assert all(a.name.startswith(f"audio-{cat.release}-") for a in b.assets)
    for w in cat.weights.values():
        assert len(w.revision) == 40 and len(w.sha256) == 64 and w.path.endswith(".gguf")


@pytest.mark.parametrize("tag,gpu,expect", [
    ("darwin-arm64", None, "macos-arm64-metal"),
    ("win32-x64", (591, 8.9), "windows-x64-cuda13.3"),
    ("win32-x64", (550, 8.9), "windows-x64-cuda12.4"),
    ("win32-x64", (591, 6.1), "windows-x64-vulkan"),  # Pascal: older than the CUDA builds support
    ("win32-x64", None, "windows-x64-vulkan"),
    ("linux-x64", None, "ubuntu-x64-vulkan-portable"),
    ("linux-x64", (595, 8.6), "ubuntu-x64-cuda12.8"),  # also in Docker, where NVIDIA Vulkan often fails (#8)
    ("linux-x64", (595, 6.1), "ubuntu-x64-vulkan-portable"),
    ("darwin-x64", None, None),
])
def test_build_choice(tag, gpu, expect):
    b = ns.pick_binary("auto", tag, nvidia=lambda: gpu)
    assert (b.id if b else None) == expect


def test_explicit_backend():
    assert ns.pick_binary("vulkan", "win32-x64").id == "windows-x64-vulkan"
    assert ns.pick_binary("metal", "win32-x64") is None


# -- safe unpacking and install ----------------------------------------------------------------------------------

def make_tar(path: Path, members: dict[str, bytes], links: dict[str, str] | None = None):
    with tarfile.open(path, "w:gz") as t:
        for name, data in members.items():
            i = tarfile.TarInfo(name)
            i.size = len(data)
            t.addfile(i, io.BytesIO(data))
        for name, target in (links or {}).items():
            i = tarfile.TarInfo(name)
            i.type, i.linkname = tarfile.SYMTYPE, target
            t.addfile(i)


def test_extract_skips_sources_and_refuses_escapes(tmp_path):
    ok = tmp_path / "ok.tar.gz"
    make_tar(ok, {"./audiocpp_server": b"x", "./src/big.cpp": b"y", "./model_specs/a.json": b"{}"})
    ns.safe_extract(ok, tmp_path / "out")
    assert (tmp_path / "out" / "audiocpp_server").exists() and not (tmp_path / "out" / "src").exists()
    bad = tmp_path / "bad.tar.gz"
    make_tar(bad, {"../evil": b"x"})
    with pytest.raises(ns.NativeError):
        ns.safe_extract(bad, tmp_path / "out2")
    assert not (tmp_path / "evil").exists()
    link = tmp_path / "link.tar.gz"
    make_tar(link, {"a": b"x"}, {"b": "../../etc/passwd"})
    with pytest.raises(ns.NativeError):
        ns.safe_extract(link, tmp_path / "out3")
    z = tmp_path / "bad.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("../evil.txt", "x")
    with pytest.raises(ns.NativeError):
        ns.safe_extract(z, tmp_path / "out4")
    assert not (tmp_path / "evil.txt").exists()


def fake_release(monkeypatch, tmp_path, server_name=None):
    """A one-asset catalog entry whose 'download' is built locally; returns the Binary."""
    data = tmp_path / "asset.tar.gz"
    make_tar(data, {f"./{server_name or ns.SERVER_NAME}": b"#!/bin/sh\n", "./model_specs/x.json": b"{}"})
    a = ns.Asset("audio-v0-bin-test.tar.gz", data.stat().st_size, hashlib.sha256(data.read_bytes()).hexdigest())
    b = ns.Binary("test-metal", "darwin-arm64", "metal", "untested", "", (a,))
    monkeypatch.setattr(ns, "platform_tag", lambda: "darwin-arm64")
    monkeypatch.setattr(ns, "candidates", lambda tag=None: [b])
    monkeypatch.setattr(ns, "version_text", lambda srv: "audio.cpp 0\nbackends: cpu,metal\n")

    def fetch(url, dest, asset, progress):
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(data, dest)
    return b, a, fetch


def test_install_verifies_and_registers_engine(tmp_path, monkeypatch):
    b, a, fetch = fake_release(monkeypatch, tmp_path)
    assert ns.server_path() is None and ns.status().problem
    p = ns.Plan(b, None, a.size, [])
    ns.install(p, None, lambda m: None, fetch=fetch)
    assert ns.server_path() is not None and ns.server_path().name == ns.SERVER_NAME
    assert ns.installed_binary()[0].id == "test-metal"
    assert ns.remove() > 0 and ns.server_path() is None


def test_download_checksum_mismatch_is_refused(tmp_path, monkeypatch):
    class R(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(ns.urllib.request, "urlopen", lambda *a, **k: R(b"not what was pinned"))
    asset = ns.Asset("x.zip", 19, "0" * 64)
    with pytest.raises(ns.NativeError, match="checksum"):
        ns._download("https://example.invalid/x.zip", tmp_path / "x.zip", asset, None)
    assert not (tmp_path / "x.zip").exists()


def test_engine_reporting_wrong_backend_is_not_installed(tmp_path, monkeypatch):
    b, a, fetch = fake_release(monkeypatch, tmp_path)
    monkeypatch.setattr(ns, "version_text", lambda srv: "backends: cpu")
    with pytest.raises(ns.NativeError, match="metal"):
        ns.install(ns.Plan(b, None, a.size, []), None, lambda m: None, fetch=fetch)
    assert ns.installed_binary() is None


def test_plan_lists_only_what_is_missing(tmp_path, monkeypatch):
    fake_release(monkeypatch, tmp_path)
    p = ns.plan("q8_0")
    assert p.binary is not None and p.weights is not None and len(p.lines) == 2
    assert p.todo_bytes == p.binary.size + ns.load().weights["q8_0"].size
    with pytest.raises(ns.NativeError):
        ns.plan("fp8")
    monkeypatch.setenv(ns.SERVER_ENV, str(FAKE))
    monkeypatch.setenv(ns.MODEL_ENV, str(FAKE))  # a build and a model of your own
    assert ns.plan("q8_0").lines == []


# -- which runtime ------------------------------------------------------------------------------------------------

def test_runtime_precedence_and_validation(tmp_path, monkeypatch):
    assert config.runtime() == "auto"
    write_config(tmp_path, '[runtime]\ninference = "native"\n')
    assert config.runtime() == "native"
    monkeypatch.setenv("FOLIVOX_RUNTIME", "pytorch")
    assert config.runtime() == "pytorch"
    assert config.runtime("native") == "native"
    with pytest.raises(ValueError):
        config.runtime("gpu")


def test_resolve_keeps_pytorch_unless_asked(tmp_path, monkeypatch):
    assert runtime.resolve("cosyvoice3") == "pytorch"  # auto, nothing recorded as checked
    assert runtime.resolve("cosyvoice3", "native") == "native"
    assert runtime.resolve("cosyvoice3", "pytorch") == "pytorch"
    assert runtime.resolve("fish-s2", "native") == "native"  # Fish: picks the native GGUF builds
    assert runtime.resolve("fish-s2") == "pytorch"  # never automatically
    assert runtime.resolve("elevenlabs", "native") == "pytorch"  # a cloud engine has nothing to switch
    b, a, fetch = fake_release(monkeypatch, tmp_path)
    ns.install(ns.Plan(b, None, a.size, []), None, lambda m: None, fetch=fetch)
    monkeypatch.setenv(ns.MODEL_ENV, str(FAKE))
    assert runtime.resolve("cosyvoice3") == "pytorch"
    monkeypatch.setattr(ns.load(), "auto_ok", ["darwin-arm64"])
    assert runtime.resolve("cosyvoice3") == "native"


def test_require_ready_explains(tmp_path):
    with pytest.raises(ProviderError, match="folivox runtime install"):
        runtime.require_ready()


def test_take_key_differs_from_pytorch(tmp_path):
    from folivox.pipeline import SYNTH_VERSION

    s = runtime.take_settings(1.0)
    assert SYNTH_VERSION not in s and s.startswith("cv3-native-") and ns.load().release in s
    assert runtime.take_settings(1.0) != runtime.take_settings(0.9)
    write_config(tmp_path, '[runtime.native]\nweight_type = "f32"\n')
    assert runtime.take_settings(1.0) != s


# -- the engine with a fake server ----------------------------------------------------------------------------------

@pytest.fixture
def fake_engine(tmp_path, monkeypatch):
    srv, model = tmp_path / "audiocpp_server", tmp_path / "m.gguf"
    srv.write_bytes(b"x")
    model.write_bytes(b"x")
    monkeypatch.setenv(ns.SERVER_ENV, str(srv))
    monkeypatch.setenv(ns.MODEL_ENV, str(model))
    monkeypatch.setattr(native, "server_command", lambda server: [sys.executable, str(FAKE)])
    monkeypatch.setattr(native.Server, "log_tail", staticmethod(lambda n=600: ""))
    monkeypatch.setenv("FAKE_LOG", str(tmp_path / "requests.jsonl"))
    wav = tmp_path / "ref.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(b"\0\0" * 24000 * 6)
    monkeypatch.setattr(native, "reference_wav", lambda voice: voice.wav)
    voice = Voice(wav, "Привет, мир.", 6.0)

    def requests():
        p = tmp_path / "requests.jsonl"
        return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()] if p.exists() else []
    return voice, requests


def test_engine_synthesises_through_the_server(fake_engine):
    voice, requests = fake_engine
    eng = native.NativeCosyVoiceEngine(backend="auto", weights="q8_0", weight_type="f16")
    try:
        assert eng.device.startswith("native:")
        prepared, _ = eng.prepare("Привет, мир! Это проверка.", voice, 7)
        audio, secs = eng.finish(prepared, voice)
        assert audio.dtype == np.float32 and eng.sample_rate == 24000
        assert abs(len(audio) / 24000 - 26 * 0.05) < 0.01
        r = requests()[0]
        assert r["seed"] == 7 and r["reference_text"] == "Привет, мир." and r["voice_ref"] == str(voice.wav)
        assert r["options"]["template_name"] == "zero_shot" and r["model"] == "cosyvoice3"
    finally:
        eng.close()


def test_server_is_shared_and_kept_between_renders(fake_engine):
    voice, requests = fake_engine
    a = native.NativeCosyVoiceEngine()
    a.synthesize("Один.", voice)
    a.close()
    b = native.NativeCosyVoiceEngine()  # within the idle time: the same process, no second model load
    b.synthesize("Два.", voice)
    b.close()
    assert len({r["pid"] for r in requests()}) == 1
    native.shutdown_all()
    assert not a._server.alive()


def test_crashed_server_is_restarted_once(fake_engine, tmp_path, monkeypatch):
    voice, requests = fake_engine
    monkeypatch.setenv("FAKE_MODE", "crash_once")
    monkeypatch.setenv("FAKE_MARK", str(tmp_path / "crashed"))
    eng = native.NativeCosyVoiceEngine()
    try:
        s = eng.synthesize("Сначала упадёт.", voice, seed=1)
        assert len(s.audio) > 0
        assert len({r["pid"] for r in requests()}) == 2
    finally:
        eng.close()


def test_server_error_is_explained(fake_engine, monkeypatch):
    voice, _ = fake_engine
    monkeypatch.setenv("FAKE_MODE", "error")
    eng = native.NativeCosyVoiceEngine()
    try:
        with pytest.raises(ns.NativeError, match="bad reference"):
            eng.synthesize("Текст.", voice)
    finally:
        eng.close()


def test_stuck_server_is_killed(fake_engine, monkeypatch):
    from folivox.tts.base import SynthTimeout

    voice, _ = fake_engine
    monkeypatch.setenv("FAKE_MODE", "hang")
    monkeypatch.setattr(native, "synth_timeout", lambda chars: 1.0)
    eng = native.NativeCosyVoiceEngine()
    try:
        with pytest.raises(SynthTimeout):
            eng.synthesize("Зависнет.", voice)
        assert not eng._server.alive()
    finally:
        eng.close()


def test_engine_without_install_says_what_to_do():
    with pytest.raises(ns.NativeError, match="folivox runtime install"):
        native.NativeCosyVoiceEngine()


def test_bad_weight_type_is_refused(fake_engine):
    with pytest.raises(ns.NativeError, match="weight_type"):
        native.NativeCosyVoiceEngine(weight_type="int3")


def test_load_engine_picks_native(fake_engine, tmp_path):
    from folivox.pipeline import Events, RenderOptions, load_engine

    eng = load_engine(RenderOptions(output=tmp_path / "o.m4b", runtime="native"), Events())
    try:
        assert isinstance(eng, native.NativeCosyVoiceEngine)
    finally:
        eng.close()


def test_decode_wav_handles_stereo_and_rejects_garbage():
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(np.array([[1000, 3000]] * 10, "<i2").tobytes())
    x, sr = native.decode_wav(buf.getvalue())
    assert sr == 16000 and x.shape == (10,) and abs(x[0] - 2000 / 32768) < 1e-6
    with pytest.raises(ns.NativeError):
        native.decode_wav(b"no audio here")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_speed_change_shortens_audio():
    x = np.sin(np.arange(24000 * 2) / 20).astype(np.float32)
    y = native.change_speed(x, 24000, 2.0)
    assert 0.45 < len(y) / len(x) < 0.55
    assert native.change_speed(x, 24000, 1.0) is x


# -- diagnostics and uninstall -------------------------------------------------------------------------------------

def test_doctor_and_uninstall_know_the_runtime(tmp_path, monkeypatch):
    from folivox import diag

    n = diag.native_summary()
    assert n["release"] == ns.load().release and "error" not in n
    assert "Native:" in diag.render_report_text({**diag.report(), "native_runtime": n})
    data = tmp_path / "data"
    (data / "native" / "bin").mkdir(parents=True)
    (data / "native" / "bin" / "f").write_bytes(b"x" * 5)
    monkeypatch.setattr(uninstall, "data_dir", lambda: data)
    assert any("Native runtime" in i.label for i in uninstall.plan())


# -- Fish models on the native runtime ------------------------------------------------------------------------------

def test_native_fish_builds_are_in_the_catalog():
    from folivox.tts import builds

    cat = builds.load()
    nat = [b for b in cat.builds if b.backend == "native"]
    assert {b.id for b in nat} == {"s2pro-native-q8", "s2pro-native-bf16", "drama3-native-q2", "drama3-native-q3"}
    for b in nat:
        assert b.status == "untested" and b.files[0]["path"].endswith(".gguf") and len(b.files[0]["sha256"]) == 64
        assert b.size_bytes == b.files[0]["size"] and set(b.platforms) >= {"darwin-arm64", "windows-x86_64"}
    assert builds.is_trusted_org("audio-cpp/audio.cpp-gguf", cat)
    assert not builds.is_trusted_org("Jmica/Drama-3-GGUF", cat)


def test_build_choice_native_only_when_asked():
    from folivox.tts import builds

    win = builds.Hardware("windows", "x86_64", 64.0, 40.0)
    mac = builds.Hardware("darwin", "arm64", 64.0, 50.0)
    assert builds.build_for("fish-s2", None, mac).backend == "mlx"  # the default stays MLX
    assert builds.build_for("fish-s2", None, mac, runtime="native").id == "s2pro-native-q8"
    assert builds.build_for("fish-drama3", None, mac, runtime="native").id.startswith("drama3-native-")
    assert builds.build_for("fish-s2", None, win, runtime="native").id == "s2pro-native-q8"
    with pytest.raises(builds.CatalogError):  # nothing tested for Windows without the native runtime
        builds.build_for("fish-s2", None, win)
    small = builds.Hardware("darwin", "arm64", 16.0, 9.0)
    with pytest.raises(builds.CatalogError):
        builds.build_for("fish-drama3", None, small, runtime="native")


def test_native_fish_engine_round_trip(fake_engine, tmp_path, monkeypatch):
    from folivox.tts import builds, fish, fish_setup, licences

    voice, requests = fake_engine
    monkeypatch.setattr(builds, "detect_hardware", lambda: builds.Hardware("darwin", "arm64", 64.0, 50.0))
    monkeypatch.setattr(fish_setup, "is_ready", lambda b: True)
    monkeypatch.setattr(licences, "require", lambda b: None)
    monkeypatch.setattr(fish, "to_reference_wav", lambda src, dst: Path(src))
    monkeypatch.setattr(builds, "weights_dir", lambda b: tmp_path / "w")
    eng = fish.FishS2Engine(build="auto", runtime="native")
    try:
        assert eng.native and eng.build.id == "s2pro-native-q8"
        s = eng.synthesize("Это Fish.", voice, seed=3)
        assert len(s.audio) > 0 and eng.sample_rate == 24000 and eng.device.startswith("native:")
        r = requests()[0]
        assert r["model"] == "fish_audio" and r["seed"] == 3 and "options" not in r
    finally:
        eng.close()


def test_native_fish_slower_than_mlx_is_said_so():
    from folivox.tts import builds

    cat = builds.load()
    assert "MLX build was faster" in builds.speed_note(cat.build("s2pro-native-bf16"), cat)
    assert builds.speed_note(cat.build("drama3-mlx-4bit"), cat) == ""
    assert builds.measured_rtf(cat.build("s2pro-native-q8")) == 2.52


def test_sources_and_build_recipes_are_recorded():
    src = ns.sources()
    assert src["drama3"]["commit"] and len(src["drama3"]["commit"]) == 40 and src["drama3_fork"]["commit"]
    assert {"macos", "linux-cuda", "linux-vulkan", "windows-cuda", "windows-vulkan", "amd-hip"} <= set(src["build"])
    from folivox.tts import native_build

    assert any("build_metal.sh" in x for x in native_build.platform_recipe("darwin-arm64"))
    assert any("build_windows.ps1" in x for x in native_build.platform_recipe("win32-x64"))
    assert any("CUDA_ARCHITECTURES" in x for x in native_build.platform_recipe("linux-x64"))
    assert ns.load().recheck["after_days"] == 60


def test_own_build_is_found(tmp_path, monkeypatch):
    from folivox.tts import native_build

    monkeypatch.delenv(ns.SERVER_ENV, raising=False)
    assert native_build.own_server() is None
    d = native_build.own_dir("drama3")
    d.mkdir(parents=True)
    (d / ns.SERVER_NAME).write_bytes(b"x")
    (d / ns.READY).write_text('{"commit": "x"}')
    assert native_build.own_server() == d / ns.SERVER_NAME
    assert ns.own_server_for("fish_drama3") == d / ns.SERVER_NAME and ns.own_server_for("fish_audio") is None
    assert ns.remove() >= 0 and native_build.own_server() is None


def test_build_refuses_where_it_was_not_run(monkeypatch):
    from folivox.tts import native_build

    monkeypatch.setattr(ns, "platform_tag", lambda: "linux-x64")
    assert "build-help" in native_build.can_build_here()
    with pytest.raises(ns.NativeError):
        native_build.build()


def test_cli_build_help_and_status_say_where_things_come_from():
    from typer.testing import CliRunner

    from folivox.cli_runtime import runtime_app

    r = CliRunner().invoke(runtime_app, ["build-help", "linux-cuda"])
    assert r.exit_code == 0 and "CUDA" in r.output and "0xShug0/audio.cpp" in r.output
    assert "Jmica/Drama-3-GGUF" in r.output
    assert CliRunner().invoke(runtime_app, ["build-help", "beos"]).exit_code == 2
    s = CliRunner().invoke(runtime_app, [])
    assert s.exit_code == 0 and ("measure" in s.output.lower() or "faster" in s.output.lower())


def test_bench_report_is_shareable_and_private():
    from folivox import bench

    rows = [{"device": "native", "actual": "native:metal", "rtf": 0.6, "chars_per_s": 20.0, "load_s": 14,
             "warmup_s": 5, "runtime": "audio.cpp (native)", "detail": "v0.9.4-macos-arm64-metal, backend metal",
             "audio_s": 33.0, "synth_s": 19.8, "chars": 440, "ram_peak_gb": 2.5, "ram_where": "engine process",
             "gpu_card_peak_gb": 3.2},
            {"device": "cuda", "error": "no kernel image"}]
    text = bench.report_text(rows, "cosyvoice3", "ru")
    assert "RTF 0.60" in text and "FAILED no kernel image" in text and "audio.cpp release" in text
    assert "native — OK, audio.cpp (native)" in text and "1.7x faster than real time" in text
    assert "NVIDIA card +3.2 GB" in text and "RAM peak 2.5 GB (engine process)" in text
    assert "/Users/" not in text and "token" not in text.lower()


_VULKAN_LIST = """ggml_vulkan: Found 2 Vulkan devices:
ggml_vulkan: 0 = AMD Radeon(TM) Graphics (AMD proprietary driver) | uma: 1
available_devices=3
Vulkan:0 "AMD Radeon(TM) Graphics" [IGPU]
Vulkan:1 "NVIDIA GeForce RTX 3050 Laptop GPU" [GPU]
CPU:0 "AMD Ryzen 9 5900HX with Radeon Graphics        " [CPU]
select with: --backend <cuda|hip|vulkan|metal|cpu> --device <index>
"""


def _fake_run(out, code=0):
    def run(cmd, **kw):
        assert "--list-devices" in cmd
        return types.SimpleNamespace(returncode=code, stdout=out, stderr="")
    return run


def test_list_devices_parses_only_the_asked_backend(tmp_path):
    devs = ns.list_devices(tmp_path / "srv", "vulkan", run=_fake_run(_VULKAN_LIST))
    assert [(d.index, d.kind) for d in devs] == [(0, "IGPU"), (1, "GPU")]
    assert devs[1].name == "NVIDIA GeForce RTX 3050 Laptop GPU"


def test_pick_device_prefers_discrete_over_integrated(tmp_path):
    devs = ns.list_devices(tmp_path / "srv", "vulkan", run=_fake_run(_VULKAN_LIST))
    assert ns.pick_device(devs).index == 1
    assert ns.pick_device(devs, "0").index == 0  # an explicit number wins
    assert ns.pick_device(devs, "7") is None
    assert ns.pick_device(devs[:1]).index == 0  # only an integrated GPU: use it


def test_choose_device_env_and_backends(tmp_path, monkeypatch):
    lister = lambda server, be: ns.list_devices(server, be, run=_fake_run(_VULKAN_LIST))  # noqa: E731
    monkeypatch.delenv(ns.DEVICE_ENV, raising=False)
    assert ns.choose_device(tmp_path / "srv", "vulkan", "auto", lister=lister) == 1
    assert ns.choose_device(tmp_path / "srv", "cuda", "auto", lister=lister) is None
    monkeypatch.setenv(ns.DEVICE_ENV, "0")
    assert ns.choose_device(tmp_path / "srv", "vulkan", "auto", lister=lister) == 0


def test_server_passes_device(tmp_path, monkeypatch):
    seen = {}

    class P:
        def poll(self): return 1
        returncode = 1

    def popen(cmd, **kw):
        seen["cmd"] = cmd
        return P()

    monkeypatch.setattr(native.subprocess, "Popen", popen)
    s = native.Server(native.Spec(tmp_path / "srv", tmp_path / "m.gguf", "vulkan", device=1))
    with pytest.raises(ns.NativeError):
        s.start()
    assert seen["cmd"][seen["cmd"].index("--device") + 1] == "1"


def test_list_devices_unknown_output_is_none(tmp_path):
    assert ns.list_devices(tmp_path / "srv", "vulkan", run=_fake_run("something new\n")) is None


def test_require_gpu_explains_a_vulkan_without_gpu(tmp_path):
    none_found = "available_devices=1\nCPU:0 \"Some CPU\" [CPU]\n"
    lister = lambda server, be: ns.list_devices(server, be, run=_fake_run(none_found))  # noqa: E731
    with pytest.raises(ns.NativeError, match="no vulkan driver for any GPU"):
        ns.require_gpu(tmp_path / "srv", "vulkan", lister=lister, seen=lambda: None)
    with pytest.raises(ns.NativeError, match="RTX 3070.*no vulkan driver for it"):
        ns.require_gpu(tmp_path / "srv", "vulkan", lister=lister, seen=lambda: "NVIDIA RTX 3070 (seen by nvidia-smi)")
    ns.require_gpu(tmp_path / "srv", "cuda", lister=lister)  # other backends are not checked
    ns.require_gpu(tmp_path / "srv", "vulkan", lister=lambda s, b: None)  # unknown output: do not block
    ok = lambda server, be: ns.list_devices(server, be, run=_fake_run(_VULKAN_LIST))  # noqa: E731
    ns.require_gpu(tmp_path / "srv", "vulkan", lister=ok)


@pytest.mark.parametrize("code", [-4, 132, 0xC0000005 - 2**32])
def test_a_crashing_device_listing_means_no_gpu(tmp_path, code):
    lister = lambda server, be: ns.list_devices(server, be, run=_fake_run("", code))  # noqa: E731
    assert lister(tmp_path / "srv", "vulkan") == []
    with pytest.raises(ns.NativeError, match="no vulkan driver"):
        ns.require_gpu(tmp_path / "srv", "vulkan", lister=lister, seen=lambda: None)


def test_bench_report_says_slower_than_real_time_and_unmeasured_memory():
    from folivox import bench

    text = bench.report_text([{"device": "cpu", "actual": "cpu", "rtf": 2.4, "chars_per_s": 5.0, "load_s": 9,
                               "warmup_s": 3}], "cosyvoice3", "en")
    assert "2.4x SLOWER than real time" in text and "not measured for this device" in text


def test_gpu_watch_reports_the_peak_above_the_start():
    from folivox import bench

    seq = iter([1000, 1000, 3048, 2500, 2000, 2000, 2000, 2000])
    w = bench.GpuWatch(every=0.01, read=lambda: next(seq, 2000))
    w.start()
    import time as _t

    _t.sleep(0.2)
    assert abs(w.finish() - 2.0) < 1e-6  # 3048 - 1000 MiB
    assert bench.GpuWatch(read=lambda: None).finish() is None


def test_bench_header_names_the_nvidia_card_without_pytorch(monkeypatch):
    from folivox import bench, diag

    monkeypatch.setattr(diag, "hardware", lambda: {"cores": 8, "ram_gb": 16.0, "gpu": None, "vram_gb": None})
    monkeypatch.setattr(bench, "nvidia_cards", lambda: "NVIDIA GeForce RTX 3070 8.0 GB, driver 561.09")
    text = bench.report_text([{"device": "cpu", "error": "x"}], "cosyvoice3", "en")
    assert "GPU: NVIDIA GeForce RTX 3070 8.0 GB, driver 561.09" in text


def test_server_env_puts_bundled_cuda_libraries_first(tmp_path, monkeypatch):
    monkeypatch.setattr(ns.sys, "platform", "linux")
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib" / "libcudart.so.12").write_bytes(b"")
    env = ns.server_env(tmp_path / "audiocpp_server", base={"LD_LIBRARY_PATH": "/opt/x"})
    parts = env["LD_LIBRARY_PATH"].split(":")
    assert parts[0] == str(tmp_path / "lib") and parts[-1] == "/opt/x"
    monkeypatch.setattr(ns.sys, "platform", "win32")
    assert "LD_LIBRARY_PATH" not in ns.server_env(tmp_path / "audiocpp_server", base={})


@pytest.mark.parametrize("model_gb,gpu_gb,expect", [
    (15.88, 16.0, 9),  # Q2 (GiB) on a 16 GB card: upstream starts at 8
    (18.93, 16.0, 15),  # Q3: 12 upstream
    (15.88, 12.0, 21),  # RTX 3060 12 GB (#8)
    (15.88, 24.0, 0),
    (15.88, None, None),
])
def test_drama3_cpu_expert_layers(model_gb, gpu_gb, expect):
    assert ns.drama3_cpu_expert_layers(model_gb, gpu_gb) == expect


def test_drama3_options_follow_setting(tmp_path, monkeypatch):
    monkeypatch.delenv("FOLIVOX_DRAMA3_CPU_EXPERT_LAYERS", raising=False)
    gguf = tmp_path / "d.gguf"
    gguf.write_bytes(b"")
    assert ns.drama3_options(gguf, "metal", None) == ()
    assert ns.drama3_options(gguf, "metal", None, configured=6) == (("fish_drama3.cpu_expert_layers", 6),)
    monkeypatch.setenv("FOLIVOX_DRAMA3_CPU_EXPERT_LAYERS", "12")
    assert ns.drama3_options(gguf, "cuda", 0) == (("fish_drama3.cpu_expert_layers", 12),)


def test_native_fit_uses_gpu_memory():
    """#8: on an NVIDIA card the native builds are judged by its memory, not by the system's."""
    from folivox.tts import builds

    cat = builds.load()
    rtx3060 = builds.Hardware("linux", "x86_64", 42.0, 38.0, 12.0)
    assert builds.fit(cat.build("s2pro-native-q8"), rtx3060).level == "ok"
    assert builds.fit(cat.build("s2pro-native-bf16"), rtx3060).level == "no"
    d = builds.fit(cat.build("drama3-native-q2"), rtx3060)
    assert d.level == "tight" and "expert layers" in d.message
    assert builds.fit(cat.build("drama3-native-q2"), builds.Hardware("linux", "x86_64", 16.0, 14.0, 4.0)).level == "no"

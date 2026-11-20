"""FOLIVOX_NATIVE_THREADS: CPU threads for every audio.cpp server Folivox starts.

The engines pass no thread count of their own except CosyVoice's --threads, so audio.cpp ran with its default of 4,
and Fish Drama 3's CPU-expert layers are much faster with more. A fake ``audiocpp_server`` executable records its
argv and then serves as tests/native_fakes.py does.
"""

import json
import stat
import sys
from pathlib import Path

import pytest

from folivox.tts import native
from folivox.tts.native_setup import NativeError

FAKE = Path(__file__).with_name("native_fakes.py")


@pytest.fixture
def started(tmp_path, monkeypatch):
    """Starts a server for a spec, as the engines do, and returns the argv the server executable was given."""
    srv, argv_log = tmp_path / "bin" / "audiocpp_server", tmp_path / "argv.json"
    srv.parent.mkdir()
    srv.write_text(f"#!{sys.executable}\n"
                   "import json, os, runpy, sys\n"
                   f"open({str(argv_log)!r}, 'w').write(json.dumps(sys.argv[1:]))\n"
                   f"sys.argv[0] = {str(FAKE)!r}\n"
                   f"runpy.run_path({str(FAKE)!r}, run_name='__main__')\n", encoding="utf-8")
    srv.chmod(srv.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setattr(native.Server, "log_tail", staticmethod(lambda n=600: ""))
    monkeypatch.setenv("FOLIVOX_NATIVE_DIR", str(tmp_path / "native"))  # the server's log file goes under it
    monkeypatch.delenv("FOLIVOX_NATIVE_THREADS", raising=False)

    def start(**spec):
        s = native.acquire(native.Spec(srv, tmp_path / "m.gguf", "cpu", **spec))
        native.release(s, idle=0)
        return json.loads(argv_log.read_text(encoding="utf-8"))
    yield start
    native.shutdown_all()


def threads(argv):
    return argv[argv.index("--threads") + 1] if "--threads" in argv else None


@pytest.mark.parametrize("family", ["fish_drama3", "fish_audio", "cosyvoice3"])
def test_every_native_server_gets_the_threads_set_in_the_environment(started, monkeypatch, family):
    monkeypatch.setenv("FOLIVOX_NATIVE_THREADS", "16")
    assert threads(started(family=family)) == "16"


def test_unset_leaves_audio_cpps_own_default(started):
    assert threads(started(family="fish_drama3")) is None


def test_an_engines_own_thread_count_wins(started, monkeypatch):
    """CosyVoice's --threads (the CLI's, for one render) is more specific than the studio-wide setting."""
    monkeypatch.setenv("FOLIVOX_NATIVE_THREADS", "16")
    assert threads(started(family="cosyvoice3", threads=6)) == "6"


@pytest.mark.parametrize("bad", ["many", "0", "-2", "1.5"])
def test_a_value_that_is_not_a_thread_count_is_refused_by_name(started, monkeypatch, bad):
    monkeypatch.setenv("FOLIVOX_NATIVE_THREADS", bad)
    with pytest.raises(NativeError, match="FOLIVOX_NATIVE_THREADS"):
        started(family="fish_drama3")

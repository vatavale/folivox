"""CosyVoice 3 on the native runtime: a long-lived ``audiocpp_server`` (audio.cpp) on the loopback interface.

One server per (engine build, model, backend, weight type) is shared by all renders and previews of this process and
stopped after a few idle minutes (loading takes seconds to tens of seconds, so a process per fragment would waste
most of the time). A crashed server is restarted once and the fragment is repeated; a stuck one is killed.
The server listens on 127.0.0.1 only; any program on this computer can reach that port while it runs.
"""

from __future__ import annotations

import atexit
import io
import json
import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from folivox.tts import native_setup
from folivox.tts.base import SynthTimeout, TTSEngine, synth_timeout
from folivox.tts.native_setup import NativeError
from folivox.tts.voice import Voice

IDLE_SECONDS = float(os.environ.get("FOLIVOX_NATIVE_IDLE", "300"))
LOAD_TIMEOUT = 900.0  # first start: compiles GPU kernels and reads the weights
WEIGHT_TYPES = {"native": "native", "f32": "F32", "f16": "F16", "bf16": "BF16", "q8_0": "Q8_0"}


class ServerGone(NativeError):
    """The server process is not reachable (crashed or killed)."""


@dataclass(frozen=True)
class Spec:
    server: Path
    model: Path
    backend: str
    weight_type: str = "native"
    threads: int | None = None
    family: str = "cosyvoice3"  # audio.cpp model family: cosyvoice3 | fish_audio | fish_drama3
    device: int | None = None  # GPU number inside the backend (see native_setup.choose_device); None = engine default
    options: tuple[tuple[str, object], ...] = ()  # extra session options, e.g. fish_drama3.cpu_expert_layers


def server_command(server: Path) -> list[str]:
    """How to start the server executable (tests replace this with a fake)."""
    return [str(server)]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Server:
    def __init__(self, spec: Spec):
        self.spec = spec
        self.proc: subprocess.Popen | None = None
        self.port = 0
        self._tmp: Path | None = None
        self.users = 0
        self.timer: threading.Timer | None = None
        self.lock = threading.Lock()  # one request at a time: the server serialises them anyway

    def session_options(self) -> dict:
        """CosyVoice converts its weights to ``weight_type`` while loading; the Fish models run as published."""
        base = {"cosyvoice3.weight_type": self.spec.weight_type} if self.spec.family == "cosyvoice3" else {}
        return {**base, **dict(self.spec.options)}

    # -- life cycle
    def start(self) -> None:
        sp = self.spec
        self.port = free_port()
        self._tmp = Path(tempfile.mkdtemp(prefix="folivox-native-"))
        cfg = {"host": "127.0.0.1", "port": self.port, "backend": sp.backend, "lazy_load": False,
               "max_loaded_models": 1,
               "models": [{"id": sp.family, "family": sp.family, "path": str(sp.model), "task": "clon",
                           "mode": "offline", "session_options": self.session_options()}]}
        cfg_path = self._tmp / "server.json"
        cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
        log = native_setup.log_file()
        log.parent.mkdir(parents=True, exist_ok=True)
        if log.exists() and log.stat().st_size > 2_000_000:
            log.write_bytes(log.read_bytes()[-200_000:])
        cmd = [*server_command(sp.server), "--config", str(cfg_path), "--no-ui", "--log", "--log-file", str(log)]
        if sp.threads:
            cmd += ["--threads", str(sp.threads)]
        if sp.device is not None:
            cmd += ["--device", str(sp.device)]
        out = log.open("ab")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            self.proc = subprocess.Popen(cmd, stdout=out, stderr=out, stdin=subprocess.DEVNULL,  # noqa: S603
                                         cwd=str(sp.server.parent), creationflags=flags,
                                         env=native_setup.server_env(sp.server))
        except OSError as e:
            self.stop()  # removes the temporary config folder
            raise NativeError(f"Could not start the native engine {sp.server}: {e}") from e
        finally:
            out.close()
        try:
            self._wait_ready()
        except BaseException:
            self.stop()  # reap a crashed or half-loaded server and remove its temporary folder
            raise

    def _wait_ready(self) -> None:
        deadline = time.time() + LOAD_TIMEOUT
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise NativeError(f"The native engine stopped while loading (exit code {self.proc.returncode}). "
                                  f"{self.log_tail()}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/v1/models", timeout=5) as r:  # noqa: S310
                    data = json.loads(r.read()).get("data") or []
                if data and all(m.get("loaded", True) for m in data):
                    return
            except (OSError, ValueError):
                pass
            time.sleep(0.5)
        self.stop()
        raise NativeError("The native engine did not finish loading in time. " + self.log_tail())

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def _exited_soon(self, grace: float = 2.0) -> bool:
        """A dying server resets the connection a moment before the OS reports its exit (seen on Windows)."""
        try:
            return self.proc is not None and self.proc.wait(timeout=grace) is not None
        except subprocess.TimeoutExpired:
            return False

    def stop(self) -> None:
        p, self.proc = self.proc, None
        if p is not None and p.poll() is None:
            p.terminate()
            try:
                p.wait(10)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait(10)
        if self._tmp:
            shutil.rmtree(self._tmp, ignore_errors=True)
            self._tmp = None

    def restart(self) -> None:
        self.stop()
        self.start()

    @staticmethod
    def log_tail(n: int = 600) -> str:
        try:
            return "Last log lines: " + native_setup.log_file().read_text(encoding="utf-8", errors="replace")[-n:]
        except OSError:
            return ""

    # -- requests
    def speech(self, text: str, ref_wav: Path, ref_text: str, seed: int, timeout: float) -> tuple[np.ndarray, int]:
        body = {"model": self.spec.family, "input": text, "voice_ref": str(ref_wav), "reference_text": ref_text,
                "seed": int(seed), "response_format": "wav"}
        if self.spec.family == "cosyvoice3":
            body["options"] = {"template_name": "zero_shot"}
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/audio/speech",
                                     json.dumps(body, ensure_ascii=False).encode("utf-8"),
                                     {"Content-Type": "application/json"})
        with self.lock:
            if not self.alive():
                raise ServerGone("the native engine is not running")
            try:
                with urllib.request.urlopen(req, timeout=timeout or None) as r:  # noqa: S310 (loopback)
                    data, ctype = r.read(), r.headers.get("Content-Type", "")
            except urllib.error.HTTPError as e:
                raise NativeError(f"The native engine refused the request ({e.code}): "
                                  f"{e.read()[:400].decode('utf-8', 'replace')}") from None
            except TimeoutError:
                self.stop()  # a stuck generation: the next call starts a fresh server
                raise SynthTimeout(f"The native engine did not answer in {timeout:.0f} s and was stopped. Progress "
                                   "is saved: run the same command again to continue.") from None
            except (urllib.error.URLError, ConnectionError, OSError) as e:
                if not self.alive() or self._exited_soon():
                    raise ServerGone(f"the native engine stopped ({e}). {self.log_tail()}") from None
                raise NativeError(f"The native engine did not answer: {e}") from None
        if "audio" not in ctype and not data.startswith(b"RIFF"):
            raise NativeError(f"The native engine returned no audio: {data[:300].decode('utf-8', 'replace')}")
        return decode_wav(data)


def decode_wav(data: bytes) -> tuple[np.ndarray, int]:
    """float32 mono samples and the sample rate of a PCM WAV."""
    try:
        with wave.open(io.BytesIO(data)) as w:
            sr, ch, width, frames = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.readframes(w.getnframes())
    except (wave.Error, EOFError) as e:
        raise NativeError(f"The native engine returned a WAV that cannot be read: {e}") from e
    if width == 2:
        x = np.frombuffer(frames, "<i2").astype(np.float32) / 32768.0
    elif width == 4:
        x = np.frombuffer(frames, "<i4").astype(np.float32) / 2147483648.0
    elif width == 1:
        x = (np.frombuffer(frames, np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise NativeError(f"Unsupported WAV sample width: {width} bytes")
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return np.ascontiguousarray(x, dtype=np.float32), sr


# ---- shared servers -------------------------------------------------------------------------------------------

_lock = threading.RLock()
_servers: dict[Spec, Server] = {}


def acquire(spec: Spec) -> Server:
    with _lock:
        s = _servers.get(spec)
        if s is None:
            s = _servers[spec] = Server(spec)
        if s.timer:
            s.timer.cancel()
            s.timer = None
        s.users += 1
    try:
        with s.lock:
            if not s.alive():
                s.start()
    except BaseException:
        release(s, idle=0)
        raise
    return s


def release(s: Server, idle: float | None = None) -> None:
    with _lock:
        s.users = max(0, s.users - 1)
        if s.users:
            return
        delay = IDLE_SECONDS if idle is None else idle
        if delay <= 0:
            _servers.pop(s.spec, None)
            s.stop()
            return
        s.timer = threading.Timer(delay, _idle_stop, args=(s,))
        s.timer.daemon = True
        s.timer.start()


def _idle_stop(s: Server) -> None:
    with _lock:
        if s.users == 0:
            _servers.pop(s.spec, None)
            s.stop()


def shutdown_all() -> None:
    with _lock:
        for s in list(_servers.values()):
            if s.timer:
                s.timer.cancel()
            s.stop()
        _servers.clear()


atexit.register(shutdown_all)


# ---- the engine -----------------------------------------------------------------------------------------------

def reference_wav(voice: Voice) -> Path:
    """Mono 24 kHz WAV of the reference voice (audio.cpp reads WAV only); cached per voice."""
    dst = native_setup.root() / "refs" / f"{voice.id}.wav"
    if dst.is_file() and dst.stat().st_size > 0:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + ".part.wav")  # a half-written file must never be taken for a finished one
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(voice.wav), "-ac", "1", "-ar", "24000",
                        "-c:a", "pcm_s16le", str(part)], capture_output=True, text=True)  # noqa: S603, S607
    if r.returncode != 0:
        part.unlink(missing_ok=True)
        raise NativeError(f"Could not prepare the reference voice {voice.wav.name}: {r.stderr.strip()[-300:]}")
    part.replace(dst)
    return dst


def change_speed(audio: np.ndarray, sr: int, speed: float) -> np.ndarray:
    """Tempo change without changing the pitch (ffmpeg atempo); the native model has no speed option."""
    if abs(speed - 1.0) < 1e-3 or audio.size == 0:
        return audio
    filters, s = [], speed
    while s > 2.0:
        filters.append("atempo=2.0")
        s /= 2.0
    while s < 0.5:
        filters.append("atempo=0.5")
        s /= 0.5
    filters.append(f"atempo={s:.4f}")
    r = subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "-",
                        "-filter:a", ",".join(filters), "-f", "f32le", "-"],  # noqa: S603, S607
                       input=audio.astype("<f4").tobytes(), capture_output=True)
    if r.returncode != 0:
        raise NativeError(f"Could not change the reading speed: {r.stderr.decode('utf-8', 'replace')[-300:]}")
    return np.frombuffer(r.stdout, "<f4").astype(np.float32)


class NativeCosyVoiceEngine(TTSEngine):
    name = "cosyvoice3"
    languages = ("ru", "en", "zh", "ja", "ko", "de", "es", "fr", "it")
    parallel = 1  # the LLM runs inside the server; there is nothing to prepare ahead

    def __init__(self, backend: str = "auto", weights: str = "q8_0", weight_type: str = "f16",
                 threads: int | None = None, device: str | int = "auto", **_unused):
        wt = WEIGHT_TYPES.get(str(weight_type).lower())
        if wt is None:
            raise NativeError(f"weight_type must be one of {', '.join(WEIGHT_TYPES)} (got {weight_type!r})")
        server = native_setup.server_path(backend)
        model = native_setup.model_path(weights)
        if server is None or model is None:
            raise NativeError(f"The native runtime is not ready: {native_setup.status(weights).problem}. "
                              "Install it with `folivox runtime install` (or use `--runtime pytorch`).")
        be = native_setup.server_backend(backend)
        self.device = f"native:{be}"
        native_setup.require_gpu(server, be)
        self.spec = Spec(server, model, be, wt, threads, device=native_setup.choose_device(server, be, device))
        t0 = time.time()
        self._server = acquire(self.spec)
        self.load_seconds = time.time() - t0
        self._refs: dict[str, Path] = {}
        self._closed = False

    def prepare_voice(self, voice: Voice) -> Path:
        if voice.id not in self._refs:
            self._refs[voice.id] = reference_wav(voice)
        return self._refs[voice.id]

    def synthesize_audio(self, text: str, voice: Voice, seed: int, speed: float):
        ref = self.prepare_voice(voice)
        timeout = synth_timeout(len(text))
        t0 = time.time()
        for attempt in (0, 1):
            try:
                audio, sr = self._server.speech(text, ref, voice.transcript, seed, timeout)
                break
            except ServerGone:
                if attempt:
                    raise
                with self._server.lock:  # one restart per fragment, then the error reaches the caller
                    if not self._server.alive():
                        self._server.start()
        self.sample_rate = sr
        return change_speed(audio, sr, speed), time.time() - t0

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            release(self._server)


def settings_tag(weights: str, weight_type: str) -> str:
    """Part of a take's identity: other weights or precision sound slightly different."""
    return f"{weights}:{weight_type.lower()}"

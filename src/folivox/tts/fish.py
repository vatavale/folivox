"""Fish Audio engines (S2 Pro, Drama 3): a worker process in the engine's own Python environment.

Folivox never imports or bundles Fish code (its licence forbids it): ``fish_worker.py`` runs under the
interpreter of the private environment made by :mod:`folivox.tts.fish_setup` and talks JSON lines over pipes.
Because the model lives in another process, a native crash or an out-of-memory kill takes down only the worker,
and closing it gives all the memory back.
"""

from __future__ import annotations

import collections
import json
import queue
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from folivox import memory
from folivox.tts import builds, fish_setup, licences
from folivox.tts.base import SynthTimeout, TTSEngine, synth_timeout
from folivox.tts.voice import Voice
from folivox.voicegen.cloud_apis import FishAudio

WORKER = Path(__file__).with_name("fish_worker.py")
LOAD_TIMEOUT = 900.0  # reading 20+ GB of weights and the kernel warm-up


class FishError(RuntimeError):
    pass


def to_reference_wav(src: Path, dst: Path) -> Path:
    """Mono 44.1 kHz WAV of a reference voice: what the models expect, whatever format the user supplied."""
    if dst.is_file() and dst.stat().st_size > 0:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-ac", "1", "-ar", "44100",
                        str(dst)], capture_output=True, text=True)  # noqa: S603, S607
    if r.returncode != 0:
        dst.unlink(missing_ok=True)  # a partial file would be taken for a finished one next time
        raise FishError(f"Could not prepare the reference voice {src.name}: {r.stderr.strip()[-300:]}")
    return dst


class _Worker:
    """The child process and its two pipes; replies are read by a thread so a hang can time out."""

    def __init__(self, python: Path):
        self.proc = subprocess.Popen([str(python), str(WORKER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,  # noqa: S603
                                     stderr=subprocess.PIPE, text=True, encoding="utf-8", bufsize=1)
        self.replies: queue.Queue = queue.Queue()
        self.stderr_tail: collections.deque[str] = collections.deque(maxlen=40)
        threading.Thread(target=self._read_out, daemon=True, name="folivox-fish-out").start()
        threading.Thread(target=self._read_err, daemon=True, name="folivox-fish-err").start()

    def _read_out(self) -> None:
        for line in self.proc.stdout:  # type: ignore[union-attr]
            try:
                self.replies.put(json.loads(line))
            except ValueError:
                self.stderr_tail.append(line.rstrip())
        self.replies.put(None)  # EOF: the worker is gone

    def _read_err(self) -> None:
        for line in self.proc.stderr:  # type: ignore[union-attr]
            self.stderr_tail.append(line.rstrip())

    def call(self, req: dict, timeout: float) -> dict:
        if self.proc.poll() is not None:
            raise FishError(self._died())
        try:
            self.proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")  # type: ignore[union-attr]
            self.proc.stdin.flush()  # type: ignore[union-attr]
        except OSError as e:
            raise FishError(self._died()) from e
        try:
            rep = self.replies.get(timeout=timeout or None)
        except queue.Empty:
            self.kill()
            raise SynthTimeout(f"The Fish worker did not answer in {timeout:.0f} s and was stopped. Progress is "
                               "saved: run the same command again to continue. If memory is tight (the computer "
                               "swaps), close other heavy applications first.") from None
        if rep is None:
            raise FishError(self._died())
        if not rep.get("ok"):
            raise FishError(rep.get("error", "unknown error"))
        return rep

    def _died(self) -> str:
        try:
            code = self.proc.wait(timeout=2)  # the pipe closes a moment before Windows reports the exit code
        except subprocess.TimeoutExpired:
            code = self.proc.poll()
        tail = "\n".join(list(self.stderr_tail)[-8:])
        hint = " It was probably killed for lack of memory: close heavy applications and try again." \
            if code in (-9, 137) else ""
        return f"The Fish worker stopped (exit code {code}).{hint}\n{tail}".rstrip()

    def kill(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
            try:
                self.proc.wait(timeout=10)  # reap it: the memory is released only when the process is gone
            except subprocess.TimeoutExpired:
                pass

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self.proc.stdin.write('{"cmd": "quit"}\n')  # type: ignore[union-attr]
                self.proc.stdin.flush()  # type: ignore[union-attr]
                self.proc.wait(timeout=10)
            except Exception:  # noqa: BLE001
                self.kill()
        for pipe in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
            try:
                pipe.close()  # type: ignore[union-attr]
            except Exception:  # noqa: BLE001
                pass


class FishEngine(TTSEngine):
    """Base of the Fish engines; subclasses fix the provider name (the catalog says which builds belong to it)."""

    name = ""
    device = "mlx"
    # S2 Pro and Drama 3 read the same 80+ languages as the Fish Audio API; the nine with Folivox rules get
    # numbers and dates expanded, the others go to the model almost as they are
    languages = FishAudio.languages
    sample_rate = 44100

    def __init__(self, build: str | None = None, device: str = "auto", runtime: str | None = None,
                 **_ignored: Any):
        from folivox.tts import runtime as tts_runtime

        self.build = builds.build_for(self.name, build, runtime=tts_runtime.resolve(self.name, runtime))
        self.native = self.build.backend == "native"
        if self.native:
            self.device = "native"
        licences.require(self.build)
        if not fish_setup.is_ready(self.build):
            raise FileNotFoundError(f"{self.build.title} is not installed. Install it first (it downloads "
                                    f"{self.build.size_gb:.1f} GB): folivox engines install {self.build.id}")
        self._worker: _Worker | None = None
        self._server = None  # native builds: the shared audio.cpp server
        self._refs: dict[str, Path] = {}
        self._tmp = tempfile.TemporaryDirectory(prefix="folivox-fish-")
        self.peak_gb = 0.0
        self.load_seconds = 0.0
        self._calls = 0

    # -- life cycle
    def start(self, voice: Voice) -> None:
        """Load the model and warm it up with the book's own voice (the first call is slow: kernels compile)."""
        if self.native:
            return self._start_native(voice)
        if self._worker is not None:
            return
        python = fish_setup.venv_python(self.build.runtime)
        ref = self._reference(voice)
        self._worker = _Worker(python)
        t0 = time.time()
        try:
            rep = self._worker.call({"cmd": "load", "model_dir": str(builds.weights_dir(self.build)),
                                     "kind": self.build.kind, "warmup": {"wav": str(ref), "text": voice.transcript}},
                                    LOAD_TIMEOUT)
        except BaseException:
            # a failed load must not leave a model-less worker behind: the next start() would skip loading
            self._worker.close()
            self._worker = None
            raise
        self.sample_rate = int(rep["sample_rate"])
        self.load_seconds = time.time() - t0

    def _start_native(self, voice: Voice) -> None:
        """The audio.cpp server loads the GGUF and stays up for the whole render (and a few idle minutes after)."""
        if self._server is not None:
            return
        from folivox import config
        from folivox.tts import native, native_setup

        family = "fish_drama3" if self.build.kind == "drama3" else "fish_audio"
        server = native_setup.own_server_for(family) or native_setup.server_path()
        if server is None:
            raise FishError("The native engine is not installed: run  folivox runtime install  "
                            f"(or folivox engines install {self.build.id})")
        backend = native_setup.server_backend()
        self.device = f"native:{backend}"
        gguf = builds.weights_dir(self.build) / self.build.files[0]["path"]
        t0 = time.time()
        try:
            device = native_setup.choose_device(server, backend, config.native_options().get("device"))
            opts = native_setup.drama3_options(gguf, backend, device, config.native_options().get(
                "drama3_cpu_expert_layers")) if family == "fish_drama3" else ()
            self._server = native.acquire(native.Spec(server, gguf, backend, family=family, device=device,
                                                      options=opts))
        except native_setup.NativeError as e:
            raise FishError(str(e)) from e
        self.load_seconds = time.time() - t0

    def close(self) -> None:
        if self._server is not None:
            from folivox.tts import native

            native.release(self._server)
            self._server = None
        if self._worker is not None:
            self._worker.close()
            self._worker = None
        self._tmp.cleanup()
        memory.release()

    def __del__(self):  # pragma: no cover — a safety net; render() closes explicitly
        try:
            if self._worker is not None:
                self._worker.close()
        except Exception:  # noqa: BLE001
            pass

    # -- synthesis
    def _reference(self, voice: Voice) -> Path:
        if voice.id not in self._refs:
            self._refs[voice.id] = to_reference_wav(voice.wav, Path(self._tmp.name) / f"ref-{voice.id}.wav")
        return self._refs[voice.id]

    def synthesize_audio(self, text: str, voice: Voice, seed: int, speed: float) -> tuple[np.ndarray, float]:
        # Reading speed is not a parameter of these models; the pipeline warns once.
        self.start(voice)
        if self.native:
            return self._synth_native(text, voice, seed)
        assert self._worker is not None
        self._calls += 1
        out = Path(self._tmp.name) / f"out-{self._calls}.npy"
        rep = self._worker.call({"cmd": "synth", "id": self._calls, "text": text, "seed": seed, "out": str(out),
                                 "refs": [{"wav": str(self._reference(voice)), "text": voice.transcript}]},
                                synth_timeout(len(text)))
        audio = np.load(out).astype(np.float32)
        out.unlink(missing_ok=True)
        self.peak_gb = max(self.peak_gb, float(rep.get("peak_gb", 0.0)))
        return audio, float(rep["seconds"])


    def _synth_native(self, text: str, voice: Voice, seed: int) -> tuple[np.ndarray, float]:
        from folivox.tts import native
        from folivox.tts.native_setup import NativeError

        assert self._server is not None
        ref = self._reference(voice)
        t0 = time.time()
        for attempt in (0, 1):
            try:
                audio, sr = self._server.speech(text, ref, voice.transcript, seed, synth_timeout(len(text)))
                break
            except native.ServerGone:
                if attempt:
                    raise FishError("The native engine stopped twice in a row; see its log in `folivox doctor`.") \
                        from None
                with self._server.lock:
                    if not self._server.alive():
                        self._server.start()
            except NativeError as e:
                raise FishError(str(e)) from e
        self.sample_rate = sr
        return audio, time.time() - t0


class FishS2Engine(FishEngine):
    name = "fish-s2"


class FishDrama3Engine(FishEngine):
    name = "fish-drama3"

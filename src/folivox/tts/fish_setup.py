"""Install a Fish build: a private Python environment for its runtime plus the verified weights.

Everything lives in ``<user data>/folivox/engines/``: ``runtimes/<runtime id>/venv`` (shared by builds that need
the same packages) and ``<build id>/weights``. Nothing is installed without the licence accepted first, and the
weights come only from the pinned revision of the catalog entry, file by file checked against the catalog.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

from folivox.tts import builds, licences, reuse
from folivox.tts.builds import Build

READY = "ready.json"
RUN_TIMEOUT = 3600.0  # one environment step (venv, pip install)
Progress = Callable[[str, int, int], None]


class InstallError(RuntimeError):
    pass


def venv_python(rt_id: str) -> Path:
    venv = builds.runtime_dir(rt_id) / "venv"
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _runtime_ready(rt: builds.Runtime) -> bool:
    if rt.backend == "native":  # no Python environment: the audio.cpp server executable is the runtime
        from folivox.tts import native_setup

        return native_setup.server_path() is not None
    marker = builds.runtime_dir(rt.id) / "runtime.json"
    try:
        return venv_python(rt.id).is_file() and json.loads(marker.read_text(encoding="utf-8"))["packages"] == \
            list(rt.packages)
    except (OSError, ValueError, KeyError):
        return False


def is_ready(b: Build) -> bool:
    cat = builds.load()
    try:
        info = json.loads((builds.build_dir(b) / READY).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if info.get("revision") != b.revision or not _runtime_ready(cat.runtimes[b.runtime]):
        return False
    wd = builds.weights_dir(b)
    try:
        return all((wd / f["path"]).stat().st_size == f["size"] for f in b.files)
    except OSError:
        return False


def sha256_file(path: Path, hook: Callable[[int], None] | None = None) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
            if hook:
                hook(len(chunk))
    return h.hexdigest()


def verify_weights(b: Build, hook: Progress | None = None) -> list[str]:
    """Names of files that are missing or differ from the catalog (size, and sha256 where the catalog has one)."""
    wd = builds.weights_dir(b)
    total = sum(f["size"] for f in b.files if f.get("sha256"))
    done = 0
    bad = []
    for f in b.files:
        p = wd / f["path"]
        if not p.is_file() or p.stat().st_size != f["size"]:
            bad.append(f["path"])
            continue
        if f.get("sha256"):
            def tick(n: int) -> None:
                nonlocal done
                done += n
                if hook:
                    hook("Checking files", done, total)

            if sha256_file(p, tick) != f["sha256"]:
                bad.append(f["path"])
    return bad


def _run(cmd: list[str], what: str, log: Callable[[str], None]) -> None:
    log(what)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,  # noqa: S603 — argv built here, no shell
                           timeout=RUN_TIMEOUT)
    except subprocess.TimeoutExpired:
        raise InstallError(f"{what} did not finish in {RUN_TIMEOUT / 60:.0f} minutes and was stopped") from None
    except OSError as e:
        raise InstallError(f"{what} could not start: {e}") from e
    if r.returncode != 0:
        tail = "\n".join((r.stderr or r.stdout).strip().splitlines()[-12:])
        raise InstallError(f"{what} failed:\n{tail}")


def ensure_runtime(rt: builds.Runtime, log: Callable[[str], None] = print) -> Path:
    """Create the private environment and install the pinned packages (a few minutes the first time).
    The native runtime downloads the pinned audio.cpp release instead; a missing server of your own build
    (FOLIVOX_NATIVE_SERVER) is an error, not something to download."""
    if rt.backend == "native":
        from folivox.tts import native_setup

        try:
            return native_setup.ensure_engine(log=log)
        except native_setup.NativeError as e:
            raise InstallError(str(e)) from e
    if _runtime_ready(rt):
        return venv_python(rt.id)
    root = builds.runtime_dir(rt.id)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    py = venv_python(rt.id)
    uv = shutil.which("uv")
    if uv:
        _run([uv, "venv", "--python", rt.python, str(root / "venv")], f"Creating Python {rt.python} environment", log)
        _run([uv, "pip", "install", "--python", str(py), *rt.packages], "Installing the engine packages", log)
    else:
        if sys.version_info[:2] > (3, 12) or sys.version_info[:2] < (3, 10):
            raise InstallError("`uv` is needed to create the engine environment (https://docs.astral.sh/uv/): this "
                               "Python cannot be used for it")
        _run([sys.executable, "-m", "venv", str(root / "venv")], "Creating the engine environment", log)
        _run([str(py), "-m", "pip", "install", "--quiet", *rt.packages], "Installing the engine packages", log)
    (root / "runtime.json").write_text(json.dumps({"packages": list(rt.packages), "created": time.time()}),
                                       encoding="utf-8")
    return py


def free_disk_ok(b: Build, where: Path) -> str | None:
    probe = where
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    need = int(b.size_bytes * 1.05) + 3_000_000_000  # weights + the environment
    free = shutil.disk_usage(probe).free
    if free < need:
        return f"{b.title} needs about {need / 1e9:.0f} GB of free disk space; {free / 1e9:.0f} GB is free."
    return None


def install(b: Build, progress: Progress | None = None, log: Callable[[str], None] = print,
            phase: Callable[[str, int], None] | None = None) -> None:
    """Download and verify a build. Needs the licence accepted; safe to run again (it continues).

    Files already on this computer (see ``reuse``) are checked and linked or copied first; only the rest is
    downloaded. ``phase(name, bytes)`` reports "looking", "found", "downloading" or "verifying" for the studio."""
    licences.require(b)
    cat = builds.load()
    if not builds.is_trusted_org(b.repo, cat):
        log(f"Note: {b.repo} is a personal repository, not a trusted organisation; the files are checked against "
            "the sha256 values pinned in the Folivox catalog.")
    problem = free_disk_ok(b, builds.engines_root())
    if problem:
        raise InstallError(problem)
    ensure_runtime(cat.runtimes[b.runtime], log)
    wd = builds.weights_dir(b)
    wd.mkdir(parents=True, exist_ok=True)
    names = [f["path"] for f in b.files]
    if phase:
        phase("looking", 0)
    nearby = reuse.found_bytes(b)
    if nearby:
        if phase:
            phase("found", nearby)
        log(f"Found {nearby / 1e9:.1f} GB of {b.title} on this computer; checking it.")
        got = reuse.adopt(b, progress)
        if got:
            linked = sum(1 for v in got.values() if v == "linked")
            log(f"Reused {len(got)} file(s): {linked} linked, {len(got) - linked} copied.")
    have = {f["path"] for f in b.files if _file_ok(wd, f)}
    names = [n for n in names if n not in have]
    if names:
        if phase:
            phase("downloading", nearby)
        _download(b, wd, names, progress, log)
    elif phase:
        phase("verifying", 0)
    bad = verify_weights(b, progress)
    if bad:
        for name in bad:
            (wd / name).unlink(missing_ok=True)
        raise InstallError("These files do not match the checksums in the Folivox catalog and were removed: "
                           f"{', '.join(bad)}. Run the install again; if it keeps failing, the repository changed.")
    (builds.build_dir(b) / READY).write_text(json.dumps({"revision": b.revision, "installed": time.time()}),
                                             encoding="utf-8")
    log(f"{b.title} is ready.")


def _file_ok(wd: Path, f: dict) -> bool:
    """A file with a pinned sha256 that is already in place and matches it (so it need not be downloaded)."""
    p = wd / f["path"]
    try:
        return bool(f.get("sha256")) and p.stat().st_size == f["size"] and sha256_file(p) == f["sha256"]
    except OSError:
        return False


def _download(b: Build, wd: Path, names: list[str], progress: Progress | None, log: Callable[[str], None]) -> None:
    from huggingface_hub import snapshot_download

    from folivox.tts import weights

    log(f"Downloading {b.title} ({b.size_gb:.1f} GB) from huggingface.co/{b.repo} at revision {b.revision[:8]}…")
    sizes = {f["path"]: f.get("size", 0) for f in b.files}
    with weights.reporting(progress), weights._watch(b.title, wd, names, sum(sizes.get(n, 0) for n in names)):
        weights.with_retries(lambda: snapshot_download(b.repo, revision=b.revision, local_dir=str(wd),
                                                       allow_patterns=names), b.title)


def remove(b: Build) -> int:
    """Delete the weights of a build (the shared environment stays until no build needs it). Returns bytes freed."""
    d = builds.build_dir(b)
    size = sum(p.stat().st_size for p in d.rglob("*") if p.is_file()) if d.exists() else 0
    shutil.rmtree(d, ignore_errors=True)
    cat = builds.load()
    still_used = any(o.runtime == b.runtime and o.id != b.id and (builds.build_dir(o) / READY).exists()
                     for o in cat.builds)
    if not still_used:
        shutil.rmtree(builds.runtime_dir(b.runtime), ignore_errors=True)
    return size

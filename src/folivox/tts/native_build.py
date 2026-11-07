"""Building audio.cpp from source, for a newer Drama 3 than the release (releases carry it since v0.9.4).

Recipes for every platform are data in ``native_catalog.json`` (``sources.build``) and are shown by
``folivox runtime build-help``. Only macOS (Metal) is automated here, because it is the one setup that was run end to
end: audio.cpp is fetched at a pinned commit, built with the project's own script, and the server is copied into
Folivox's data folder. The code is third-party and not reviewed by Folivox: the caller must ask the user first.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

from folivox.tts import native_setup as ns
from folivox.tts.native_setup import NativeError

RUN_TIMEOUT = 3600.0  # one build step (fetch, tools, compile); a hung network must not block forever
TARGETS = {"drama3": "fish_drama3"}  # build target -> audio.cpp model family list for --models


def source(target: str) -> dict:
    if target != "drama3":
        raise NativeError(f"Unknown build target {target!r}; known: {', '.join(TARGETS)}")
    return ns.sources()["drama3"]


def own_dir(target: str) -> Path:
    return ns.root() / "own" / f"{target}-{source(target)['commit'][:8]}"


def own_server(target: str = "drama3") -> Path | None:
    """The server built by :func:`build` for the pinned commit, if it is there."""
    d = own_dir(target)
    try:
        json.loads((d / ns.READY).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    p = d / ns.SERVER_NAME
    return p if p.is_file() else None


def platform_recipe(tag: str | None = None) -> list[str]:
    """Plain build steps for this computer (all platforms; see ``folivox runtime build-help``)."""
    tag = tag or ns.platform_tag()
    b = ns.sources()["build"]
    if tag.startswith("darwin"):
        return b["macos"]
    if tag.startswith("win32"):
        return b["windows-cuda"] + [""] + b["windows-vulkan"] + [""] + b["amd-hip"]
    return b["linux-cuda"] + [""] + b["linux-vulkan"] + [""] + b["amd-hip"]


def can_build_here() -> str | None:
    """A reason the automatic build cannot run here, or None."""
    if ns.platform_tag() != "darwin-arm64":
        return ("Automatic building is available on Apple Silicon only; for this computer follow the steps from "
                "`folivox runtime build-help`")
    for tool in ("git",):
        if shutil.which(tool) is None:
            return f"`{tool}` is needed to fetch the source"
    r = subprocess.run(["xcrun", "-find", "metal"], capture_output=True, text=True)  # noqa: S603, S607
    if r.returncode != 0:
        return "The Metal compiler is missing: install Xcode or run `xcode-select --install`"
    if shutil.which("uv") is None and not (shutil.which("cmake") and shutil.which("ninja")):
        return "`uv` (or both cmake and ninja) is needed to get the build tools"
    return None


def _run(cmd: list[str], cwd: Path | None, what: str, log: Callable[[str], None], env: dict | None = None) -> None:
    log(what)
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=env,  # noqa: S603 — argv built here
                           timeout=RUN_TIMEOUT)
    except subprocess.TimeoutExpired:
        raise NativeError(f"{what} did not finish in {RUN_TIMEOUT / 60:.0f} minutes and was stopped") from None
    except OSError as e:
        raise NativeError(f"{what} could not start: {e}") from e
    if r.returncode != 0:
        tail = "\n".join((r.stderr or r.stdout).strip().splitlines()[-12:])
        raise NativeError(f"{what} failed:\n{tail}")


def build(target: str = "drama3", log: Callable[[str], None] = print) -> Path:
    """Fetch the pinned commit, compile ``audiocpp_server`` (several minutes) and keep it in the data folder."""
    import os

    why = can_build_here()
    if why:
        raise NativeError(why)
    src = source(target)
    work = ns.root() / "build" / f"{target}-{src['commit'][:8]}"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    repo = work / "src"
    repo.mkdir()
    env = dict(os.environ)
    try:
        _run(["git", "init", "-q"], repo, "Preparing the source folder", log)
        _run(["git", "fetch", "-q", "--depth", "1", src["repo"], src["commit"]], repo,
             f"Fetching {src['repo']} at {src['commit'][:8]}", log)
        _run(["git", "checkout", "-q", "FETCH_HEAD"], repo, "Checking out the pinned commit", log)
        if not (shutil.which("cmake") and shutil.which("ninja")):
            _run(["uv", "venv", "-q", str(work / "tools")], None, "Getting CMake and Ninja", log)
            _run(["uv", "pip", "install", "-q", "--python", str(work / "tools" / "bin" / "python"), "cmake", "ninja"],
                 None, "Installing CMake and Ninja", log)
            env["PATH"] = f"{work / 'tools' / 'bin'}{os.pathsep}{env['PATH']}"
        _run(["scripts/build_metal.sh", "--build-type", "Release", "--model-set", "custom", "--models",
              TARGETS[target], "--target", "audiocpp_server"], repo,
             "Compiling (several minutes; Metal build with the Drama 3 model)", log, env)
        built = repo / "build" / "macos-metal-release" / "bin" / "audiocpp_server"
        if not built.is_file():
            raise NativeError("The build finished but audiocpp_server was not found")
        dest = own_dir(target)
        shutil.rmtree(dest, ignore_errors=True)
        dest.mkdir(parents=True)
        shutil.copy2(built, dest / ns.SERVER_NAME)
        ver = ns.version_text(dest / ns.SERVER_NAME)
        if "metal" not in ver:
            raise NativeError(f"The built server does not report the metal backend ({ver.strip()[:200]})")
        (dest / ns.READY).write_text(json.dumps({"commit": src["commit"], "built": time.time()}), encoding="utf-8")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    log(f"Built: {dest / ns.SERVER_NAME}")
    return dest / ns.SERVER_NAME


def remove(target: str = "drama3") -> int:
    d = own_dir(target)
    freed = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) if d.exists() else 0
    shutil.rmtree(d, ignore_errors=True)
    return freed


if __name__ == "__main__":  # pragma: no cover
    print(*platform_recipe(), sep="\n", file=sys.stdout)

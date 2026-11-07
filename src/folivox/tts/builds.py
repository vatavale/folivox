"""Catalog of the Fish Audio builds Folivox can offer (``catalog.json``) and the choice by hardware.

Only builds listed in the catalog can be installed: each has a Hugging Face repo, a pinned revision and
the size and sha256 of its files, and only safetensors weights (no pickle) are accepted. The catalog is
data, so a new quantisation is a new entry; ``tools/fish_catalog.py watch`` finds candidates.
"""

from __future__ import annotations

import json
import os
import platform
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from platformdirs import user_data_dir

CATALOG_FILE = Path(__file__).with_name("catalog.json")
IGNORE_MEMORY_ENV = "FOLIVOX_IGNORE_MEMORY"  # skip the free-memory check (at your own risk)
WEIGHT_SUFFIXES = (".safetensors", ".gguf")  # tensor containers, no pickle and no code
# Files a build may contain besides safetensors: plain text/config. Anything else (pickle, scripts) is refused.
TEXT_SUFFIXES = (".json", ".md", ".txt", ".jinja", ".gitattributes")
TEXT_NAMES = ("LICENSE", "NOTICE", ".gitattributes")


class CatalogError(ValueError):
    pass


@dataclass(frozen=True)
class Licence:
    id: str
    name: str
    version: str
    commercial: bool
    text_url: str
    text_sha256: str
    page: str


@dataclass(frozen=True)
class Runtime:
    id: str
    backend: str
    python: str
    platforms: tuple[str, ...]
    packages: tuple[str, ...]
    note: str = ""


@dataclass(frozen=True)
class Build:
    id: str
    engine: str  # provider name in folivox.tts.ENGINES
    title: str
    backend: str  # mlx | cuda | cpu
    kind: str  # model family the worker needs to know: s2 | drama3
    platforms: tuple[str, ...]
    repo: str
    revision: str
    runtime: str
    licence: str
    quant: str
    rank: int
    status: str  # verified | untested
    size_bytes: int
    weights_gb: float
    peak_gb: float
    min_total_gb: float
    languages: tuple[str, ...]
    files: tuple[dict, ...]
    tested_on: tuple[dict, ...] = ()
    notes: str = ""

    @property
    def size_gb(self) -> float:
        return self.size_bytes / 1e9


@dataclass(frozen=True)
class Catalog:
    builds: tuple[Build, ...]
    licences: dict[str, Licence]
    runtimes: dict[str, Runtime]
    trusted_orgs: tuple[str, ...]
    extra: dict = field(default_factory=dict)

    def build(self, build_id: str) -> Build:
        for b in self.builds:
            if b.id == build_id:
                return b
        raise CatalogError(f"Unknown build {build_id!r}; available: {', '.join(b.id for b in self.builds)}")

    def for_engine(self, engine: str) -> list[Build]:
        return [b for b in self.builds if b.engine == engine]

    def engines(self) -> list[str]:
        seen: list[str] = []
        for b in self.builds:
            if b.engine not in seen:
                seen.append(b.engine)
        return seen


def parse(data: dict) -> Catalog:
    """Validate and load a catalog document; raises :class:`CatalogError` on anything unsafe."""
    if data.get("schema") != 1:
        raise CatalogError("Unsupported catalog schema")
    licences = {k: Licence(k, v["name"], v["version"], bool(v["commercial"]), v["text_url"], v["text_sha256"],
                           v["page"]) for k, v in data["licences"].items()}
    runtimes = {k: Runtime(k, v["backend"], v["python"], tuple(v["platforms"]), tuple(v["packages"]),
                           v.get("note", "")) for k, v in data["runtimes"].items()}
    trusted = tuple(data["trusted_orgs"])
    builds = []
    for b in data["builds"]:
        mem = b["memory_gb"]
        files = tuple(b["files"])
        build = Build(id=b["id"], engine=b["engine"], title=b["title"], backend=b["backend"], kind=b["kind"],
                      platforms=tuple(b["platforms"]), repo=b["repo"], revision=b["revision"],
                      runtime=b["runtime"], licence=b["licence"], quant=b["quant"], rank=int(b["rank"]),
                      status=b["status"], size_bytes=int(b["size_bytes"]), weights_gb=float(mem["weights"]),
                      peak_gb=float(mem["peak"]), min_total_gb=float(mem["min_total"]),
                      languages=tuple(b["languages"]), files=files, tested_on=tuple(b.get("tested_on", ())),
                      notes=b.get("notes", ""))
        _validate(build, licences, runtimes)
        builds.append(build)
    if len({b.id for b in builds}) != len(builds):
        raise CatalogError("Duplicate build ids")
    return Catalog(tuple(builds), licences, runtimes, trusted)


def _validate(b: Build, licences: dict, runtimes: dict) -> None:
    where = f"build {b.id}"
    if len(b.revision) != 40 or any(c not in "0123456789abcdef" for c in b.revision):
        raise CatalogError(f"{where}: the revision must be a full commit hash (pinned)")
    if b.licence not in licences:
        raise CatalogError(f"{where}: unknown licence {b.licence!r}")
    if b.runtime not in runtimes:
        raise CatalogError(f"{where}: unknown runtime {b.runtime!r}")
    if b.status not in ("verified", "untested"):
        raise CatalogError(f"{where}: status must be verified or untested")
    if "/" not in b.repo:
        raise CatalogError(f"{where}: repo must look like org/name")
    for f in b.files:
        name = f["path"]
        base = name.rsplit("/", 1)[-1]
        if name.startswith("/") or ".." in name.split("/"):
            raise CatalogError(f"{where}: unsafe file name {name!r}")
        if not (name.endswith(WEIGHT_SUFFIXES) or name.endswith(TEXT_SUFFIXES) or base in TEXT_NAMES):
            raise CatalogError(f"{where}: {name!r} is not a safetensors, gguf or text file "
                               "(pickle and scripts are refused)")
        if name.endswith(WEIGHT_SUFFIXES) and not f.get("sha256"):
            raise CatalogError(f"{where}: {name} has no sha256")
    if not any(f["path"].endswith(WEIGHT_SUFFIXES) for f in b.files):
        raise CatalogError(f"{where}: no safetensors or gguf weights")


@lru_cache(maxsize=1)
def load() -> Catalog:
    return parse(json.loads(CATALOG_FILE.read_text(encoding="utf-8")))


def measured_rtf(b: Build) -> float | None:
    """Real-time factor (synthesis time / audio time) of our own run of this build, if there is one."""
    for t in b.tested_on:
        if t.get("rtf"):
            return float(t["rtf"])
    return None


def speed_note(b: Build, cat: Catalog | None = None) -> str:
    """A native build that was measured slower than the MLX build of the same engine says so (our runs, same Mac)."""
    if b.backend != "native":
        return ""
    mine = measured_rtf(b)
    for other in (cat or load()).for_engine(b.engine):
        theirs = measured_rtf(other)
        if other.backend == "mlx" and mine and theirs and mine > theirs * 1.1:
            return (f"In our test on Apple Silicon the MLX build was faster (real-time factor {theirs:g} against "
                    f"{mine:g}); the native build is meant for computers without MLX.")
    return ""


def is_trusted_org(repo: str, cat: Catalog | None = None) -> bool:
    return repo.split("/")[0] in (cat or load()).trusted_orgs


# -- hardware ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Hardware:
    system: str  # darwin | linux | windows
    arch: str  # arm64 | x86_64 …
    total_gb: float | None
    available_gb: float | None
    gpu_gb: float | None = None  # memory of the NVIDIA card (GiB) the native builds run on; None = none or unknown

    @property
    def platform_tag(self) -> str:
        return f"{self.system}-{self.arch}"


def detect_hardware() -> Hardware:
    from folivox import diag, memory

    system = {"darwin": "darwin", "win32": "windows"}.get(sys.platform, "linux")
    arch = platform.machine().lower()
    arch = "arm64" if arch in ("arm64", "aarch64") else "x86_64" if arch in ("amd64", "x86_64") else arch
    gpu = None
    if system != "darwin":  # Apple GPUs share the system memory
        from folivox.tts import native_setup

        gpu = native_setup.nvidia_memory_gb()
    return Hardware(system, arch, diag.ram_gb(), memory.available_gb(), gpu)


def supported_here(b: Build, hw: Hardware, cat: Catalog | None = None) -> bool:
    rt = (cat or load()).runtimes[b.runtime]
    return hw.platform_tag in b.platforms and hw.platform_tag in rt.platforms


@dataclass(frozen=True)
class Fit:
    level: str  # ok | tight | no  (tight = starts, but close other applications)
    message: str


def fit(b: Build, hw: Hardware) -> Fit:
    """Does the build suit this computer? ``no`` when the total memory is too small or the platform is wrong,
    ``tight`` when the free memory right now is below the measured peak."""
    if hw.platform_tag not in b.platforms:
        return Fit("no", f"{b.title} runs on {', '.join(b.platforms)}, not on {hw.platform_tag}.")
    if b.backend == "native" and hw.gpu_gb:  # the model sits in the card's own memory, not in the system's
        return _gpu_fit(b, hw)
    if hw.total_gb is not None and hw.total_gb < b.min_total_gb:
        return Fit("no", f"{b.title} needs a computer with at least {b.min_total_gb:.0f} GB of memory "
                         f"(this one has {hw.total_gb:.0f} GB).")
    if hw.available_gb is not None and hw.available_gb < b.peak_gb:
        return Fit("tight", f"{b.title} needs up to {b.peak_gb:.0f} GB while it works and only "
                            f"{hw.available_gb:.0f} GB is free now: close other heavy "
                            "applications, otherwise the computer starts swapping and the render crawls or stalls.")
    return Fit("ok", "")


def _gpu_fit(b: Build, hw: Hardware) -> Fit:
    """A native build on an NVIDIA card. Drama 3 keeps the expert layers that do not fit on the CPU (slower, and
    they take system memory); the other models must fit the card."""
    gpu = hw.gpu_gb or 0.0
    if b.kind == "drama3":
        from folivox.tts import native_setup

        cpu_layers = native_setup.drama3_cpu_expert_layers(b.weights_gb, gpu) or 0
        on_cpu = b.weights_gb * native_setup.DRAMA3_EXPERT_SHARE * cpu_layers / 40
        if hw.total_gb is not None and hw.total_gb < on_cpu + 4:
            return Fit("no", f"{b.title} would keep {cpu_layers} of its 40 expert layers ({on_cpu:.0f} GB) in system "
                             f"memory next to the {gpu:.0f} GB GPU, and this computer has {hw.total_gb:.0f} GB.")
        if cpu_layers:
            return Fit("tight", f"{b.title} is larger than the {gpu:.0f} GB GPU: {cpu_layers} of its 40 expert layers "
                                f"({on_cpu:.0f} GB) run on the CPU, which is slower.")
        return Fit("ok", "")
    if b.peak_gb > gpu:  # the peak (weights, context, activations) is what has to fit on the card
        return Fit("no", f"{b.title} needs up to {b.peak_gb:.0f} GB of GPU memory and the card has {gpu:.0f} GB.")
    return Fit("ok", "")


def memory_gate(b: Build, hw: Hardware) -> str | None:
    """A reason not to start the render at all (the weights alone do not fit in the free memory), else None."""
    if os.environ.get(IGNORE_MEMORY_ENV) == "1":
        return None
    if hw.available_gb is not None and hw.available_gb < b.weights_gb:
        return (f"Only {hw.available_gb:.0f} GB of memory is free and {b.title} needs {b.weights_gb:.0f} GB just for "
                f"its weights (up to {b.peak_gb:.0f} GB at the peak). Close heavy applications (virtual "
                f"machines, containers) and try again, or set {IGNORE_MEMORY_ENV}=1 to start anyway.")
    return None


# -- installed state --------------------------------------------------------------------------------------

def engines_root() -> Path:
    return Path(user_data_dir("folivox", appauthor=False)) / "engines"


def build_dir(b: Build) -> Path:
    return engines_root() / b.id


def runtime_dir(rt_id: str) -> Path:
    return engines_root() / "runtimes" / rt_id


def weights_dir(b: Build) -> Path:
    return build_dir(b) / "weights"


def is_installed(b: Build) -> bool:
    """Weights verified and the runtime environment ready (written by ``folivox engines install``)."""
    from folivox.tts import fish_setup

    return fish_setup.is_ready(b)


# -- choice -----------------------------------------------------------------------------------------------

def candidates(hw: Hardware | None = None, cat: Catalog | None = None) -> list[Build]:
    """Verified builds that can run on this computer, most capable (and fastest) first."""
    cat = cat or load()
    hw = hw or detect_hardware()
    ok = [b for b in cat.builds if b.status == "verified" and supported_here(b, hw, cat) and fit(b, hw).level != "no"]
    return sorted(ok, key=lambda b: -b.rank)


ROOM = 0.75  # "automatic" counts on this share of the total memory: what is left when heavy applications are closed


def roomy(b: Build, hw: Hardware) -> bool:
    """The build's peak is at most ``ROOM`` of the total memory. The memory free right now plays no part: on a
    working computer it is mostly taken, while a render is meant to run with the heavy applications closed."""
    return hw.total_gb is None or b.peak_gb <= hw.total_gb * ROOM


def pick(pool: list[Build], hw: Hardware, installed=None) -> Build | None:
    """What "automatic" chooses from ``pool`` (most capable first): the most capable build with room to spare on
    this computer. Among the installed builds when there are any (none roomy: the lightest, the render warns);
    otherwise the one to download (none roomy: the first)."""
    installed = installed or is_installed
    have = [b for b in pool if installed(b)]
    if have:
        return next((b for b in have if roomy(b, hw)), min(have, key=lambda b: b.peak_gb))
    return next((b for b in pool if roomy(b, hw)), pool[0] if pool else None)


def auto_build(engine: str, hw: Hardware | None = None, cat: Catalog | None = None, installed=None) -> Build | None:
    """The build ``build_id=auto`` runs for an engine (never a native one), or None when nothing fits."""
    cat = cat or load()
    hw = hw or detect_hardware()
    return pick([b for b in candidates(hw, cat) if b.engine == engine and b.backend != "native"], hw, installed)


def recommend(hw: Hardware | None = None, accepted: dict[str, str] | None = None, installed=None,
              cat: Catalog | None = None, runtime: str | None = None) -> Build | None:
    """The default Fish build for this computer, or None (then the default engine stays CosyVoice 3).

    Among the builds that fit the total memory, whose licence the user accepted and that are installed:
    the most capable one with room to spare (Drama 3 4-bit > Drama 3 3-bit > S2 Pro 8-bit), else the lightest.
    Only builds the current inference runtime can run, as :func:`build_for` would pick them: native builds when the
    runtime resolves to native for that engine, the others otherwise. ``runtime``: requested mode (None → config).
    """
    from folivox.tts import licences
    from folivox.tts import runtime as tts_runtime

    cat = cat or load()
    hw = hw or detect_hardware()
    accepted = licences.accepted() if accepted is None else accepted
    installed = installed or is_installed
    ready = [b for b in candidates(hw, cat) if licences.is_accepted(cat.licences[b.licence], accepted) and installed(b)
             and (b.backend == "native") == (tts_runtime.resolve(b.engine, runtime) == "native")]
    return next((b for b in ready if roomy(b, hw)), min(ready, key=lambda b: b.peak_gb) if ready else None)


def default_engine(hw: Hardware | None = None) -> str:
    """Engine used when none is chosen (CLI, config or the studio): see :func:`recommend`."""
    try:
        b = recommend(hw)
    except Exception:  # noqa: BLE001 — a broken catalog or probe must never stop the default engine from working
        b = None
    return b.engine if b else "cosyvoice3"


def is_fish_engine(name: str) -> bool:
    return name in load().engines()


def build_for(engine: str, build_id: str | None = None, hw: Hardware | None = None,
              runtime: str | None = None) -> Build:
    """The build a Fish engine runs: the one named, else the best installed build of that engine that fits.

    ``runtime="native"`` (the inference switch, see :mod:`folivox.tts.runtime`) picks among the native GGUF builds,
    tested or not; any other value never picks one automatically (name the build to use it)."""
    cat = load()
    if build_id and build_id != "auto":
        b = cat.build(build_id)
        if b.engine != engine:
            raise CatalogError(f"Build {build_id} belongs to the engine {b.engine}, not {engine}")
        return b
    hw = hw or detect_hardware()
    if runtime == "native":
        pool = sorted((b for b in cat.builds if b.engine == engine and b.backend == "native"
                       and supported_here(b, hw, cat) and fit(b, hw).level != "no"), key=lambda b: -b.rank)
        if not pool:
            raise CatalogError(f"No native {engine} build fits this computer. `folivox engines` lists what is "
                               "available; use `--runtime pytorch` for the default build.")
    else:
        pool = [b for b in candidates(hw, cat) if b.engine == engine and b.backend != "native"]
    b = pick(pool, hw)
    if b:
        return b
    raise CatalogError(f"No tested {engine} build fits this computer. `folivox engines` lists what is available.")

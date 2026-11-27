"""User configuration: ``config.toml`` in the user config dir (or ``$FOLIVOX_CONFIG``).

Example::

    [providers]
    tts = "cosyvoice3"
    asr = "auto"
    llm = "ollama"
    voicegen = "qwen3-voicedesign"

    [runtime]                        # optional native C++ runtime (audio.cpp): auto | pytorch | native
    inference = "auto"

    [llm.ollama]
    model = "gemma4:31b-it-qat"

    [llm.openai]                     # any OpenAI-compatible endpoint
    base_url = "https://openrouter.ai/api/v1"
    model = "some/model"
    api_key_env = "OPENROUTER_API_KEY"   # the key itself stays in the environment

    [llm.anthropic]                  # Claude through Anthropic's API (key: ANTHROPIC_API_KEY)
    model = "claude-opus-5-5"
    effort = "medium"                # optional: low | medium | high | xhigh | max (default: high, older models none)

    [voicegen.qwen3-voicedesign]
    python = "~/.local/share/tts-bench/qwen3tts/.venv/bin/python"
    model = "~/.local/share/tts-bench/qwen3tts/models/VoiceDesign"

API keys are never read from this file — only from environment variables or the OS keychain.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

from platformdirs import user_config_dir


def config_path() -> Path:
    env = os.environ.get("FOLIVOX_CONFIG")
    return Path(env).expanduser() if env else Path(user_config_dir("folivox", appauthor=False)) / "config.toml"


@lru_cache(maxsize=1)
def load() -> dict:
    path = config_path()
    if not path.exists():
        return {}
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib
    with path.open("rb") as f:
        try:
            data = tomllib.load(f)
        except tomllib.TOMLDecodeError as e:
            raise ValueError(f"{path}: not valid TOML ({e})") from e
    _reject_keys(data)
    return data


def _reject_keys(data: dict, where: str = "") -> None:
    for k, v in data.items():
        if isinstance(v, dict):
            _reject_keys(v, f"{where}{k}.")
        elif k.lower() in ("api_key", "key", "token", "secret"):
            raise ValueError(f"{config_path()}: {where}{k} — put API keys in environment variables "
                             f"and reference them with api_key_env")


def section(dotted: str) -> dict:
    """``section("llm.ollama")`` → the ``[llm.ollama]`` table (empty if absent)."""
    node: object = load()
    for part in dotted.split("."):
        node = node.get(part, {}) if isinstance(node, dict) else {}
    return dict(node) if isinstance(node, dict) else {}


def provider(kind: str, cli_value: str | None = None) -> str | None:
    """CLI option > ``FOLIVOX_<KIND>`` env var > ``[providers]`` in config > registry default (None)."""
    return cli_value or os.environ.get(f"FOLIVOX_{kind.upper()}") or section("providers").get(kind)


RUNTIMES = ("auto", "pytorch", "native")


def runtime(cli_value: str | None = None) -> str:
    """Inference runtime: CLI option > ``FOLIVOX_RUNTIME`` > ``[runtime] inference`` in config > ``auto``."""
    value = (cli_value or os.environ.get("FOLIVOX_RUNTIME") or section("runtime").get("inference")
             or "auto")
    value = str(value).strip().lower()
    if value not in RUNTIMES:
        raise ValueError(f"Unknown inference runtime {value!r}; choose one of {', '.join(RUNTIMES)}")
    return value


def save_runtime(value: str) -> None:
    """Write ``[runtime] inference`` to config.toml (a ``.bak`` copy is kept: comments are lost on rewrite)."""
    import shutil

    import tomli_w

    if value not in RUNTIMES:
        raise ValueError(f"Unknown inference runtime {value!r}; choose one of {', '.join(RUNTIMES)}")
    path = config_path()
    data = dict(load())
    old = data.get("runtime")
    data["runtime"] = {**(old if isinstance(old, dict) else {}), "inference": value}
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
    path.write_text(tomli_w.dumps(data), encoding="utf-8")
    load.cache_clear()


def cloud_limits(provider_name: str) -> dict:
    """``[cloud.<provider>]``: ``requests_per_minute`` and ``parallel`` for one cloud engine (both optional);
    ``FOLIVOX_<PROVIDER>_RPM`` / ``FOLIVOX_<PROVIDER>_PARALLEL`` win over the file. 0 for rpm = no limit."""
    found = section(f"cloud.{provider_name}")
    out: dict = {}
    for key, env in (("requests_per_minute", "RPM"), ("parallel", "PARALLEL")):
        raw = os.environ.get(f"FOLIVOX_{provider_name.upper()}_{env}", found.get(key))
        if raw in (None, ""):
            continue
        try:
            out[key] = float(raw) if key == "requests_per_minute" else int(raw)
        except (TypeError, ValueError):
            raise ValueError(f"{key} for the {provider_name} engine must be a number, got {raw!r}") from None
        if out[key] < 0 or (key == "parallel" and out[key] < 1):
            raise ValueError(f"{key} for the {provider_name} engine must be positive, got {raw!r}")
    return out


def native_options() -> dict:
    """``[runtime.native]``: ``backend`` (auto|metal|cuda|vulkan), ``weights`` (q8_0|f32), ``weight_type``,
    ``device`` (auto = the discrete GPU if there is one, or a device number from `folivox runtime`)."""
    return {"backend": "auto", "weights": "q8_0", "weight_type": "f16", "device": "auto",
            **section("runtime.native")}


KEYCHAIN_SERVICE = "folivox"


def keychain_get(name: str) -> str | None:
    """A key saved in the OS keychain (optional ``keyring`` package; the web UI stores keys there)."""
    try:
        import keyring

        return keyring.get_password(KEYCHAIN_SERVICE, name) or None
    except Exception:  # noqa: BLE001 — no keyring installed or no backend: fall back to the environment only
        return None


def keychain_set(name: str, value: str) -> None:
    import keyring

    keyring.set_password(KEYCHAIN_SERVICE, name, value)


def keychain_delete(name: str) -> None:
    import keyring

    try:
        keyring.delete_password(KEYCHAIN_SERVICE, name)
    except keyring.errors.PasswordDeleteError:
        pass


def env_key(name: str) -> str | None:
    """A key given through the environment: ``NAME`` itself, else the file named by ``NAME_FILE`` (Docker and
    Kubernetes secrets, e.g. ``OPENAI_API_KEY_FILE=/run/secrets/openai``). An explicit ``NAME`` wins over ``NAME_FILE``.
    The file's surrounding whitespace is cut. An unreadable or empty file raises ``ValueError`` (the value is never
    part of a message); None if neither variable is set."""
    if value := os.environ.get(name):
        return value
    path = os.environ.get(f"{name}_FILE", "").strip()
    if not path:
        return None
    try:
        value = Path(path).read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError) as e:
        why = getattr(e, "strerror", None) or type(e).__name__
        raise ValueError(f"{name}_FILE: cannot read {path} ({why})") from None
    if not value:
        raise ValueError(f"{name}_FILE: {path} is empty")
    return value


def env_key_set(name: str) -> bool:
    """Is the key given through the environment (``NAME`` or ``NAME_FILE``)? Does not read the file."""
    return bool(os.environ.get(name) or os.environ.get(f"{name}_FILE", "").strip())


def api_key(env_names: list[str] | str, provider_name: str) -> str:
    """Environment variable (or its ``_FILE``) first, then the OS keychain. Never the config file."""
    names = [env_names] if isinstance(env_names, str) else env_names
    for n in names:
        if value := env_key(n):
            return value
    for n in names:
        if value := keychain_get(n):
            return value
    raise ValueError(f"{provider_name}: set the API key in the environment variable {names[0]} (or {names[0]}_FILE "
                     "with the path of a file holding it) or save it in the web interface (`folivox ui`); it is "
                     "used at your own cost and never stored in a file")


def output_dir() -> Path:
    """Where the studio puts finished audiobooks unless told otherwise: ``FOLIVOX_OUTPUT_DIR`` or ``~/Audiobooks``."""
    return Path(os.environ.get("FOLIVOX_OUTPUT_DIR") or Path.home() / "Audiobooks").expanduser()

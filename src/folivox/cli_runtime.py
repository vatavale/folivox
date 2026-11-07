"""``folivox runtime``: the optional native C++ runtime (audio.cpp): status, install, remove."""

from __future__ import annotations

import os

import typer
from rich.console import Console

runtime_app = typer.Typer(help="Optional native C++ runtime (audio.cpp): status, install, remove.",
                          no_args_is_help=False, invoke_without_command=True, rich_markup_mode="rich",
                          context_settings={"help_option_names": ["-h", "--help"]})
console = Console()


def describe() -> list[str]:
    """Plain lines about the native runtime on this computer (used by `runtime`, `setup` and `doctor`)."""
    from folivox import config
    from folivox.tts import native_setup as ns

    st = ns.status(config.native_options()["weights"])
    cat = ns.load()
    out = [f"audio.cpp {cat.release} · this computer: {st.platform}"]
    if st.binary:
        out.append(f"engine for this computer: {st.binary.id} ({st.binary.backend}, {st.binary.size / 1e6:.0f} MB, "
                   f"{'checked' if st.binary.status == 'verified' else 'not yet checked on this platform'})")
        if st.binary.note:
            out.append(f"needs: {st.binary.note}")
    else:
        out.append("no prebuilt engine for this computer (own build: set FOLIVOX_NATIVE_SERVER and "
                   "FOLIVOX_NATIVE_MODEL)")
    out.append(f"engine installed: {st.server if st.server else 'no'}")
    out += device_lines(st)
    if os.environ.get(ns.MODEL_ENV):
        out.append(f"model file: {os.environ[ns.MODEL_ENV]} (from {ns.MODEL_ENV})")
    else:
        out.append("model files: " + ", ".join(f"{k} {'yes' if v else 'no'}" for k, v in st.weights.items()))
    out.append(f"`--runtime auto` uses it: {'yes' if st.auto else 'no (PyTorch)'}")
    out += advice_lines()
    return out


def device_lines(st) -> list[str]:
    """Which GPU the engine computes on when the computer has several (Vulkan: an iGPU next to a discrete card)."""
    from folivox import config
    from folivox.tts import native_setup as ns

    if not st.server or not st.binary or st.binary.backend not in ns._MULTI_DEVICE:
        return []
    devs = [d for d in ns.list_devices(st.server, st.binary.backend) or [] if d.kind != "CPU"]
    if len(devs) < 2:
        return []
    chosen = ns.pick_device(devs, ns.device_setting(config.native_options().get("device")))
    out = ["GPUs the engine can use: " + "; ".join(
        f"{d.index} {d.name} ({'integrated' if d.kind == 'IGPU' else 'discrete'})" for d in devs)]
    if chosen:
        out.append(f"computing on: {chosen.index} {chosen.name} (automatic: a discrete GPU first; to choose another "
                   f"set `device = N` under [runtime.native] in config.toml or {ns.DEVICE_ENV}=N)")
    return out


def advice_lines() -> list[str]:
    """What our own measurements say here, in plain words (the studio shows the same facts)."""
    from folivox.tts import runtime

    a = runtime.advice()
    if a["level"] == "try":
        lo, hi = a["speedup"]
        out = [f"our test on {a['hardware']} ({a['date']}, {a['fragment']}): the native runtime was {lo:g}-{hi:g}x "
               f"faster than PyTorch (real-time factor {a['native_rtf'][0]:g}-{a['native_rtf'][1]:g} against "
               f"{a['pytorch_rtf'][0]:g}-{a['pytorch_rtf'][1]:g}); quality check score {a['qa_native']:g} against "
               f"{a['qa_pytorch']:g} (0 is perfect). It is suggested, not the default."]
        if a["note"]:
            out.append(f"note: {a['note']}")
        if a["stale"]:
            out.append(f"this was measured {a['age_days']} days ago and audio.cpp changes fast: measure again with "
                       f"`{a['bench']}`")
        return out
    return [f"not measured on {a['platform']} yet. You can try it at your own risk and measure it: "
            f"`{a['bench']}`; then share the report it prints (see docs/native-runtime.md)."]


@runtime_app.callback(help="Show whether the native runtime is installed and what `--runtime auto` does here.")
def runtime_status(ctx: typer.Context):
    if ctx.invoked_subcommand:
        return
    from folivox import config

    console.print(f"Inference runtime now: [bold]{config.runtime()}[/] (set with --runtime, FOLIVOX_RUNTIME or "
                  "`\\[runtime] inference` in config.toml)", highlight=False)
    for line in describe():
        console.print(line, markup=False, highlight=False)
    console.print("Install: folivox runtime install   ·   use: folivox render BOOK --runtime native")


@runtime_app.command("install", help="Download the native engine and the CosyVoice 3 model (asks first).")
def runtime_install(
        weights: str = typer.Option("q8_0", "--weights", help="Model file: q8_0 (2.3 GB, converted to fp16 on "
                                    "load) or f32 (7 GB, the cleanest source for fp16)."),
        backend: str = typer.Option("auto", "--backend", help="auto, metal, cuda or vulkan."),
        yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask for confirmation.")):
    from folivox.cli import RichEvents
    from folivox.tts import native_setup as ns

    try:
        p = ns.plan(weights, backend)
    except ns.NativeError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1) from None
    if not p.lines:
        console.print("[green]✓[/] Already installed. See `folivox runtime`.")
        return
    console.print("This will download:")
    for line in p.lines:
        console.print(f"  • {line}", highlight=False)
    if p.binary and p.binary.status != "verified":
        console.print(f"[yellow]This build was not yet checked by the Folivox team on {ns.platform_tag()}.[/] "
                      "PyTorch stays the default; use `--runtime native` to try it and report how it went.")
    problem = ns.disk_problem(p.todo_bytes)
    if problem:
        console.print(f"[red]{problem}[/]")
        raise typer.Exit(1)
    if not yes and not typer.confirm(f"Download {p.todo_bytes / 1e9:.1f} GB now?", default=False):
        console.print("Cancelled, nothing was downloaded.")
        raise typer.Exit(1)
    events = RichEvents()
    try:
        ns.install(p, events.download, lambda m: console.print(m, markup=False, highlight=False))
    except (ns.NativeError, RuntimeError, OSError) as e:
        events.close()
        console.print(f"[red]Install stopped:[/] {e}\nRun the same command again to continue.")
        raise typer.Exit(1) from None
    events.close()
    console.print("[green]✓[/] Native runtime installed. Try it: folivox render BOOK --runtime native --preview 400")


@runtime_app.command("build-help", help="Where everything comes from and how to build audio.cpp on each platform.")
def runtime_build_help(
        target: str = typer.Argument("", help="macos, linux-cuda, linux-vulkan, windows-cuda, windows-vulkan or "
                                    "amd-hip (default: this computer).")):
    from folivox.tts import native_build
    from folivox.tts import native_setup as ns

    src = ns.sources()
    up, fk = src["upstream"], src["drama3"]
    console.print(f"[bold]Where things come from[/]\n  engine: {up['repo']} (release {up['release']}, Apache-2.0)\n"
                  f"  models: {up['gguf']} (each model keeps its own licence)\n"
                  f"  Drama 3 (released since v0.9.4): {fk['repo']} {fk['branch']}, commit {fk['commit'][:8]};\n"
                  f"    weights {fk['gguf']}; docs {fk['docs']}", highlight=False)
    steps = src["build"].get(target) if target else None
    if target and steps is None:
        console.print(f"[red]Unknown target {target!r}; known: {', '.join(src['build'])}[/]")
        raise typer.Exit(2)
    lines = steps if steps is not None else native_build.platform_recipe()
    console.print(f"\n[bold]Build steps{f' ({target})' if target else f' for {ns.platform_tag()}'}[/] "
                  "(replace MODELS with cosyvoice3, fish_audio or fish_drama3):", highlight=False)
    for line in lines:
        console.print(f"  {line}" if line else "", markup=False, highlight=False)
    console.print("\nAfter building: set FOLIVOX_NATIVE_SERVER to the audiocpp_server file and FOLIVOX_NATIVE_MODEL to "
                  "a GGUF if you use your own model. On Apple Silicon `folivox runtime build drama3` does it for "
                  "Drama 3.", markup=False)


@runtime_app.command("build", help="Compile the Drama 3 server from the pinned audio.cpp source (Apple Silicon).")
def runtime_build(target: str = typer.Argument("drama3", help="drama3"),
                  yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask for confirmation.")):
    from folivox.tts import native_build
    from folivox.tts import native_setup as ns

    try:
        why = native_build.can_build_here()
        if why:
            console.print(f"[red]{why}[/]")
            raise typer.Exit(1)
        src = native_build.source(target)
    except ns.NativeError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(2) from None
    if native_build.own_server(target):
        console.print(f"[green]✓[/] Already built: {native_build.own_server(target)}")
        return
    console.print(f"This downloads the audio.cpp source at a pinned commit ({src['repo']}, "
                  f"{src['commit'][:8]}, newer than the last release) and compiles it on this computer (several "
                  "minutes, about 1 GB of temporary space). Folivox has not reviewed that code; it runs with your "
                  "permissions.",
                  highlight=False)
    if not yes and not typer.confirm("Build it now?", default=False):
        console.print("Cancelled.")
        raise typer.Exit(1)
    try:
        path = native_build.build(target, lambda m: console.print(m, markup=False, highlight=False))
    except ns.NativeError as e:
        console.print(f"[red]Build stopped:[/] {e}")
        raise typer.Exit(1) from None
    console.print(f"[green]✓[/] Built {path}. No Folivox build uses it yet: "
                  "native Drama 3 builds wait for an audio.cpp release")


@runtime_app.command("remove", help="Delete the downloaded native engine and model (PyTorch files are not touched).")
def runtime_remove(yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask for confirmation.")):
    from folivox.tts import native_setup as ns

    if not yes and not typer.confirm("Delete the native engine and model files?", default=False):
        raise typer.Exit(1)
    from folivox.tts import native

    native.shutdown_all()
    console.print(f"Freed {ns.remove() / 1e9:.1f} GB.")

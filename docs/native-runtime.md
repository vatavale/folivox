# Native runtime (optional)

By default CosyVoice 3 runs on PyTorch. Folivox can instead run it on **[audio.cpp](https://github.com/0xShug0/audio.cpp)**,
a C++/GGML inference engine (Apache-2.0, ShugoAI LLC) with Metal, CUDA and Vulkan backends. It is optional, needs a
one-time download, and never replaces the PyTorch path: if anything about it fails, `--runtime pytorch` still works.

```bash
folivox runtime                      # what is installed, which build fits this computer
folivox runtime install              # asks, then downloads the engine and the model (about 2.3 GB on a Mac)
folivox render book.epub --runtime native --preview 400
```

## The switch

One general switch for the whole product, not per model:

| Where | How |
|---|---|
| CLI | `--runtime auto\|pytorch\|native` |
| Environment | `FOLIVOX_RUNTIME` |
| `config.toml` | `[runtime] inference = "auto"` |
| Studio | Advanced settings → *Inference runtime* (saved at once; also installs and removes the engine) |

* `pytorch`: always PyTorch.
* `native`: always the C++ engine; if it is not installed the render stops with a clear message (no silent switch).
* `auto` (default): native only on a platform whose entry in `src/folivox/tts/native_catalog.json` (`auto_ok`) records
  that it passed the quality check, and only when installed; otherwise PyTorch. **`auto_ok` is empty today, so `auto` means
  PyTorch everywhere.** It is changed by data once measurements exist (see "What is verified").

Only engines with a native implementation are affected (CosyVoice 3 today). Fish and the cloud engines ignore the
switch. `--device` is a PyTorch setting; the native engine picks its own GPU.

## How it runs

Folivox starts one `audiocpp_server` on `127.0.0.1` (a free port, no UI, one model loaded), sends it one HTTP request
per fragment and reads the WAV back. A process per fragment would pay loading time (about 13 s on a Mac: GPU kernel
compilation plus the weights) every time, so the server is kept: it is shared by renders and previews and stopped after
5 idle minutes (`FOLIVOX_NATIVE_IDLE` seconds) or when Folivox exits. A crashed server is restarted once and the
fragment repeated; a stuck one is killed after the usual per-fragment limit and the render can be resumed. The server
log is `native/logs/audiocpp_server.log` in the Folivox data folder (it goes into `folivox doctor --bundle`).
While it runs, any program on this computer can reach its port.

The reference voice is converted to a mono 24 kHz WAV (audio.cpp reads WAV only). Reading speed is applied after
synthesis (ffmpeg `atempo`), because the native model has no speed option. `--llm` is ignored: there is one
published GGUF of the model.

Takes made natively have their own journal key (engine release, weights file, precision), so switching runtime
re-reads only what must be re-read and never mixes up audio silently.

## What gets installed

`folivox runtime install` shows the plan (what, how many GB) and asks. Everything is pinned in
`native_catalog.json`, with sizes and SHA-256; archives are checked while they download and the model after it,
and archives are unpacked only into `<data folder>/native/` (paths and links that would escape it are refused).

| Platform | Build | Download | Status |
|---|---|---|---|
| macOS, Apple Silicon | release `macos-arm64-metal` | 32 MB | checked on one M3 Pro |
| Windows x64, NVIDIA (driver 580+, RTX 20 and newer) | `cuda13.3` + its `cudart` archive | 0.85 GB | **not tested** |
| Windows x64, NVIDIA, older driver | `cuda12.4` + `cudart` | 1.07 GB | **not tested** |
| Windows x64, any GPU with Vulkan 1.1 (AMD, Intel, NVIDIA) | `vulkan` | 63 MB | **not tested** |
| Linux x64, NVIDIA (RTX 20 and newer), also in Docker | `ubuntu-x64-cuda12.8` + `cudart` | 1.0 GB | **not tested** |
| Linux x64, Vulkan | `ubuntu-x64-vulkan-portable` | 74 MB | **not tested** (the plain build was dropped: it crashes with an illegal instruction on a Ryzen 9 5900HX, no AVX-512) |

Model (Hugging Face `audio-cpp/audio.cpp-gguf`, pinned revision, Apache-2.0 weights): `cosyvoice3-q8_0.gguf`
(2.26 GB, default) or `cosyvoice3-f32.gguf` (7 GB; `--weights f32`). The engine converts the weights to
`weight_type` (default `f16`) while loading, so the Q8 file means "fp16 of Q8-rounded weights" and the F32 file is the
cleaner source; which one sounds better is part of the quality check below.

Linux with CUDA uses the upstream build made for Colab: CUDA 12.8, kernels for compute capability 7.5 to 9.0
(RTX 20 to 40 series, A100, H100), no CPU-specific instructions. RTX 50 cards are not in it; they get the Vulkan build.
Not offered automatically: AMD HIP on Windows (community builds in a fork; not pinned by Folivox), Intel Macs.

**Your own build.** `FOLIVOX_NATIVE_SERVER=/path/to/audiocpp_server` and `FOLIVOX_NATIVE_MODEL=/path/to/cosyvoice3.gguf`
are used instead of the installed files (a CUDA or HIP build from source, for example); `folivox runtime` shows them.

## Which GPU it computes on

With the Vulkan (and HIP) build the engine takes device 0, and on a laptop that is often the integrated GPU in front of
the discrete card (about 5 times slower in our test). Folivox asks the engine for its devices (`--list-devices`) and
passes the first discrete GPU to it; `folivox runtime` lists the GPUs and says which one is used. To choose another one:

```toml
[runtime.native]
device = 0        # a number from `folivox runtime`; "auto" (default) = the discrete GPU if there is one
```

or `FOLIVOX_NATIVE_DEVICE=0`. If a GPU backend finds no driver for any GPU, the render stops with an explanation that names the card it can see (from `nvidia-smi` or `/dev/dxg`) and says the driver is what is missing. Docker Desktop
on Windows (WSL2) gives containers CUDA and Direct3D but no Vulkan driver, so the Vulkan build cannot use the GPU in a
container there; use the default PyTorch engine, which does use CUDA.

## Where everything comes from, and how to build it

`folivox runtime build-help [macos|linux-cuda|linux-vulkan|windows-cuda|windows-vulkan|amd-hip]` prints this from the
catalog (`src/folivox/tts/native_catalog.json`, section `sources`):

* engine: <https://github.com/0xShug0/audio.cpp> (release `v0.9.4`, Apache-2.0); prebuilt archives are listed above
* models: <https://huggingface.co/audio-cpp/audio.cpp-gguf> (CosyVoice 3, Fish S2 Pro); every model keeps its own licence
* Drama 3: in the release binaries since v0.9.4; weights are in `Jmica/Drama-3-GGUF` (the V2 checkpoint, pinned
  commit and sha256 in the catalog). On an NVIDIA card smaller than the file, Folivox keeps part of the 40 expert
  layers on the CPU (`fish_drama3.cpu_expert_layers`, estimated from the card's memory; override with
  `drama3_cpu_expert_layers` under `[runtime.native]` or `FOLIVOX_DRAMA3_CPU_EXPERT_LAYERS`). The earlier home of the port,
  `JarodMica/audio.cpp` branch `drama-3-native`, is kept in the catalog as a fallback.
* build recipes for macOS (Metal), Linux (CUDA, Vulkan), Windows (CUDA, Vulkan) and AMD (community HIP packages) are
  data in the catalog and are the steps the upstream documentation gives; only the macOS one is automated.

`folivox runtime build drama3` (Apple Silicon): fetches the pinned commit, gets CMake and Ninja through `uv` if they are
missing, compiles `audiocpp_server` with the project's `scripts/build_metal.sh` (several minutes, about 1 GB
of temporary space) and keeps the server in the Folivox data folder. It asks first: the source is third-party code that
Folivox has not reviewed. On other systems build it by hand and set `FOLIVOX_NATIVE_SERVER`.

## Re-checking (audio.cpp changes fast)

audio.cpp is a young project. Releases, packages, speed and model support change within days (Drama 3 appeared in
`main` while this was being written). Every measurement here carries its date, the studio and `folivox runtime` mark
those older than 60 days, and `python tools/native_catalog.py watch` (read-only) reports a newer release, a moved
fork or Drama 3 branch, changed model repositories, whether Drama 3 reached a release, and which measurements are old.
Run it every month or two; after a change worth adopting, update the catalog pins, run `folivox bench` and record the
numbers.

## What the measurements change in the product

* `folivox runtime` and *Advanced settings → Inference runtime* show what our own runs found on your platform (speed-up,
  quality-check score, date) and suggest the native runtime where it was clearly faster; where nothing was measured they
  say so and show the command to measure. A suggestion never switches anything on by itself: `auto` stays PyTorch until
  the quality gate below is met.
* `folivox setup` shows the same facts when it offers the download.
* The Fish card in the studio and `folivox engines` show the measured speed of each build, and warn when the native
  build was slower than the MLX build of the same engine (render warning too).
* `folivox bench` measures any engine and runtime (`--tts fish-drama3 --runtime native`) and prints a report to
  paste into an issue, without private data.

## Fish Audio models (S2 Pro, Drama 3)

`--runtime native` also selects the native GGUF builds of the Fish engines (`s2pro-native-q8`, `s2pro-native-bf16`;
see [Fish Audio engines](fish-engines.md#native-builds-audiocpp-any-platform-experimental)). Measured once on the M3 Pro (see the Fish page): S2 Pro Q8 RTF 2.52 and bf16 RTF 4.03, QA scores 0.004 and
0.002; both slower than the MLX builds there. Without the switch
Fish keeps using its MLX builds on Apple Silicon. S2 Pro and Drama 3 (`drama3-native-q2`, `drama3-native-q3`) use the stock audio.cpp release (Drama 3 since v0.9.4).

## Licences

audio.cpp is Apache-2.0 (the engine only; the downloaded release carries its LICENSE). Weights keep their own licence:
the CosyVoice 3 weights are Apache-2.0. Fish S2 Pro (Research License, non-commercial) and Drama 3 (Fish Community
License) are not part of this runtime and keep their existing notices and acceptance flow.

## What is verified

Measured on one MacBook Pro (M3 Pro, 36 GB), the standard Russian test fragment (898 characters, 74 s of audio),
voice `russlib`, CosyVoice 3, Metal, model weights Q8 converted to fp16:

| | RTF (synthesis time / audio time) |
|---|---|
| PyTorch on MPS | 1.93 (earlier measurement) |
| native, engine kept running (what Folivox does) | about 0.6 (45 s for 74 s of audio, model load 14 s extra, once) |
| native, `audiocpp_cli` per fragment (not used) | 0.95 (includes a start of about 13 s per call) |
| native with the default Q8 weights, no conversion | 2.93 (slow on Metal; this is why `weight_type = f16` is the default) |

The numbers above are single runs on one machine (a second PyTorch run on the same fragment gave RTF 1.40, so
the speed-up is between about 2x and 3x, not a precise figure).

**Quality is not established.** The only check so far is the Whisper QA of `folivox render` on those six fragments
(one take each, no retakes): mean error score 0.012 for the native runtime and 0.004 for PyTorch. Five native fragments
were clean; in one the model dropped the last clause of a long sentence ("что там в огороде"), a known
CosyVoice weakness that PyTorch shows as well and that the QA retake normally catches. Nobody has listened critically
and there is no comparison over many fragments. The Folivox quality gate for making native the default on a
platform: character error rate no worse than PyTorch by more than 1 percentage point on the same fragments, no runaway,
empty or truncated takes over at least 30 fragments, and a listening comparison.

**Measured once on Windows** (2026-10-20, Ryzen 9 5900HX laptop with a Radeon iGPU and an RTX 3050 Laptop 4 GB, driver 561,
one 440-character `ru-hard` preview, quality check not run): CUDA 12.4 build real-time factor 0.49 (model load 23 s);
Vulkan build on the RTX 3050 0.66 (load 15 s); Vulkan build on the Radeon iGPU 2.37. The 4 GB card held the model.
That Radeon was device 0, so the engine alone would have used it; Folivox now picks the discrete GPU (next section).

**Not verified at all:** CUDA 13.3 on Windows, anything on Linux, other AMD and Intel GPUs (Vulkan), Windows
as a whole (paths with non-Latin user names, PowerShell installer path, antivirus and unsigned executables), the F32
file on Metal, Intel Macs. audio.cpp's own documentation calls CUDA its optimised path and Vulkan/Metal "portability";
its published speed tables (RTX 5090, against Python) do not include CosyVoice 3, so the speed on NVIDIA relative to
PyTorch CUDA is unknown.

### Help test it

```bash
folivox runtime install
folivox bench --devices native,cuda        # or cpu / mps: compares the runtimes on your machine
folivox render book.epub --runtime native --preview 600 --voice your-voice.wav
folivox doctor
```

Please report the `folivox bench` table, `folivox runtime`, `folivox doctor` output, and whether the preview sounds like
PyTorch's (same voice, same text). AMD and Intel owners: the Vulkan build is the interesting one
([#3](https://github.com/vatavale/folivox/issues/3) tried a Vulkan C++ build of CosyVoice on a Ryzen 780M).

## Pitfalls we met (from audio.cpp's docs and our runs)

* Always give the model family (`--family cosyvoice3` / `"family"` in the server config); without it startup is slow.
* The reference must be WAV; FLAC is rejected. Bundled voices are converted automatically.
* Lazy-loaded models stay in memory until the server exits; Folivox loads at start and stops the server when idle.
* Quantised weights are model- and route-specific: Q8 on Metal was 3x slower than fp16 here, and audio.cpp warns that
  low precision can change quality or even length.
* Flash attention is built in for CosyVoice 3 (no switch); it was fine on Metal, but another C++ project produced
  runaway audio with it, so the QA step stays on by default.
* The CUDA builds need the separate `cudart` archive unpacked next to the binaries (Folivox does it) and an NVIDIA
  driver 580+ (CUDA 13.3 build) with a GPU of compute capability 7.5 or newer.

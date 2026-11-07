<div align="center">

# Folivox

**The audiobook workhorse. A powerful tool in a simple interface, free.**

Drop in a book, pick a voice, press Render. Get a chaptered audiobook read in any voice you like,
on your own computer.

FB2 · EPUB · DOCX · PDF · TXT &nbsp;→&nbsp; `.m4b` / `.mp3` with chapters, cover and metadata

![Python 3.10–3.12](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue)
![License Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-green)
![Platforms](https://img.shields.io/badge/platform-macOS%20%7C%20Windows%20%7C%20Linux-lightgrey)
![Status early development](https://img.shields.io/badge/status-early%20development-orange)

<img src="docs/img/studio-progress.png" alt="Folivox studio rendering a book: progress, chapter lanes, live fragments" width="820">

</div>

## What is this

Folivox turns a whole book, many hours of it, into a finished audiobook. Out of the box it reads with
[CosyVoice 3](https://github.com/FunAudioLLM/CosyVoice), an open model that clones a voice from a short sample,
running entirely on your machine: no account, no subscription, no per-character fee, nothing uploaded.

It is one application with many engines behind it. The book can be read by the free local CosyVoice 3, by the
local [Fish Audio Drama 3](docs/fish-engines.md) (80+ languages, several voices in one scene; needs a Mac with enough memory),
or by ElevenLabs, OpenAI, Gemini, Cartesia or Fish Audio online with your own API key. A narrator voice can come from the bundled
catalog, from your own recording, or be designed for this particular book. Everything around the voice,
from text clean-up to quality control, chapters and a crash-safe journal, is the same whichever engine you choose.

Use the **web studio** (`folivox ui`, in 49 interface languages) or the **command line**: same pipeline, same
results.

## Quick start

### 1. Install

Pick one way. Each needs no administrator rights and is safe to run again.

**One command** (recommended). Open a terminal (macOS: **Terminal** in Applications → Utilities; Windows:
**PowerShell** from the Start menu; Linux: any terminal), paste the line for your system and press Enter.

macOS and Linux:

```bash
curl -fsSL https://raw.githubusercontent.com/vatavale/folivox/main/scripts/install.sh | bash
```

Windows (PowerShell):

```powershell
irm https://raw.githubusercontent.com/vatavale/folivox/main/scripts/install.ps1 | iex
```

<details>
<summary><b>With git</b> (if you want the source code)</summary>

macOS and Linux:

```bash
git clone https://github.com/vatavale/folivox.git
cd folivox
bash scripts/install.sh
```

Windows (PowerShell):

```powershell
git clone https://github.com/vatavale/folivox.git
cd folivox
powershell -ExecutionPolicy Bypass -File scripts\install.ps1
```

</details>

<details>
<summary><b>With Docker</b> (home server or NAS)</summary>

Optional, not needed on a normal computer. Create a stack from [docker-compose.yml](docker-compose.yml) (or
[docker-compose.nvidia.yml](docker-compose.nvidia.yml) for an NVIDIA GPU), set `FOLIVOX_UI_TOKEN` to a password, deploy
and open `http://<server>:8765/`. Works from Portainer, Dockhand or `docker compose up -d`. Details:
[docs/docker.md](docs/docker.md).

</details>

### 2. Start

Open a new terminal window and run:

```bash
folivox ui
```

Your browser opens the studio. Drop a book, choose a voice, press **Render audiobook**. To stop the studio,
press `Ctrl+C` in the terminal.

The installer sets up everything for you: `uv`, ffmpeg, the PyTorch build that fits your hardware (NVIDIA GPU,
Apple Silicon or CPU) and Folivox. Then it checks your computer and downloads the models (about 6 GB for the
voice and 1.6 GB for the quality check) with a progress bar; an interrupted download continues where it
stopped. Prefer the terminal to the studio? `folivox render book.fb2` puts the audiobook next to the book.

### 3. Update

```bash
folivox update
```

Whichever way you installed, this fetches the latest version and keeps your models, voices, settings and
unfinished renders. Check the version with `folivox --version`.

### 4. Uninstall

```bash
folivox uninstall               # delete the downloaded models (asks first; --dry-run only shows what would go)
folivox uninstall --everything  # also delete your voices and settings
uv tool uninstall folivox       # last step: remove the program itself
```

It removes only what Folivox created. Your books, audiobooks, ffmpeg and uv are never touched.

Details, options, manual install and troubleshooting: [docs/install.md](docs/install.md).

## Choose how your book is read

| | Free, on your computer | With an online engine (your API key) |
|---|---|---|
| **Who reads** | CosyVoice 3 or Fish Audio Drama 3 (80+ languages), local and offline | ElevenLabs, OpenAI, Gemini, Cartesia or Fish Audio |
| **Voice** | Clones any sample: bundled, recorded, uploaded, cut from a video or generated; several voices in one book | The provider's voices, or a voice designed or localised for the book |
| **Cost** | Free (Drama 3 needs a Mac with enough memory) | Estimated before you start, confirmed by you, with an optional spending cap |
| **Your text** | Never leaves your computer | Sent to that provider |

Paste a key for any of the five providers in the studio (it is stored in your OS keychain) and they appear
next to the local engine. With the model set to `auto`, Folivox asks the provider which models your key can
use and takes the newest one, so new releases are picked up without updating Folivox.

## A voice for every book

- **Ready to use:** a bundled catalog of public-domain narrator voices for the book languages, with audition.
- **Your own voice:** record it with the microphone right in the studio or upload a 6–15 second sample;
  optional gentle noise reduction.
- **Designed for this book:** describe the narrator, or press **Suggest from my book** and a text LLM writes
  the description from the book itself. An online provider (or the local Qwen3 voice-design model) reads a short
  text, Whisper ranks the takes, and you keep the best one as a cloning prompt for the free local engine.
- **From a video or a long recording:** drop an MP4 or a long audio file on **Add your own voice** and Folivox
  finds the cleanest 6–15 s stretches (single speaker, no music), shows them with their text and lets you pick one.
  The studio also checks the recognised text of a sample and warns when it does not fit the recording.
- **Your library:** every voice you add is kept on your computer, with its source and licence, searchable by
  name and language; star your favourites.

Full guide: [docs/voices.md](docs/voices.md).

### Recommended: the almost-free workflow

The best results we get come from combining one cheap online sample with the free local engine:

1. **Generate a voice sample with Gemini 3.8.** In the studio: voice gallery → **Generate a voice sample** →
   Gemini, describe the narrator (or press **Suggest from my book**), generate a few short takes. With a paid
   Gemini key this costs well under one cent (about $0.0023 per 10 s of audio on the paid tier); Google also has
   a free tier.
2. **Keep the best take** (Whisper ranks them) as a voice in your library.
3. **Render the whole book locally** with CosyVoice (or Drama 3) cloning that voice: free, unlimited, offline.

You pay once for about ten seconds of a top-quality voice, and the open model reads the remaining hours
with it. CLI equivalent:

```bash
export GEMINI_API_KEY=…
folivox voices design narrator -g gemini --mode design --lang en --describe "calm, warm audiobook narrator" --keep takes/
folivox render book.epub --voice narrator
```

### The very best quality: a full online render

Reading the whole book with a commercial engine still sounds better than local cloning. For that we recommend
**Gemini 3.8 Flash TTS**: the best narration in our experience and at the same time one of the cheapest engines,
about $0.8 per hour of audio at list prices (October 2026; a ten-hour book is roughly $8). Folivox shows the
estimate before you start and can stop at a spending cap:

```bash
folivox render book.epub --tts gemini --tts-mode design --tts-describe "calm, warm audiobook narrator" --max-cost 10
```

Prices, limits and terms of all five engines: [docs/cloud-engines.md](docs/cloud-engines.md).

## Features

- **Many formats in.** FB2 (also `.fb2.zip`), EPUB, DOCX, PDF (text layer, no OCR) and TXT. Chapters, nested sections, poems and epigraphs are kept; PDF headers, footers and page numbers are removed and hyphenation is undone.
- **Fish Audio Drama 3.** Directed in plain words (`[calm, warm narration]`), 80+ languages, several speakers in one scene. Runs locally on a Mac with enough memory; an online variant with your key is optional. [Details](docs/fish-engines.md).
- **10 languages with full rules, 80+ with Drama 3.** Russian, English, German, French, Spanish, Italian, Chinese, Japanese, Korean and Dutch get their own text rules (CosyVoice reads the first nine; Dutch needs a Fish or online engine). Any other language Fish Audio Drama 3 reads, also locally (84 in all, e.g. Polish, Portuguese, Arabic, Hindi, Thai), works too. Detected from the book's text; `--lang` overrides it. Books with passages in other languages are read fragment by fragment in the right language.
- **Text prepared for speech.** Numbers, dates, times, money, units, roman numerals and abbreviations read in full, with case and gender agreement where the language has it (Russian cases, Spanish `veintiuna`, French `une`, German `eine`, Korean native counters, Chinese `两`); sentences split and packed into fragments; dialogue pauses. [Per-language table](docs/how-it-works.md#what-the-normaliser-reads-out-per-language).
- **Stress, `ё` and your own pronunciations.** For Russian, `ё` is restored automatically with ruaccent, and full stress marks are available (`--stress all`, experimental). Names and invented words: a plain-text dictionary next to the book (`Kvothe = Quothe`), also editable in the studio. [Details](docs/pronunciation.md).
- **Quality control.** Every fragment is listened back with Whisper. If it does not match the text it is read again, and the best take is kept (`--takes`, default 3).
- **Real audiobook files.** One `.m4b` with chapter markers, cover and metadata (title, author, narrator, year, description, series), or one `.mp3` per chapter. Loudness-normalised (−18 LUFS).
- **Several voices in one book.** A narrator and characters, each with their own voice and expression tags (`[sighs]`). A local LLM can mark the speakers automatically; check and fix them in the studio's markup editor. [Details](docs/markup.md).
- **A voice from any recording.** Drop a video or a long recording: Folivox finds clean 6–15 s stretches and shows them with their text. [Details](docs/voices.md).
- **Built for 10-hour books.** Every finished fragment is journalled. Stop, crash or lose power, then run again and it continues. Edit the text and only changed fragments are redone. The previous file is never replaced until the new one is complete.
- **Studio with live progress.** Chapter lanes, the fragment being read, elapsed and remaining time, retakes, playback of finished chapters, a quick preview before the long run. Local only, token-protected.
- **Runs on what you have.** Apple Silicon (GPU via Metal, Whisper via MLX), NVIDIA CUDA, or a plain CPU; If the GPU fails, the CPU takes over. `folivox doctor` reviews your hardware and gives advice; `folivox bench` measures the real speed.
- **Interface in 49 languages.** English, Russian, German, French, Spanish, Italian, Chinese, Japanese, Korean, Portuguese, Dutch, Catalan, Greek, Albanian, Swedish, Danish, Norwegian, Finnish, Icelandic, Estonian, Latvian, Lithuanian, Polish, Czech, Slovak, Hungarian, Romanian, Ukrainian, Bulgarian, Serbian, Croatian, Slovenian, Macedonian, Turkish, Arabic, Hebrew, Persian, Urdu, Hindi, Bengali, Marathi, Telugu, Tamil, Gujarati, Indonesian, Malay, Vietnamese, Thai and Swahili (menu in the header). Arabic, Hebrew, Persian and Urdu mirror the page right to left. [Full list](docs/ui-languages.md).
- **Free, with no limits.** The local engine reads any number of books of any length at no cost. Only if you choose a paid online engine: a cost estimate first, your confirmation, a clean stop at `--max-cost`, no fragment paid for twice.
- **Sturdy and open.** Early ffmpeg and disk checks, resumable downloads, retries with backoff, waiting out network outages, a redacted support bundle. Pluggable providers via entry points.

### A closer look

<table>
<tr>
<td width="50%"><img src="docs/img/studio-book.png" alt="A loaded book with cover, author, language shares and three chapters"><br><sub>Drop a book: cover, author, languages with their shares, chapters and estimated length.</sub></td>
<td width="50%"><img src="docs/img/studio-engine.png" alt="Reading engine: Fish Audio Drama 3, free and local, with its builds"><br><sub>Who reads the book: a free local engine (CosyVoice 3, Fish Audio Drama 3 and its builds) or an online one.</sub></td>
</tr>
<tr>
<td width="50%"><img src="docs/img/studio-voices.png" alt="Voice step: one voice or several voices, characters with their voices"><br><sub>One voice for the whole book, or several: the characters found in the book each get their own voice.</sub></td>
<td width="50%"><img src="docs/img/studio-generate.png" alt="Generate a voice sample: choose an online service"><br><sub>No good recording? Generate a ten-second voice sample with an online service for a few cents and use it for free, offline.</sub></td>
</tr>
</table>

<sub>Demo render on a public-domain text (<i>Alice's Adventures in Wonderland</i>); timings are illustrative.</sub>

<p align="center"><img src="docs/img/studio-markup.png" alt="Markup editor: a short story with dialogue highlighted by character, speakers with their voices" width="900"><br><sub>Several voices: the program finds the characters and gives each their lines; check, fix and listen in the editor. The Russian lullaby line is tagged <code>RU</code> and read in its own language.</sub></p>

## Why Folivox

- **Free for real.** The local engine has no limits and no bill; online engines are an option, not a requirement.
- **A workhorse for long books.** Resumable, crash-safe, quality-checked, with a time forecast: start a
  ten-hour book and walk away.
- **Speaks every book's language properly.** A text normaliser for each of the 10 languages with rules reads numbers,
  dates, money, units and roman numerals the way a narrator would, with gender and case agreement.
- **Simple outside, deep inside.** A few clicks in the studio; every option is still there in the CLI.
- **Any voice, consistent to the last page.** Yours, a bundled one, or one designed for the book.
- **Honest about what leaves your computer.** Online use is opt-in, labelled, estimated and capped.
- **Open and replaceable.** Apache-2.0 and small provider interfaces: when a better model appears, plug it in.

## Usage in short

```bash
folivox inspect book.epub                     # chapter tree, size, estimated duration
folivox book.fb2 -v me.wav -o book.m4b        # render the whole book with your voice
folivox book.fb2 -c 1-2 --preview 600         # a quick test: chapters 1–2, first 600 characters each
folivox book.pdf --mp3 -o out/                # one mp3 per chapter
folivox render book.epub --tts openai --tts-voice coral --max-cost 5   # an online engine with a cap
folivox voices                                # your voice library
folivox doctor --bundle                       # check the setup / write a redacted support bundle
folivox bench                                 # speed test: characters per second on each device (CPU, CUDA, MPS)
```

`folivox bench` synthesises one short fragment on every device it finds, each in its own process, and prints
characters per second, RTF and peak GPU memory, so you can see what actually helps on your machine and compare
machines (`--devices cpu,cuda` picks devices; CUDA is tried even if `auto` would skip a small GPU).

`folivox --help`, `folivox render --help` and `folivox help <topic>` (`voices`, `providers`, `config`,
`languages`, `resuming`, `quality`, `speed`) explain every option. Full list: [docs/cli.md](docs/cli.md).

## FAQ

**What hardware do I need?** Python 3.10–3.12, ffmpeg and about 7.6 GB for the models. A Mac with Apple
Silicon, or Windows/Linux with an NVIDIA GPU or just a CPU. The CPU works but is slower. `folivox setup` and
`folivox doctor` look at your machine and tell you what to expect. (AMD and Intel graphics cards: an experimental
route exists, see [docs/native-runtime.md](docs/native-runtime.md).)

**How long does a book take?** It depends on your hardware, so we do not quote a number. The studio shows the
audio length before you start, estimates the render time from your own previous runs, and shows the remaining
time while it works. Try a preview first to hear the voice on a few hundred characters.

**Is my book uploaded anywhere?** Not with the local engine (the default): everything runs on your computer
and the studio listens on `127.0.0.1` only; the network is used to download the models. Online engines,
online voice generation and OpenAI-compatible recognition or LLM providers **send text (or audio) to that
provider and may cost money**; Folivox says so and asks first. Keys live in environment variables or your OS
keychain, never in a config file.

**Can I stop and continue?** Yes, at any time, even after a crash or a power cut: run the same command (or
press Render) again.

**Whose voice may I clone?** Your own, or one you have explicit permission to use. Folivox ships no celebrity
or third-party voices and has no feature for imitating real people.
The bundled voices are real readers' public-domain recordings from LibriVox; a reader can ask for their recording to be removed, and the voice will then be dropped.

**How do I remove it?** `folivox uninstall` deletes the models Folivox downloaded (it shows the list and asks first;
your voices and settings stay unless you add `--everything`), then `uv tool uninstall folivox` removes the program.
Your books, audiobooks, ffmpeg and anything you had before are never touched. Details: [docs/install.md](docs/install.md).

## Documentation

| Page | What is in it |
|---|---|
| [docs/install.md](docs/install.md) | One-command install, starting the studio, update and removal |
| [docs/docker.md](docs/docker.md) | Optional: run the studio in Docker on a home server or NAS |
| [docs/cli.md](docs/cli.md) | All `folivox render` options and examples |
| [docs/pronunciation.md](docs/pronunciation.md) | Custom pronunciations: names, invented words, Russian stress, phonemes |
| [docs/voices.md](docs/voices.md) | Voice library, your own voice, the voice sample, designing and generating voices |
| [docs/cloud-engines.md](docs/cloud-engines.md) | Online reading engines: prices, limits, terms, spending cap |
| [docs/markup.md](docs/markup.md) | Several voices in one book, expression tags, text/JSON markup |
| [docs/fish-engines.md](docs/fish-engines.md) | Optional Fish Audio engines (S2 Pro, Drama 3): licences, memory, catalog |
| [docs/native-runtime.md](docs/native-runtime.md) | Optional native (audio.cpp) runtime for faster local reading |
| [docs/providers.md](docs/providers.md) | Pluggable providers and `config.toml` |
| [docs/ui.md](docs/ui.md) | The web studio: design, screens and API |
| [docs/ui-languages.md](docs/ui-languages.md) | The 49 interface languages, right-to-left support, translations |
| [docs/how-it-works.md](docs/how-it-works.md) | The pipeline in five steps |
| [docs/support.md](docs/support.md) | Resuming, retries, `folivox doctor` and the support bundle |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Development setup and code structure |

## Development

```bash
uv sync --all-extras
uv run pytest -q
uv run ruff check src tests
```

The CI workflow (`.github/workflows/ci.yml`) runs lint and the unit tests on macOS, Windows and Linux without
model inference. Notes for working on the code: [CONTRIBUTING.md](CONTRIBUTING.md).

## Feedback

Found a bug, a mispronounced word or a book that does not open? Please
[open an issue](https://github.com/vatavale/folivox/issues) with the steps to reproduce; for crashes attach
the redacted bundle from `folivox doctor --bundle`. Ideas and feature requests are welcome there too.

Folivox is maintained by one person, so before writing code for a larger change, please open an issue
first to talk it over.

## Acknowledgements

Folivox stands on the work of many people who shared it openly. Thank you:

- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) by the FunAudioLLM team (Alibaba Tongyi Lab): the
  voice-cloning model and inference code at the heart of Folivox.
- [Fish Audio](https://fish.audio/) for the [S2 Pro](https://huggingface.co/fishaudio/s2-pro) and
  [Drama 3](https://huggingface.co/fishaudio/drama-3) models (Fish Audio Community License), and
  [audio.cpp](https://github.com/0xShug0/audio.cpp) for running them natively.
- [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS) and [Whisper](https://github.com/openai/whisper)
  (OpenAI), parts of which are vendored; [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and
  [mlx-whisper](https://github.com/ml-explore/mlx-examples) for the quality check.
- [LibriVox](https://librivox.org/) and its volunteer readers, who record public-domain books and give the
  recordings to the public domain; the bundled voices come from LibriVox recordings.
- [Qwen3-TTS](https://github.com/QwenLM) for local voice design.
- [ruaccent](https://github.com/Den4ikAI/ruaccent) (stress and `ё`), [razdel](https://github.com/natasha/razdel)
  (Natasha project, sentence splitting) and [num2words](https://github.com/savoirfairelinux/num2words).
- [FFmpeg](https://ffmpeg.org/), [PyTorch](https://pytorch.org/), [Hugging Face](https://huggingface.co/),
  [PyMuPDF](https://pymupdf.readthedocs.io/), [EbookLib](https://github.com/aerkalov/ebooklib),
  [python-docx](https://github.com/python-openxml/python-docx), [noisereduce](https://github.com/timsainb/noisereduce),
  [Starlette](https://www.starlette.io/), [Typer](https://typer.tiangolo.com/), [Rich](https://github.com/Textualize/rich)
  and [uv](https://github.com/astral-sh/uv).
- [Project Gutenberg](https://www.gutenberg.org/) for the public-domain text in the screenshots.

## License and status

Status: early development. Folivox is licensed under **Apache-2.0**. It vendors a trimmed copy of the
CosyVoice inference code (Apache-2.0) and parts of Matcha-TTS and Whisper (MIT); see
[`src/folivox/vendor/README.md`](src/folivox/vendor/README.md) and [NOTICE](NOTICE). The CosyVoice 3 weights are
Apache-2.0 ([model card](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512)) and are downloaded, not
redistributed. Audio produced by commercial APIs is never bundled; its use is governed by the provider's terms.

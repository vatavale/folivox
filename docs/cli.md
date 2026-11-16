# Command line reference


```bash
folivox inspect book.epub                     # chapter tree, size, estimated duration
folivox book.fb2 -v me.wav -o book.m4b        # render the whole book
folivox book.fb2 -v me.wav -c 1-2             # only chapters 1–2
folivox book.pdf -v me.wav --mp3 -o out/      # one mp3 per chapter
```

Run `folivox` with no arguments for a quick start, `folivox --help` / `folivox render --help` for
all options (grouped by purpose, with defaults and examples), and `folivox help <topic>` for
longer explanations (`voices`, `providers`, `config`, `languages`, `resuming`, `quality`, `speed`).

`folivox render` options (the same groups as in `--help`):

| Group | Option | Default | |
|---|---|---|---|
| Input | `--lang` | from the book | `ru en de fr es it zh ja ko`; online engines (Fish Audio) also take other ISO 639-1 codes (`pl`, `pt`, `ar`, …), read without Folivox's number rules |
| | `--chapters`, `-c` | all | `1-3,5` |
| | `--depth` | 2 | how deep to split the book into chapters |
| | `--preview N` | off | only the first N characters of each chapter (quick test) |
| Voice | `--voice`, `-v` | by book language | catalog id or a WAV/FLAC file |
| | `--voice-text` | `.txt` next to the sample, else Whisper | transcript of the voice sample |
| | `--speed` | 1.0 | speaking rate |
| Text | `--max-chars` | 300 (online engines: their own, 2000 for Gemini, OpenAI and ElevenLabs) | longest fragment sent to the model at once; above 300 on an online engine, neighbouring paragraphs share one request |
| | `--stress` | `yo` | `none`, `yo` (restore ё), `all` (acute stress marks, experimental) |
| Quality control | `--qa` / `--no-qa` | on | check each fragment with speech recognition (faster without) |
| | `--takes` | 3 | max takes per fragment when QA finds a mismatch |
| Providers | `--tts` | best installed Fish engine you accepted, else `cosyvoice3` | TTS provider (`folivox providers`, `folivox engines`) |
| | `--tts-build` | `auto` | Fish engines: build id from `folivox engines` |
| | `--asr` | `auto` | speech recognition for QA: `mlx-whisper`, `faster-whisper`, `openai` |
| | `--llm` | `rl` | CosyVoice language-model weights: `rl` or `base` |
| | `--model-dir` | downloaded on first run | folder with CosyVoice 3 weights |
| Performance | `--device` | `auto` | `cpu`, `cuda`, `mps` |
| | `--threads` | PyTorch default | CPU threads |
| Output | `--output`, `-o` | `<book>.m4b` | output file (a folder with `--mp3`) |
| | `--mp3` | | one mp3 per chapter instead of an m4b |
| | `--bitrate` | `64k` | AAC/MP3 bitrate |
| Book details | `--title` `--author` `--narrator` `--series` `--series-number` `--year` `--description` | from the book | what the player shows; the series is written as the Grouping tag |
| | `--cover` | the book's own | an image (JPEG, PNG, WebP) to embed instead, or `none`; shrunk to 1600 px |
| | `--work-dir` | `<output>.folivox/` | cache and journal; delete it when done |
| | `--verbose` | | print every fragment |

## Speakers and expression tags

```bash
folivox markup export book.epub -o book.txt    # edit it, then:
folivox markup import book.epub book.txt
folivox render book.epub --tts fish-drama3 --voice me.wav     # picks up book.markup.json by itself
```

Automatic: `markup models|analyse|characters|auto` (a local Gemma 4 or any LLM). Also `markup speaker|speakers|check|tags|clear`, and for `render`: `--markup FILE`, `--no-markup`,
`--no-expression-tags`. See [Speakers and expression tags](markup.md).

## Fish Audio engines

```bash
folivox engines                         # builds, size, memory, licence, fit here, install state
folivox engines licence BUILD           # the full licence text
folivox engines accept fish-community   # read and accept a licence (also asked by `install`)
folivox engines install BUILD           # download + verify (asks for the licence first)
folivox engines remove BUILD            # delete the downloaded weights
folivox engines best                    # id of the best tested build that fits this computer
folivox render book.epub --tts fish-drama3 --tts-build drama3-mlx-4bit --voice my-voice.wav
```

`--tts-build auto` (default) uses the best installed build that fits. Details, licences and memory:
[Fish Audio engines](fish-engines.md).

## Native runtime (optional)

```bash
folivox runtime                          # installed? which build fits this computer? what does `auto` do here?
folivox runtime install                  # download the audio.cpp engine + the CosyVoice 3 GGUF model (asks first)
folivox runtime install --weights f32    # the 7 GB file (the cleanest source for fp16 precision)
folivox runtime remove                   # delete them (PyTorch files are not touched)
folivox runtime build-help               # where everything comes from; build steps for this computer (or a named one)
folivox runtime build drama3             # Apple Silicon: compile the Drama 3 server from the pinned audio.cpp commit
folivox render book.epub --runtime native
```

`--runtime auto|pytorch|native` (also `FOLIVOX_RUNTIME` and `[runtime] inference` in `config.toml`) chooses what runs
CosyVoice 3. The default is `auto`, which is PyTorch until a platform is recorded as checked. Details, the
fallback rules and what is verified: [Native runtime](native-runtime.md).

## Speed test

```bash
folivox bench                         # every device found: cpu, cuda, mps (and native, when installed)
folivox bench --devices cpu,cuda      # only these (cuda is tried even if `auto` would skip a small GPU)
folivox bench --lang en --voice me.wav --timeout 600
folivox bench --tts fish-s2 --runtime native --voice me.wav   # a Fish engine; prints a report to share
```

The report at the end is meant to be pasted into an issue. For every device it says whether the run worked, which
runtime did it (PyTorch or the native audio.cpp engine) and on which GPU, the speed (RTF below 1 is faster than real
time, and the line says "x times faster" or "SLOWER" in words), how long the model took to load, and the peak memory:
RAM of the process that holds the model, and for NVIDIA cards the extra GPU memory taken while it ran. The header says
whether Folivox ran inside a container, which matters for GPU access. Quality is not part of it.

Synthesises one short fragment per device, each in its own process (a crash or an out-of-memory error ends only
that line) and prints characters per second, RTF (synthesis time / audio length; below 1 is faster than real
time), model load and warm-up time, and the peak GPU memory. Run it with the same voice and language on
different machines to compare them. Speech recognition (the QA step) is not measured. The studio and the
console also show the device and the current `chars/s` during a render.

## Interface language

The command line interface is in English only.

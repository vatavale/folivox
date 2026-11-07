"""Texts for ``folivox`` (quick start), ``folivox help <topic>`` and the --help epilogs."""

from __future__ import annotations

QUICKSTART = """\
[bold]1.[/] Look at the book:        [cyan]folivox inspect book.epub[/]
[bold]2.[/] Try 30 seconds of it:    [cyan]folivox book.epub --preview 400 -c 1[/]
[bold]3.[/] Render the whole book:   [cyan]folivox book.epub -o book.m4b[/]

Press Ctrl+C at any time; run the same command again to continue where it stopped."""

MAIN_EPILOG = """\
[bold]Examples[/]

  [cyan]folivox inspect book.fb2[/]                       chapter tree, size, estimated duration
  [cyan]folivox book.fb2 -v me.wav -o book.m4b[/]         render with your own voice sample
  [cyan]folivox book.fb2 -c 1-3,5 --preview 500[/]        quick test: chapters 1–3 and 5, 500 chars each
  [cyan]folivox book.pdf --mp3 -o out/[/]                 one mp3 per chapter
  [cyan]folivox voices[/]                                 voices you can use with --voice

[dim]`folivox BOOK` is short for `folivox render BOOK`.  More: `folivox help` lists topics.[/]"""

RENDER_EPILOG = """\
[bold]Examples[/]

  [cyan]folivox render book.fb2[/]                                  default voice for the book language
  [cyan]folivox render book.fb2 -v me.wav --voice-text me.txt[/]    clone a voice from a 6–15 s sample
  [cyan]folivox render book.epub -c 1-2 --preview 600[/]            test run: 600 chars of chapters 1–2
  [cyan]folivox render book.txt --lang de --speed 0.95 --no-qa[/]   German, slightly slower, no QA pass
  [cyan]folivox render book.fb2 --asr faster-whisper[/]               pick a QA provider

[dim]Interrupted? Run the same command again.  More: folivox help resuming | voices | quality[/]"""

TOPICS: dict[str, tuple[str, str]] = {
    "cloud": ("Paid cloud reading engines", """\
ElevenLabs, OpenAI, Gemini API, Cartesia and Fish Audio (Drama 3) can read the whole book instead of the local model.
The book text is sent to the provider and [bold]billed to your account[/] (a whole book can cost tens of dollars).

  [cyan]folivox render book.epub --tts elevenlabs --tts-voice Rachel --preview 600[/]   cheap test first
  [cyan]folivox render book.epub --tts openai --tts-voice coral --max-cost 10 --yes[/]   cap the spend; resumable

Before every cloud run Folivox shows an estimate and asks; [cyan]--yes[/] skips the question,
[cyan]--max-cost N[/] stops cleanly at N USD (run again with a higher cap to continue).
Voices: [cyan]folivox tts-voices ENGINE[/].
Retakes after a QA mismatch are billed, so cloud runs use 2 takes by default. Reading speed and stress marks
do not apply. Check the provider's terms before selling the audio; free plans are usually personal use only."""),
    "voices": ("Choosing and adding voices", """\
Without [cyan]--voice[/] Folivox picks a catalog voice in the book's language.

  [cyan]folivox voices[/]                              list the catalog (bundled + yours)
  [cyan]folivox book.fb2 --voice russlib[/]            use a catalog voice by id
  [cyan]folivox book.fb2 --voice me.wav[/]             use a WAV/FLAC file directly
  [cyan]folivox voices add anna anna.wav --text anna.txt --gender f[/]
  [cyan]folivox voices design anna --describe "calm native female narrator"[/]   generate one
  [cyan]folivox voices sample-text book.fb2[/]         passages that are easy for a generator to read
  [cyan]folivox voices suggest-prompt book.fb2 -g elevenlabs[/]   a voice-design prompt written by a text LLM
  [cyan]folivox voices speakers -g gemini[/]           preset voices of a generator
  [cyan]folivox voices design anna -g gemini --speaker Kore --lang en[/]   cloud provider (your key, paid)
  [cyan]folivox voices design anna -g elevenlabs --describe "calm warm woman, low voice"[/]
  [cyan]folivox voices design anna -g cartesia --mode localize --speaker <id> --lang ru[/]   native accent
  [cyan]--model NAME[/] pins a cloud model; the default [cyan]auto[/] uses the newest the provider lists.

Cloud generators ([cyan]openai[/], [cyan]gemini[/], [cyan]elevenlabs[/], [cyan]cartesia[/], [cyan]fish[/]) need an
API key in [cyan]OPENAI_API_KEY[/], [cyan]GEMINI_API_KEY[/], [cyan]ELEVENLABS_API_KEY[/], [cyan]CARTESIA_API_KEY[/],
[cyan]FISH_API_KEY[/]
(or the OS keychain via [cyan]folivox ui[/]). The text is sent to the provider and billed to you;
the voice is stored as a [bold]commercial API[/] voice for your own use and is never bundled.
Check each provider's terms.

A good sample is 6–15 s of clean, evenly read speech with an exact transcript (pass it with
[cyan]--voice-text[/], or put a .txt next to the audio; otherwise Whisper transcribes it).
Clone only your own voice or one you have permission to use."""),
    "providers": ("Pluggable backends", """\
TTS, speech recognition (QA), voice generators and text LLMs are providers selected by name.

  [cyan]folivox providers[/]                           what exists and what is selected

Choose per run ([cyan]--tts[/], [cyan]--asr[/], [cyan]--llm[/]…), with [cyan]FOLIVOX_TTS[/] / [cyan]FOLIVOX_ASR[/] /
[cyan]FOLIVOX_VOICEGEN[/] / [cyan]FOLIVOX_LLM[/], or permanently in the config file (see [cyan]folivox help config[/]).
Cloud providers run at your own cost and send your text or audio off your computer."""),
    "config": ("Configuration file and environment", """\
File: [cyan]config.toml[/] in the user config directory (path shown by [cyan]folivox providers[/];
override with [cyan]FOLIVOX_CONFIG[/]).

  [providers]
  llm = "ollama"

  [llm.ollama]
  model = "gemma4:31b-it-qat"

  [llm.openai]
  base_url = "https://openrouter.ai/api/v1"
  model = "some/model"
  api_key_env = "OPENROUTER_API_KEY"

API keys come from environment variables or the OS keychain (save them in [cyan]folivox ui[/]); the config
refuses [cyan]api_key[/] fields, so keys never land in a file.

Environment: FOLIVOX_VOICE, FOLIVOX_VOICE_TEXT, FOLIVOX_MODEL_DIR, FOLIVOX_TTS, FOLIVOX_ASR,
FOLIVOX_VOICEGEN, FOLIVOX_LLM, FOLIVOX_CONFIG.
Precedence: command line > environment > config file > built-in default."""),
    "languages": ("Supported languages", """\
Rules for numbers, dates and abbreviations: Russian, English, German, French, Spanish, Italian,
Chinese, Japanese, Korean (codes: ru en de fr es it zh ja ko). CosyVoice reads only these.

Fish engines (fish-s2, fish-drama3, local or online) and the other online engines also read many more
languages. Dutch (nl) has its own rules too (numbers, dates, money, units, abbreviations); for the others,
Portuguese (pt) and Polish (pl) among them, the text goes to the model almost as it is and the model reads
numbers and abbreviations by itself. Set the language with [cyan]--lang[/].

The language comes from the book metadata (FB2/EPUB/DOCX) or is detected from the text and
titles; [cyan]--lang[/] overrides it. [cyan]folivox inspect[/] shows what was detected.
Russian gets the most work: numbers, dates and units with case agreement, ё restoration
([cyan]--stress[/]), dialogue pauses. Other languages get basic normalisation and sentence splitting.
A voice sample in another language than the book may carry its accent over."""),
    "resuming": ("Interrupting and resuming", """\
Every synthesised fragment is cached in the work directory (default [cyan]<output>.folivox/[/]) and
recorded in a journal.

  • Ctrl+C is safe. Run the same command again to continue.
  • Edit the book text and re-run: only changed fragments are re-synthesised.
  • Changing the voice, speed or TTS settings starts those fragments over.
  • Delete the work directory when the book is done (or to start from scratch).
  • [cyan]--work-dir[/] puts the cache elsewhere, e.g. on a bigger disk."""),
    "pronunciation": ("Custom pronunciations: names, invented words, stress", """\
A text file next to the book, [cyan]book.pronounce.txt[/] (or [cyan]pronounce.txt[/] for a series), one entry per line:

  [cyan]Kvothe = Quothe[/]                  respell the word the way it sounds (every engine)
  [cyan]Tolkien* = Tolkeen[/]               [cyan]*[/] keeps any ending: Tolkien's, Tolkiens
  [cyan]Kvothe = /K W OW1 TH/ | Quothe[/]   phonemes for the local engine | respelling for the online ones

Stress: [cyan]+[/] before the stressed vowel ([cyan]Mir+andel[/]) or [cyan]'[/] right after it ([cyan]Mira'ndel[/]).
Russian works the same way (and also takes one capital letter). Not an apostrophe after the syllable.
Enter each form of a word yourself: stress can move between forms. Online engines ignore stress marks and
phonemes. Check by ear: [cyan]folivox book.epub --preview 400 -c 1[/].

  [cyan]folivox names book.epub[/]           list the names in the book (a start for the file)
  [cyan]folivox names book.epub --write[/]   add them to book.pronounce.txt
  [cyan]--pronounce FILE[/]                  use another file (repeat the option for several)

In the studio the same file is edited under the book details. More: docs/pronunciation.md"""),
    "quality": ("Quality control (QA)", """\
Each fragment is re-transcribed with speech recognition and compared with the source text. On a
mismatch the fragment is synthesised again (up to [cyan]--takes[/] times) and the best take is kept.

  [cyan]--no-qa[/]        skip the check: faster, but no retakes
  [cyan]--takes N[/]      more takes = fewer artefacts, slower
  [cyan]--verbose[/]      print every fragment with its scores
  [cyan]--max-chars N[/]  shorter fragments are safer for the model, longer ones sound smoother"""),
    "speed": ("Hardware and speed test", """\
[cyan]folivox doctor[/] shows what the machine offers (CPU, GPU, memory) and which device [cyan]auto[/] will use.
[cyan]folivox bench[/] measures the real speed: one short fragment on every device found (cpu, cuda, mps), \
each in its own process, so a crash or out-of-memory on one device ends only that line.

  [cyan]folivox bench[/]                      all devices found
  [cyan]folivox bench --devices cpu,cuda[/]   only these; cuda is tried even if [cyan]auto[/] would skip a small GPU

The table shows characters per second (more is faster), RTF (synthesis time / audio length; below 1 is faster \
than real time), model load and warm-up time, and the peak GPU memory. Run it with the same voice and language \
on different machines to compare them. The speech-recognition check is not included.

[cyan]auto[/] uses CUDA only after a self-test and with at least 4 GB of GPU memory; Whisper goes to the GPU \
only with 8 GB or more. Force a device with [cyan]--device cpu|cuda|mps[/]. During a render the studio and \
the console show the device in use and the current chars/s."""),
}

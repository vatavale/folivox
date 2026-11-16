"""Command line interface.

``folivox BOOK [options]`` renders an audiobook (the ``render`` command is implied);
``folivox inspect BOOK`` shows the chapter tree and estimates; ``folivox voices`` manages voices;
``folivox providers`` lists the pluggable backends.
"""

from __future__ import annotations

import os
import sys
import time
from enum import Enum
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich.tree import Tree

from folivox import __version__
from folivox.book import Book, Section
from folivox.helptext import MAIN_EPILOG, QUICKSTART, RENDER_EPILOG, TOPICS
from folivox.pipeline import Events

app = typer.Typer(add_completion=False, no_args_is_help=False, invoke_without_command=True,
                  rich_markup_mode="rich", epilog=MAIN_EPILOG, context_settings={"help_option_names": ["-h", "--help"]},
                  help="Turn e-books (FB2, EPUB, DOCX, PDF, TXT, or ZIP of one) into m4b audiobooks, locally.")
console = Console()
COMMANDS = {"render", "inspect", "names", "voices", "engines", "runtime", "markup", "providers", "help", "ui", "doctor",
            "setup", "update", "uninstall", "bench", "--help", "-h", "--version"}

# Option panels of `folivox render --help`
P_INPUT, P_VOICE, P_TEXT, P_QA = "Input", "Voice", "Text", "Quality control"
P_PROVIDERS, P_PERF, P_OUT, P_META = "Providers", "Performance", "Output", "Book details"
P_CLOUD = "Cloud engines (paid)"


class Stress(str, Enum):
    none = "none"
    yo = "yo"
    all = "all"


class LlmWeights(str, Enum):
    rl = "rl"
    base = "base"


class Device(str, Enum):
    auto = "auto"
    cpu = "cpu"
    cuda = "cuda"
    mps = "mps"


def fmt_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h} h {m:02d} min" if h else f"{m} min {s:02d} s"


def num(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def _load(path: Path, lang: str | None = None) -> Book:
    from folivox.readers import UnsupportedFormat, read_book

    if not path.exists():
        console.print(f"[red]File not found:[/] {path}")
        raise typer.Exit(2)
    try:
        book = read_book(path)
    except UnsupportedFormat as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(2) from None
    if lang:
        book.lang = _check_lang(lang)
    else:
        from folivox.readers import WARNING_TEXT

        for w in book.warnings:
            console.print(f"[yellow]⚠ {WARNING_TEXT.get(w, w)}[/]")
    return book


def _check_lang(code: str) -> str:
    from folivox import lang

    try:
        return lang.get(code).code
    except ValueError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(2) from None


def _lang_name(code: str) -> str:
    from folivox import lang

    try:
        language = lang.get(code)
        return f"{language.native} ({language.code})"
    except ValueError:
        return code or "—"


@app.command(help="Show the book structure, chapters and estimated audio length.", rich_help_panel="Main commands",
             epilog="[bold]Example[/]\n\n  [cyan]folivox inspect book.epub --depth 1 --text[/]")
def inspect(
    book_path: Path = typer.Argument(..., help="Book file (FB2, EPUB, DOCX, PDF, TXT, or ZIP of one)."),
    depth: int = typer.Option(2, "--depth", help="How deep to split the book into chapters (1 = top level only)."),
    lang: str | None = typer.Option(None, "--lang", show_default="from book / detected",
                                    help="Book language: ru en de fr es it zh ja ko; others online."),
    text: bool = typer.Option(False, "--text", help="Add a column with the beginning of every chapter."),
):
    """Show the book structure, chapters and estimated audio length."""
    book = _load(book_path, lang)
    meta = Table.grid(padding=(0, 2))
    meta.add_row("Title", book.title)
    meta.add_row("Author", book.author or "—")
    meta.add_row("Language", _lang_name(book.lang))
    meta.add_row("Format", book.source_format.upper())
    if book.year:
        meta.add_row("Year", book.year)
    meta.add_row("Cover", f"yes ({len(book.cover) // 1024} KB)" if book.cover else "no")
    words = sum(len(b.text.split()) for ch in book.chapters(depth) for b in ch.blocks)
    meta.add_row("Size", f"{num(book.chars)} characters, {num(words)} words")
    meta.add_row("Audio (estimate)", f"≈ {fmt_duration(book.estimated_seconds)}")
    console.print(meta)
    console.print()

    tree = Tree(f"[bold]{book.title}[/]")

    def add(node, sec: Section):
        label = sec.title or "[dim](untitled)[/]"
        n = node.add(f"{label} [dim]· {num(sec.chars)} chars[/]")
        for c in sec.children:
            add(n, c)

    for s in book.sections:
        add(tree, s)
    console.print(tree)
    console.print()

    chapters = book.chapters(depth)
    cps = book.chars / book.estimated_seconds if book.estimated_seconds else 14.0
    t = Table(title=f"Audiobook chapters ({len(chapters)})", show_lines=False)
    t.add_column("#", justify="right")
    t.add_column("Chapter")
    t.add_column("Characters", justify="right")
    t.add_column("≈ Time", justify="right")
    if text:
        t.add_column("Beginning")
    for i, ch in enumerate(chapters, 1):
        row = [str(i), ch.title, num(ch.chars), fmt_duration(ch.chars / cps)]
        if text:
            first = next((b.text for b in ch.blocks), "")
            row.append(first[:60] + ("…" if len(first) > 60 else ""))
        t.add_row(*row)
    console.print(t)


class RichEvents(Events):
    """Console progress for :func:`folivox.pipeline.render`."""

    def __init__(self, verbose: bool = False):
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            TaskProgressColumn,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )

        self.verbose = verbose
        self.progress = Progress(TextColumn("[bold]{task.description}"), BarColumn(), TaskProgressColumn(),
                                 MofNCompleteColumn(), TextColumn("chars"), TimeElapsedColumn(), TextColumn("·"),
                                 TimeRemainingColumn(), TextColumn("{task.fields[info]}"), console=console,
                                 transient=False)
        self.task = None
        self.audio = self.synth = 0.0
        self.retakes = 0
        self._dl = None
        self._dl_task: dict = {}

    def stage(self, name, detail=""):
        console.log(f"[cyan]{detail or name}[/]")

    def warning(self, message):
        console.log(f"[yellow]⚠ {message}[/]")

    def plan(self, total_chars, done_chars, total_chunks, pending_chunks):
        console.log(f"Fragments: {num(total_chunks)}, left {num(pending_chunks)}; "
                    f"text {num(total_chars)} chars (done {num(done_chars)})")
        if pending_chunks:
            self.progress.start()
            self.task = self.progress.add_task("Rendering", total=total_chars, completed=done_chars, info="")

    def chunk_started(self, chunk):
        pass

    def chunk_done(self, chunk, take, attempts):
        self.audio += take.duration
        self.synth += take.synth_seconds
        self.chars = getattr(self, "chars", 0) + len(chunk.source)
        if len(attempts) > 1:
            self.retakes += len(attempts) - 1
        rtf = self.synth / self.audio if self.audio else 0
        info = f"RTF {rtf:.2f} · audio {fmt_duration(self.audio)}"
        if self.synth > 0:
            info += f" · {self.chars / self.synth:.0f} chars/s"
        if self.spent is not None:
            info = f"audio {fmt_duration(self.audio)} · ≈ ${self.spent:.2f}" + (
                f" of ${self.cap:.2f}" if self.cap else "")
        if self.retakes:
            info += f" · retakes {self.retakes}"
        if self.task is not None:
            self.progress.update(self.task, advance=len(chunk.source), info=info)
        if self.verbose or (take.score is not None and len(attempts) > 1):
            sc = ", ".join("—" if a is None else f"{a:.2f}" for a in attempts)
            self.progress.console.log(f"[dim]ch.{chunk.chapter + 1} #{chunk.idx}[/] [{sc}] {chunk.text[:70]}")

    spent: float | None = None
    cap: float | None = None

    def spend(self, chars, usd, cap):
        self.spent, self.cap = usd, cap

    downloaded = False

    def download(self, name, done_bytes, total_bytes):
        from rich.progress import (
            BarColumn,
            DownloadColumn,
            Progress,
            TextColumn,
            TimeRemainingColumn,
            TransferSpeedColumn,
        )

        self.downloaded = True
        if self._dl is None:
            self._dl = Progress(TextColumn("[cyan]{task.description}"), BarColumn(), DownloadColumn(),
                                TransferSpeedColumn(), TimeRemainingColumn(), console=console, transient=True)
            self._dl.start()
            self._dl_task = {}
        if name not in self._dl_task:
            self._dl_task[name] = self._dl.add_task(f"Downloading {name}", total=total_bytes or None)
        self._dl.update(self._dl_task[name], completed=done_bytes, total=total_bytes or None)
        if total_bytes and done_bytes >= total_bytes:
            self._dl.stop()
            self._dl = None
            console.log(f"[green]✓[/] Downloaded {name} ({total_bytes / 1e9:.1f} GB)")

    def chapter_built(self, index, title, duration):
        if self.task is not None and self.progress.live.is_started:
            self.progress.stop()
        console.log(f"[green]✓[/] {title} — {fmt_duration(duration)}")

    def close(self):
        if self.progress.live.is_started:
            self.progress.stop()
        if self._dl is not None:
            self._dl.stop()
            self._dl = None


@app.command(help="List the names in a book that may need a custom pronunciation (a start for the dictionary).",
             rich_help_panel="Main commands",
             epilog="[bold]Examples[/]\n\n  [cyan]folivox names book.epub[/]\n"
                    "  [cyan]folivox names book.epub --write[/]   adds them to book.pronounce.txt\n\n"
                    "[dim]More: folivox help pronunciation[/]")
def names(
    book_path: Path = typer.Argument(..., help="Book file (FB2, EPUB, DOCX, PDF, TXT, or ZIP of one)."),
    lang: str | None = typer.Option(None, "--lang", show_default="from book / detected",
                                    help="Book language: ru en de fr es it zh ja ko; others online."),
    min_count: int = typer.Option(3, "--min", min=1, help="Only names that occur at least this many times."),
    limit: int = typer.Option(150, "--limit", min=1, help="At most this many names (the most frequent first)."),
    write: bool = typer.Option(False, "--write", help="Add the new names to <book>.pronounce.txt instead of printing."),
):
    from folivox.text import names as found
    from folivox.text.pronounce import Lexicon

    book = _load(book_path, lang)
    target = book_path.with_name(book_path.stem + ".pronounce.txt")
    known = {e.word.rstrip("*").lower() for e in Lexicon.load([target] if target.is_file() else []).entries}
    fresh = [c for c in found.candidates(book, min_count=min_count) if c.word.lower() not in known][:limit]
    lines = [f"{c.word} = {c.word}   # {c.count}x" for c in fresh]
    if not write:
        if not lines:
            console.print("[dim]No names found (try a lower --min).[/]")
        for line in lines:
            typer.echo(line)
        return
    if not lines:
        console.print("No new names to add.")
        return
    head = "" if not target.is_file() else ("\n" if not target.read_text(encoding="utf-8").endswith("\n") else "")
    note = "# Found by `folivox names`: replace the right-hand side with how to read the name.\n"
    with target.open("a", encoding="utf-8") as f:
        f.write(head + ("\n" if target.stat().st_size else "") + note + "\n".join(lines) + "\n")
    console.print(f"Added {len(lines)} names to [bold]{target}[/]. Edit the right-hand sides, then render as usual.")


@app.command(help="Render a book into an audiobook (m4b with chapters and cover).", epilog=RENDER_EPILOG,
             rich_help_panel="Main commands")
def render(
    book_path: Path = typer.Argument(..., help="Book file (FB2, EPUB, DOCX, PDF, TXT, or ZIP of one)."),
    lang: str | None = typer.Option(None, "--lang", rich_help_panel=P_INPUT, show_default="from book / detected",
                                    help="Book language: ru en de fr es it zh ja ko; others online."),
    chapters: str | None = typer.Option(None, "--chapters", "-c", rich_help_panel=P_INPUT, show_default="all",
                                        help="Chapters to render, e.g. 1-3,5 (numbers from `folivox inspect`)."),
    depth: int = typer.Option(2, "--depth", rich_help_panel=P_INPUT,
                              help="How deep to split the book into chapters (1 = top level only)."),
    preview: int | None = typer.Option(None, "--preview", rich_help_panel=P_INPUT, show_default="off",
                                       help="Render only the first N characters of each chapter (quick test)."),
    voice: str = typer.Option("", "--voice", "-v", envvar="FOLIVOX_VOICE", rich_help_panel=P_VOICE,
                              show_default="by book language",
                              help="Catalog voice id (`folivox voices`) or a WAV/FLAC file of 6–15 s of clean speech."),
    voice_text: str | None = typer.Option(None, "--voice-text", envvar="FOLIVOX_VOICE_TEXT", rich_help_panel=P_VOICE,
                                          show_default=".txt next to sample, else Whisper",
                                          help="Transcript of the voice sample (text or a .txt file)."),
    speed: float = typer.Option(1.0, "--speed", rich_help_panel=P_VOICE, help="Speaking rate (0.9 = slower)."),
    max_chars: int = typer.Option(None, "--max-chars", rich_help_panel=P_TEXT,
                                  show_default="300 local, per engine online",
                                  help="Longest fragment sent to the model at once. Online engines rate-limit "
                                       "per request, so a longer fragment means fewer requests."),
    stress: Stress = typer.Option(Stress.yo, "--stress", rich_help_panel=P_TEXT,
                                  help="Russian only: none | yo (restore ё) | all (stress marks, experimental)."),
    pronounce: list[Path] = typer.Option(None, "--pronounce", rich_help_panel=P_TEXT, show_default=False,
                                         help="Custom pronunciations (name = how to read it). Default: "
                                              "<book>.pronounce.txt and pronounce.txt next to the book."),
    markup: Path = typer.Option(None, "--markup", rich_help_panel=P_VOICE, show_default=False,
                                help="Speakers and expression tags (`folivox markup`). Default: <book>.markup.json "
                                     "next to the book."),
    no_markup: bool = typer.Option(False, "--no-markup", rich_help_panel=P_VOICE, help="Ignore the book's markup."),
    expression_tags: bool = typer.Option(True, "--expression-tags/--no-expression-tags", rich_help_panel=P_VOICE,
                                         show_default=False,
                                         help="Write the markup's expression tags into the text (Fish engines; on)."),
    qa: bool = typer.Option(True, "--qa/--no-qa", rich_help_panel=P_QA, show_default=False,
                            help="Check every fragment with speech recognition and retake mismatches (on)."),
    qa_level: str = typer.Option("standard", "--qa-level", rich_help_panel=P_QA,
                                 help="How strict QA is: low | standard | high (stricter means more retakes)."),
    takes: int | None = typer.Option(None, "--takes", rich_help_panel=P_QA, show_default="3 local, 2 cloud",
                                     help="Max takes per fragment when QA finds a mismatch (cloud takes are billed)."),
    tts: str | None = typer.Option(None, "--tts", rich_help_panel=P_PROVIDERS,
                                   show_default="best installed Fish engine, else cosyvoice3",
                                   help="Speech synthesis provider (`folivox providers`, `folivox engines`)."),
    tts_build: str = typer.Option("auto", "--tts-build", rich_help_panel=P_PROVIDERS,
                                  help="Fish engines: build id from `folivox engines` (auto = best installed)."),
    tts_model: str = typer.Option("auto", "--tts-model", rich_help_panel=P_CLOUD,
                                  help="Cloud engine model id, or auto = the newest the provider offers."),
    tts_voice: str = typer.Option("", "--tts-voice", rich_help_panel=P_CLOUD,
                                  help="Cloud voice id or name (`folivox tts-voices ENGINE`)."),
    tts_style: str | None = typer.Option(None, "--tts-style", rich_help_panel=P_CLOUD,
                                         show_default="neutral audiobook narration",
                                         help="Narration note (OpenAI, Gemini; ElevenLabs v3/v4 tag)."),
    tts_mode: str = typer.Option("preset", "--tts-mode", rich_help_panel=P_CLOUD,
                                 help="preset | design (describe a voice) | localize | clone (your own voice)."),
    tts_describe: str = typer.Option("", "--tts-describe", rich_help_panel=P_CLOUD,
                                     help="Voice description for --tts-mode design."),
    own_voice: bool = typer.Option(False, "--own-voice", rich_help_panel=P_CLOUD,
                                   help="Confirm that the --voice you clone is your own (or you have permission)."),
    max_cost: float | None = typer.Option(None, "--max-cost", rich_help_panel=P_CLOUD, show_default="no cap",
                                          help="Stop cleanly (resumable) before spending more than N USD."),
    tts_parallel: int | None = typer.Option(None, "--tts-parallel", rich_help_panel=P_CLOUD, min=1,
                                            show_default="engine's own (Gemini 1, others 3)",
                                            help="Online engines: requests sent at the same time. Raise it if your "
                                                 "account tier allows more."),
    tts_rpm: float | None = typer.Option(None, "--tts-rpm", rich_help_panel=P_CLOUD, min=0,
                                         show_default="engine's own (Gemini 10, others none)",
                                         help="Online engines: requests started per minute (0 = no limit of "
                                              "our own). Parallel requests only help while this allows them."),
    keep_voice: bool = typer.Option(False, "--keep-voice", rich_help_panel=P_CLOUD,
                                    help="Keep the voice made for the book to reuse it later (--tts-voice NAME)."),
    voice_name: str = typer.Option("", "--tts-voice-name", rich_help_panel=P_CLOUD,
                                   help="Name for a kept voice (default: from the description)."),
    yes: bool = typer.Option(False, "--yes", "-y", rich_help_panel=P_CLOUD,
                             help="Agree to send the text to a paid cloud engine without asking."),
    asr: str | None = typer.Option(None, "--asr", rich_help_panel=P_PROVIDERS, show_default="auto",
                                   help="Speech recognition provider used by QA."),
    llm: LlmWeights = typer.Option(LlmWeights.rl, "--llm", rich_help_panel=P_PROVIDERS,
                                   help="CosyVoice language-model weights: rl (fewer text errors) or base."),
    model_dir: Path | None = typer.Option(None, "--model-dir", envvar="FOLIVOX_MODEL_DIR",
                                          rich_help_panel=P_PROVIDERS, show_default="downloaded on first run",
                                          help="Folder with CosyVoice 3 weights."),
    runtime: str | None = typer.Option(None, "--runtime", rich_help_panel=P_PERF, show_default="auto (PyTorch)",
                                       help="Inference runtime for CosyVoice 3: auto, pytorch, or native "
                                            "(C++, `folivox runtime install`)."),
    device: Device = typer.Option(Device.auto, "--device", rich_help_panel=P_PERF,
                                  help="Compute device for PyTorch (the native runtime picks its own)."),
    threads: int | None = typer.Option(None, "--threads", rich_help_panel=P_PERF, show_default="PyTorch default",
                                       help="CPU threads for PyTorch."),
    output: Path | None = typer.Option(None, "--output", "-o", rich_help_panel=P_OUT,
                                       show_default="<book>.m4b next to the book",
                                       help="Output .m4b file (a folder with --mp3)."),
    mp3: bool = typer.Option(False, "--mp3", rich_help_panel=P_OUT,
                             help="Write one MP3 per chapter instead of an m4b."),
    bitrate: str = typer.Option("64k", "--bitrate", rich_help_panel=P_OUT, help="AAC/MP3 bitrate."),
    cover: str | None = typer.Option(None, "--cover", rich_help_panel=P_OUT, show_default="the book's own cover",
                                     help="Cover image (JPEG, PNG, WebP) or `none` for no cover."),
    title: str | None = typer.Option(None, "--title", rich_help_panel=P_META, show_default="from the book",
                                     help="Title written to the audiobook."),
    author: str | None = typer.Option(None, "--author", rich_help_panel=P_META, show_default="from the book",
                                      help="Author."),
    narrator: str | None = typer.Option(None, "--narrator", rich_help_panel=P_META, help="Narrator (composer tag)."),
    series: str | None = typer.Option(None, "--series", rich_help_panel=P_META, help="Series name."),
    series_number: str | None = typer.Option(None, "--series-number", rich_help_panel=P_META,
                                             help="Number in the series (e.g. 2)."),
    year: str | None = typer.Option(None, "--year", rich_help_panel=P_META, show_default="from the book",
                                    help="Year."),
    description: str | None = typer.Option(None, "--description", rich_help_panel=P_META,
                                           show_default="from the book", help="Description / annotation."),
    work_dir: Path | None = typer.Option(None, "--work-dir", rich_help_panel=P_OUT, show_default="<output>.folivox",
                                         help="Cache and journal folder (makes runs resumable)."),
    verbose: bool = typer.Option(False, "--verbose", rich_help_panel=P_OUT, help="Print every fragment."),
):
    """Render a book into an audiobook (m4b with chapters and cover)."""
    from folivox import config
    from folivox.pipeline import QA_LEVELS, CostCapReached, RenderOptions
    from folivox.pipeline import render as run_render
    from folivox.providers import ProviderError
    book = _load(book_path, lang)
    try:
        runtime_name = config.runtime(runtime)
    except ValueError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(2) from None
    from folivox.tts import builds as fish_builds

    explicit_tts = config.provider("tts", tts)
    tts_name = explicit_tts or fish_builds.default_engine()
    cloud = _is_cloud(tts_name)
    voice_path, voice_lang, voice_ref = None, "", None
    if not cloud:
        voice_path, voice_text, voice_lang = _resolve_voice(voice, voice_text, book.lang)
        if not explicit_tts and tts_name != "cosyvoice3":
            from folivox.tts import fish_check

            if fish_check.is_bundled_voice(voice_path) and not fish_check.bundled_voices_allowed():
                console.print("[yellow]The bundled voices are real readers, so Fish engines do not use them: "
                              "reading with CosyVoice 3. Give your own voice with --voice to use Fish.[/]")
                tts_name = "cosyvoice3"
    elif tts_mode == "clone":
        voice_ref = _clone_source(voice, own_voice)
    if voice_lang and voice_lang != book.lang:
        console.print(f"[yellow]⚠ The voice sample is in {_lang_name(voice_lang)}, the book in "
                      f"{_lang_name(book.lang)}: the accent may carry over.[/]")
    cover_path = None
    if cover and cover.lower() != "none":
        cover_path = Path(cover).expanduser()
        if not cover_path.is_file():
            console.print(f"[red]Cover image not found: {cover_path}[/]")
            raise typer.Exit(2)
    if output is None:
        output = book_path.with_suffix("" if mp3 else ".m4b")
    from folivox.text import pronounce as pronunciations

    lexicon_files = tuple(pronounce) if pronounce else tuple(pronunciations.find(book_path))
    for f in lexicon_files:
        if not f.is_file():
            console.print(f"[red]Pronunciation file not found:[/] {f}")
            raise typer.Exit(2)
    if lexicon_files:
        console.print(f"Pronunciations: {', '.join(f.name for f in lexicon_files)}")
    from folivox import markup as markups

    markup_path = None if no_markup else (markup or markups.find(book_path))
    if markup and not markup.is_file():
        console.print(f"[red]Markup file not found:[/] {markup}")
        raise typer.Exit(2)
    if markup_path:
        console.print(f"Markup: {markup_path.name}")
    opts = RenderOptions(output=output, pronounce=lexicon_files, markup=markup_path, expression_tags=expression_tags,
                         voice_wav=voice_path, voice_text=voice_text, work_dir=work_dir,
                         chapters=chapters, preview=preview, depth=depth, max_chars=max_chars, stress=stress.value,
                         lang=book.lang, speed=speed, device=device.value, threads=threads,
                         model_dir=str(model_dir) if model_dir else None, qa=qa,
        qa_threshold=QA_LEVELS.get(qa_level, QA_LEVELS["standard"]),
                         max_takes=takes or (2 if cloud else 3), mp3=mp3,
                         bitrate=bitrate, llm=llm.value, tts=tts_name, runtime=runtime_name, tts_build=tts_build,
                         tts_model=tts_model, tts_voice=tts_voice,
                         tts_style=tts_style, tts_mode=tts_mode, tts_description=tts_describe, voice_ref=voice_ref,
                         max_cost=max_cost, keep_voice=keep_voice,
                         tts_parallel=tts_parallel, tts_rpm=tts_rpm, tts_voice_name=voice_name,
                         asr=config.provider("asr", asr) or "auto", cover=cover_path,
                         no_cover=bool(cover) and cover.lower() == "none",
                         meta={k: v for k, v in {"title": title, "author": author, "narrator": narrator,
                                                 "series": series, "series_part": series_number, "year": year,
                                                 "description": description}.items() if v})
    console.print(f"[bold]{book.title}[/] — {book.author or 'unknown author'}; {_lang_name(book.lang)}; "
                  f"estimated audio ≈ {fmt_duration(book.estimated_seconds)}")
    if cloud:
        _confirm_cloud(book, opts, yes)
    events = RichEvents(verbose)
    t0 = time.time()
    from folivox import diag

    diag.event("render started", book=diag.snippet(book.title, 30), options=diag.options_summary(opts),
               versions=diag.report()["packages"])
    try:
        result = run_render(book, opts, events)
        diag.event("render finished", seconds=round(time.time() - t0, 1))
    except CostCapReached as e:
        events.close()
        console.print(f"\n[yellow]{e}[/]")
        raise typer.Exit(3) from None
    except KeyboardInterrupt:
        events.close()
        console.print("\n[yellow]Stopped. Progress is saved — run the same command again to continue.[/]")
        raise typer.Exit(130) from None
    except Exception as e:  # noqa: BLE001
        events.close()
        from folivox.tts.voice import VoiceError

        diag.event("render failed", level=40, exc_info=True, error=str(e))
        from folivox.tts.base import SynthTimeout

        if isinstance(e, SynthTimeout):
            console.print(f"[red]Error:[/] {e}")
            sys.stdout.flush()
            os._exit(1)  # the stuck worker thread cannot be stopped and would keep the process alive

        if isinstance(e, (VoiceError, ValueError, FileNotFoundError, RuntimeError, ProviderError)):
            console.print(f"[red]Error:[/] {e}")
            raise typer.Exit(1) from None
        raise
    events.close()
    console.print(f"[bold green]Done:[/] {result}  ({fmt_duration(time.time() - t0)})")
    if events.downloaded:
        console.print("\n[bold]Setup finished.[/] The models are stored once and reused. Next time:\n"
                      "  [cyan]folivox ui[/]                  the web studio (voices, preview, progress)\n"
                      "  [cyan]folivox render book.fb2[/]     straight from the terminal\n"
                      "  [cyan]folivox doctor[/]              check ffmpeg, hardware and models\n"
                      "  [cyan]folivox bench[/]               measure the speed on each device (CPU, CUDA, MPS)")
    if events.audio:
        console.print(f"Synthesis: {fmt_duration(events.synth)} for {fmt_duration(events.audio)} of audio, "
                      f"RTF {events.synth / events.audio:.2f}")
    _print_flagged(opts.resolved_work_dir())


def _print_flagged(work_dir: Path) -> None:
    """Fragments that still fail QA after every retake: worth a listen before publishing."""
    from folivox.journal import Journal

    if not (work_dir / "journal.sqlite").exists():
        return
    j = Journal(work_dir)
    try:
        flagged = list(j.flagged())
    finally:
        j.close()
    if not flagged:
        return
    console.print(f"\n[yellow]{len(flagged)} fragment(s) still differ from the text after every retake[/] — "
                  "listen to them (the studio lists them under “Needs review”):")
    for c, t in flagged[:20]:
        console.print(f"  [dim]ch.{c.chapter + 1} #{c.idx} · {t.score:.2f}[/] {c.source[:70]}\n"
                      f"    [dim]heard:[/] {(t.asr or '')[:70]}")
    if len(flagged) > 20:
        console.print(f"  [dim]… and {len(flagged) - 20} more[/]")


def _is_cloud(name: str) -> bool:
    from folivox.pipeline import is_cloud

    return is_cloud(name)


def _clone_source(voice: str, own_voice: bool) -> Path:
    from folivox import voices as catalog

    mine = next((v for v in catalog.user() if v.id == voice), None)
    if mine is None:
        console.print("[red]--tts-mode clone needs --voice = the id of one of your own voices (`folivox voices`)[/]")
        raise typer.Exit(2)
    if not own_voice and not typer.confirm(f"Is {mine.id!r} your own voice, or do you have the speaker's permission?",
                                           default=False):
        raise typer.Exit(1)
    return mine.path


def _confirm_cloud(book, opts, yes: bool) -> None:
    """Show what a paid render will send and cost; ask unless --yes."""
    from folivox.tts import cost
    from folivox.tts.cloud import generator_class

    cls = generator_class(opts.tts)
    from folivox.pipeline import parse_selection

    chosen = [i + 1 for i in parse_selection(opts.chapters, len(book.chapters(opts.depth)))]
    chars = cost.book_chars(book, opts.depth, chosen, opts.preview)
    model = opts.tts_model if opts.tts_model != "auto" else (cls.models[0]["id"] if cls.models else "")
    est = cost.estimate(opts.tts, chars, model, opts.max_takes, opts.qa, opts.tts_mode, opts.lang)
    console.print(f"[bold yellow]Paid cloud engine: {cls.label}[/] ({'model ' + model if model else 'model auto'})")
    console.print(f"  {cost.describe(est)}")
    if opts.max_cost is not None:
        console.print(f"  Spending cap: ${opts.max_cost:.2f} — the run stops cleanly and can be resumed.")
    console.print(f"  The book text leaves this computer and is billed to your {cls.label} account. {cls.price_note}\n"
                  f"  Check the provider's terms before publishing the result: {cls.terms_url}")
    if opts.preview is None:
        console.print("  [dim]Tip: try `--preview 600` first — a minute of audio for a few cents.[/]")
    if not yes and not typer.confirm("Send the text and pay for it?", default=False):
        raise typer.Exit(1)
    opts.accept_cloud = True


def _version(value: bool):
    if value:
        console.print(f"folivox {__version__}")
        raise typer.Exit()


def _resolve_voice(voice: str, voice_text: str | None, book_lang: str = "ru") -> tuple[Path, str | None, str]:
    """Catalog id or path → (audio file, transcript, sample language or "")."""
    from folivox import voices as catalog

    entry = catalog.find(voice) if voice else catalog.default_for(book_lang)
    if entry is not None:
        return entry.path, voice_text or entry.transcript, entry.lang
    path = Path(voice).expanduser()
    if not path.exists():
        known = ", ".join(v.id for v in catalog.all_voices())
        console.print(f"[red]Voice {voice!r} not found.[/] Catalog: {known}. Or give a WAV/FLAC path.")
        raise typer.Exit(2)
    return path, voice_text, ""


from folivox.cli_engines import engines_app  # noqa: E402
from folivox.cli_markup import markup_app  # noqa: E402
from folivox.cli_runtime import runtime_app  # noqa: E402

app.add_typer(markup_app, name="markup", rich_help_panel="Setup")

app.add_typer(engines_app, name="engines", rich_help_panel="Setup")
app.add_typer(runtime_app, name="runtime", rich_help_panel="Setup")

voices_app = typer.Typer(help="Voice catalog: list, add, remove, design.", no_args_is_help=False,
                         invoke_without_command=True, rich_markup_mode="rich",
                         context_settings={"help_option_names": ["-h", "--help"]})
app.add_typer(voices_app, name="voices", rich_help_panel="Setup")


@voices_app.callback(help="List available voices.")
def voices_list(ctx: typer.Context,
                lang: str | None = typer.Option(None, "--lang", help="Only voices in this language (ru en de …).")):
    """List available voices."""
    if ctx.invoked_subcommand:
        return
    from folivox import voices as catalog

    t = Table(title="Voices")
    for col in ("id", "Name", "Sex", "Language", "Length", "License", "Where"):
        t.add_column(col)
    for v in catalog.all_voices():
        if lang and v.lang != lang:
            continue
        try:
            from folivox.tts.voice import audio_duration

            dur = f"{audio_duration(v.path):.1f} s"
        except Exception:  # noqa: BLE001
            dur = "?"
        star = " *" if v.id == catalog.DEFAULT_VOICE else ""
        t.add_row(v.id + star, v.name, {"f": "f", "m": "m"}.get(v.gender, "—"), v.lang, dur, v.license or "—",
                  "bundled" if v.bundled else "mine")
    console.print(t)
    console.print(f"[dim]* default voice. Your voices: {catalog.user_voices_dir()}[/]")


@voices_app.command("add", help="Add your own voice to the catalog (stays on this computer).")
def voices_add(
    voice_id: str = typer.Argument(..., help="Id: lowercase latin letters, digits, - and _."),
    audio: Path = typer.Argument(..., help="WAV/FLAC/MP3 sample of clean speech: 6–15 s (CosyVoice), 10–30 s (Fish)."),
    text: str | None = typer.Option(None, "--text", help="Transcript of the sample (text or a .txt file).",
                                  show_default=".txt next to sample, else Whisper"),
    name: str = typer.Option("", "--name", help="Human-readable name."),
    gender: str = typer.Option("", "--gender", help="Speaker sex: f or m (informational)."),
    lang: str = typer.Option("ru", "--lang", help="Language of the sample (ru en de …)."),
    license: str = typer.Option("", "--license", help="License / origin of the sample."),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing voice with the same id."),
    engine: str = typer.Option("cosyvoice3", "--engine", help="Engine the voice is meant for; sets the length that "
                               "is recommended (cosyvoice3, fish-s2, fish-drama3…)."),
    denoise: bool = typer.Option(False, "--denoise/--no-denoise", show_default=False,
                                 help="Reduce steady background noise in the sample first (off)."),
    pick: int = typer.Option(1, "--pick", min=1, help="For a video or a long recording: which found stretch "
                                                      "to use (1 = the best)."),
    find_from: str | None = typer.Option(None, "--from", help="For a video or a long recording: search from "
                                                              "here (seconds, mm:ss or h:mm:ss)."),
    find_to: str | None = typer.Option(None, "--to", help="For a video or a long recording: search up to here."),
    list_only: bool = typer.Option(False, "--list", help="For a video or a long recording: only show the found "
                                                         "stretches, add nothing."),
):
    """Add your own voice to the catalog (stays on this computer).

    ``audio`` may also be a video or a long recording: a clean 6–15 s stretch of speech is found in it."""
    import tempfile

    lang = _check_lang(lang)
    if not audio.exists():
        console.print(f"[red]File not found: {audio}[/]")
        raise typer.Exit(2)
    from folivox.voices import search

    searching = search.needs_search(audio)
    if (find_from or find_to or list_only or pick > 1) and not searching:
        console.print("[red]--from, --to, --list and --pick are for videos and recordings longer than 30 s.[/]")
        raise typer.Exit(2)
    with tempfile.TemporaryDirectory(prefix="folivox-found-") as found_dir:
        if searching:
            audio, text = _find_in_recording(audio, lang, text, pick, find_from, find_to, list_only,
                                             Path(found_dir), engine)
        _voices_add_sample(voice_id, audio, text, name, gender, lang, license, force, denoise, engine)


def _find_in_recording(audio: Path, lang: str, text: str | None, pick: int, find_from: str | None,
                       find_to: str | None, list_only: bool, tmp: Path, engine: str) -> tuple[Path, str]:
    """Search a video or long recording for a voice sample; returns the cut WAV and its recognised text."""
    from rich.progress import Progress

    from folivox.qa.asr import Transcriber
    from folivox.tts.voice import good_range
    from folivox.voices import search

    if text:
        console.print("[red]--text cannot be used with a video or a long recording: the text is recognised "
                      "for the stretch that is found.[/]")
        raise typer.Exit(2)
    try:
        t0 = search.parse_time(find_from) if find_from else 0.0
        t1 = search.parse_time(find_to) if find_to else None
    except ValueError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(2) from None
    try:
        with Progress(console=console, transient=True) as bar:
            task = bar.add_task("Looking for speech", total=100)
            found = search.find_samples(
                audio, lang, Transcriber(language=lang), count=max(3, pick), start=t0, end=t1,
                length=good_range(engine),
                progress=lambda f, what: bar.update(task, completed=f * 100, description=what.capitalize()))
    except (search.NothingFound, FileNotFoundError, RuntimeError) as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(2) from None

    def stamp(s: float) -> str:
        return f"{int(s // 3600)}:{int(s % 3600 // 60):02d}:{s % 60:04.1f}"

    for i, c in enumerate(found, 1):
        console.print(f"{i}. {stamp(c.start)} – {stamp(c.end)} ({c.duration:.1f} s, score {c.score:.2f}): «{c.text}»")
        for w in c.warnings:
            console.print(f"   [yellow]⚠ {w}[/]")
    if list_only:
        raise typer.Exit(0)
    if pick > len(found):
        console.print(f"[red]Only {len(found)} stretches were found.[/]")
        raise typer.Exit(2)
    chosen = found[pick - 1]
    console.print("Take only a voice you own or have permission to use, and listen to the result: "
                  "one speaker, no music behind.")
    return search.cut(audio, chosen, tmp / "found.wav"), chosen.text


def _voices_add_sample(voice_id: str, audio: Path, text: str | None, name: str, gender: str, lang: str,
                       license: str, force: bool, denoise: bool, engine: str) -> None:
    from folivox import voices as catalog
    from folivox.tts.voice import load_voice

    transcribe = None
    if not text and not audio.with_suffix(".txt").exists():
        from folivox.qa.asr import Transcriber

        transcribe = Transcriber(language=lang).transcribe_file
    from folivox.tts.voice import VoiceError

    try:
        v = load_voice(audio, text, transcribe=transcribe)
    except VoiceError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(2) from None
    from folivox.tts.voice import transcript_warnings

    for w in v.warnings_for(engine) + transcript_warnings(v.transcript, v.duration, lang):
        console.print(f"[yellow]⚠ {w}[/]")
    import tempfile

    with tempfile.TemporaryDirectory(prefix="folivox-voice-") as tmp:
        source = audio
        if denoise:
            import soundfile as sf

            from folivox.audio.denoise import reduce_noise

            data, sr = sf.read(str(audio), dtype="float32", always_2d=True)
            source = Path(tmp) / "denoised.wav"
            sf.write(str(source), reduce_noise(data.mean(axis=1), sr), sr)
            console.print("Background noise reduced.")
        try:
            entry = catalog.add(voice_id, source, v.transcript, name=name, gender=gender, lang=lang,
                                license=license, overwrite=force)
        except (ValueError, FileExistsError) as e:  # bad id, id taken
            console.print(f"[red]{e}[/]")
            raise typer.Exit(2) from None
    console.print(f"[green]Voice added:[/] {entry.id}: «{entry.transcript}»")


@voices_app.command("remove", help="Remove one of your voices from the catalog.")
def voices_remove(voice_id: str = typer.Argument(..., help="Id of one of your voices.")):
    """Remove one of your voices from the catalog."""
    from folivox import voices as catalog

    try:
        catalog.remove(voice_id)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(2) from None
    console.print(f"Removed {voice_id}")


@voices_app.command("design", help=("Generate a reference voice with a voice generator, pick the best take "
                                     "by speech recognition."))
def voices_design(
    voice_id: str = typer.Argument(..., help="Id of the new voice."),
    describe: str = typer.Option("", "--describe", "-d", help=("Voice description for a design generator, e.g. "
                                                                "\"calm native Russian female audiobook "
                                                                "narrator\"")),
    speaker: str = typer.Option("", "--speaker", help="Preset speaker name (for preset generators)."),
    lang: str = typer.Option("ru", "--lang", help="Language of the sample (ru en de …)."),
    gender: str = typer.Option("", "--gender", help="Speaker sex: f or m (informational)."),
    text: list[str] = typer.Option([], "--text", help="Text to read (repeatable).",
                                  show_default="built-in easy texts"),
    takes: int = typer.Option(3, "--takes", help="Takes generated per text; the best one is kept."),
    generator: str | None = typer.Option(None, "--generator", "-g",
                                         help="Voice generator provider (`folivox providers`).",
                                         show_default="qwen3-voicedesign"),
    keep: Path | None = typer.Option(None, "--keep", show_default="temporary",
                                 help="Folder to keep every take for listening."),
    add: bool = typer.Option(True, "--add/--no-add", help="Add the best take to the catalog."),
    model: str | None = typer.Option(None, "--model", show_default="auto = newest the provider offers",
                                     help="Cloud provider model id (see `folivox providers`)."),
    mode: str | None = typer.Option(None, "--mode", show_default="from --describe / --speaker",
                                    help="preset, design (Gemini, ElevenLabs, Fish), localize (Cartesia), "
                                         "clone (Cartesia, Fish)."),
    reference: str | None = typer.Option(None, "--reference", help="Your own catalog voice to clone (clone mode)."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask before sending text to a cloud provider."),
):
    """Generate a reference voice with a voice generator, pick the best take by speech recognition."""
    import tempfile

    from folivox import config
    from folivox import voices as catalog
    from folivox.providers import ProviderError
    from folivox.qa.asr import Transcriber
    from folivox.voicegen import GENERATORS
    from folivox.voices.design import generate_takes, safe_texts, score_takes

    lang = _check_lang(lang)
    try:
        gen = GENERATORS.create(config.provider("voicegen", generator), model=model)
    except (ProviderError, ValueError, TypeError) as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1) from None
    modes = getattr(gen, "modes", (gen.mode,))
    if mode:
        if mode not in modes:
            console.print(f"[red]{gen.name} supports: {', '.join(modes)}[/]")
            raise typer.Exit(2)
        gen.mode = mode
    elif describe and "design" in modes:
        gen.mode = "design"
    elif speaker and "preset" in modes:
        gen.mode = "preset"
    ref_path = None
    if gen.mode == "clone":
        mine = next((v for v in catalog.user() if v.id == reference), None)
        if mine is None:
            console.print("[red]--reference must be the id of one of your own voices (`folivox voices`)[/]")
            raise typer.Exit(2)
        if not yes and not typer.confirm(f"Is {mine.id!r} your own voice, or do you have the speaker's permission?",
                                         default=False):
            raise typer.Exit(1)
        ref_path = mine.path
    if gen.mode == "design" and not describe:
        console.print(f"[red]--describe is required for {gen.name}[/]")
        raise typer.Exit(2)
    if gen.mode in ("preset", "localize") and not speaker:
        try:
            names = ", ".join(gen.speakers())
        except ProviderError as e:
            names = str(e)
        console.print(f"[red]--speaker is required for {gen.name}: {names}[/]")
        raise typer.Exit(2)
    texts = text or safe_texts(lang)
    if not texts:
        console.print(f"[red]No built-in texts for {lang}; pass --text[/]")
        raise typer.Exit(2)
    if not gen.local:
        est = type(gen).estimate_for(sum(len(t) for t in texts), takes, gen.mode, gen.model)
        cost = f", about ${est['usd']:.2f}" if est.get("usd") is not None else ""
        console.print(f"[yellow]{gen.label or gen.name}: {est['chars']} characters will be sent to the provider and "
                      f"billed to your account{cost}.[/] {est['note']}")
        console.print("[dim]Some providers' terms restrict using their output to build competing models; the audio "
                      "stays on this computer and is never bundled.[/]")
        if not yes and not typer.confirm("Continue?", default=False):
            raise typer.Exit(1)
    out_dir = keep or Path(tempfile.mkdtemp(prefix="folivox-design-"))
    total = len(texts) * takes
    done = [0]

    def progress(job, seconds):
        done[0] += 1
        console.log(f"take {done[0]}/{total} ({seconds:.0f} s)")

    console.log(f"Generating {total} takes with {gen.name}" + (f" ({gen.model})" if hasattr(gen, "requested") else "")
                + "…")
    try:
        paths = generate_takes(gen, texts, lang, out_dir, description=describe, speaker=speaker, takes=takes,
                               prefix=voice_id, progress=progress, reference=ref_path)
    except ProviderError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1) from None
    by_path = {p: texts[int(p.stem.split("_t")[-1].split("_")[0]) - 1] for p in paths}
    console.log("Checking takes with speech recognition…")
    ranked = score_takes(paths, by_path, Transcriber(language=lang), lang)
    t = Table(title=voice_id)
    for col in ("#", "File", "Length", "Score", "Recognised"):
        t.add_column(col)
    for i, c in enumerate(ranked, 1):
        t.add_row(str(i), c.path.name, f"{c.duration:.1f} s", "—" if c.score is None else f"{c.score:.3f}",
                  c.transcript[:60])
    console.print(t)
    best = ranked[0]
    if add:
        commercial = bool(getattr(gen, "commercial", False))
        source = (f"{gen.label or gen.name}{' ' + gen.model if commercial else ''}"
                  f"{' (commercial API)' if commercial else ''}: {describe or speaker or gen.mode}")
        entry = catalog.add(voice_id, best.path, best.text, gender=gender, lang=lang, source=source,
                            license=gen.license, overwrite=True, commercial=commercial)
        console.print(f"[green]Voice added:[/] {entry.id} ← {best.path.name}")
    if keep:
        console.print(f"All takes: {out_dir}")


@voices_app.command("suggest-prompt", help=("Write a voice-design prompt for a book with a text LLM "
                                             "(for `voices design --describe`)."))
def voices_suggest_prompt(
    book_path: Path = typer.Argument(..., help="Book file."),
    generator: str | None = typer.Option(None, "--generator", "-g", show_default="qwen3-voicedesign",
                                         help="Voice generator the prompt is for (`folivox providers`)."),
    lang: str | None = typer.Option(None, "--lang", help="Book language.", show_default="from book / detected"),
    llm: str | None = typer.Option(None, "--llm", show_default="ollama",
                                 help="Text LLM provider (`folivox providers`)."),
    model: str | None = typer.Option(None, "--model", show_default="from config",
                                   help="Model name for the LLM provider."),
):
    """Write a voice-design prompt for a book with a text LLM."""
    from folivox import config
    from folivox.llm import LLMS
    from folivox.providers import ProviderError
    from folivox.voicegen import GENERATORS
    from folivox.voices.suggest import suggest_design_prompt

    name = config.provider("voicegen", generator)
    caps = None
    try:
        from folivox.voicegen.cloud import cloud_providers

        cls = cloud_providers().get(name or GENERATORS.default)
        caps = cls.capabilities() if cls else None
        GENERATORS.get(name)
    except ProviderError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(2) from None
    if caps and "design" not in caps["modes"]:
        console.print(f"[yellow]{caps['label']} has no voice design; it only offers preset voices "
                      f"(`folivox voices speakers -g {caps['name']}`). The prompt can still serve as instructions "
                      f"where the provider accepts them.[/]")
    book = _load(book_path, lang)
    text = "\n".join(b.text for ch in book.chapters() for b in ch.blocks)
    try:
        console.print(suggest_design_prompt(text, book.lang, LLMS.create(config.provider("llm", llm), model=model),
                                            title=book.title or "", author=book.author or ""))
    except (ProviderError, ValueError) as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1) from None


@voices_app.command("speakers", help="List the preset voices of a voice generator.")
def voices_speakers(
    generator: str | None = typer.Option(None, "--generator", "-g", show_default="qwen3-voicedesign",
                                         help="Voice generator provider (`folivox providers`)."),
    lang: str | None = typer.Option(None, "--lang", help="Only voices for this language."),
):
    """List the preset voices of a voice generator (cloud providers need their API key)."""
    from folivox import config
    from folivox.providers import ProviderError
    from folivox.voicegen import GENERATORS

    try:
        gen = GENERATORS.create(config.provider("voicegen", generator))
        speakers = gen.speakers()
    except (ProviderError, ValueError) as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1) from None
    t = Table(title=f"{gen.name} voices")
    for col in ("Id", "Language", "Sex"):
        t.add_column(col)
    for k, (lg, sex) in speakers.items():
        if not lang or lg in (lang, ""):
            t.add_row(k, lg or "any", sex or "—")
    console.print(t)


@voices_app.command("sample-text", help=("Pick passages from a book that are safe for a voice generator to "
                                          "read (by a text LLM)."))
def voices_sample_text(
    book_path: Path = typer.Argument(..., help="Book file."),
    n: int = typer.Option(3, "-n", help="How many passages to pick."),
    lang: str | None = typer.Option(None, "--lang", help="Book language.", show_default="from book / detected"),
    llm: str | None = typer.Option(None, "--llm", show_default="ollama",
                                 help="Text LLM provider (`folivox providers`)."),
    model: str | None = typer.Option(None, "--model", show_default="from config",
                                   help="Model name for the LLM provider."),
):
    """Pick passages from a book that are safe for a voice generator to read (by a text LLM)."""
    from folivox import config
    from folivox.llm import LLMS
    from folivox.voices.design import pick_book_texts

    book = _load(book_path, lang)
    text = "\n".join(b.text for ch in book.chapters() for b in ch.blocks)
    found = pick_book_texts(text, book.lang, LLMS.create(config.provider("llm", llm), model=model), n=n)
    for s in found:
        console.print(f"• {s}")
    if not found:
        console.print("[yellow]Nothing suitable found.[/]")


@app.command("tts-voices", help="List the voices of a cloud reading engine (needs its API key).",
             rich_help_panel="Setup")
def tts_voices(engine: str = typer.Argument(..., help="elevenlabs, openai, gemini, cartesia or fish."),
               lang: str | None = typer.Option(None, "--lang", help="Only voices for this language (Cartesia)."),
               delete: str | None = typer.Option(None, "--delete", metavar="NAME_OR_ID",
                                                 help="Delete a kept voice from your account and forget it.")):
    """List the voices of a cloud reading engine, and the voices you kept from earlier books."""
    from folivox.providers import ProviderError
    from folivox.tts import ENGINES, saved

    try:
        eng = ENGINES.create(engine)
        if eng.local:
            console.print(f"[red]{engine} is a local engine and has no voice list.[/]")
            raise typer.Exit(2)
        if delete:
            mine = saved.find(engine, delete)
            if mine is None:
                console.print(f"[red]No kept voice {delete!r} for {engine}.[/]")
                raise typer.Exit(2)
            eng.gen.teardown_voices([mine["id"]])
            saved.remove(engine, mine["id"])
            console.print(f"Deleted “{mine['name']}” from your account.")
            return
        rows = eng.gen.speakers(lang)
    except (ProviderError, ValueError) as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1) from None
    mine = saved.listing(engine)
    if mine:
        k = Table(title=f"Your kept {engine} voices (use --tts-voice NAME; delete with --delete NAME)")
        for col in ("Name", "Kind", "Description", "Id"):
            k.add_column(col)
        for v in mine:
            k.add_row(v["name"], v["mode"], v.get("description", "")[:60], v["id"])
        console.print(k)
    t = Table(title=f"{engine} voices (use --tts-voice)")
    for col in ("Id", "Name", "Sex", "Traits"):
        t.add_column(col)
    for vid, (_lang, sex) in rows.items():
        d = getattr(eng.gen, "details", {}).get(vid, {})
        t.add_row(vid, getattr(eng.gen, "names", {}).get(vid, ""), sex, d.get("traits", ""))
    console.print(t)
    if getattr(eng.gen, "voices_note", ""):
        console.print(f"[dim]{eng.gen.voices_note}[/]")


@app.command(help="List pluggable backends: TTS, speech recognition, voice generators, text LLMs.",
             rich_help_panel="Setup")
def providers():
    """List pluggable backends: TTS, speech recognition, voice generators, text LLMs."""
    from folivox import config
    from folivox.llm import LLMS
    from folivox.qa.asr import ASR_PROVIDERS
    from folivox.tts import ENGINES
    from folivox.voicegen import GENERATORS

    t = Table(title="Providers")
    for col in ("Kind", "Name", "Description", "Runs", "Needs"):
        t.add_column(col)
    for reg in (ENGINES, ASR_PROVIDERS, GENERATORS, LLMS):
        current = config.provider(reg.kind) or reg.default
        for p in reg.providers():
            mark = " *" if p.name == current else ""
            t.add_row(reg.kind, p.name + mark, p.description, "locally" if p.local else "cloud API (your key)",
                      p.needs)
    console.print(t)
    from folivox.voicegen.cloud import cloud_providers

    c = Table(title="Cloud voice generation (`folivox voices design -g NAME`)")
    for col in ("Name", "Modes", "Models (--model; default auto = newest)", "API key"):
        c.add_column(col)
    for cls in cloud_providers().values():
        c.add_row(cls.name, " + ".join(cls.modes), ", ".join(m["id"] for m in cls.models), cls.key_env)
    console.print(c)
    from folivox.tts.cost import estimate

    r = Table(title="Cloud reading engines (`folivox render --tts NAME`): approximate list price")
    for col in ("Name", "Models (--tts-model; default auto)", "Per hour of audio", "Request limit", "Voice modes"):
        r.add_column(col)
    for cls in cloud_providers().values():
        hour = estimate(cls.name, 14 * 3600, cls.models[0]["id"] if cls.models else "", 1, False)["per_hour"]
        r.add_row(cls.name, ", ".join(m["id"] for m in cls.models), f"≈ ${hour:.2f}" if hour else "?",
                  f"{cls.max_chars:,} chars", " + ".join(cls.modes))
    console.print(r)
    console.print(f"[dim]* selected. Config: {config.config_path()}[/]")


@app.command("ui", help="Start the local web interface (127.0.0.1 unless --host says otherwise; needs folivox[ui]).",
             rich_help_panel="Main commands",
             epilog="[bold]Examples[/]\n\n  [cyan]folivox ui --port 9000 --no-open[/]\n"
                    "  [cyan]folivox ui --restart[/]   replace the running studio (after an update)\n"
                    "  [cyan]folivox ui --stop[/]      stop the running studio")
def ui(port: int = typer.Option(8765, "--port", envvar="FOLIVOX_UI_PORT", help="Port to listen on."),
       host: str = typer.Option("127.0.0.1", "--host", envvar="FOLIVOX_UI_HOST",
                                help="Address to listen on. Anything but 127.0.0.1 (for example 0.0.0.0 in Docker) "
                                     "opens the studio to the network and requires FOLIVOX_UI_TOKEN "
                                     "(or FOLIVOX_UI_AUTH=off for a trusted home network, see docs/docker.md)."),
       open_browser: bool = typer.Option(True, "--open/--no-open", show_default=False,
                                         help="Open the browser automatically (on)."),
       restart: bool = typer.Option(False, "--restart", help="Stop the studio already running on this port, "
                                                              "then start this version there."),
       stop: bool = typer.Option(False, "--stop", help="Stop the studio running on this port and exit."),
       force: bool = typer.Option(False, "--force", help="With --restart/--stop: do it even if a render is running "
                                                          "(its progress is saved).")):
    """Start the local web interface."""
    try:
        import starlette  # noqa: F401
        import uvicorn  # noqa: F401
    except ImportError:
        console.print("[red]The web interface needs extra packages:[/] pip install \"folivox[ui]\"")
        raise typer.Exit(1) from None
    from folivox.ui import run, stop_running

    if stop:
        info = stop_running("127.0.0.1", port, force)
        console.print(f"Stopped the Folivox studio on port {port} (pid {info.get('pid', '?')}).")
        return
    allowed = {h.strip().lower() for h in os.environ.get("FOLIVOX_UI_ALLOWED_HOSTS", "").split(",") if h.strip()}
    run(host=host, port=port, open_browser=open_browser, restart=restart, force=force, allowed_hosts=allowed or None)


GUIDE = """[bold green]Folivox is ready.[/]

[bold]How to start it whenever you need it[/]
  1. Open a terminal (macOS: Terminal; Windows: PowerShell; Linux: any terminal).
  2. Type  [cyan]folivox ui[/]  and press Enter. Your browser opens the studio (if not, open the address it prints).
  3. Work in the studio. To stop it, return to the terminal and press [cyan]Ctrl+C[/].
  Nothing runs in the background and nothing starts by itself: when you close the terminal, Folivox is off.

[bold]Without the browser[/]
  [cyan]folivox render book.fb2[/]   one command, the audiobook appears next to the book
  Interrupted? Run the same command again: it continues where it stopped.

[bold]If something goes wrong[/]
  [cyan]folivox doctor[/]            checks ffmpeg, hardware and models
  [cyan]folivox doctor --bundle ~/Desktop[/]   saves a file to send to support
  [cyan]folivox bench[/]             speed test: characters per second on CPU / GPU, to compare machines
  [cyan]folivox help[/]              short guides: voices, providers, resuming, quality, speed

Update: [cyan]folivox update[/] (models and voices are kept)    Remove: [cyan]folivox uninstall[/]
"""


@app.command(help="First-time setup: check the machine, download the models with progress, show how to start.",
             rich_help_panel="Main commands")
def setup(skip_models: bool = typer.Option(False, "--skip-models", help="Only check the machine; download models "
                                           "later, on first use."),
          skip_asr: bool = typer.Option(False, "--skip-asr", help="Do not download the speech-recognition model "
                                        "(the quality check is then off until it is available).")):
    """Safe to run again: whatever is already downloaded is kept."""
    from folivox import diag

    console.rule("[bold]Your computer")
    data = diag.report()
    console.print(diag.render_report_text(data), markup=False, highlight=False)
    if not data["ffmpeg"]["found"]:
        console.print("\n[red]ffmpeg is missing.[/] Install it (macOS: brew install ffmpeg · Windows: winget install "
                      "ffmpeg · Linux: apt install ffmpeg) and run [cyan]folivox setup[/] again.")
        raise typer.Exit(1)
    if not skip_models:
        events = RichEvents()
        console.rule("[bold]Models (downloaded once, about 6 GB for the voice)")
        try:
            from folivox.tts import weights

            with weights.reporting(events.download):
                model_dir = weights.resolve_model_dir()
                weights.ensure_llm(model_dir, "rl")
            console.print(f"[green]✓[/] Voice model ready: {model_dir}")
            if not skip_asr:
                from folivox.qa.asr import ASR_PROVIDERS

                console.print("Speech recognition model (quality check, about 1.6 GB)…")
                ASR_PROVIDERS.create("auto", language="en")
                console.print("[green]✓[/] Speech recognition model ready")
        except ImportError as e:
            console.print(f"[red]Missing component:[/] {e}. Reinstall with the installer script, or "
                          "pip install \"folivox[all]\".")
            raise typer.Exit(1) from None
        except Exception as e:  # noqa: BLE001
            events.close()
            console.print(f"[red]Setup stopped:[/] {e}\nRun [cyan]folivox setup[/] again: the download continues "
                          "where it stopped.")
            raise typer.Exit(1) from None
        events.close()
        _offer_native()
    console.rule()
    console.print(GUIDE)


def _offer_native() -> None:
    """After the PyTorch models: one line about the optional native runtime, and an offer where it exists."""
    from folivox.cli_runtime import describe
    from folivox.tts import native_setup as ns

    try:
        st = ns.status()
        if st.ready or st.binary is None:
            return
        console.rule("[bold]Native runtime (optional, can be much faster)")
        for line in describe():
            console.print(line, markup=False, highlight=False)
        if not sys.stdin.isatty():
            console.print("Install it later with: folivox runtime install")
            return
        if typer.confirm("Download it now? (PyTorch stays the default either way)", default=False):
            from folivox.cli_runtime import runtime_install

            runtime_install(weights="q8_0", backend="auto", yes=False)
        else:
            console.print("Later: folivox runtime install")
    except typer.Exit:
        pass
    except Exception as e:  # noqa: BLE001 — an optional extra must not fail the setup
        console.print(f"[yellow]Native runtime check skipped: {e}[/]")


INSTALLER_URL = "https://raw.githubusercontent.com/vatavale/folivox/main/scripts/install.{ext}"


@app.command(help="Update Folivox to the latest version from GitHub (models, voices and settings are kept).",
             rich_help_panel="Other",
             epilog="[bold]Example[/]\n\n  [cyan]folivox update[/]   or, for a specific version, "
                    "[cyan]FOLIVOX_SOURCE=git+https://github.com/vatavale/folivox@<tag> folivox update[/]")
def update(yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask for confirmation.")):
    """Installs the latest Folivox release with the official installer (a source checkout is updated with git)."""
    import subprocess
    import tempfile
    import urllib.request

    if _dev_checkout():
        console.print("This Folivox runs from a development checkout, so the installer is not used. Update it with "
                      "[cyan]git pull[/] and [cyan]uv sync[/] in the checkout. Nothing was changed.")
        raise typer.Exit(1)
    windows = sys.platform == "win32"
    url = INSTALLER_URL.format(ext="ps1" if windows else "sh")
    console.print(f"Installed now: folivox {__version__}. The latest installer is fetched from [cyan]{url}[/]")
    if not yes and not typer.confirm("Update now?", default=True):
        console.print("Cancelled, nothing was changed.")
        raise typer.Exit(1)
    try:
        with urllib.request.urlopen(url, timeout=30) as r:  # noqa: S310 (fixed https URL)
            script = r.read()
    except OSError as e:
        console.print(f"[red]Could not download the installer:[/] {e}\nCheck the network and try again.")
        raise typer.Exit(1) from None
    if windows:
        # Windows will not let uv replace Scripts\\folivox.exe while it is running, so the installer
        # goes to its own console and waits for this process (and the launcher that started it) to exit.
        path = Path(tempfile.mkdtemp(prefix="folivox-update-")) / "install.ps1"
        path.write_bytes(script)
        env = {**os.environ, "FOLIVOX_NO_START": "1", "FOLIVOX_WAIT_PID": f"{os.getpid()},{os.getppid()}",
               "FOLIVOX_SELF_DELETE": "1"}
        try:
            subprocess.Popen(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(path)],
                             env=env, creationflags=subprocess.CREATE_NEW_CONSOLE)  # type: ignore[attr-defined]
        except OSError as e:
            console.print(f"[red]Could not start PowerShell to run the installer:[/] {e}. Nothing was changed.")
            raise typer.Exit(1) from None
        console.print("The installer continues in a new window; this one closes so the files can be replaced. "
                      "Check the version afterwards with [cyan]folivox --version[/].")
        raise typer.Exit(0)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / ("install.ps1" if windows else "install.sh")
        path.write_bytes(script)
        env = {**os.environ, "FOLIVOX_NO_START": "1"}
        cmd = (["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(path)] if windows
               else ["bash", str(path)])
        try:
            code = subprocess.call(cmd, env=env)
        except OSError as e:
            console.print(f"[red]Could not start {cmd[0]} to run the installer:[/] {e}. Nothing was changed.")
            raise typer.Exit(1) from None
    if code:
        console.print("[red]Update failed.[/] It is safe to run [cyan]folivox update[/] again.")
        raise typer.Exit(code)
    console.print("[green]✓[/] Updated. Check the version with [cyan]folivox --version[/].")


def _dev_checkout() -> bool:
    """True when Folivox runs from a source checkout or an editable install (the installer would not update it)."""
    import json
    from importlib import metadata

    try:
        raw = metadata.distribution("folivox").read_text("direct_url.json")
        if raw and json.loads(raw).get("dir_info", {}).get("editable"):
            return True
    except (metadata.PackageNotFoundError, ValueError, AttributeError, OSError):
        pass
    pkg = Path(__file__).resolve().parent
    return any((d / ".git").exists() for d in (pkg.parent, pkg.parent.parent))


@app.command(help="Remove what Folivox downloaded (models; with --everything also voices and settings). "
                  "Never touches books, audiobooks, ffmpeg or uv.", rich_help_panel="Other")
def uninstall(everything: bool = typer.Option(False, "--everything",
                                              help="Also remove your voices and settings (without it only the "
                                                   "downloaded models go)."),
              yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask for confirmation."),
              dry_run: bool = typer.Option(False, "--dry-run", help="Only show what would be removed.")):
    """Step 1 of removal; step 2 is the last line it prints."""
    from folivox import uninstall as un

    items = un.plan(everything)
    if items:
        t = Table(box=None, show_header=False, padding=(0, 2))
        for it in items:
            t.add_row(it.label, f"{it.size / 1e9:.1f} GB" if it.size >= 1e8 else f"{it.size / 1e6:.1f} MB",
                      f"[dim]{it.path}[/]")
        console.print("[bold]Will be removed:[/]")
        console.print(t)
        console.print(f"Frees about {sum(i.size for i in items) / 1e9:.1f} GB.")
    else:
        console.print("Nothing to remove: no downloaded models" + (" or settings" if everything else "") + " found.")
    if not everything:
        console.print("[dim]Kept: your voices and settings (add --everything to remove them too).[/]")
    console.print("[dim]Never touched: your books, BOOK.folivox work folders and finished audiobooks, ffmpeg, uv, "
                  "and any model folder you set yourself (FOLIVOX_MODEL_DIR).[/]")
    if dry_run:
        return
    if items:
        if not yes and not typer.confirm("Remove these now?", default=False):
            console.print("Cancelled, nothing was removed.")
            raise typer.Exit(1)
        errors = un.remove(items, everything)
        for e in errors:
            console.print(f"[red]Could not remove:[/] {e}")
        if errors:
            console.print("Close anything that uses these files (the studio, a render) and run it again.")
            raise typer.Exit(1)
        console.print("[green]✓[/] Removed.")
    console.print("\nLast step, remove the program itself: [cyan]uv tool uninstall folivox[/]\n"
                  "(Optional, only if you installed them for Folivox and use them for nothing else: "
                  "[cyan]uv self uninstall[/], [cyan]winget uninstall Gyan.FFmpeg[/], [cyan]brew uninstall ffmpeg[/].)")


@app.command(help="Check the setup (ffmpeg, model, devices, disk) and optionally write a redacted support bundle.",
             rich_help_panel="Main commands",
             epilog="[bold]Example[/]\n\n  [cyan]folivox doctor --bundle ~/Desktop[/]")
def doctor(bundle: Path | None = typer.Option(None, "--bundle", "-b", help="Write a redacted zip (versions, log tail, "
                                              "progress counters; no keys, no book text) to this file or folder."),
           work: Path | None = typer.Option(None, "--work", help="Work folder of a render to summarise "
                                            "(NAME.folivox next to the output).")):
    """Print a health report; with --bundle also save a zip to attach to a bug report."""
    from folivox import diag

    data = diag.report(work)
    console.print(diag.render_report_text(data), markup=False, highlight=False)
    if bundle:
        try:
            path = diag.support_bundle(bundle, work)
        except OSError as e:
            console.print(f"\nCould not write the support bundle to {bundle}: {e.strerror or e}", markup=False)
            raise typer.Exit(1) from None
        console.print(f"\nSupport bundle: {path}\nIt holds no API keys and no book text; look inside before sharing.",
                      markup=False)


@app.command(help="Measure synthesis speed (characters per second) on each device, to compare machines.",
             rich_help_panel="Other",
             epilog="[bold]Example[/]\n\n  [cyan]folivox bench[/]   or   [cyan]folivox bench --devices cpu,cuda[/]")
def bench(devices: str = typer.Option("", "--devices", help="Comma-separated: cpu, cuda, mps, native (default: all "
                                      "found; native only when installed). cuda is tried even if "
                                      "`auto` would skip it."),
          voice: str = typer.Option("", "--voice", "-v", show_default="by language", help="Voice for the test."),
          lang: str = typer.Option("ru", "--lang", help="Language of the test text (ru or en)."),
          timeout: int = typer.Option(1800, "--timeout", help="Seconds one device may take before it is given up."),
          tts: str = typer.Option("cosyvoice3", "--tts", help="Engine to measure: cosyvoice3, fish-s2 or fish-drama3 "
                                  "(Fish needs your own voice, not a bundled one)."),
          tts_build: str = typer.Option("auto", "--tts-build", help="Fish build id from `folivox engines`."),
          runtime: str | None = typer.Option(None, "--runtime", help="For Fish: native picks the native GGUF builds.")):
    """One short fragment per device, each in its own process, so a crash on one device does not stop the rest."""
    from folivox import bench as b
    from folivox.tts import weights

    fish = tts != "cosyvoice3"
    if not fish and weights.needs_download(None, "rl"):
        console.print("[red]The model is not downloaded yet.[/] Run `folivox setup` first.")
        raise typer.Exit(1)
    wav, vtext, _ = _resolve_voice(voice, None, lang)
    spec = None
    if fish:
        from folivox.tts import builds, fish_check
        from folivox.tts import runtime as tts_runtime

        if fish_check.is_bundled_voice(wav) and not fish_check.bundled_voices_allowed():
            console.print(f"[red]{fish_check.BUNDLED_VOICES_NOTE}[/]")
            raise typer.Exit(2)
        mode = tts_runtime.resolve(tts, runtime)
        try:
            bld = builds.build_for(tts, tts_build, runtime=mode)
        except builds.CatalogError as e:
            console.print(f"[red]{e}[/]")
            raise typer.Exit(2) from None
        spec = {"tts": tts, "build": bld.id, "runtime": mode}
        devices = devices or bld.backend
    devs = [d.strip() for d in devices.split(",") if d.strip()] or b.available_devices()
    text = b.sample_text(lang)
    console.print(f"Test fragment: {len(text)} characters. Devices: {', '.join(devs)}. "
                  "The first run of a slow device can take minutes.")
    rows = []
    for d in devs:
        with console.status(f"{d}: loading the model and synthesising…"):
            rows.append(b.run_device(d, str(wav), vtext, text, timeout, spec))
    table = Table("device", "engine", "chars/s", "RTF", "load, s", "warm-up, s", "GPU peak, GB", "RAM peak, GB",
                  "result")
    for r in rows:
        if "error" in r:
            table.add_row(r["device"], "–", "–", "–", "–", "–", "–", "–", f"[red]{r['error']}[/]")
        else:
            gpu = r.get("gpu_card_peak_gb") if r.get("gpu_card_peak_gb") is not None else r.get("peak_vram_gb")
            ram = r.get("ram_peak_gb")
            table.add_row(r["device"], r.get("runtime", "") or str(r.get("actual", "")), f"{r['chars_per_s']:.1f}",
                          f"{r['rtf']:.2f}", f"{r['load_s']:.0f}", f"{r['warmup_s']:.0f}",
                          f"{gpu:.1f}" if gpu is not None else "–", f"{ram:.1f}" if ram is not None else "–",
                          "[green]ok[/]")
    console.print(table)
    console.print("[dim]chars/s: characters of text per second of synthesis (more is faster). RTF: synthesis time / "
                  "audio length (below 1 is faster than real time). Whisper's quality check is not included.[/]")
    what = f"{tts} {spec['build']}" if spec else "cosyvoice3"
    console.print("\n[bold]To share the result[/] (paste this into an issue or a message; it holds no private data):")
    console.print(b.report_text(rows, what, lang), markup=False, highlight=False)
    ok = [r for r in rows if "error" not in r]
    if len(ok) > 1 and not fish:
        best = max(ok, key=lambda r: r["chars_per_s"])
        console.print(f"Fastest: [bold]{best['device']}[/]. Use it with `--device {best['device']}`.")


@app.command("help", help="Explain a topic: voices, providers, config, languages, resuming, quality, speed.",
             rich_help_panel="Help")
def help_topic(topic: str | None = typer.Argument(None, help="Topic name; omit to list topics.", show_default=False)):
    """Explain a topic in more detail than --help can."""
    from rich.panel import Panel

    if topic is None or topic not in TOPICS:
        if topic:
            console.print(f"[red]Unknown topic {topic!r}.[/]")
        t = Table(box=None, show_header=False, padding=(0, 2))
        for name, (title, _body) in TOPICS.items():
            t.add_row(f"[cyan]folivox help {name}[/]", title)
        console.print(t)
        raise typer.Exit(2 if topic else 0)
    title, body = TOPICS[topic]
    console.print(Panel(body, title=f"[bold]{title}[/]", title_align="left", padding=(1, 2)))


@app.callback()
def _root(ctx: typer.Context,
          version: bool | None = typer.Option(None, "--version", callback=_version, is_eager=True,
                                              help="Show the version and exit.")):
    if ctx.invoked_subcommand is not None:
        return
    from rich.panel import Panel

    console.print(f"[bold]Folivox[/] {__version__} — local audiobooks from e-books\n")
    console.print(Panel(QUICKSTART, title="Quick start", title_align="left", padding=(1, 2)))
    t = Table(box=None, show_header=False, padding=(0, 2), title="Commands", title_justify="left")
    rows = [("folivox BOOK [options]", "render a book (short for `render`)"),
            ("folivox inspect BOOK", "chapters, size, estimated duration"),
            ("folivox ui", "local web interface in your browser"),
            ("folivox voices", "list, add, design voices"),
            ("folivox providers", "TTS / ASR / LLM backends and config"),
            ("folivox doctor", "check ffmpeg, hardware and models"),
            ("folivox bench", "speed test on each device (CPU, CUDA, MPS)"),
            ("folivox help [topic]", "voices · providers · config · languages · resuming · quality · speed")]
    for cmd, desc in rows:
        t.add_row(f"[cyan]{cmd}[/]", desc)
    console.print(t)
    console.print("\n[dim]All options: `folivox render --help`[/]")


def main():
    # `folivox book.fb2 ...` is shorthand for `folivox render book.fb2 ...`
    args = sys.argv[1:]
    if args and args[0] not in COMMANDS and not args[0].startswith("-"):
        sys.argv.insert(1, "render")
    app()


if __name__ == "__main__":
    main()

"""Book → audiobook pipeline. UI-agnostic: progress is reported through :class:`Events`."""

from __future__ import annotations

import hashlib
import os
import re
import time
import zlib
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from folivox.audio import postprocess, save
from folivox.audio.assemble import BookMeta, ChapterAudio, build_chapter, write_m4b, write_mp3s
from folivox.book import Book, Chapter
from folivox.journal import Journal, PlannedChunk, Take
from folivox.text.chunk import ChunkOptions, MarkupContext, finalize, plan_chapter
from folivox.text.pronounce import Lexicon

# Bump when synthesis changes in a way that invalidates cached audio.
SYNTH_VERSION = "cv3-1"
FISH_SYNTH_VERSION = "fish-1"  # Fish engines (S2 Pro, Drama 3): own counter, independent of CosyVoice


# QA strictness: highest speech-recognition error (CER + rate penalty) a take may have without a retake
# Measured on CosyVoice 3 (ru): good takes score 0–0.07, a take that drops one sentence of a fragment 0.15–0.25.
QA_LEVELS = {"low": 0.14, "standard": 0.10, "high": 0.07}


@dataclass
class RenderOptions:
    output: Path
    voice_wav: Path | None = None  # reference recording (local engines)
    voice_text: str | None = None
    work_dir: Path | None = None
    chapters: str | None = None  # "1-3,5" (1-based)
    preview: int | None = None  # only the first N characters of each selected chapter
    depth: int = 2
    # None: 300 locally (CosyVoice 3 drops the last sentence of longer fragments), the engine's own default online
    max_chars: int | None = None
    stress: str = "yo"
    lang: str | None = None  # book language (ISO 639-1); None → from the book metadata / detection
    speed: float = 1.0
    device: str = "auto"
    tts: str = "cosyvoice3"  # provider name in folivox.tts.ENGINES
    runtime: str = ""  # "" = from the settings (auto | pytorch | native); see folivox.tts.runtime
    tts_build: str = "auto"  # Fish engines: build id from the catalog (auto = the best installed one that fits)
    asr: str = "auto"  # provider name in folivox.qa.asr.ASR_PROVIDERS
    llm: str = "rl"  # CosyVoice LLM weights: "rl" (fewer content errors, per upstream eval) or "base"
    threads: int | None = None
    model_dir: str | None = None
    qa: bool = True
    max_takes: int = 3
    qa_threshold: float = QA_LEVELS["standard"]
    mp3: bool = False
    bitrate: str = "64k"
    narrator: str = ""
    cover: Path | None = None  # an image to use instead of the book's own cover
    no_cover: bool = False  # write no cover at all
    meta: dict[str, str] = field(default_factory=dict)  # edited book details; an empty value keeps the book's own
    chunk: ChunkOptions = field(default_factory=ChunkOptions)
    pronounce: tuple[Path, ...] = ()  # custom pronunciation files, the book's own last (see folivox.text.pronounce)
    markup: Path | None = None  # speakers and expression tags for the book (see folivox.markup)
    expression_tags: bool = True  # write the markup's expression tags into the model input (Fish engines only)
    sample_exact: bool = False  # a preview of a chosen passage: read it from its first word, skip nothing
    # cloud engines (paid, the text leaves the computer)
    tts_model: str = "auto"
    tts_voice: str = ""  # provider voice id or name
    tts_style: str | None = None  # narration note; None → the engine's neutral default
    tts_mode: str = "preset"  # preset | design | localize | clone
    tts_description: str = ""  # voice description (design mode)
    voice_ref: Path | None = None  # your own recording to clone (clone mode)
    max_cost: float | None = None  # stop cleanly before spending more than this many USD
    accept_cloud: bool = False  # the caller showed the estimate and the user agreed
    keep_voice: bool = False  # keep a voice made in the provider account (and list it for other books)
    tts_voice_name: str = ""  # the name a kept voice is listed under

    def resolved_work_dir(self) -> Path:
        if self.work_dir:
            return Path(self.work_dir).expanduser()
        if os.environ.get("FOLIVOX_WORK_DIR"):  # Docker: keep journals and takes out of the audiobook library
            return Path(os.environ["FOLIVOX_WORK_DIR"]).expanduser() / (self.output.stem + ".folivox")
        return self.output.with_name(self.output.stem + ".folivox")


class Cancelled(Exception):
    """Raised by :func:`render` when ``Events.cancelled()`` returns True; progress is saved in the journal."""


class CostCapReached(Cancelled):
    """The spending cap was reached; progress is saved, raise the cap and run again to continue."""


class Events:
    """Override what you need. All methods are called from the rendering thread."""

    def cancelled(self) -> bool:
        """Polled between fragments; return True to stop cleanly (resume later by running again)."""
        return False

    def stage(self, name: str, detail: str = "") -> None: ...

    def warning(self, message: str) -> None: ...

    def plan(self, total_chars: int, done_chars: int, total_chunks: int, pending_chunks: int) -> None: ...

    def chunk_started(self, chunk: PlannedChunk) -> None: ...

    def chunk_done(self, chunk: PlannedChunk, take: Take, attempts: list[float | None]) -> None: ...

    def chapter_built(self, index: int, title: str, duration: float) -> None: ...

    def layout(self, chapters: list[dict]) -> None:
        """Every fragment of every selected chapter: ``{index, title, fragments: [[chars, key]], done: "1010…"}``."""

    def step(self, step: str, attempt: int = 1, attempts: int = 1) -> None:
        """What happens to the current fragment: ``synth`` (take n of m), ``qa`` (recognising it), ``split``."""

    def download(self, name: str, done_bytes: int, total_bytes: int) -> None:
        """Model weights are being downloaded (``total_bytes`` is 0 when unknown)."""

    def spend(self, chars: int, usd: float | None, cap: float | None) -> None:
        """Cloud engines: characters sent so far in this run and the approximate spend."""


def parse_selection(spec: str | None, count: int) -> list[int]:
    """'1-3,5' → [0, 1, 2, 4] (0-based); None → all."""
    if not spec:
        return list(range(count))
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        m = re.fullmatch(r"(\d+)?\s*-\s*(\d+)?|(\d+)", part)
        if not m:
            raise ValueError(f"Invalid chapter number: {part!r}")
        if m.group(3):
            a = b = int(m.group(3))
        else:
            a, b = int(m.group(1) or 1), int(m.group(2) or count)
        out.update(i - 1 for i in range(a, b + 1) if 1 <= i <= count)
    if not out:
        raise ValueError(f"The book has no chapters {spec!r} (there are {count})")
    return sorted(out)


def load_markup(book: Book, opts: RenderOptions, events: Events):
    """The book's markup bound to its chapters (problems are warnings: the narrator reads what does not fit)."""
    from folivox import markup as mk

    if not opts.markup:
        return None
    try:
        raw = mk.Markup.load(Path(opts.markup).expanduser())
    except (OSError, mk.MarkupError) as e:
        raise RuntimeError(f"Cannot use the markup {opts.markup}: {e}") from None
    if raw.depth != opts.depth:
        events.warning(f"The markup was made for chapter depth {raw.depth} but this render uses {opts.depth}: "
                       "chapters may not line up; the parts that do not match are read by the narrator")
    clean, problems = mk.bind(raw, book.chapters(raw.depth))
    for w in problems[:10]:
        events.warning(w)
    if len(problems) > 10:
        events.warning(f"…and {len(problems) - 10} more markup problems")
    return clean


def speaker_voices(markup, main_voice, asr, opts: RenderOptions, events: Events) -> dict:
    """{speaker id: Voice} for the speakers that read something and have a voice of their own."""
    from folivox import voices as catalog
    from folivox.tts.voice import VoiceError, load_voice

    out: dict = {}
    if markup is None:
        return out
    for sid in sorted(markup.used_speakers() - {0}):
        sp = markup.speaker(sid)
        ref = (sp.voice if sp else "").strip()
        if not ref:
            continue
        entry = catalog.find(ref)
        if entry is not None:
            wav, text = entry.path, entry.transcript
        else:
            wav, text = Path(ref).expanduser(), None
        try:
            out[sid] = load_voice(wav, text, transcribe=asr.transcribe_file if asr else None)
        except (VoiceError, OSError) as e:  # deleted catalog voice, moved or unreadable sample
            events.warning(f"{sp.name}: the voice {ref} cannot be used ({e}); the narrator reads this character")
            continue
        for w in out[sid].warnings:
            events.warning(f"{sp.name}: {w}")
    return out


def engine_languages(name: str, engine=None) -> frozenset[str] | None:
    """The languages an engine can read (an instance knows its own; a local engine's class is read without loading
    the model). ``None`` when unknown: nothing is filtered then."""
    langs = getattr(engine, "languages", None)
    if not langs:
        import importlib

        from folivox.tts import ENGINES

        try:
            module, _, attr = ENGINES.get(name).target.partition(":")
            langs = getattr(getattr(importlib.import_module(module), attr), "languages", ())
        except Exception:  # noqa: BLE001 — an unknown engine is checked when it is created
            langs = ()
    return frozenset(langs) or None


def plan(book: Book, opts: RenderOptions, journal: Journal, voice_id: str, events: Events, engine=None,
         markup=None, voices: dict | None = None) -> list[str]:
    chapters = book.chapters(opts.depth)
    selected = parse_selection(opts.chapters, len(chapters))
    events.stage("plan", "Preparing the preview text" if opts.preview else
                 f"Preparing text: {len(selected)} of {len(chapters)} chapters")
    cloud = engine is not None and not engine.local
    stress = opts.stress
    if cloud and stress == "all":  # stress marks are CosyVoice markup; a cloud voice would read them aloud
        events.warning("Stress marks are not used with cloud engines; restoring ё only")
        stress = "yo"
    if cloud:  # never above the provider's request limit
        max_chars = min(opts.max_chars or engine.fragment_chars, engine.max_chars)
    else:
        max_chars = opts.max_chars or 300
    copts = ChunkOptions(max_chars=max_chars, min_chars=min(opts.chunk.min_chars, max_chars), stress=stress,
                         lang=opts.lang or book.lang or "ru", pauses=opts.chunk.pauses,
                         speak_headings=opts.chunk.speak_headings, local_engine=not cloud, mixed=opts.chunk.mixed,
                         join_paragraphs=cloud and max_chars > 300,
                         speaks=engine_languages(opts.tts, engine))
    if opts.pronounce:
        copts.lexicon = Lexicon.load(opts.pronounce)
        for w in copts.lexicon.warnings:
            events.warning(f"Pronunciations: {w}")
        for w in copts.lexicon.unusable_here(not cloud):
            events.warning(f"Pronunciations: «{w}» is given as phonemes only; an online engine reads it as written")
    tags_ok = markup is not None and takes_tags(opts.tts) and opts.expression_tags
    if markup is not None and opts.expression_tags and not takes_tags(opts.tts) and any(
            c.marks or any(s.tags for s in c.spans) for c in markup.chapters.values()):
        events.warning("Expression tags are written only for Fish engines; this engine reads the text without them")

    def ctx(i: int):
        return MarkupContext(markup, i, tags_ok) if markup is not None else None

    planned = [(chapters[i].title, _plan_preview(chapters[i], copts, opts.preview, ctx(i), opts.sample_exact))
               for i in selected]
    settings = f"{SYNTH_VERSION}|llm={opts.llm}|speed={opts.speed}"
    from folivox.tts import runtime

    if runtime.cosyvoice_native(opts.tts, opts.runtime or None):
        settings = runtime.take_settings(opts.speed)
    elif opts.tts != "cosyvoice3":  # keep cache keys of existing CosyVoice renders valid
        settings = f"{opts.tts}|speed={opts.speed}"
        if cloud:
            settings = f"{opts.tts}|{engine.model}"
        elif is_fish(opts.tts):  # another build or revision sounds different
            from folivox.tts import builds

            b = builds.build_for(opts.tts, opts.tts_build, runtime=runtime.resolve(opts.tts, opts.runtime or None))
            settings = f"{opts.tts}|{FISH_SYNTH_VERSION}|{b.id}@{b.revision[:12]}"
    for code, n in sorted(copts.fallbacks.items(), key=lambda kv: -kv[1]):
        events.warning(f"{n} fragment(s) in {_language_name(code)} will be read in {_language_name(copts.lang)}: "
                       f"{opts.tts} cannot read {_language_name(code)} here")
    journal.set_plan(planned, voice_id, settings, {sid: v.id for sid, v in (voices or {}).items()})
    journal.set_meta(book_title=book.title, chapters_selected=[i + 1 for i in selected], settings=settings,
                     lang=copts.lang, qa_threshold=opts.qa_threshold if opts.qa else None)
    return [t for t, _ in planned]


LOOKAHEAD = 2  # fragments whose LLM tokens are generated ahead of time


def voice_accent_notes(journal: Journal, markup, opts: RenderOptions) -> list[str]:
    """One note per voice of the catalog that reads fragments in a language other than its own: a cloned voice
    speaks every language the engine knows, but keeps the accent of its sample."""
    from folivox import voices as catalog

    count: dict[tuple[int, str], int] = {}
    for c in journal.chunks():
        if c.lang:
            count[(c.speaker, c.lang)] = count.get((c.speaker, c.lang), 0) + 1
    main = next((v for v in catalog.all_voices() if opts.voice_wav and Path(v.path) == Path(opts.voice_wav)), None)
    notes = []
    for (sid, lang), n in sorted(count.items()):
        sp = markup.speaker(sid) if markup is not None else None
        entry = catalog.find(sp.voice) if sp and sp.voice else main
        if entry is not None and entry.lang != lang:
            who = f"{sp.name}: " if sid and sp else ""
            notes.append(f"{who}{n} fragment(s) in {_language_name(lang)} are read by a {_language_name(entry.lang)} "
                         f"voice ({entry.name}) and keep its accent; a {_language_name(lang)} voice reads them better")
    return notes


def _language_name(code: str) -> str:
    from folivox.markup_llm import LANG_NAMES

    return LANG_NAMES.get(code, code)


def _seed(key: str, attempt: int) -> int:
    return zlib.crc32(key.encode()) + attempt * 7919


def _prose_start(chunks) -> int:
    """Index of the first fragment that reads like running text, so a preview skips the title and
    copyright page (names, years, publishers) that synthesise slowly and say nothing about the voice."""
    for i, c in enumerate(chunks):
        src = c.source
        if len(src) < 120 or not src.rstrip().endswith((".", "!", "?", "…", "»", "”", '"')):
            continue
        odd = sum(ch.isdigit() or ("a" <= ch.lower() <= "z") for ch in src)
        if odd / len(src) < 0.08:
            return i
    return 0


PREVIEW_HEAD = 6000  # characters of a chapter planned first for a preview (room to skip the title pages)


def _head(chapter: Chapter, budget: int) -> tuple[Chapter, bool]:
    """The start of ``chapter`` of about ``budget`` characters, and whether anything was cut off."""
    blocks, n = [], 0
    for b in chapter.blocks:
        if n >= budget:
            return Chapter(chapter.title, blocks, chapter.headings), True
        if n + len(b.text) > budget:  # one huge block: cut at a word boundary
            cut = b.text[:budget - n]
            cut = cut[:cut.rfind(" ")] if " " in cut else cut
            blocks.append(replace(b, text=cut))
            return Chapter(chapter.title, blocks, chapter.headings), True
        blocks.append(b)
        n += len(b.text)
    return chapter, False


def _plan_preview(chapter: Chapter, copts: ChunkOptions, preview: int | None, ctx: MarkupContext | None = None,
                  exact: bool = False):
    """Plan a chapter; for a preview only its beginning, so a huge book is not split into thousands of
    fragments to keep the first few hundred characters. The window grows if the title pages eat it."""
    if not preview:
        return plan_chapter(chapter, copts, ctx)
    budget = max(PREVIEW_HEAD, preview * 4)
    bare = replace(copts, stress="none", lexicon=None)  # stress is a neural model: run it only on the kept fragments
    while True:
        part, cut = _head(chapter, budget)
        out = _limit(plan_chapter(part, bare, ctx), preview, copts, exact)
        if not cut or sum(len(c.source) for c in out) >= preview:
            break
        budget *= 4
    if copts.stress != "none" or copts.lexicon:
        for c in out:
            c.text, c.source = finalize(c.text, copts)
    return out


def _limit(chunks, preview: int | None, copts: ChunkOptions, exact: bool = False):
    """Keep fragments until ``preview`` characters are covered; end with a chapter pause."""
    if not preview:
        return chunks
    if not exact:
        chunks = chunks[_prose_start(chunks):]
    out, n = [], 0
    for c in chunks:
        if n >= preview:
            break
        out.append(c)
        n += len(c.source)
    if out:
        out[-1].pause_after = copts.pauses.chapter_end
    return out


FOREIGN_SHARE = 0.12  # share of Latin letters at which a fragment of a non-Latin language is not QA-checked


def _mostly_foreign(source: str, lang: str) -> bool:
    """True for e-mails, URLs and publisher names in a Cyrillic/CJK text: Whisper transcribes them
    phonetically, so the comparison with the written form fails however well the take was read."""
    if lang not in ("ru", "zh", "ja", "ko"):
        return False
    letters = [c for c in source if c.isalpha()]
    if not letters:
        return False
    return sum("a" <= c.lower() <= "z" for c in letters) / len(letters) >= FOREIGN_SHARE


_PHONEMES = re.compile(r"(?:\[[A-Z]{1,2}[0-2]?\])+")  # custom-pronunciation markup, not text to compare with
SPLIT_PAUSE = 0.25  # seconds between the halves of a fragment split after failed takes


def _best_take(prefetch, asr, opts: RenderOptions, key: str, text: str, source: str, depth: int = 0,
               events: Events | None = None, voice=None, fragment_lang: str = ""):
    """Synthesise ``text`` up to ``opts.max_takes`` times, keep the best take by QA score.

    If every take fails QA and the fragment has several sentences, split it in two at a
    sentence boundary and render the halves separately: CosyVoice tends to drop the last
    sentence of long, list-like paragraphs, and shorter inputs fix that reliably.
    Returns (score, audio, asr_text, per-attempt scores). ``fragment_lang`` is the fragment's language when it is
    not the book's: the engine is told, and the check uses that language (recogniser, canonical form, split).
    """
    from folivox.qa import score, threshold_for
    from folivox.text.chunk import sentences
    from folivox.text.stress import strip_marks

    events = events or Events()
    best = None
    attempts: list[float | None] = []
    lang = fragment_lang or opts.lang or "ru"
    limit = threshold_for(source, opts.qa_threshold, lang)
    takes = max(1, opts.max_takes if asr else 1)
    if asr is not None and _mostly_foreign(source, lang):
        asr, takes = None, 1  # the recogniser writes Latin words in Cyrillic: the check could only ever fail
    for attempt in range(takes):
        events.step("synth", attempt + 1, takes)
        if attempt:  # a retake of a cloud fragment is billed: stop cleanly instead of overspending
            _check_cap(prefetch.engine, len(text))
        extra = {"voice": voice} if voice is not None else {}  # a speaker with a voice of their own
        if fragment_lang:
            extra["lang"] = fragment_lang
        syn = prefetch.synthesize(key, text, _seed(key, attempt), speed=opts.speed, **extra)
        audio = postprocess(syn.audio, syn.sample_rate)
        dur = len(audio) / syn.sample_rate
        sc = hyp = None
        if asr is not None:
            events.step("qa", attempt + 1, takes)
            try:
                spoken = {"language": fragment_lang} if fragment_lang else {}
                hyp = asr.transcribe(audio, syn.sample_rate, **spoken) if dur > 0.05 else ""
                sc = score(source, hyp, dur, lang)
            except Exception as e:  # noqa: BLE001 — a failed check must not lose an hour of synthesis
                events.warning(f"Speech recognition failed for one fragment ({e}); keeping it unchecked")
                hyp = sc = None
        attempts.append(sc)
        if best is None or (sc is not None and sc < best[0]):
            best = (sc if sc is not None else 0.0, audio, hyp)
        if sc is None or sc <= limit:
            return best[0], best[1], best[2], attempts

    parts = sentences(text, lang)
    # a split re-renders the fragment: free locally, billed again with a cloud engine
    if len(parts) < 2 or depth >= (2 if getattr(prefetch.engine, "local", True) else 0):
        return best[0], best[1], best[2], attempts
    events.step("split")
    # split into two halves of similar length at a sentence boundary
    total, acc, cut = sum(len(x) for x in parts), 0, 1
    for i, part in enumerate(parts[:-1], 1):
        acc += len(part)
        if acc >= total / 2:
            cut = i
            break
    halves = [" ".join(parts[:cut]), " ".join(parts[cut:])]
    # a half's check text is rebuilt from the model input: phoneme markup of custom pronunciations is dropped
    results = [_best_take(prefetch, asr, opts, f"{key}/{i}", h, _PHONEMES.sub("", strip_marks(h)), depth + 1, events,
                          voice, fragment_lang)
               for i, h in enumerate(halves)]
    sr = prefetch.engine.sample_rate
    audio = np.concatenate([results[0][1], np.zeros(int(sr * SPLIT_PAUSE), np.float32), results[1][1]])
    joined = f"{results[0][2] or ''} {results[1][2] or ''}".strip()
    split_score = score(source, joined, len(audio) / sr, lang) if asr is not None else 0.0
    attempts += results[0][3] + results[1][3]
    if split_score < best[0]:
        return split_score, audio, joined, attempts
    return best[0], best[1], best[2], attempts


def _check_cap(engine, next_chars: int) -> None:
    check = getattr(engine, "check_budget", None)
    if check is not None:
        check(next_chars)


def _report_spend(engine, events: Events, opts: RenderOptions) -> None:
    if not getattr(engine, "local", True):
        events.spend(engine.chars_sent, engine.spent_usd, opts.max_cost)


def synthesize_pending(journal: Journal, engine, voice, asr, opts: RenderOptions, events: Events,
                       voices: dict | None = None) -> None:
    from folivox.tts.base import Prefetcher

    pending = journal.pending()
    st = journal.stats()
    events.plan(st["total_chars"], st["done_chars"], len(journal.chunks()), len(pending))
    pending_keys = {c.key for c in pending}
    layout = []
    for ci, title in journal.chapters():
        chunks = journal.chunks(ci)
        layout.append({"index": ci, "title": title, "fragments": [[len(c.source), c.key[:10]] for c in chunks],
                       "done": "".join("0" if c.key in pending_keys else "1" for c in chunks)})
    events.layout(layout)
    todo: list[PlannedChunk] = []
    keys: set[str] = set()
    for c in pending:  # the same sentence may occur twice in a book
        if c.key not in keys:
            keys.add(c.key)
            todo.append(c)

    from folivox.providers import ProviderError
    from folivox.tts.cloud import CostCapReached as CapHit

    prefetch = Prefetcher(engine, voice)
    try:
        clock = time.time()
        for pos, chunk in enumerate(todo):
            if events.cancelled():
                raise Cancelled
            _check_cap(engine, len(chunk.text))
            for nxt in todo[pos: pos + 1 + max(LOOKAHEAD, engine.parallel)]:
                prefetch.submit(nxt.key, nxt.text, _seed(nxt.key, 0), (voices or {}).get(nxt.speaker), nxt.lang)
            events.chunk_started(chunk)
            sc, audio, hyp, attempts = _best_take(prefetch, asr, opts, chunk.key, chunk.text, chunk.source,
                                                  events=events, voice=(voices or {}).get(chunk.speaker),
                                                  fragment_lang=chunk.lang)
            sr = engine.sample_rate
            if not len(audio):
                # an empty take (the model or the trimmer left nothing) would be written as a 0-byte file that
                # never counts as done: re-synthesised (and billed) on every resume, its chapter never assembled
                audio = np.zeros(max(1, int(sr * 0.05)), np.float32)
            dur = len(audio) / sr
            save(journal.audio_path(chunk.key), audio, sr)
            now = time.time()
            # wall time since the previous fragment: with prefetching, stage times overlap
            take = Take(chunk.key, dur, sc if asr is not None else None, hyp, len(attempts), now - clock)
            clock = now
            journal.record(take)
            events.chunk_done(chunk, take, attempts)
            _report_spend(engine, events, opts)
    except CapHit as e:
        raise CostCapReached(str(e)) from None
    except ProviderError:
        if events.cancelled():  # a pause during an outage was interrupted by the user
            raise Cancelled from None
        raise
    finally:
        prefetch.close()


def _cached_sample_rate(journal: Journal) -> int | None:
    """Sample rate of the planned takes on disk. The journal's ``sample_rate`` is that of the last run, which
    may have used another engine (Fish 44.1 kHz after CosyVoice 24 kHz) whose takes are not in this plan."""
    import soundfile as sf

    for c in journal.chunks():
        try:
            return int(sf.info(str(journal.audio_path(c.key))).samplerate)
        except Exception:  # noqa: BLE001
            continue
    return None


def _measured(path: Path, fallback: float) -> float:
    from folivox.audio import audio_seconds

    return audio_seconds(path) or fallback


def _chapter_ok(path: Path, duration: float | None) -> bool:
    """A cached chapter file is reused only if it decodes and has the recorded length."""
    from folivox.audio import audio_seconds

    got = audio_seconds(path) if path.exists() else None
    return got is not None and duration is not None and abs(got - duration) <= max(0.5, duration * 0.02)


def preflight(opts: RenderOptions, events: Events) -> None:
    """Fail early, before hours of synthesis, on problems that would only show at the final step."""
    import shutil

    from folivox.audio.assemble import ffmpeg

    ffmpeg()  # raises FFmpegMissing with install instructions
    for folder in (opts.output.parent, opts.resolved_work_dir()):
        folder = Path(folder).expanduser()
        try:
            folder.mkdir(parents=True, exist_ok=True)
            probe = folder / ".folivox-write-test"
            probe.write_bytes(b"ok")
            probe.unlink()
        except OSError as e:
            raise RuntimeError(f"Cannot write to {folder}: {e.strerror or e}. Pick another output folder") from None
    free = shutil.disk_usage(opts.resolved_work_dir()).free
    if free < 500 * 1024 * 1024:
        events.warning(f"Only {free / 1e6:.0f} MB free on the disk with the work folder; a long book needs several GB")


def device_label(engine) -> str:
    """What the engine really runs on, e.g. «CUDA · NVIDIA RTX 4070», «MPS · Apple GPU», «CPU», «Cloud»."""
    dev = getattr(engine, "device", "") or ""
    if dev == "cloud":
        return "Cloud"
    if dev.startswith("cuda"):
        try:
            import torch

            return f"CUDA · {torch.cuda.get_device_name(0)}"
        except Exception:  # noqa: BLE001
            return "CUDA"
    if dev.startswith("native:"):
        return f"Native C++ (audio.cpp) · {dev[7:].capitalize()}"
    if dev == "mps":
        return "MPS · Apple GPU"
    if dev == "mlx":
        return "MLX · Apple GPU"
    return dev.upper() or "CPU"


def load_engine(opts: RenderOptions, events: Events):
    """Create the local engine; a failed GPU load falls back to the CPU, other failures get a clear message."""
    from folivox.tts import ENGINES, runtime

    if runtime.cosyvoice_native(opts.tts, opts.runtime or None):
        from folivox import config
        from folivox.tts.native import NativeCosyVoiceEngine

        if opts.llm != "rl":
            events.warning("The native runtime has one CosyVoice 3 model; --llm is ignored")
        return NativeCosyVoiceEngine(threads=opts.threads, **config.native_options())
    last = None
    if opts.device == "auto":
        from folivox import gpu

        why = gpu.cuda_skip_reason()
        if why:
            events.warning(f"CUDA is not used: {why}. Running on the CPU (slower); `--device cuda` forces the GPU")
    devices = [opts.device]
    if opts.device not in ("cpu",):
        devices.append("cpu")  # "auto" may pick a broken mps/cuda: the CPU is slow but always works
    for i, device in enumerate(devices):
        try:
            return ENGINES.create(opts.tts, **{**_engine_options(opts), "device": device})
        except (FileNotFoundError, KeyboardInterrupt):
            raise
        except Exception as e:  # noqa: BLE001
            last = e
            if i + 1 < len(devices):
                events.warning(f"Could not load the model on {device} ({e}); retrying on the CPU (slower)")
    raise RuntimeError(f"Could not load the {opts.tts} model: {last}. If the weights are damaged, delete the "
                       "models folder (see `folivox doctor`) and run again; they are downloaded once more") from last


def assemble(journal: Journal, book: Book, opts: RenderOptions, sample_rate: int, events: Events) -> Path:
    work = journal.dir
    built: list[ChapterAudio] = []
    for ci, title in journal.chapters():
        chunks = journal.chunks(ci)
        if not chunks or not all(journal.is_done(c.key) for c in chunks):
            events.warning(f"Chapter “{title}” is not fully rendered — skipping")
            continue
        sig = hashlib.sha1("|".join(f"{c.key}:{c.pause}" for c in chunks).encode()).hexdigest()
        path = work / "chapters" / f"{ci + 1:03d}.flac"
        if journal.meta(f"chapter_sig_{ci}") == sig and _chapter_ok(path, journal.meta(f"chapter_dur_{ci}")):
            dur = journal.meta(f"chapter_dur_{ci}")
        else:
            events.stage("chapter", f"Building chapter {ci + 1}: {title}")
            dur = build_chapter([(journal.audio_path(c.key), c.pause) for c in chunks], path, sample_rate)
            journal.set_meta(**{f"chapter_sig_{ci}": sig, f"chapter_dur_{ci}": dur})
        # chapter marks follow the file as written (older journals stored the pre-loudnorm length)
        dur = _measured(path, dur)
        built.append(ChapterAudio(title, path, dur))
        events.chapter_built(ci, title, dur)
    if not built:
        raise RuntimeError("Nothing to assemble: no chapter is fully rendered")
    m = opts.meta
    meta = BookMeta(m.get("title") or book.title, m.get("author") or book.author, m.get("year") or book.year,
                    m.get("description") or book.annotation, m.get("narrator") or opts.narrator,
                    m.get("series", ""), m.get("series_part", ""), m.get("genre", ""))
    cover = None if opts.no_cover else (opts.cover.read_bytes() if opts.cover else book.cover)
    events.stage("encode", "Encoding {fmt}".format(fmt="MP3" if opts.mp3 else "M4B"))
    if opts.mp3:
        out_dir = opts.output if opts.output.suffix == "" else opts.output.with_suffix("")
        write_mp3s(built, out_dir, meta, cover)
        return out_dir
    write_m4b(built, opts.output, meta, cover, opts.bitrate)
    return opts.output


def _engine_options(opts: RenderOptions) -> dict:
    if opts.tts == "cosyvoice3":
        return {"model_dir": opts.model_dir, "device": opts.device, "threads": opts.threads, "llm": opts.llm}
    if is_fish(opts.tts):
        return {"build": opts.tts_build, "device": opts.device, "runtime": opts.runtime or None}
    return {"device": opts.device}


def is_fish(name: str) -> bool:
    from folivox.tts import builds

    return builds.is_fish_engine(name)


def takes_tags(name: str) -> bool:
    """Engines that read ``[calm]`` style directions in the text: the local Fish builds and Fish Audio online."""
    return is_fish(name) or name == "fish"


def is_cloud(name: str) -> bool:
    from folivox.tts import ENGINES

    return not ENGINES.get(name).local


def _default_voice_name(engine, opts: RenderOptions) -> str:
    """A readable name for a kept voice when the user gave none: never a bare provider id."""
    from folivox import lang as languages

    if opts.tts_mode == "design" and opts.tts_description:
        return opts.tts_description[:40].strip(" ,.")
    base = getattr(engine.gen, "names", {}).get(opts.tts_voice, "")
    language = languages.get(opts.lang).name
    if opts.tts_mode == "localize" and base:
        return f"{base} · {language}"
    if opts.tts_mode == "clone" and opts.voice_ref:
        return f"{Path(opts.voice_ref).parent.name or 'My voice'} (clone)"
    return f"My {language} voice"


def _cloud_voice(engine, opts: RenderOptions, journal: Journal, events: Events):
    """The provider voice for the whole run → ``(voice, owned)``.

    A voice kept from an earlier book (``tts_voice`` = its id or name) is reused as it is and never deleted here.
    A voice made for this book (design / localize / clone) is remembered in the journal so an interrupted run
    continues with the same voice; it is deleted after a finished run unless ``keep_voice`` lists it for later."""
    from folivox.tts import saved

    mine = saved.find(opts.tts, opts.tts_voice)
    if mine is not None:
        voice = engine.make_voice("", opts.tts_style, opts.lang, mine["mode"], mine.get("description", ""), None,
                                  mine["id"])
        events.stage("voice", f"Using your saved voice “{mine['name']}”")
        return voice, False
    kind = f"{opts.tts}|{opts.tts_mode}|{opts.tts_voice}|{opts.tts_description}|{opts.voice_ref}"
    remembered = journal.meta("cloud_voice") if journal.meta("cloud_voice_for") == kind else ""
    voice = engine.make_voice(opts.tts_voice, opts.tts_style, opts.lang, opts.tts_mode, opts.tts_description,
                              opts.voice_ref, remembered or "")
    if opts.tts_mode == "preset":
        if not voice.speaker and opts.tts in ("elevenlabs", "cartesia", "fish"):
            raise RuntimeError(f"Pick a voice for {opts.tts} (--tts-voice NAME or id; `folivox tts-voices {opts.tts}`)")
        return voice, False
    journal.set_meta(cloud_voice=voice.speaker, cloud_voice_for=kind)
    if not remembered:
        events.stage("voice", f"Created a voice in your {type(engine.gen).label} account for this book")
    if opts.keep_voice:
        name = opts.tts_voice_name or _default_voice_name(engine, opts)
        saved.add(opts.tts, voice.speaker, name, opts.tts_mode, opts.tts_description, opts.lang)
        events.stage("voice", f"Voice “{name}” is kept in your account and listed for other books")
    return voice, True


SPEAKER_VOICE_KINDS = {"fish:": "preset", "fish-design:": "design", "fish-clone:": "clone"}


def cloud_speaker_voices(markup, engine, opts: RenderOptions, journal: Journal, events: Events):
    """``({speaker id: CloudVoice}, [(speaker, voice)])`` for an online engine (Fish Audio only).

    A speaker's voice is written ``fish:<voice id>`` (library or own voice), ``fish-design:<description>`` or
    ``fish-clone:<catalog voice or recording>``; the last two create a voice in the account, remembered in the
    journal so an interrupted run goes on with the same one. The second list holds the voices made for this run."""
    out: dict = {}
    made: list = []
    if markup is None:
        return out, made
    used = sorted(markup.used_speakers() - {0})
    if used and opts.tts != "fish":
        raise RuntimeError("Several speakers need a local engine or Fish Audio; this online engine reads the whole "
                           "book in one voice")
    from folivox import voices as catalog

    memory = dict(journal.meta("cloud_speaker_voices") or {})
    for sid in used:
        sp = markup.speaker(sid)
        ref = (sp.voice if sp else "").strip()
        prefix = next((p for p in SPEAKER_VOICE_KINDS if ref.startswith(p)), None)
        if not ref:
            continue
        if prefix is None:
            events.warning(f"{sp.name}: the voice {ref} is a local recording and cannot be used with Fish Audio; "
                           "the narrator reads this character")
            continue
        value, mode = ref[len(prefix):].strip(), SPEAKER_VOICE_KINDS[prefix]
        if not value:
            continue
        if mode == "preset":
            out[sid] = engine.make_voice(value, opts.tts_style, opts.lang, "preset")
            continue
        reference, description = None, ""
        if mode == "clone":
            entry = catalog.find(value)
            reference = str(entry.path if entry is not None else Path(value).expanduser())
        else:
            description = value
        kind = f"{mode}|{value}|{opts.lang}"
        entry = memory.get(str(sid)) or {}
        remembered = entry.get("id", "") if entry.get("for") == kind else ""
        voice = engine.make_voice("", opts.tts_style, opts.lang, mode, description, reference, remembered)
        memory[str(sid)] = {"for": kind, "id": voice.speaker}
        journal.set_meta(cloud_speaker_voices=memory)
        if not remembered:
            events.stage("voice", f"Created a voice for {sp.name} in your {type(engine.gen).label} account")
        out[sid] = voice
        made.append((sp, voice))
    return out, made


def _release_speaker_voices(engine, made: list, opts: RenderOptions, journal: Journal, events: Events) -> None:
    from folivox.tts import saved

    for sp, voice in made:
        if opts.keep_voice:
            saved.add(opts.tts, voice.speaker, f"{sp.name} ({opts.tts_voice_name or 'book'})".strip(), voice.mode,
                      voice.description, opts.lang)
            events.stage("voice", f"Voice for {sp.name} is kept in your account and listed for other books")
        else:
            engine.release(voice)
    journal.set_meta(cloud_speaker_voices={})


def render(book: Book, opts: RenderOptions, events: Events | None = None) -> Path:
    from folivox.qa.asr import ASR_PROVIDERS
    from folivox.tts import ENGINES
    from folivox.tts.voice import load_voice

    events = events or Events()
    from folivox import lang as languages

    opts.lang = languages.get(opts.lang or book.lang or "ru").code
    preflight(opts, events)
    journal = Journal(opts.resolved_work_dir())
    voice, owned, engine = None, False, None
    try:
        bad = journal.audit()
        if bad:
            events.warning(f"{len(bad)} saved fragment(s) were damaged and will be synthesised again")
        cloud = is_cloud(opts.tts)
        from folivox.tts import runtime as tts_runtime

        if tts_runtime.cosyvoice_native(opts.tts, opts.runtime or None):
            tts_runtime.require_ready()
        if is_fish(opts.tts):
            from folivox.tts import fish_check

            fish_check.check(opts, events)
        if cloud and not opts.accept_cloud:
            from folivox.providers import ProviderError

            raise ProviderError(f"{opts.tts} is a paid cloud engine: the book text is sent to the provider and billed "
                                "to your account. Confirm the cost estimate first (--yes in the CLI, the checkbox in "
                                "the studio)")
        asr = None
        if opts.qa:
            events.stage("asr", f"Loading speech recognition for QA ({opts.asr})")
            try:
                asr = ASR_PROVIDERS.create(opts.asr, language=opts.lang,
                                           **({"device": "cpu"} if opts.device == "cpu" and opts.asr == "auto" else {}))
            except Exception as e:  # noqa: BLE001 — QA is optional; do not block the whole render
                events.warning(f"Speech recognition could not be loaded ({e}); rendering without the quality check")
                asr = None
        if cloud:
            if opts.speed != 1.0:
                events.warning("Reading speed is not applied by cloud engines; the provider's own pace is used")
            engine = ENGINES.create(opts.tts, model=opts.tts_model, max_cost=opts.max_cost)
            engine.should_stop = events.cancelled
            engine.on_wait = events.warning
            voice, owned = _cloud_voice(engine, opts, journal, events)
            for w in voice.warnings_for(opts.tts):
                events.warning(w)
            events.stage("model", f"{type(engine.gen).label}, model {engine.model}")
        else:
            voice = load_voice(opts.voice_wav, opts.voice_text,
                               transcribe=asr.transcribe_file if asr else None)
            for w in voice.warnings_for(opts.tts):
                events.warning(w)
        markup = load_markup(book, opts, events)
        made: list = []
        if cloud:
            voices, made = cloud_speaker_voices(markup, engine, opts, journal, events)
        else:
            voices = speaker_voices(markup, voice, asr, opts, events)
        if voices and not cloud and is_fish(opts.tts):
            from folivox.tts import fish_check

            if not fish_check.bundled_voices_allowed():
                for sid, v in voices.items():
                    if fish_check.is_bundled_voice(v.wav):
                        from folivox.providers import ProviderError

                        raise ProviderError(f"Speaker {sid}: " + fish_check.BUNDLED_VOICES_NOTE)
        plan(book, opts, journal, voice.id, events, engine, markup, voices)
        if not cloud:
            for note in voice_accent_notes(journal, markup, opts):
                events.warning(note)
        if journal.pending():
            from folivox.tts import weights

            if engine is not None:
                sample_rate = engine.sample_rate
            else:
                if opts.tts == "cosyvoice3" and not tts_runtime.cosyvoice_native(opts.tts, opts.runtime or None) \
                        and weights.needs_download(opts.model_dir, opts.llm):
                    events.stage("download", "Preparing the model download (first run)")
                else:
                    events.stage("model", f"Loading TTS model into memory ({opts.tts})")
                t0 = time.time()

                with weights.reporting(events.download):
                    engine = load_engine(opts, events)
                start = getattr(engine, "start", None)
                if start is not None:  # Fish: load the worker's model and warm it up with the book's voice
                    events.stage("model", f"Warming up {engine.build.title} (the first call takes a while)")
                    start(voice)
                events.stage("model", f"Model loaded in {time.time() - t0:.0f} s ({engine.device})")
                sample_rate = engine.sample_rate
            events.stage("device", f"Device: {device_label(engine)}")
            synthesize_pending(journal, engine, voice, asr, opts, events, voices)
        else:
            sample_rate = _cached_sample_rate(journal) or journal.meta("sample_rate", 24000)
        journal.set_meta(sample_rate=sample_rate)
        result = assemble(journal, book, opts, sample_rate, events)
        if cloud and owned and not opts.keep_voice:
            engine.release(voice)  # only after a finished run: an interrupted one resumes with the same voice
            journal.set_meta(cloud_voice="", cloud_voice_for="")
        if cloud and made:
            _release_speaker_voices(engine, made, opts, journal, events)
        return result
    finally:
        closer = getattr(engine, "close", None)
        if closer is not None:
            closer()  # a Fish worker holds tens of GB: stop it when the render stops
        journal.close()

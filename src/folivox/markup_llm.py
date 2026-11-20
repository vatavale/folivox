"""Automatic markup with a text LLM: a character bible (who is in the book) and per-paragraph speaker marking.

Two passes, both resumable and cached:

1. **Analyse** reads the whole book once, window by window, and builds the *character bible*: names with
   aliases, gender, role, age, speech traits, relations and how many lines each speaks. The bible tells how many
   distinct voices the book needs, before any voice is chosen.
2. **Annotate** walks the paragraphs that contain speech and asks the model to mark who speaks each part (and,
   if asked, where an expression tag belongs). The model sees the bible entries relevant to the window and a short
   context of the previous paragraphs, so "he said" and pronouns resolve across chapters; it may add or correct
   bible entries as it reads.

The model only *annotates*. It is asked to repeat each paragraph with ``<|speaker:N|>`` / ``[tag]`` inserted; the
reply is accepted only if removing the annotations gives exactly the book's paragraph
(:func:`folivox.markup.parse_block`). Anything else, an unknown speaker or tag, a failed call: the narrator reads
that paragraph and a warning is counted. Offsets are always computed here, never taken from the model.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
import unicodedata
import urllib.error
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from folivox import markup as mk
from folivox.lang import has_speech
from folivox.lang.attribution import NARRATOR_NAMES, PRONOUNS, RU_SAID

PROMPT_VERSION = "2"
WINDOW_CHARS = 5000
MIN_LINES_FOR_A_VOICE = 3  # a character who speaks less than this is not counted as needing a voice of their own
# With more than 15 speakers (the markup's ids are 1..15) the rarest share two ids, named so in the markup
SHARED_NAMES = {14: "Other men", 15: "Other women"}
MAX_RELEVANT = 25
# a paragraph that opens with speech (not one that merely quotes a sign)
DIALOGUE_START = re.compile(r"^\s*[—–―\"«„“”'‘「『]|^\s*-\s")
SPEECH_MARKS = re.compile(r"[«»„“”\"‘’「」『』]|^\s*[—–―]|^\s*-\s|\bthink|\bthought", re.I | re.M)

Progress = Callable[[str, int, int, str], None]  # stage, done, total, message


class LLMError(RuntimeError):
    pass


class ShortReply(ValueError):
    """A usable reply that leaves out paragraphs with speech (a model that answers the first paragraphs and stops).
    It is asked again like any bad reply; when every try stays short, the most complete one is kept."""

    def __init__(self, reply: dict, missing: list[int]):
        super().__init__(f"the reply left out {len(missing)} paragraph(s) with speech")
        self.reply, self.missing = reply, missing

    def retry_note(self) -> str:
        shown = ", ".join(f"B{b}" for b in self.missing[:12]) + (", …" if len(self.missing) > 12 else "")
        return (f"\n\nYour previous answer left out paragraphs that contain speech ({shown}). Answer for every "
                "paragraph, with the JSON object only.")


class CostCapReached(LLMError):
    """The next model call would pass the run's spending cap; what is done is saved and the run can be resumed."""


# -- character bible ------------------------------------------------------------------------------------------

def norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.casefold().replace("\u0451", "\u0435"))  # yo -> ye
    return "".join(c for c in s if c.isalnum() or c == " ").strip()


def same_name(a: str, b: str) -> bool:
    """Equal names, or the same name in another case/declension (Anna/Anny/Annu, Ivan/Ivana)."""
    a, b = norm_name(a), norm_name(b)
    if not a or not b:
        return False
    if a == b:
        return True
    lo = min(len(a), len(b))
    if lo < 3 or " " in a or " " in b or abs(len(a) - len(b)) > 3:
        return False
    common = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        common += 1
    return common >= max(3, lo - 1)


@dataclass
class Character:
    name: str
    aliases: list[str] = field(default_factory=list)
    gender: str = ""  # f | m | n (neutral/group) | "" unknown
    role: str = "minor"  # main | minor
    age: str = ""
    traits: str = ""  # how the character speaks
    relations: list[dict] = field(default_factory=list)  # [{"with": name, "kind": "mother"}]
    lines: int = 0  # lines of direct speech seen so far
    first: int = -1  # chapter where the character first appears
    unnamed: bool = False
    id: int = 0  # speaker id in the markup (0 = not assigned)
    voice_prompts: dict = field(default_factory=dict)  # voice-design descriptions: simple / standard / detailed
    shared: bool = False  # the id is one of the two shared by the rarest speakers (SHARED_NAMES), not their own
    votes: dict = field(default_factory=dict)  # {"gender": {"f": 3, "m": 1}, "age": {...}}: what each window said

    def vote(self, what: str, value: str, weight: int = 1) -> None:
        """Count what a window (or a closer look) said about the gender or age; the answer is the majority, so one
        wrong first impression does not stick. A person's own correction is a heavy vote."""
        if not value or value == "?":
            return
        v = self.votes.setdefault(what, {})
        if not v and getattr(self, what):  # a book analysed before votes existed: its answer counts once
            v[getattr(self, what)] = 1
        v[value] = v.get(value, 0) + weight
        best = max(v.values())
        if v.get(getattr(self, what), 0) < best:  # keep the current answer on a tie
            setattr(self, what, next(k for k, n in v.items() if n == best))

    def unsure(self) -> list[str]:
        """Which of gender and age are still a guess: unknown, a single vote, or nearly tied."""
        out = []
        for what in ("gender", "age"):
            v = sorted(self.votes.get(what, {}).values(), reverse=True)
            known = getattr(self, what)
            if not known or sum(v) < 2 or (len(v) > 1 and v[0] - v[1] < 2):
                out.append(what)
        return out

    def names(self) -> list[str]:
        return [self.name, *self.aliases]

    def is_narrator(self) -> bool:
        """Listed under the name "Narrator": the narrator's own voice (speaker 0) reads them, not a speaker id."""
        return norm_name(self.name) in {norm_name(n) for n in NARRATOR_NAMES}

    def matches(self, name: str) -> bool:
        return any(norm_name(n) == norm_name(name) for n in self.names())

    def matches_loosely(self, name: str, gender: str = "") -> bool:
        """Same name in another case form; never across two known, different genders (Mark / Mara)."""
        if self.gender and gender and self.gender != gender:
            return False
        return any(same_name(n, name) for n in self.names())


def other_person(c: Character, name: str) -> bool:
    """``name`` is a full name (several words) that is none of ``c``'s and fits none of its full names: Robert Prentiss
    is not Robert Cohn, though both are called "Robert". One name inside the other (Elizabeth Bennet, Miss Elizabeth
    Bennet) still fits, and a short name (Darcy) never makes two people."""
    words = set(norm_name(name).split())
    fulls = [set(norm_name(n).split()) for n in c.names() if len(norm_name(n).split()) > 1]
    return (len(words) > 1 and bool(fulls) and not c.matches(name)
            and not any(words <= f or f <= words for f in fulls))


@dataclass
class Bible:
    lang: str = ""
    narration: str = ""  # first | third | mixed | ""
    characters: list[Character] = field(default_factory=list)
    model: str = ""
    done_windows: int = 0
    total_windows: int = 0
    sampled: bool = False  # only a spread sample of the book was read (a preview), not every window

    def find(self, name: str, gender: str = "") -> Character | None:
        name = (name or "").strip()
        if not name:
            return None
        return (next((c for c in self.characters if c.matches(name)), None)
                or next((c for c in self.characters if c.matches_loosely(name, gender)), None))

    def merge(self, entry: dict, chapter: int = -1) -> Character | None:
        """Add what the model reported about one character; the same person under another name is recognised."""
        name = str(entry.get("name") or "").strip()
        if not name or len(name) > 60:
            return None
        aliases = [str(a).strip() for a in entry.get("aliases") or [] if str(a).strip() and len(str(a)) <= 60]
        gender = str(entry.get("gender") or "").strip().lower()[:1]
        gender = gender if gender in ("f", "m", "n") else ""
        # an alias finds the character only if the full names agree: a first name two people share is no proof
        c = self.find(name, gender) or next((x for x in (self.find(a, gender) for a in aliases)
                                             if x and not other_person(x, name)), None)
        if c is None:
            c = Character(name=name, first=chapter)
            self.characters.append(c)
        for a in [name, *aliases]:
            if not c.matches(a):
                c.aliases.append(a)
        c.vote("gender", gender)
        if str(entry.get("role") or "") == "main":
            c.role = "main"
        c.vote("age", str(entry.get("age") or "").strip()[:30])
        traits = str(entry.get("traits") or "").strip()[:200]
        if traits and traits.lower() not in c.traits.lower():
            c.traits = (c.traits + "; " + traits).strip("; ")[:300]
        for r in entry.get("relations") or []:
            if isinstance(r, dict) and r.get("with") and r not in c.relations and len(c.relations) < 12:
                c.relations.append({"with": str(r["with"])[:60], "kind": str(r.get("kind", ""))[:30]})
        try:
            c.lines += max(0, int(entry.get("lines") or 0))
        except (TypeError, ValueError):
            pass
        c.unnamed = c.unnamed or bool(entry.get("unnamed"))
        if c.first < 0:
            c.first = chapter
        return c

    def speakers(self) -> list[Character]:
        return sorted((c for c in self.characters if c.lines > 0), key=lambda c: -c.lines)

    def voices_needed(self, min_lines: int = MIN_LINES_FOR_A_VOICE) -> int:
        """Narrator plus every character who speaks at least ``min_lines`` lines."""
        return 1 + sum(1 for c in self.characters if c.lines >= min_lines and not c.is_narrator())

    def assign_ids(self) -> None:
        """Stable speaker ids 1..15 (the markup's range): keep existing ones, give the next free id to new speakers
        by lines spoken. With more than 15, the rarest share the ids 14 (men / unknown) and 15 (women) and are
        marked ``shared``. A shared id is never one that a single character already holds: a speaker who would
        land there keeps no id (the narrator reads them) and is listed by :meth:`unassigned`. A character listed as
        "Narrator" gets no id: speaker 0 is the narrator."""
        holders: dict[int, list[Character]] = {}
        for c in self.characters:
            if c.id:
                holders.setdefault(c.id, []).append(c)
        for group in holders.values():  # a list saved before ``shared`` existed: an id held by several is shared
            if len(group) > 1:
                for c in group:
                    c.shared = True
        overflow = sum(1 for c in self.characters if c.lines > 0 and not c.is_narrator()) > mk.MAX_SPEAKERS - 1
        for c in sorted(self.characters, key=lambda c: -c.lines):
            if c.id or c.lines <= 0 or c.is_narrator():
                continue
            free = next((i for i in range(1, mk.MAX_SPEAKERS)
                         if i not in holders and not (overflow and i in SHARED_NAMES)), None)
            if free is not None:
                c.id, c.shared = free, False
            else:
                pool = 15 if c.gender == "f" else 14
                if not all(x.shared for x in holders.get(pool, [])):
                    continue  # one character's own id: better the narrator than that character's voice
                c.id, c.shared = pool, True
            holders.setdefault(c.id, []).append(c)

    def unassigned(self) -> list[Character]:
        """Speakers left without an id because all 15 are taken (the narrator reads them)."""
        return sorted((c for c in self.characters if c.lines > 0 and not c.id and not c.is_narrator()),
                      key=lambda c: -c.lines)

    def to_dict(self) -> dict:
        return {"version": 1, "lang": self.lang, "narration": self.narration, "model": self.model,
                "done_windows": self.done_windows, "total_windows": self.total_windows,
                "sampled": self.sampled, "voices_needed": self.voices_needed(),
                "characters": [{**c.__dict__, "unsure": c.unsure()} for c in self.characters]}

    @classmethod
    def from_dict(cls, d: dict) -> Bible:
        b = cls(str(d.get("lang", "")), str(d.get("narration", "")), model=str(d.get("model", "")),
                done_windows=int(d.get("done_windows", 0)), total_windows=int(d.get("total_windows", 0)),
                sampled=bool(d.get("sampled", False)))
        for c in d.get("characters", []):
            known = {k: c[k] for k in Character.__dataclass_fields__ if k in c}
            if known.get("name"):
                b.characters.append(Character(**known))
        return b

    def save(self, path: Path) -> None:
        tmp = Path(path).with_suffix(".tmp")
        tmp.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> Bible:
        try:
            return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError, AttributeError) as e:
            raise LLMError(f"Cannot read the character list {path}: {e}") from None


def bible_path(book_path: Path) -> Path:
    p = Path(book_path)
    return p.with_name(p.stem + ".characters.json")


def cache_path(book_path: Path) -> Path:
    p = Path(book_path)
    return p.with_name(p.stem + ".llm-cache.sqlite")


# -- cache and calls -------------------------------------------------------------------------------------------

class Cache:
    """Model replies by hash of (model, prompt version, prompts): an interrupted run continues where it stopped
    and an unchanged window is never asked twice."""

    def __init__(self, path: Path | None):
        self.db = sqlite3.connect(str(path) if path else ":memory:")
        self.db.execute("CREATE TABLE IF NOT EXISTS replies (key TEXT PRIMARY KEY, reply TEXT NOT NULL)")

    def get(self, key: str) -> str | None:
        row = self.db.execute("SELECT reply FROM replies WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def put(self, key: str, reply: str) -> None:
        self.db.execute("INSERT OR REPLACE INTO replies VALUES (?, ?)", (key, reply))
        self.db.commit()

    def close(self) -> None:
        self.db.close()


def extract_json(text: str) -> dict:
    """The JSON object in a model reply (tolerates code fences, prose around it and trailing commas)."""
    text = re.sub(r"```(?:json)?", "", text)
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b <= a:
        raise ValueError("no JSON object in the reply")
    raw = text[a:b + 1]
    try:
        out = json.loads(raw)
    except ValueError:
        out = json.loads(re.sub(r",\s*([}\]])", r"\1", raw))
    if not isinstance(out, dict):
        raise ValueError("the reply is not a JSON object")
    return out


class Runner:
    """Calls the model with caching, JSON extraction and a few retries (a warmer second try)."""

    BUSY_TRIES = 5  # attempts in all when the server is busy (429, 5xx) or the connection drops
    BUSY_WAIT_CAP = 60.0  # seconds, the longest single wait

    def __init__(self, llm, model: str, cache: Cache, tries: int = 2,
                 cancelled: Callable[[], bool] = lambda: False, sleep: Callable[[float], None] = time.sleep,
                 budget=None):
        self.llm, self.model, self.cache, self.tries = llm, model, cache, tries
        self.cancelled, self.sleep = cancelled, sleep
        self.budget = budget  # folivox.markup_cost.Budget: counts what the calls cost and stops at a spending cap
        self.calls = self.cached = self.failures = 0

    def _wait(self, seconds: float) -> None:
        """Sleep in short steps so a cancel is noticed."""
        while seconds > 0 and not self.cancelled():
            step = min(1.0, seconds)
            self.sleep(step)
            seconds -= step

    def _complete(self, prompt: str, system: str, temperature: float) -> str:
        """One model reply; waits out a busy server, fails at once on a wrong key, model name or URL.

        A request the server refuses for its content (400, 413, 422: too long a window, say) raises ValueError,
        so the caller's retries and fallbacks handle it like a bad reply."""
        for attempt in range(self.BUSY_TRIES):
            try:
                return self.llm.complete_json(prompt, system=system, temperature=temperature)
            except urllib.error.HTTPError as e:
                code = e.code
                if code in (401, 403):
                    raise LLMError(f"The language model server refused the request (HTTP {code}): "
                                   "check the API key.") from None
                if code == 404:
                    raise LLMError("The language model server answered HTTP 404: check the model name and "
                                   "the server address.") from None
                if code in (400, 413, 422):
                    raise ValueError(f"the server refused this request (HTTP {code})") from None
                if code != 429 and code < 500:
                    raise LLMError(f"The language model did not answer: HTTP {code}") from None
                retry_after = e.headers.get("Retry-After") if e.headers is not None else None
                last: Exception = LLMError(f"The language model did not answer: HTTP {code}")
            except (ConnectionError, TimeoutError, urllib.error.URLError) as e:
                retry_after = None
                last = LLMError(f"The language model did not answer: {e}")
            except OSError as e:
                raise LLMError(f"The language model did not answer: {e}") from None
            if attempt == self.BUSY_TRIES - 1 or self.cancelled():
                break
            try:
                delay = float(retry_after) if retry_after else 2.0 * 2 ** attempt
            except ValueError:  # an HTTP date: not worth parsing
                delay = 2.0 * 2 ** attempt
            self._wait(max(0.0, min(delay, self.BUSY_WAIT_CAP)))
            if self.cancelled():
                break
        raise last

    def ask(self, system: str, prompt: str, validate: Callable[[dict], dict]) -> dict:
        """Reply parsed and checked by ``validate`` (which raises ValueError on a bad one)."""
        key = hashlib.sha1("\x1f".join((PROMPT_VERSION, self.model, system, prompt)).encode()).hexdigest()
        hit = self.cache.get(key)
        if hit is not None:
            try:
                out = validate(extract_json(hit))
                self.cached += 1
                return out
            except ValueError:
                pass
        last: Exception | None = None
        best: ShortReply | None = None  # the most complete of the short replies, never cached
        for attempt in range(self.tries):
            self.calls += 1
            extra = ("" if attempt == 0 else last.retry_note() if isinstance(last, ShortReply)
                     else "\n\nYour previous answer was not valid. Answer with the JSON object only.")
            sent = len(system) + len(prompt) + len(extra)
            if self.budget is not None:
                self.budget.check(sent)  # raises CostCapReached before a call that would pass the cap
            try:
                self.llm.last_usage = None  # a reply that reports no usage must not inherit the last one's
            except AttributeError:
                pass
            try:
                reply = self._complete(prompt + extra, system, 0.1 + 0.2 * attempt)
            except (ValueError, KeyError, TypeError, AttributeError) as e:
                # a reply the provider sent and the client rejected (a refusal, an answer cut off at the token
                # limit) is billed too: count the usage it reported before the retry is checked against the cap
                usage = getattr(self.llm, "last_usage", None)
                if self.budget is not None and usage:
                    self.budget.record(usage, sent, 0)
                last = e
                continue
            if self.budget is not None:  # a reply is paid for even when it is unusable
                self.budget.record(getattr(self.llm, "last_usage", None), sent, len(reply))
            try:
                out = validate(extract_json(reply))
            except ShortReply as e:
                last = e
                best = e if best is None or len(e.missing) < len(best.missing) else best
                continue
            except (ValueError, KeyError, TypeError, AttributeError) as e:
                last = e
                continue
            self.cache.put(key, reply)
            return out
        self.failures += 1
        if best is not None:
            raise best
        raise ValueError(f"no usable answer after {self.tries} tries: {last}")


# -- windows --------------------------------------------------------------------------------------------------

@dataclass
class Window:
    chapter: int
    blocks: list[tuple[int, str]]  # (block index, text)

    @property
    def chars(self) -> int:
        return sum(len(t) for _, t in self.blocks)

    def text(self) -> str:
        return "\n".join(f"[B{b}] {t}" for b, t in self.blocks)


_LOOSE_SPEAKER = re.compile(r"<\|?\s*speaker\s*:\s*(\d{1,2})\s*\|?>")


def _fix_speaker_tags(line: str) -> str:
    """Small models often drop a bar (``<|speaker:1>``); read those as the tag they meant."""
    return _LOOSE_SPEAKER.sub(r"<|speaker:\1|>", line)


def windows(chapters: list, max_chars: int = WINDOW_CHARS, only_speech: bool = False) -> list[Window]:
    """Consecutive paragraphs of a chapter, up to ``max_chars`` (a paragraph is never cut)."""
    out: list[Window] = []
    for ci, ch in enumerate(chapters):
        cur: list[tuple[int, str]] = []
        size = 0
        for bi, b in enumerate(ch.blocks):
            t = b.text.strip()
            if not has_speech(t) or "\n" in t or (only_speech and not SPEECH_MARKS.search(t)):
                continue
            if cur and size + len(t) > max_chars:
                out.append(Window(ci, cur))
                cur, size = [], 0
            cur.append((bi, t))
            size += len(t)
        if cur:
            out.append(Window(ci, cur))
    return out


LANG_NAMES = {"ru": "Russian", "en": "English", "de": "German", "fr": "French", "es": "Spanish", "it": "Italian",
              "zh": "Chinese", "ja": "Japanese", "ko": "Korean"}

QUOTE_HINTS = ("Direct speech may be written with dialogue dashes (— …), guillemets («…»), curly or straight "
               "quotes (“…”, \"…\"), corner brackets (「…」『…』), or with no marks at all in some translations. "
               "Inner thoughts and letters read aloud also belong to a character, not to the narrator.")

# -- analyse ---------------------------------------------------------------------------------------------------

ANALYSE_SYSTEM = ("You are a careful literary analyst. You prepare books for multi-voice audiobooks. You answer "
                  "with one JSON object and nothing else. You never invent facts that are not in the text.")

ANALYSE_PROMPT = """Book language: {lang}.
{hints}

Characters already known (reuse the exact "name" when it is the same person, also under a nickname, a
patronymic, a title or another grammatical form; put new forms into "aliases"):
{known}

Read the excerpt (paragraphs are numbered [B..]). List every character who SPEAKS (direct speech), THINKS
(inner monologue) or is clearly important in it. Do not list places, objects, groups of people that never speak,
or the narrator. A speaker with no name gets a short descriptive name in the book's language ("old guard") and
"unnamed": true. Keep names exactly as spelled in the book (nominative case).

Reply with this JSON:
{{"narration_person": "first" | "third" | "mixed",
 "characters": [{{"name": str, "aliases": [str], "gender": "f" | "m" | "n" | "?", "role": "main" | "minor",
   "age": "child" | "young" | "adult" | "old" | "?", "traits": "how this character speaks, 6 words at most",
   "relations": [{{"with": str, "kind": str}}], "lines": <number of their direct-speech lines in this excerpt>,
   "unnamed": bool}}]}}

Excerpt:
{text}"""


def _known(bible: Bible, relevant: list[Character]) -> str:
    rows = [{"name": c.name, "aliases": c.aliases[:6], "gender": c.gender or "?", "age": c.age or "?",
             "role": c.role, "speaks": c.traits[:50]} for c in relevant]
    return json.dumps(rows, ensure_ascii=False) if rows else "(none yet)"


def relevant_for(bible: Bible, window_text: str, recent: list[str]) -> list[Character]:
    """Characters named in the window or active in the previous ones, plus the busiest, at most ``MAX_RELEVANT``."""
    low = norm_name(window_text)
    named = [c for c in bible.characters if any(norm_name(n) and norm_name(n)[: max(3, len(norm_name(n)) - 1)] in low
                                                for n in c.names())]
    active = [c for c in bible.characters if c.name in recent]
    busiest = sorted(bible.characters, key=lambda c: -c.lines)[:5]
    seen, out = set(), []
    for c in [*named, *active, *busiest]:
        if id(c) not in seen:
            seen.add(id(c))
            out.append(c)
    return out[:MAX_RELEVANT]


def analyse(chapters: list, lang: str, runner: Runner, bible: Bible | None = None, *, max_windows: int | None = None,
            save: Callable[[Bible], None] | None = None, progress: Progress | None = None,
            cancelled: Callable[[], bool] = lambda: False, window_chars: int = WINDOW_CHARS) -> Bible:
    """Read the book once and fill the character bible. Resumable: windows done before are skipped."""
    bible = bible or Bible()
    bible.lang, bible.model = lang, runner.model
    wins = windows(chapters, window_chars, only_speech=False)
    full_done = not bible.sampled and 0 < bible.total_windows <= bible.done_windows
    # a preview after a completed full reading has nothing to add: the full reading stays
    sampled = bool(max_windows and len(wins) > max_windows and not full_done)
    if sampled:  # spread a sample over the whole book
        step = len(wins) / max_windows
        wins = [wins[int(i * step)] for i in range(max_windows)]
    if bible.sampled != sampled:  # a full reading after a preview, or a preview after a full one, starts over
        bible.done_windows = bible.total_windows = 0  # (the characters are kept); the window lists differ
    bible.sampled = sampled
    bible.total_windows = len(wins)
    recent: list[str] = []
    narr: dict[str, int] = {}
    for n, w in enumerate(wins):
        if cancelled():
            break
        if n < bible.done_windows:
            continue
        text = w.text()
        prompt = ANALYSE_PROMPT.format(lang=LANG_NAMES.get(lang, lang), hints=QUOTE_HINTS,
                                       known=_known(bible, relevant_for(bible, text, recent)), text=text)

        def check(d: dict) -> dict:
            if not isinstance(d.get("characters"), list):
                raise ValueError("no characters list")
            return d

        try:
            out = runner.ask(ANALYSE_SYSTEM, prompt, check)
        except ValueError as e:
            if progress:
                progress("analyse", n + 1, len(wins), f"window {n + 1} skipped: {e}")
            bible.done_windows = n + 1
            if save:
                save(bible)
            continue
        narr[str(out.get("narration_person", ""))] = narr.get(str(out.get("narration_person", "")), 0) + 1
        recent = []
        for e in out["characters"]:
            if isinstance(e, dict):
                c = bible.merge(e, w.chapter)
                if c:
                    recent.append(c.name)
        bible.done_windows = n + 1
        if narr:
            top = max(narr, key=narr.get)
            bible.narration = top if top in ("first", "third", "mixed") else bible.narration
        if save:
            save(bible)
        if progress:
            progress("analyse", n + 1, len(wins), f"{len(bible.characters)} characters")
    bible.assign_ids()
    if save:
        save(bible)
    return bible


# -- consolidate: a second look at the characters once the whole book has been read -------------------------------

MERGE_PROMPT = """Book language: {lang}.
These are the characters found in a book, with the names and nicknames each was met under:
{rows}

Some may be the same person under different names (a nickname, a patronymic, a title, a diminutive). List the
groups of names that are one person. Do not join two different people, even relatives with the same surname.

Reply with this JSON: {{"same": [["name", "name"], ...]}}  (an empty list when there are none)"""

LOOK_PROMPT = """Book language: {lang}.
Character: "{name}" (also: {aliases}). Known so far: gender {gender}, age {age}, speaks: {traits}.

Passages from different places of the book that mention or quote this character:
{snippets}

Decide from the passages (pronouns, endings of verbs and adjectives, forms of address, who is addressed as a
child, a parent, a grandfather and so on). Use "?" where the passages really do not tell.

Reply with this JSON: {{"gender": "f" | "m" | "n" | "?", "age": "child" | "young" | "adult" | "old" | "?",
 "traits": "how this character speaks, 6 words at most"}}"""

CONSOLIDATE_SYSTEM = ("You know the whole book well and read it with care. "
                      "You answer with one JSON object and nothing else.")


def _same_list(d: dict) -> dict:
    if not isinstance(d.get("same"), list):
        raise ValueError("no list")
    return d


def _full(names: list[str]) -> bool:
    return any(len(norm_name(n).split()) > 1 for n in names)


def merge_characters(bible: Bible, keep: Character, other: Character) -> None:
    """Fold ``other`` into ``keep`` (the same person met under two names)."""
    for a in other.names():
        if not keep.matches(a):
            keep.aliases.append(a)
    for what in ("gender", "age"):
        for value, n in other.votes.get(what, {}).items():
            keep.vote(what, value, n)
        if not keep.votes.get(what) and getattr(other, what):
            keep.vote(what, getattr(other, what))
    keep.lines += other.lines
    keep.role = "main" if "main" in (keep.role, other.role) else keep.role
    if other.traits and other.traits.lower() not in keep.traits.lower():
        keep.traits = (keep.traits + "; " + other.traits).strip("; ")[:300]
    keep.relations += [r for r in other.relations if r not in keep.relations][:12 - len(keep.relations)]
    keep.first = min([x for x in (keep.first, other.first) if x >= 0], default=-1)
    if not keep.id:
        keep.id, keep.shared = other.id, other.shared
    bible.characters = [c for c in bible.characters if c is not other]


def _snippets(chapters: list, c: Character, n: int = 8, width: int = 320) -> list[str]:
    """Up to ``n`` paragraphs that name the character, spread over the whole book."""
    keys = [norm_name(x) for x in c.names() if len(norm_name(x)) >= 3]
    hits = []
    for ci, ch in enumerate(chapters):
        for b in ch.blocks:
            low = norm_name(b.text)
            if any(k[: max(3, len(k) - 1)] in low for k in keys):
                hits.append((ci, b.text))
    if len(hits) > n:
        step = len(hits) / n
        hits = [hits[int(i * step)] for i in range(n)]
    return [f"(chapter {ci + 1}) " + t[:width] for ci, t in hits]


def consolidate(chapters: list, lang: str, bible: Bible, runner: Runner, *, max_look: int = 25,
                save: Callable[[Bible], None] | None = None, progress: Progress | None = None,
                cancelled: Callable[[], bool] = lambda: False) -> Bible:
    """After the whole book is read: join the names that are one person, then look again, with passages from all
    over the book, at every character whose gender or age is still a guess."""
    names = [c for c in bible.characters if c.lines > 0 or c.role == "main"]
    if len(names) >= 3 and not cancelled():
        rows = json.dumps([{"name": c.name, "aliases": c.aliases[:5], "gender": c.gender or "?", "lines": c.lines}
                           for c in names[:60]], ensure_ascii=False)
        try:
            out = runner.ask(CONSOLIDATE_SYSTEM, MERGE_PROMPT.format(lang=LANG_NAMES.get(lang, lang), rows=rows),
                             _same_list)
        except ValueError:
            out = {"same": []}
        for group in out["same"]:
            found: dict[int, tuple[Character, list[str]]] = {}  # each character with the names that found it
            for n in (str(x) for x in group if isinstance(x, str)):
                hits = [c for c in bible.characters if c.matches(n)]
                if len(hits) > 1:  # a name two characters share (a first name) points at neither
                    continue
                c = hits[0] if hits else bible.find(n)
                if c:
                    found.setdefault(id(c), (c, []))[1].append(n)
            if len(found) < 2:
                continue
            keep, keep_by = max(found.values(), key=lambda f: f[0].lines)
            for other, by in list(found.values()):
                if other is keep or (keep.gender and other.gender and keep.gender != other.gender):
                    continue
                # two different full names are one person only when the model named both in full
                if other_person(keep, other.name) and not (_full(by) and _full(keep_by)):
                    continue
                merge_characters(bible, keep, other)
        bible.assign_ids()
        if save:
            save(bible)
    todo = sorted((c for c in bible.characters if c.lines > 0 and c.unsure()), key=lambda c: -c.lines)[:max_look]
    for i, c in enumerate(todo):
        if cancelled():
            break
        snippets = _snippets(chapters, c)
        if not snippets:
            continue
        prompt = LOOK_PROMPT.format(
            lang=LANG_NAMES.get(lang, lang), name=c.name, aliases=", ".join(c.aliases[:5]) or "-",
            gender=c.gender or "unknown", age=c.age or "unknown", traits=c.traits[:80] or "-",
            snippets="\n".join(f"- {x}" for x in snippets))
        try:
            out = runner.ask(CONSOLIDATE_SYSTEM, prompt, lambda d: d)
        except ValueError:
            continue
        g = str(out.get("gender") or "").strip().lower()[:1]
        c.vote("gender", g if g in ("f", "m", "n") else "", 3)
        c.vote("age", str(out.get("age") or "").strip()[:30], 3)
        traits = str(out.get("traits") or "").strip()[:200]
        if traits and traits.lower() not in c.traits.lower():
            c.traits = (c.traits + "; " + traits).strip("; ")[:300]
        if save:
            save(bible)
        if progress:
            progress("consolidate", i + 1, len(todo), c.name)
    return bible


# -- annotate --------------------------------------------------------------------------------------------------

ANNOTATE_SYSTEM = ("You mark up books for multi-voice audiobooks. You never change the text: you only insert "
                   "speaker markers. You answer with one JSON object and nothing else.")

ANNOTATE_PROMPT = """Book language: {lang}. {hints}
Narration is {person}-person.

Speakers (id = marker number; 0 is the narrator and needs no marker):
{speakers}
{recent}
For every numbered paragraph below that contains direct speech or a character's thoughts, repeat the paragraph
EXACTLY (same words, spaces and punctuation) inserting <|speaker:N|> in front of each part spoken by speaker N and
<|speaker:0|> where the narrator continues. Attribution such as "said Anna" is the narrator's. Do not mark
paragraphs that are pure narration. Resolve pronouns and "he said" from the context. If a speaker is not in the
list, leave that part to the narrator and add the speaker to "new_characters" (they get a marker for the next
paragraphs); if you learn something about a listed speaker (a new alias, the real gender) put it in
"updates".{expression}

Reply with this JSON:
{{"blocks": [{{"b": <paragraph number>, "text": "<the paragraph with markers>"}}],
 "new_characters": [{{"name": str, "aliases": [str], "gender": "f" | "m" | "n" | "?", "unnamed": bool}}],
 "updates": [{{"name": str, "aliases": [str], "gender": "f" | "m" | "n" | "?"}}]}}

Paragraphs:
{text}"""

# The model never repeats the text in this method: the paragraph is cut into numbered parts here and the model
# only says who says each part, so punctuation it would "correct" (Spanish dialogue dashes) cannot break a paragraph.
PARTS_SYSTEM = ("You mark up books for multi-voice audiobooks. You are given paragraphs cut into numbered parts and "
                "say who says each part. You answer with one JSON object and nothing else.")

PARTS_PROMPT = """Book language: {lang}. {hints}
Narration is {person}-person.

Speakers (id; 0 is the narrator):
{speakers}
{recent}
Every paragraph below is cut into numbered parts. For each paragraph say which parts are spoken (or thought) by a
character and by whom: "who" maps a part number to the id of its speaker. Leave out the parts that are narration,
including attributions such as "said Anna". A character's speech often continues after the attribution: give that
later part too. Resolve pronouns and "he said" from the context. If a speaker is not in the list, leave that part
out and add the speaker to "new_characters"; if you learn something about a listed speaker (a new alias, the real
gender) put it in "updates".{lang_rule}

Example. A paragraph cut into parts:
  1| "Come in,"
  2| said Ben,
  3| "it is cold."
If Ben is speaker 2 the answer for it is {{"b": 7, "who": {{"1": 2, "3": 2}}}}.

Reply with this JSON:
{{"blocks": [{{"b": <paragraph number>, "who": {{"<part number>": <speaker id>}}{lang_json}}}],
 "new_characters": [{{"name": str, "aliases": [str], "gender": "f" | "m" | "n" | "?", "unnamed": bool}}],
 "updates": [{{"name": str, "aliases": [str], "gender": "f" | "m" | "n" | "?"}}]}}

Paragraphs:
{text}"""

# Added to the prompt only for a book that has more than one language (the same JSON, one more optional key).
PARTS_LANG_RULE = """
The book is in {lang} but also contains {others}. If a part is written in a language other than {lang} (a phrase or
a sentence in another language, not a name or a single borrowed word), add "lang": it maps that part number to the
ISO 639-1 code of its language, for example {{"2": "{example}"}}. Say nothing about parts in {lang}, and mark such a
part even when it is narration."""
PARTS_LANG_JSON = ', "lang": {{"<part number>": "<ISO 639-1 code>"}}'
OTHER_LANG_SHARE = 0.02  # a language with at least this share of the book is worth asking the model about


def other_languages(chapters: list, lang: str) -> list[str]:
    """Languages besides ``lang`` that make up at least 2% of the book's letters (empty for a one-language book)."""
    from folivox.lang.detect import shares

    text = "\n".join(b.text for ch in chapters for b in ch.blocks)
    return [k for k, v in shares(text).items() if k != lang and v >= OTHER_LANG_SHARE]


def parts_prompt(lang: str, others: list[str] | tuple[str, ...] = (), **fields) -> str:
    """PARTS_PROMPT for a book in ``lang``; ``others`` (more languages in the book) adds the language question."""
    rule = json_part = ""
    if others:
        names = [LANG_NAMES.get(o, o) for o in others]
        rule = PARTS_LANG_RULE.format(lang=LANG_NAMES.get(lang, lang), others=", ".join(names),
                                      example="en" if lang != "en" else others[0])
        json_part = PARTS_LANG_JSON
    return PARTS_PROMPT.format(lang=LANG_NAMES.get(lang, lang), lang_rule=rule, lang_json=json_part, **fields)


def part_languages(text: str, parts: list[tuple[int, int]], claimed: object, book_lang: str) -> list[str]:
    """The language of each part from the model's ``{"part number": "code"}``: only codes on the allow-list that
    differ from the book's and fit the part's letters; anything else is ignored (the book language reads it)."""
    from folivox.lang.codes import valid
    from folivox.lang.runs import plausible

    out = [""] * len(parts)
    if not isinstance(claimed, dict):
        return out
    for k, v in claimed.items():
        try:
            i = int(k) - 1
        except (TypeError, ValueError):
            continue
        code = valid(v)
        if 0 <= i < len(parts) and code and code != valid(book_lang) and plausible(text[parts[i][0]:parts[i][1]], code,
                                                                                    book_lang):
            out[i] = code
    return out


_DASHES = "—–"
_QUOTES = "«»„“”\""


def split_parts(text: str) -> list[tuple[int, int]]:
    """Cut a paragraph where speech may start or stop: before dashes and opening quotes, after closing quotes and
    after sentence ends. Returns (start, end) of the non-blank parts, in order."""
    cuts = {0, len(text)}
    for i, ch in enumerate(text):
        if ch in _DASHES:
            cuts.add(i)
        elif ch in _QUOTES:  # a quote after a space (or at the start) opens, any other closes
            opens = ch == "«" or ch == "„" or i == 0 or text[i - 1].isspace() or text[i - 1] in _DASHES
            cuts.add(i if opens and ch not in "»" else i + 1)
        elif ch in ".!?…" and i + 1 < len(text) and text[i + 1].isspace():
            cuts.add(i + 1)
    edges = sorted(cuts)
    parts = []
    for a, b in zip(edges, edges[1:], strict=False):
        chunk = text[a:b]
        if chunk.strip():
            parts.append((a + len(chunk) - len(chunk.lstrip()), b - (len(chunk) - len(chunk.rstrip()))))
    return parts


def parts_text(blocks: list[tuple[int, str]]) -> str:
    """The window as the model sees it: only paragraphs with speech, each cut into numbered parts."""
    out = []
    for b, text in blocks:
        if not SPEECH_MARKS.search(text):
            continue
        out.append(f"[B{b}]")
        out += [f"  {i}| {text[s:e]}" for i, (s, e) in enumerate(split_parts(text), 1)]
    return "\n".join(out)


def part_speakers(b: int, text: str, who: object) -> tuple[list[tuple[int, int]], list[int]]:
    """The parts of a paragraph and the speaker id of each, from the model's answer ``{"part number": speaker id}``
    (a list with one id per part is accepted too). Anything else is refused; part numbers that do not exist are
    ignored."""
    parts = split_parts(text)
    if isinstance(who, list):
        if len(who) != len(parts):
            raise mk.MarkupError(f"B{b}: expected {len(parts)} speakers")
        who = {str(i): x for i, x in enumerate(who, 1)}
    if not isinstance(who, dict):
        raise mk.MarkupError(f"B{b}: no speakers")
    try:
        ids = {int(k): int(v) for k, v in who.items()}
    except (TypeError, ValueError):
        raise mk.MarkupError(f"B{b}: speakers must be numbers") from None
    return parts, [ids.get(i, 0) for i in range(1, len(parts) + 1)]


def spans_from_ids(b: int, text: str, parts: list[tuple[int, int]], ids: list[int],
                   langs: list[str] | None = None) -> list[mk.Span]:
    """One span per run of parts with the same speaker and language (the narrator reading in the book's language
    has none)."""
    spans: list[mk.Span] = []
    for n, ((s, e), sp) in enumerate(zip(parts, ids, strict=False)):
        lg = langs[n] if langs else ""
        if not sp and not lg:
            continue
        if spans and spans[-1].speaker == sp and spans[-1].lang == lg and not text[spans[-1].end:s].strip():
            spans[-1] = mk.Span(b, spans[-1].start, e, sp, (), lg)
        else:
            spans.append(mk.Span(b, s, e, sp, (), lg))
    return spans


def spans_from_parts(b: int, text: str, who: object) -> list[mk.Span]:
    parts, ids = part_speakers(b, text, who)
    return spans_from_ids(b, text, parts, ids)


# -- the rule check: does what the model marked agree with what the text itself says? ---------------------------

def _gender_hint(lang: str, narration: str) -> str:
    """"f" or "m" when a piece of narration points at one person of one gender (a pronoun, a Russian speech verb)."""
    seen = set()
    m_set, f_set = PRONOUNS.get(lang, (set(), set()))
    for w in re.findall(r"\w+", narration.lower()):
        if w in m_set:
            seen.add("m")
        elif w in f_set:
            seen.add("f")
    if lang == "ru":
        for m in RU_SAID.finditer(narration):
            seen.add("f" if m.group(2) else "m")
    return seen.pop() if len(seen) == 1 else ""


def _base_names(c: Character) -> set[str]:
    """The ways the text names a character in the base form: the name, the aliases, the first word of a full name."""
    out = {norm_name(n) for n in c.names() if len(n) >= 3}
    if " " in c.name.strip():
        out.add(norm_name(c.name.split()[0]))
    return {n for n in out if n and " " not in n}


def check_paragraph(text: str, parts: list[tuple[int, int]], ids: list[int], bible: Bible, lang: str) -> list[dict]:
    """Doubts about one marked paragraph, as review entries. Two rules that read only the text: the name in an
    attribution ("said Boris") must be the speaker of the speech it belongs to, and the pronoun or the verb
    ending in it must not contradict the speaker's gender."""
    by_id = {c.id: c for c in bible.characters if c.id}
    names = set().union(*(_base_names(c) for c in by_id.values())) if by_id else set()
    out: list[dict] = []
    for i, (s, e) in enumerate(parts):
        if ids[i]:
            continue
        piece = text[s:e]
        # the speech this narration belongs to: the part right before it, else the part right after it
        target = (ids[i - 1] if i else 0) or (ids[i + 1] if i + 1 < len(ids) else 0)
        if not target or len(piece.split()) > 12 or not SPEECH_MARKS.search(text):
            continue
        # the attribution is the first clause ("said Boris"); a name further on ("so quietly that Anna ...") is
        # someone else's, and only the base form of a name counts (Russian "Анне" or "Анну" is the listener)
        clause = re.split(r"[,;:]", piece, maxsplit=1)[0]
        words = {norm_name(w) for w in re.findall(r"\w+", clause) if len(w) >= 3}
        named = {c.id for c in bible.characters if c.id and words & _base_names(c)}
        y = by_id.get(target)
        if len(named) == 1 and target not in named:
            x = by_id.get(next(iter(named)))
            out.append({"b": -1, "code": "attribution", "named": x.name if x else "", "marked": y.name if y else ""})
            continue
        # the pronoun rule is only safe when the narration names nobody (a name means someone else may be meant)
        if any(norm_name(w) in names for w in re.findall(r"\w+", piece) if len(w) >= 3):
            continue
        hint = _gender_hint(lang, piece)
        if hint and y and y.gender in ("f", "m") and "gender" not in y.unsure() and y.gender != hint:
            out.append({"b": -1, "code": "gender", "marked": y.name, "says": hint})
    return out[:1]  # one note per paragraph is enough to send the person there


EXPRESSION_RULES = """
Also put an expression tag in square brackets, for example [sighs] or [whispering], at the exact spot where the
text clearly asks for it (a sigh, a laugh, whispering, shouting, a long pause). Use only these tags:
{tags}
At most one tag per sentence, none when the text does not call for it; most paragraphs get none."""


@dataclass
class AnnotateStats:
    blocks: int = 0
    marked: int = 0
    fell_back: int = 0
    unknown_speakers: int = 0
    dropped_tags: int = 0
    failed_windows: int = 0
    short_windows: int = 0  # windows whose every reply left out paragraphs with speech (the most complete is kept)
    skipped: int = 0  # paragraphs with speech the model left out: the narrator reads them, they are on the review list
    new_characters: int = 0
    flagged: int = 0  # paragraphs sent to the review list


def _speakers_json(bible: Bible, relevant: list[Character]) -> str:
    rows = []
    for c in relevant:
        if not c.id:
            continue
        rows.append({"id": c.id, "name": c.name, "aliases": c.aliases[:5], "gender": c.gender or "?",
                     "age": c.age, "traits": c.traits[:80]})
    return json.dumps(rows, ensure_ascii=False) if rows else "(no characters)"


def _tag_list() -> str:
    """The sound and tone tags offered to the model: short, common, documented for the Fish models."""
    names = sorted({t["tag"] for t in mk.tag_database()["tags"] if t["category"] in ("sound", "tone")})
    return ", ".join(f"[{t}]" for t in names[:30])


PARTS_NOTE = """
Look again, carefully. A first reading of paragraph B{b} gave doubtful speakers: {why}. Use the context above and
below (who is talking to whom, who answers whom) and answer for that paragraph only.

The paragraphs just before it, with their speakers:
{before}
The paragraphs just after it:
{after}
"""


def _why(r: dict) -> str:
    code = r.get("code")
    if code == "attribution":
        return f"the narration names {r.get('named')} but the speech went to {r.get('marked')}"
    if code == "gender":
        who = "woman" if r.get("says") == "f" else "man"
        return f"the narration points at a {who} but the speech went to {r.get('marked')}"
    return "the first reading did not give a usable answer"


def second_opinion(chapters: list, lang: str, bible: Bible, runner: Runner, markup: mk.Markup, *,
                   save: Callable[[mk.Markup, Bible], None] | None = None, progress: Progress | None = None,
                   cancelled: Callable[[], bool] = lambda: False) -> tuple[int, int]:
    """Ask again, with a few paragraphs of context on both sides, about every place the rule check doubted. A new
    answer is kept only if the rule check accepts it; otherwise the first one stays and the place stays on the list.
    Returns (settled, still doubtful)."""
    todo = [(ci, dict(r)) for ci, cm in sorted(markup.chapters.items()) for r in cm.review]
    others = other_languages(chapters, lang) if todo else []
    settled = 0
    by_name = {c.id: c.name for c in bible.characters if c.id}
    for i, (ci, r) in enumerate(todo):
        if cancelled():
            break
        ch, cm = chapters[ci], markup.chapters[ci]
        b = int(r["b"])
        if not 0 <= b < len(ch.blocks):
            continue
        text = ch.blocks[b].text

        def spoke(j: int, cm=cm, ch=ch) -> str:
            who = [by_name.get(x.speaker, "?") for x in cm.spans if x.block == j and x.speaker]
            return f"[B{j}] {ch.blocks[j].text[:220]}" + (f"  -> {', '.join(dict.fromkeys(who))}" if who else "")

        before = "\n".join(spoke(j) for j in range(max(0, b - 6), b)) or "(none)"
        after = "\n".join(f"[B{j}] {ch.blocks[j].text[:220]}"
                          for j in range(b + 1, min(len(ch.blocks), b + 3))) or "(none)"
        relevant = relevant_for(bible, text + " " + before, [])
        note = PARTS_NOTE.format(b=b, why=_why(r), before=before, after=after)
        prompt = parts_prompt(lang, others, hints=QUOTE_HINTS, person=bible.narration or "third",
                              speakers=_speakers_json(bible, relevant), recent=note, text=parts_text([(b, text)]))

        def check(d: dict) -> dict:
            if not isinstance(d.get("blocks"), list):
                raise ValueError("no blocks list")
            return d

        try:
            out = runner.ask(PARTS_SYSTEM, prompt, check)
            item = next(x for x in out["blocks"] if int(str(x.get("b")).lstrip("B")) == b)
            parts, ids = part_speakers(b, text, item.get("who"))
        except (ValueError, StopIteration, KeyError, TypeError, AttributeError, mk.MarkupError):
            continue
        valid = {c.id for c in bible.characters if c.id} | {0}
        if any(x not in valid for x in ids) or check_paragraph(text, parts, ids, bible, lang):
            continue
        cm.spans = [x for x in cm.spans if x.block != b] + spans_from_ids(
            b, text, parts, ids, part_languages(text, parts, item.get("lang"), lang) if others else None)
        cm.spans.sort(key=lambda x: (x.block, x.start))
        cm.review = [x for x in cm.review if x.get("b") != b]
        settled += 1
        if save:
            save(markup, bible)
        if progress:
            progress("review", i + 1, len(todo), f"{settled} settled")
    return settled, sum(len(c.review) for c in markup.chapters.values())


def sync_speakers(markup: mk.Markup, bible: Bible) -> None:
    """Give every character with an id its speaker in the markup. A shared id (more than 15 speakers) is named as
    shared and lists the names of everyone on it, so it is never mistaken for one of them."""
    for c in bible.characters:
        if not c.id:
            continue
        if c.shared:
            on_it = [x for x in bible.characters if x.id == c.id]
            label = SHARED_NAMES.get(c.id, f"Speaker {c.id} (shared)")
            sp = markup.ensure_speaker(c.id, label)
            if not sp.name or sp.name.startswith("Speaker ") or any(sp.name == x.name for x in on_it):
                sp.name = label  # also replaces the name of the first character put there before
                sp.notes = "" if sp.notes in {x.traits for x in on_it} else sp.notes
            genders = {x.gender for x in on_it}
            sp.gender = sp.gender if sp.gender in genders else (genders.pop() if len(genders) == 1 else "")
            sp.aliases = [n for x in on_it for n in x.names()][:8]
            continue
        sp = markup.ensure_speaker(c.id, c.name)
        if sp.name.startswith("Speaker ") or not sp.name:
            sp.name = c.name
        sp.gender = sp.gender or c.gender
        sp.aliases = c.aliases[:8]
        sp.notes = sp.notes or c.traits


def annotate(chapters: list, lang: str, bible: Bible, runner: Runner, markup: mk.Markup | None = None, *,
             expression_tags: bool = False, chapter_filter: set[int] | None = None,
             save: Callable[[mk.Markup, Bible], None] | None = None, progress: Progress | None = None,
             cancelled: Callable[[], bool] = lambda: False, window_chars: int = WINDOW_CHARS,
             depth: int = 2, method: str | None = None,
             max_windows: int | None = None, other_langs: list[str] | None = None) -> tuple[mk.Markup, AnnotateStats]:
    """Mark who speaks in every paragraph with speech. Existing spans in the paragraphs processed are replaced.

    ``method``: "parts" (the default: the paragraph is cut here and the model only says which parts each character
    speaks; no expression tags) or "text" (the model repeats each paragraph with markers; used with expression tags)."""
    method = method or os.environ.get("FOLIVOX_MARKUP_METHOD", "parts")
    by_parts = method == "parts" and not expression_tags
    markup = markup or mk.Markup(depth=depth)
    markup.depth = depth
    bible.assign_ids()
    stats = AnnotateStats()
    wins = [w for w in windows(chapters, window_chars, only_speech=True)
            if chapter_filter is None or w.chapter in chapter_filter]
    sampled = bool(max_windows and len(wins) > max_windows)
    if sampled:  # a preview: windows spread over the book
        step = len(wins) / max_windows
        wins = [wins[int(i * step)] for i in range(max_windows)]
    tags_known = mk.known_tags() if expression_tags else set()
    tag_list = _tag_list() if expression_tags else ""
    others = other_languages(chapters, lang) if by_parts and other_langs is None else list(other_langs or ())
    recent_names: list[str] = []
    recent_ctx: list[str] = []

    last_window = {w.chapter: n for n, w in enumerate(wins)}
    for n, w in enumerate(wins):
        if cancelled():
            break
        text = w.text()
        relevant = relevant_for(bible, text, recent_names)
        recent = ""
        if recent_ctx:
            recent = "Context, the previous paragraphs and who spoke them:\n" + "\n".join(recent_ctx[-4:]) + "\n"
        if by_parts:
            prompt = parts_prompt(lang, others, hints=QUOTE_HINTS, person=bible.narration or "third",
                                  speakers=_speakers_json(bible, relevant), recent=recent, text=parts_text(w.blocks))
        else:
            prompt = ANNOTATE_PROMPT.format(
                lang=LANG_NAMES.get(lang, lang), hints=QUOTE_HINTS, person=bible.narration or "third",
                speakers=_speakers_json(bible, relevant), recent=recent, text=text,
                expression=EXPRESSION_RULES.format(tags=tag_list) if expression_tags else "")
        originals = dict(w.blocks)
        must = [b for b, t in w.blocks if DIALOGUE_START.search(t)]  # paragraphs that open with speech

        def check(d: dict, must=must) -> dict:
            if not isinstance(d.get("blocks"), list):
                raise ValueError("no blocks list")
            answered = set()
            for item in d["blocks"]:
                try:
                    answered.add(int(str(item["b"]).lstrip("B")))
                except (KeyError, TypeError, ValueError):
                    pass
            missing = [b for b in must if b not in answered]
            if missing:
                raise ShortReply(d, missing)
            return d

        try:
            out = runner.ask(PARTS_SYSTEM if by_parts else ANNOTATE_SYSTEM, prompt, check)
        except ShortReply as e:  # asked again and still short: keep what it answered, the rest goes on the list
            out = e.reply
            stats.short_windows += 1
            if progress:
                progress("annotate", n + 1, len(wins),
                         f"window {n + 1}: the model left out {len(e.missing)} paragraph(s) with speech")
        except ValueError:
            stats.failed_windows += 1
            stats.blocks += len(w.blocks)
            stats.fell_back += len(w.blocks)
            if progress:
                progress("annotate", n + 1, len(wins), f"window {n + 1}: the narrator reads it")
            continue
        # characters the model met for the first time
        found = [x for k in ("new_characters", "updates") if isinstance(out.get(k), list) for x in out[k]]
        for e in found:
            if isinstance(e, dict):
                before = len(bible.characters)
                bible.merge({**e, "lines": 0}, w.chapter)
                stats.new_characters += len(bible.characters) - before
        bible.assign_ids()
        for c in bible.characters:  # a new speaker needs an id and a line to get one
            if not c.id and c.lines <= 0:
                c.lines = 1
        bible.assign_ids()
        sync_speakers(markup, bible)
        ids = {c.id for c in bible.characters if c.id} | {0}
        by_name = {c.id: c.name for c in bible.characters if c.id}
        seen: set[int] = set()
        recent_names = []
        cm = markup.chapter(w.chapter, chapters[w.chapter].title, mk.chapter_sig(chapters[w.chapter]))
        for item in out["blocks"]:
            try:
                b = int(str(item["b"]).lstrip("B"))
                line = "" if by_parts else _fix_speaker_tags(str(item["text"]))
            except (KeyError, TypeError, ValueError):
                continue
            original = originals.get(b)
            if original is None or b in seen:
                continue
            seen.add(b)
            stats.blocks += 1
            flags: list[dict] = []
            try:
                if by_parts:
                    parts, ids_ = part_speakers(b, original, item.get("who"))
                    langs_ = part_languages(original, parts, item.get("lang"), lang) if others else None
                    spans, marks = spans_from_ids(b, original, parts, ids_, langs_), []
                    flags = check_paragraph(original, parts, ids_, bible, lang)
                else:
                    spans, marks = mk.parse_block(line, original, loose=True)
            except mk.MarkupError:
                stats.fell_back += 1
                cm.review = [r for r in cm.review if r.get("b") != b] + [{"b": b, "code": "unparsed"}]
                stats.flagged += 1
                continue
            good_spans = []
            for s in spans:
                if s.speaker not in ids:
                    stats.unknown_speakers += 1
                    continue
                tags = tuple(t for t in s.tags if t in tags_known)
                stats.dropped_tags += len(s.tags) - len(tags)
                if s.speaker or tags or s.lang:
                    good_spans.append(mk.Span(b, s.start, s.end, s.speaker, tags, s.lang))
            good_marks = []
            for m in marks:
                if m.tag in tags_known:
                    good_marks.append(mk.Mark(b, m.at, m.tag))
                else:
                    stats.dropped_tags += 1
            cm.review = [r for r in cm.review if r.get("b") != b] + [{**f, "b": b} for f in flags]
            stats.flagged += bool(flags)
            cm.spans = [x for x in cm.spans if x.block != b] + good_spans
            cm.marks = [x for x in cm.marks if x.block != b] + good_marks
            if good_spans or good_marks:
                stats.marked += 1
                names = [by_name.get(s.speaker, "narrator") for s in good_spans if s.speaker] or ["narrator"]
                recent_names += [by_name[s.speaker] for s in good_spans if s.speaker in by_name]
                recent_ctx.append(f"{original[:140]} -> {', '.join(dict.fromkeys(names))}")
        missing = [b for b, _ in w.blocks if b not in seen]
        stats.blocks += len(missing)  # paragraphs the model skipped: the narrator reads them
        for b, t in w.blocks:  # a paragraph with speech that the model left out is a place to look at
            if b in missing and DIALOGUE_START.search(t):
                cm.review = [r for r in cm.review if r.get("b") != b] + [{"b": b, "code": "skipped"}]
                stats.flagged += 1
                stats.skipped += 1
        if last_window[w.chapter] == n and not sampled:
            cm.done = True  # every window of this chapter has been through the model
        cm.spans.sort(key=lambda s: (s.block, s.start))
        cm.marks.sort(key=lambda m: (m.block, m.at))
        if save:
            save(markup, bible)
        if progress:
            progress("annotate", n + 1, len(wins), f"{stats.marked} paragraphs marked")
    return markup, stats


# -- choosing and reaching the model ------------------------------------------------------------------------------

def make_llm(provider: str | None = None, model: str | None = None):
    """(llm, model name) from the config and options. For Ollama the model defaults to the largest Gemma 4 that
    suits this computer, and a missing model or a stopped server is reported with the command that fixes it."""
    from folivox import config, diag
    from folivox.llm import LLMS, recommend_model

    name = config.provider("llm", provider) or LLMS.default
    cfg = config.section(f"llm.{name}")
    if name == "ollama":
        model = model or cfg.get("model") or recommend_model(diag.ram_gb())
        llm = LLMS.create(name, model=model)
        have = llm.installed()
        if not have:
            raise LLMError("Ollama is not running (start it, or install it from https://ollama.com), or has no "
                           f"models. Then:  ollama pull {model}")
        if not any(h == model or h.split(":")[0] == model for h in have):
            raise LLMError(f"The model {model} is not installed. Run:  ollama pull {model}   (installed: "
                           f"{', '.join(have)})")
        return llm, model
    llm = LLMS.create(name, **({"model": model} if model else {}))
    return llm, getattr(llm, "model", name)

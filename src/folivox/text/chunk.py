"""Turn chapters into TTS chunks with pauses.

A chunk is what the model synthesizes in one call: one or a few sentences from a single
paragraph. Short sentences are packed together (CosyVoice is unstable on 1–3 word inputs),
long ones are split at clause boundaries. Pauses between chunks depend on structure:
sentence < paragraph < heading / scene break < chapter.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field, replace

from folivox import lang as languages
from folivox.book import Block, BlockKind, Chapter
from folivox.lang import has_speech
from folivox.text.pronounce import Lexicon
from folivox.text.stress import strip_marks


@dataclass
class Pauses:
    """Seconds of silence after a chunk."""

    sentence: float = 0.25  # between chunks of one paragraph
    paragraph: float = 0.65
    dialogue: float = 0.45  # between consecutive dialogue lines
    heading: float = 1.2
    scene: float = 1.8  # «* * *»
    chapter_end: float = 2.5


_DOUBLE_QUOTES = "«»„“”‟\"‹›「」『』"
_QUOTE_SENTINEL = "\ue000"
_LETTER = r"[^\W\d_]"


def _keeping_quotes(normalize):
    """Wrap a language's ``normalize`` so quotation marks survive as a plain ``"``. The rules strip them, but the
    voice models read the text and take quotes as a cue for direct speech (a local A/B showed no harm: same CER,
    no spoken quote marks). An apostrophe between letters stays in its word (the rules handle it). A closing ’
    counts as a quote only in a paragraph that also has an opening ‘, since ’ after a letter can be a
    possessive (dogs’ bowls)."""
    def run(text: str) -> str:
        marks = _DOUBLE_QUOTES + ("‘’" if "‘" in text else "")
        protected = re.sub(f"(?<!{_LETTER})[{marks}]|[{marks}](?!{_LETTER})|‘", _QUOTE_SENTINEL, text)
        return normalize(protected).replace(_QUOTE_SENTINEL, '"')

    return run


@dataclass
class ChunkOptions:
    max_chars: int = 300
    min_chars: int = 30
    stress: str = "yo"
    lang: str = "ru"
    pauses: Pauses = field(default_factory=Pauses)
    speak_headings: bool = True
    lexicon: Lexicon | None = None  # the book's custom pronunciations
    local_engine: bool = True  # False: an online engine, which cannot take stress marks or phonemes
    mixed: bool = True  # find stretches of another script (Latin in Russian text, …) and read them in their language
    speaks: frozenset[str] | None = None  # languages the engine can read (None: not checked)
    fallbacks: Counter = field(default_factory=Counter)  # {language: fragments the engine cannot read in it}
    join_paragraphs: bool = False  # online engines: put neighbouring paragraphs into one request (fewer requests)


@dataclass
class Chunk:
    index: int  # within the chapter
    text: str  # model input (normalised, maybe with stress marks)
    source: str  # same text without stress marks, used for QA
    pause_after: float
    kind: str = "p"
    speaker: int = 0  # who reads it (0 = narrator); see folivox.markup
    markup: str = ""  # expression tags written in front of the text, e.g. "[sighs]" (part of the cache key)
    lang: str = ""  # the fragment's language when it is not the book's ("" = the book's)
    block: int = -1  # the chapter block it comes from (-1: a chapter heading); used to join fragments
    role: str = ""  # "dialogue", "poem" or "text": what kind of passage the block is, for joining


@dataclass
class MarkupContext:
    """What ``plan_chapter`` needs to apply a book's markup to one chapter."""
    markup: object  # folivox.markup.Markup
    chapter: int
    tags: bool  # the engine understands expression tags; otherwise they are dropped


_CLAUSE_SPLIT = re.compile(r"(?<=[,;:])\s+|\s+(?=—\s)|(?<=[，；：、])")


def sentences(text: str, lang: str = "ru") -> list[str]:
    """Sentences of the language module, re-joined where a dialogue attribution was split off.

    «Ты куда? — спросила соседка.» stays one sentence.
    """
    language = languages.get(lang)
    out: list[str] = []
    for s in language.sentences(text):
        if not s:
            continue
        if out and (language.is_continuation(s) or not has_speech(s)):
            out[-1] = f"{out[-1]}{_sep(out[-1])}{s}"
        else:
            out.append(s)
    return out


def _sep(prev: str) -> str:
    """No space after CJK full-width punctuation."""
    return "" if prev and prev[-1] in "。！？，；：、」』）" else " "


def split_long(sentence: str, max_chars: int) -> list[str]:
    if len(sentence) <= max_chars:
        return [sentence]
    parts = [p for p in _CLAUSE_SPLIT.split(sentence) if p]
    if len(parts) == 1:
        words = sentence.split()
        mid = len(words) // 2
        parts = [" ".join(words[:mid]), " ".join(words[mid:])]
        if len(words) < 2:
            return [sentence]
    # greedy re-pack of clauses up to max_chars
    out: list[str] = []
    for p in parts:
        if out and len(out[-1]) + 1 + len(p) <= max_chars:
            out[-1] = f"{out[-1]} {p}"
        else:
            out.append(p)
    res: list[str] = []
    for p in out:
        res += split_long(p, max_chars) if len(p) > max_chars and p != sentence else [p]
    return res


def pack(sents: list[str], max_chars: int, min_chars: int) -> list[str]:
    pieces: list[str] = []
    for s in sents:
        pieces += split_long(s, max_chars)
    out: list[str] = []
    for p in pieces:
        if out and len(out[-1]) + 1 + len(p) <= max_chars:
            out[-1] = f"{out[-1]} {p}"
        else:
            out.append(p)
    return out


def _is_dialogue(block: Block) -> bool:
    return block.kind == BlockKind.PARAGRAPH and block.text.lstrip().startswith(("—", "–", "- "))


def _heading_text(h: str) -> str:
    h = h.strip()
    return h if re.search(r"[.!?…]$", h) else h + "."


def _poem_text(stanza: str) -> str:
    lines = [ln.strip() for ln in stanza.split("\n") if ln.strip()]
    return " ".join(ln if re.search(r"[,.;:!?…—-]$", ln) else ln + "," for ln in lines).rstrip(",") + (
        "" if re.search(r"[.!?…]$", lines[-1]) else ".") if lines else ""


def finalize(text: str, opts: ChunkOptions) -> tuple[str, str]:
    """Model input (stress marks, custom pronunciations) and the plain text that quality control compares with."""
    tts = languages.get(opts.lang).stress(text, opts.stress)
    source = strip_marks(tts)
    if opts.lexicon:
        tts = opts.lexicon.apply(tts, local=opts.local_engine)
        source = opts.lexicon.apply(source, quality_control=True)
    return tts, source


def fragment_lang(seg, opts: ChunkOptions) -> str:
    """The language a segment is read in when it is not the book's, else ``""``. A language the engine cannot
    read, or one with no text rules here, stays with the book language (counted in ``opts.fallbacks``); so does
    text a custom pronunciation covers (the entry says how to read it in the book's language)."""
    lang = seg.lang
    if not lang or lang == opts.lang:
        return ""
    try:
        languages.get(lang)
        known = True
    except ValueError:
        known = False
    if not known or (opts.speaks is not None and lang not in opts.speaks):
        opts.fallbacks[lang] += 1
        return ""
    if opts.lexicon and opts.lexicon.apply(seg.text, local=opts.local_engine) != seg.text:
        return ""
    return lang


def _plan_segments(raw, segs, ctx, end_pause, opts, normalize, dialogue, p) -> None:
    """One block read by several speakers (or with tags): every segment is packed on its own, so a fragment
    never mixes two voices. Tags go in front of the first fragment of their segment."""
    from folivox.markup import norm_tag

    def tag_text(seg) -> str:
        return "".join(f"[{norm_tag(t)}]" for t in seg.tags) if ctx.tags and ctx.markup.expression_tags else ""

    pieces, carry = [], ""
    for seg in segs:
        if has_speech(seg.text):
            edge_trimmed = seg.text.strip().strip("—–―").strip()  # dialogue dashes at the edges are not read
            pieces.append((seg, edge_trimmed, carry + tag_text(seg), fragment_lang(seg, opts)))
            carry = ""
        else:  # a segment without speech (a stray space): its tags go to the next one
            carry += tag_text(seg)
    for k, (seg, text, tags, lang) in enumerate(pieces):
        language = languages.get(lang or opts.lang)  # the text rules of the language the segment is read in
        normalize, dialogue = language.normalize, language.dialogue
        spoken = dialogue(_keeping_quotes(normalize)(text))
        parts = pack(sentences(spoken, lang or opts.lang), opts.max_chars, opts.min_chars)
        nxt_speaker = pieces[k + 1][0].speaker if k + 1 < len(pieces) else None
        last_pause = end_pause if nxt_speaker is None else (p.dialogue if nxt_speaker != seg.speaker else p.sentence)
        for j, part in enumerate(parts):
            raw.append((dialogue(part), last_pause if j == len(parts) - 1 else p.sentence, "p", seg.speaker,
                        tags if j == 0 else "", lang))


_QUOTE_START = tuple("«“„\"‹「『")


def _role(block: Block) -> str:
    if block.kind == BlockKind.POEM:
        return "poem"
    if _is_dialogue(block) or block.text.lstrip().startswith(_QUOTE_START):
        return "dialogue"
    return "text"


def _size(units) -> int:
    return sum(u.size for u in units) + 2 * max(0, len(units) - 1)


@dataclass
class _Unit:
    """A block (paragraph, line of dialogue, stanza), or one piece of a block too long for a single request."""
    chunks: list
    role: str

    @property
    def size(self) -> int:
        return sum(len(c.text) for c in self.chunks) + max(0, len(self.chunks) - 1)


def _balanced(units: list, limit: int) -> list[list]:
    """Split ``units`` (in order) into as few groups as fit ``limit``, as even in size as possible: a long passage
    becomes two halves rather than a full request and a scrap."""
    if _size(units) <= limit:
        return [units]
    k = 1  # the fewest groups, from a greedy fill
    size = -2
    for u in units:
        size += u.size + 2
        if size > limit:
            k, size = k + 1, u.size
    n, inf = len(units), float("inf")
    pre = [0]
    for u in units:
        pre.append(pre[-1] + u.size)

    def cost(a: int, b: int) -> int:  # units a..b-1 in one group
        return pre[b] - pre[a] + 2 * (b - a - 1)

    while True:
        best = [[inf] * (n + 1) for _ in range(k + 1)]  # best[g][j]: smallest largest group, j units in g groups
        cut = [[0] * (n + 1) for _ in range(k + 1)]
        best[0][0] = 0
        for g in range(1, k + 1):
            for b in range(1, n + 1):
                for a in range(g - 1, b):
                    c = cost(a, b)
                    if c > limit:
                        continue
                    v = max(best[g - 1][a], c)
                    if v < best[g][b]:
                        best[g][b], cut[g][b] = v, a
        if best[k][n] < inf or k >= n:
            break
        k += 1
    if best[k][n] == inf:  # a unit larger than the limit: keep every unit apart
        return [[u] for u in units]
    groups, b = [], n
    for g in range(k, 0, -1):
        a = cut[g][b]
        groups.append(units[a:b])
        b = a
    return groups[::-1]


def _pack_passage(units: list, limit: int) -> list[list]:
    """Group the blocks of one passage (between headings, scene breaks and changes of speaker) into requests.

    The passage is read as sections of one kind: a run of dialogue (short narration lines between replies count
    as dialogue), a poem, a run of narrative paragraphs. A section that fits one request is never split, and a
    request ends where one section ends and another begins, unless the next section fits in the same request
    entirely. So a conversation goes in one piece and the narration after it starts a new request, rather than
    the conversation plus one paragraph and then the rest. A section longer than a request is split evenly at
    block boundaries."""
    for i in range(1, len(units) - 1):  # «Помолчали.» between two lines of dialogue belongs to the conversation
        if units[i].role == "text" and units[i].size <= 120 and units[i - 1].role == units[i + 1].role == "dialogue":
            units[i].role = "dialogue"
    sections: list[list] = []
    for u in units:
        if sections and sections[-1][-1].role == u.role:
            sections[-1].append(u)
        else:
            sections.append([u])
    parts: list[list] = []
    carry: list = []
    for sec in sections:
        if not carry or _size(carry) + 2 + _size(sec) <= limit:
            if _size(carry + sec) <= limit:
                carry += sec
                continue
        if _size(sec) <= limit:  # the section fits on its own: keep it whole
            if carry:
                parts.append(carry)
            carry = sec
            continue
        if carry and _size(carry) >= limit // 4:  # a long section starts a request of its own
            parts.append(carry)
            carry = []
        groups = _balanced(carry + sec, limit)
        parts += groups[:-1]
        carry = groups[-1]
    if carry:
        parts.append(carry)
    # a scrap of a request (one short line) goes with the request before it when that still fits
    out: list[list] = []
    for g in parts:
        if out and _size(g) < limit // 8 and _size(out[-1]) + 2 + _size(g) <= limit:
            out[-1] = out[-1] + g
        else:
            out.append(g)
    return out


def join_chunks(chunks: list[Chunk], opts: ChunkOptions) -> list[Chunk]:
    """Merge fragments of running text into longer requests for an online engine (up to ``max_chars``), choosing
    the cuts by sense rather than filling each request to the limit (see :func:`_pack_passage`). The silence
    between merged blocks is then up to the voice model (a line break between replies, a blank line between
    paragraphs). Headings, scene breaks, the end of a chapter, another speaker or language and fragments with
    expression tags always start a new request and keep their own pause."""
    p = opts.pauses
    soft = max(p.sentence, p.paragraph, p.dialogue)
    limit = opts.max_chars

    def linked(a: Chunk, b: Chunk) -> bool:
        return (a.kind in ("p", "poem") and b.kind in ("p", "poem") and not a.markup and not b.markup
                and a.speaker == b.speaker and a.lang == b.lang and a.pause_after <= soft)

    passages: list[list[Chunk]] = []
    for c in chunks:
        if passages and linked(passages[-1][-1], c):
            passages[-1].append(c)
        else:
            passages.append([c])
    out: list[Chunk] = []
    for passage in passages:
        units: list[_Unit] = []
        for c in passage:
            if units and units[-1].chunks[-1].block == c.block and c.block >= 0:
                units[-1].chunks.append(c)
            else:
                units.append(_Unit([c], c.role or "text"))
        flat: list[_Unit] = []
        for u in units:  # a block too long for one request stays in its pieces
            flat += [_Unit([c], u.role) for c in u.chunks] if u.size > limit else [u]
        for group in _pack_passage(flat, limit):
            parts = [c for u in group for c in u.chunks]
            first = parts[0]
            text, source = first.text, first.source
            for prev, c in zip(parts, parts[1:], strict=False):
                sep = " " if prev.block == c.block else "\n" if prev.role == c.role == "dialogue" else "\n\n"
                text, source = f"{text}{sep}{c.text}", f"{source}{sep}{c.source}"
            kind = "poem" if all(c.kind == "poem" for c in parts) else first.kind
            out.append(replace(first, index=len(out), text=text, source=source, pause_after=parts[-1].pause_after,
                               kind=kind))
    return out


def plan_chapter(chapter: Chapter, opts: ChunkOptions | None = None, ctx: MarkupContext | None = None) -> list[Chunk]:
    opts = opts or ChunkOptions()
    p = opts.pauses
    language = languages.get(opts.lang)
    normalize, dialogue = language.normalize, language.dialogue
    raw: list[tuple[str, float, str, int, str, str]] = []  # (text, pause, kind, speaker, tags, language)

    if opts.speak_headings:
        for h in chapter.headings:
            if has_speech(h):
                raw.append((normalize(_heading_text(h)), p.heading, "heading", 0, "", ""))

    blocks = chapter.blocks
    where: dict[int, tuple[int, str]] = {}  # raw index → (block, role)
    for i, b in enumerate(blocks):
        start = len(raw)
        _plan_block(i, b, blocks, raw, opts, ctx, p, normalize, dialogue)
        for k in range(start, len(raw)):
            where[k] = (i, _role(b))

    chunks: list[Chunk] = []
    for k, (text, pause, kind, speaker, tags, lang) in enumerate(raw):
        if not has_speech(text):
            continue
        tts, source = finalize(text, replace(opts, lang=lang) if lang else opts)
        if tags:
            tts = f"{tags} {tts}"
        block, role = where.get(k, (-1, ""))
        chunks.append(Chunk(len(chunks), tts, source, pause, kind, speaker, tags, lang, block, role))
    if opts.join_paragraphs:
        chunks = join_chunks(chunks, opts)
    if chunks:
        chunks[-1].pause_after = p.chapter_end
    return chunks


def _plan_block(i, b, blocks, raw, opts, ctx, p, normalize, dialogue) -> None:
    """Raw fragments of block ``i`` (appended to ``raw``); a scene break only lengthens the pause before it."""
    if not has_speech(b.text):  # scene break «* * *»
        if raw:
            text, pause, kind, spk, tg, lg = raw[-1]
            raw[-1] = (text, max(pause, p.scene), kind, spk, tg, lg)
        return
    if b.kind == BlockKind.SUBTITLE:
        raw.append((normalize(_heading_text(b.text)), p.heading, "heading", 0, "", ""))
        return
    nxt = blocks[i + 1] if i + 1 < len(blocks) else None
    end_pause = p.dialogue if (_is_dialogue(b) and nxt is not None and _is_dialogue(nxt)) else p.paragraph
    if b.kind in (BlockKind.EPIGRAPH, BlockKind.AUTHOR) and not (nxt and nxt.kind in (
            BlockKind.EPIGRAPH, BlockKind.AUTHOR)):
        end_pause = p.heading
    segs = None
    if b.kind != BlockKind.POEM:
        from folivox.markup import Markup

        here = ctx or MarkupContext(Markup(), 0, False)  # no markup: only the script check can split a block
        segs = here.markup.segments(here.chapter, i, b.text, opts.lang if opts.mixed else "")
    if segs and (len(segs) > 1 or segs[0].speaker or segs[0].tags or segs[0].lang):
        _plan_segments(raw, segs, here, end_pause, opts, normalize, dialogue, p)
        return
    text = dialogue(normalize(_poem_text(b.text) if b.kind == BlockKind.POEM else b.text))
    parts = pack(sentences(text, opts.lang), opts.max_chars, opts.min_chars)
    for j, part in enumerate(parts):
        raw.append((dialogue(part), end_pause if j == len(parts) - 1 else p.sentence, b.kind.value, 0, "", ""))


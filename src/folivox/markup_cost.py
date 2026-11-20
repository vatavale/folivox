"""What automatic markup costs with a paid language model: an estimate before a run, a spending cap during one.

The estimate builds the prompts this book would send (the windows and templates of :mod:`folivox.markup_llm`) and
counts their characters; the replies are sized from the paragraphs with speech. During a run the spend is counted
from the token usage each reply reports, and :class:`Budget` stops the run before the call that would pass the cap
(the reply cache lets the same command carry on later).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# Measured on English novels with the markup prompts (2026-11-20): Claude 2.55 characters per token, OpenAI 3.6,
# Gemma 3.4. Counting with the smallest errs high for every provider.
CHARS_PER_TOKEN = 2.5
ANALYSE_REPLY_SHARE = 0.35  # an analysis reply (the characters met in a window) is about a third of the window's text
# output tokens per token of the JSON answer: models reason before answering, counted (and billed) as output.
# Measured: 1.6x Claude Sonnet 5.5, 2x Claude Opus 5.5, 4.2x Claude Haiku 5.5 and gpt-6-luna.
THINKING_LIKELY, THINKING_MAX = 2.0, 4.5
FIRST_OUTPUT_SHARE = 0.8  # before the first reply of a run: output tokens per prompt token (measured 0.3 to 0.8)
KNOWN_GUESS = 8  # characters in the list sent with each window when the book has not been analysed yet
SHORT_CALL = 0.25  # a second-look call (one character, one paragraph) is about a quarter of a marking call

PRICES_READ = "2026-11-20"
# List prices, USD per million tokens: input, output, cached input read, cache write. Read on PRICES_READ from the
# providers' pricing pages; config.toml's price_in / price_out override them (prices change, check yours).
PRICES: dict[str, tuple[float, float, float, float]] = {
    "claude-opus-5-5": (4.0, 20.0, 0.20, 5.0),
    "claude-sonnet-5-5": (2.0, 10.0, 0.10, 2.5),
    "claude-haiku-5-5": (0.10, 0.50, 0.01, 0.125),
    "gpt-6.2-astra": (8.0, 40.0, 0.40, 8.0),
    "gpt-6.2-sol": (2.0, 10.0, 0.10, 2.0),
    "gpt-6-luna": (0.10, 0.50, 0.01, 0.10),
}


@dataclass(frozen=True)
class Price:
    """USD per million tokens."""

    input: float
    output: float
    cache_read: float
    cache_write: float
    source: str = ""  # "list" (the table above), "config" or "local"

    @property
    def free(self) -> bool:
        return self.source == "local"

    def usd(self, u: Usage) -> float:
        return (u.input * self.input + u.output * self.output + u.cache_read * self.cache_read
                + u.cache_write * self.cache_write) / 1e6


FREE = Price(0.0, 0.0, 0.0, 0.0, "local")


@dataclass
class Usage:
    """Tokens of a run so far (or of an estimate): ``input`` is the uncached prompt."""

    calls: int = 0
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    estimated: int = 0  # calls whose reply reported no usage (counted from the characters instead)

    def add(self, d: dict) -> None:
        self.calls += 1
        for k in ("input", "output", "cache_read", "cache_write"):
            setattr(self, k, getattr(self, k) + max(0, int(d.get(k) or 0)))

    @property
    def prompt(self) -> int:
        return self.input + self.cache_read + self.cache_write

    @property
    def tokens(self) -> int:
        return self.prompt + self.output


def _key(model: str) -> str:
    """``anthropic/claude-sonnet-5.5`` (a router's name) and ``claude-sonnet-5-5`` find the same entry."""
    return str(model or "").strip().lower().rsplit("/", 1)[-1].replace(".", "-")


_TABLE = {_key(k): v for k, v in PRICES.items()}


def _number(raw: object, what: str, provider: str) -> float:
    try:
        n = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ValueError(f"{what} for the {provider} language model must be a number (USD per million tokens), "
                         f"got {raw!r}") from None
    if n < 0 or math.isnan(n):
        raise ValueError(f"{what} for the {provider} language model must be zero or more, got {raw!r}")
    return n


def is_local(provider: str) -> bool:
    from folivox.llm import LLMS

    try:
        return LLMS.get(provider).local
    except Exception:  # noqa: BLE001 — an unknown name: not ours to call free
        return False


def price_of(provider: str, model: str) -> Price | None:
    """The price of ``model`` through ``provider``: free on this computer, else ``price_in`` / ``price_out`` from
    ``[llm.<provider>]`` in config.toml, else the list price of a well-known model; None when it is not known."""
    from folivox import config

    if is_local(provider):
        return FREE
    cfg = config.section(f"llm.{provider}")
    known = _TABLE.get(_key(model))
    if cfg.get("price_in") is not None or cfg.get("price_out") is not None:
        if cfg.get("price_in") is None or cfg.get("price_out") is None:
            raise ValueError(f"[llm.{provider}] in config.toml: set both price_in and price_out "
                             "(USD per million tokens)")
        pin, pout = _number(cfg["price_in"], "price_in", provider), _number(cfg["price_out"], "price_out", provider)
        # cache rates: the list's for a known model, else the input price (errs high)
        return Price(pin, pout, known[2] if known else pin, known[3] if known else pin, "config")
    return Price(*known, "list") if known else None


# model names a provider is known to serve, for the studio's model list (the configured model always comes first)
FAMILIES = {"anthropic": "claude-", "openai": "gpt-"}


def known_models(provider: str) -> list[str]:
    prefix = FAMILIES.get(provider)
    return [m for m in PRICES if prefix and m.startswith(prefix)]


def default_model(provider: str) -> str:
    """The model a provider uses when none is chosen: ``model`` in its config section, else its own default."""
    import inspect

    from folivox import config
    from folivox.llm import LLMS

    if model := config.section(f"llm.{provider}").get("model"):
        return str(model)
    try:
        p = LLMS.get(provider)
        module, _, attr = p.target.partition(":")
        import importlib

        factory = getattr(importlib.import_module(module), attr)
        default = inspect.signature(factory).parameters["model"].default
        return default if isinstance(default, str) else ""
    except Exception:  # noqa: BLE001 — a plugin without a model parameter
        return ""


# -- the spending cap -----------------------------------------------------------------------------------------------

class Budget:
    """Counts what a run's calls cost (from each reply's usage) and stops before the call that would pass ``max_cost``.

    No cap by default. A cap needs a known price: without one the run refuses to start rather than guess."""

    def __init__(self, price: Price | None, max_cost: float | None = None,
                 hint: str = "raise --max-cost and run the same command again to continue"):
        if max_cost is not None and price is None:
            raise ValueError("A spending cap needs the model's price: set price_in and price_out (USD per million "
                             "tokens) in the [llm.<provider>] section of config.toml")
        if max_cost is not None and max_cost <= 0:
            raise ValueError("The spending cap must be more than zero")
        self.price, self.max_cost, self.hint = price, max_cost, hint
        self.spent = Usage()
        self.chars = 0  # prompt characters of the calls whose usage was reported
        self.longest = 0  # output tokens of the longest reply so far

    @property
    def usd(self) -> float | None:
        return self.price.usd(self.spent) if self.price else None

    def _next(self, chars: int) -> Usage:
        """A guess of the next call: its prompt at this run's tokens per character and a reply as long as the longest
        so far (before the first reply: FIRST_OUTPUT_SHARE of the prompt)."""
        per = self.spent.prompt / self.chars if self.chars and self.spent.prompt else 1 / CHARS_PER_TOKEN
        prompt = math.ceil(chars * per)
        out = self.longest if self.spent.calls else math.ceil(prompt * FIRST_OUTPUT_SHARE)
        return Usage(1, prompt, out)

    def check(self, chars: int) -> None:
        """Raise :class:`~folivox.markup_llm.CostCapReached` when sending ``chars`` characters would pass the cap."""
        if self.max_cost is None or self.price is None:
            return
        nxt = self._next(chars)
        after = self.price.usd(self.spent) + self.price.usd(nxt)
        if after > self.max_cost:
            from folivox.markup_llm import CostCapReached

            raise CostCapReached(f"Spending cap ${self.max_cost:.2f} reached (≈ ${self.usd or 0:.2f} spent, "
                                 f"{self.spent.tokens:,} tokens in {self.spent.calls} calls). Progress is saved: "
                                 f"{self.hint}.")

    def record(self, usage: dict | None, chars: int, reply_chars: int) -> None:
        """Count one reply: the usage it reported, or (when the server said nothing) its size in characters."""
        if not usage:
            usage = {"input": math.ceil(chars / CHARS_PER_TOKEN),
                     "output": math.ceil(reply_chars / CHARS_PER_TOKEN * THINKING_LIKELY)}
            self.spent.estimated += 1
        else:
            self.chars += chars
        self.spent.add(usage)
        self.longest = max(self.longest, int(usage.get("output") or 0))

    def summary(self) -> str:
        u = self.spent
        if not u.calls:
            return "No paid calls were made (every answer came from the saved replies)."
        cached = f", {u.cache_read:,} of them read from the provider's cache" if u.cache_read else ""
        rough = f" ({u.estimated} replies reported no usage and were counted by size)" if u.estimated else ""
        cost = f"≈ ${self.usd:.2f}" if self.usd is not None else "price not known"
        return (f"Spent {cost}: {u.prompt:,} prompt tokens{cached} and {u.output:,} output tokens in {u.calls} "
                f"calls{rough}.")


# -- the estimate ---------------------------------------------------------------------------------------------------

@dataclass
class Estimate:
    analyse_calls: int = 0
    annotate_calls: int = 0
    extra_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0  # likely
    output_tokens_max: int = 0  # if the model reasons at length before every answer
    price: Price | None = None
    model: str = ""
    provider: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def calls(self) -> int:
        return self.analyse_calls + self.annotate_calls + self.extra_calls

    @property
    def usd(self) -> float | None:
        return self.price.usd(Usage(input=self.input_tokens, output=self.output_tokens)) if self.price else None

    @property
    def usd_max(self) -> float | None:
        return self.price.usd(Usage(input=self.input_tokens, output=self.output_tokens_max)) if self.price else None

    def to_dict(self) -> dict:
        return {"provider": self.provider, "model": self.model, "calls": self.calls,
                "analyse_calls": self.analyse_calls, "annotate_calls": self.annotate_calls,
                "extra_calls": self.extra_calls, "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens, "output_tokens_max": self.output_tokens_max,
                "free": bool(self.price and self.price.free), "price_known": self.price is not None,
                "price_source": self.price.source if self.price else "",
                "usd": None if self.usd is None else round(self.usd, 4),
                "usd_max": None if self.usd_max is None else round(self.usd_max, 4), "approximate": True}


def _stand_ins(bible):
    """The bible to size the character lists with: the real one, or KNOWN_GUESS made-up characters before analysis."""
    from folivox import markup_llm as ml

    if bible.characters:
        return bible
    fake = ml.Bible(lang=bible.lang)
    for i in range(KNOWN_GUESS):
        fake.characters.append(ml.Character(name=f"Character {i + 1}", aliases=["Alias"], gender="f", age="adult",
                                            traits="short, formal sentences", lines=3, id=i + 1))
    return fake


def _sample(wins: list, n: int | None) -> list:
    if n and len(wins) > n:  # the spread sample the passes take
        step = len(wins) / n
        return [wins[int(i * step)] for i in range(n)]
    return wins


def analyse_windows(chapters: list, bible, max_windows: int | None = None) -> list:
    """The windows :func:`folivox.markup_llm.analyse` would still send (it resumes where it stopped)."""
    from folivox import markup_llm as ml

    wins = ml.windows(chapters, ml.WINDOW_CHARS, only_speech=False)
    full_done = not bible.sampled and 0 < bible.total_windows <= bible.done_windows
    sampled = bool(max_windows and len(wins) > max_windows and not full_done)
    wins = _sample(wins, max_windows) if sampled else wins
    start = bible.done_windows if bible.sampled == sampled else 0  # as analyse(): a preview and a full run differ
    return wins[start:]


def annotate_windows(chapters: list, chapter_filter: set[int] | None = None, max_windows: int | None = None) -> list:
    from folivox import markup_llm as ml

    wins = [w for w in ml.windows(chapters, ml.WINDOW_CHARS, only_speech=True)
            if chapter_filter is None or w.chapter in chapter_filter]
    return _sample(wins, max_windows)


def _parts_reply(blocks: list[tuple[int, str]]) -> int:
    """Characters of a marking answer: one ``{"b": n, "who": {...}}`` per paragraph with speech (half its parts)."""
    from folivox import markup_llm as ml

    n = 60  # the empty new_characters / updates lists
    for _b, t in blocks:
        if ml.SPEECH_MARKS.search(t):
            n += 16 + 9 * math.ceil(len(ml.split_parts(t)) / 2)
    return n


def estimate(chapters: list, lang: str, bible, price: Price | None, *, analyse: list | None = None,
             annotate: list | None = None, extra_calls: int = 0, expression_tags: bool = False,
             method: str | None = None, model: str = "", provider: str = "") -> Estimate:
    """Tokens and USD of a run that sends the ``analyse`` and ``annotate`` windows (from :func:`analyse_windows` /
    :func:`annotate_windows`) and ``extra_calls`` short second-look calls, built as the passes build them."""
    import os

    from folivox import markup_llm as ml

    est = Estimate(price=price, model=model, provider=provider)
    sized = _stand_ins(bible)
    person = bible.narration or "third"
    in_chars = reply_chars = 0
    for w in analyse or []:
        text = w.text()
        prompt = ml.ANALYSE_PROMPT.format(lang=ml.LANG_NAMES.get(lang, lang), hints=ml.QUOTE_HINTS,
                                          known=ml._known(sized, ml.relevant_for(sized, text, [])), text=text)
        in_chars += len(ml.ANALYSE_SYSTEM) + len(prompt)
        reply_chars += int(len(text) * ANALYSE_REPLY_SHARE)
        est.analyse_calls += 1
    method = method or os.environ.get("FOLIVOX_MARKUP_METHOD", "parts")
    by_parts = method == "parts" and not expression_tags
    others = ml.other_languages(chapters, lang) if by_parts and annotate else []
    tag_list = ml._tag_list() if expression_tags else ""
    mark_in = mark_reply = 0
    prev: list[str] = []
    for w in annotate or []:
        text = w.text()
        speakers = ml._speakers_json(sized, ml.relevant_for(sized, text, []))
        recent = ("Context, the previous paragraphs and who spoke them:\n" + "\n".join(prev[-4:]) + "\n"
                  if prev else "")
        if by_parts:
            prompt = ml.parts_prompt(lang, others, hints=ml.QUOTE_HINTS, person=person, speakers=speakers,
                                     recent=recent, text=ml.parts_text(w.blocks))
            system, reply = ml.PARTS_SYSTEM, _parts_reply(w.blocks)
        else:
            prompt = ml.ANNOTATE_PROMPT.format(
                lang=ml.LANG_NAMES.get(lang, lang), hints=ml.QUOTE_HINTS, person=person, speakers=speakers,
                recent=recent, text=text,
                expression=ml.EXPRESSION_RULES.format(tags=tag_list) if expression_tags else "")
            system = ml.ANNOTATE_SYSTEM
            reply = 60 + sum(len(t) + 40 for _b, t in w.blocks if ml.SPEECH_MARKS.search(t))
        mark_in += len(system) + len(prompt)
        mark_reply += reply
        prev = [f"{t[:140]} -> Character" for _b, t in w.blocks if ml.SPEECH_MARKS.search(t)][-4:] or prev
        est.annotate_calls += 1
    in_chars += mark_in
    reply_chars += mark_reply
    if extra_calls:  # each about a quarter of an average marking call
        per_in = (mark_in / est.annotate_calls) if est.annotate_calls else ml.WINDOW_CHARS
        per_reply = (mark_reply / est.annotate_calls) if est.annotate_calls else 400
        in_chars += int(extra_calls * per_in * SHORT_CALL)
        reply_chars += int(extra_calls * per_reply * SHORT_CALL)
        est.extra_calls = extra_calls
    est.input_tokens = math.ceil(in_chars / CHARS_PER_TOKEN)
    reply_tokens = reply_chars / CHARS_PER_TOKEN
    est.output_tokens = math.ceil(reply_tokens * THINKING_LIKELY)
    est.output_tokens_max = math.ceil(reply_tokens * THINKING_MAX)
    return est


def describe(e: Estimate) -> str:
    """One line for the CLI: calls, tokens and USD (or why the USD is missing)."""
    calls = (f"{e.calls} calls ({e.analyse_calls} reading the characters, {e.annotate_calls} marking"
             + (f", {e.extra_calls} second looks" if e.extra_calls else "") + ")")
    tokens = (f"≈ {e.input_tokens:,} tokens in, {e.output_tokens:,} out "
              f"(up to {e.output_tokens_max:,} if the model reasons at length)")
    if e.price is not None and e.price.free:
        return f"{calls}, {tokens}: runs on this computer, free."
    if e.price is None:
        return (f"{calls}, {tokens}. The price of {e.model or 'this model'} is not known, so the cost is not "
                f"shown: set price_in and price_out (USD per million tokens) in [llm.{e.provider or '<provider>'}] in "
                "config.toml.")
    src = "your config prices" if e.price.source == "config" else f"approximate list prices read {PRICES_READ}"
    return f"{calls}, {tokens}: ≈ ${e.usd:.2f} (up to ${e.usd_max:.2f}). {src[0].upper() + src[1:]}."

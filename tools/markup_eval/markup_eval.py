#!/usr/bin/env python3
"""Speaker attribution of "folivox markup auto" measured against the Project Dialogism Novel Corpus (PDNC).

    python tools/markup_eval/markup_eval.py fetch                  # PDNC at a pinned commit (CC BY 4.0), ~9 MB
    python tools/markup_eval/markup_eval.py prepare                # cut the excerpts, the gold quotations
    python tools/markup_eval/markup_eval.py run haiku --llm anthropic --model claude-haiku-5-5
    python tools/markup_eval/markup_eval.py score haiku
    python tools/markup_eval/markup_eval.py report                 # every scored run, side by side

A run calls the real "folivox markup auto" (any provider and model, more options after "--") once per excerpt
through a small logging proxy on 127.0.0.1, which records the token usage of every reply (never the request
headers, so never a key) for the cost columns. Work files go to ./markup-eval (or --work / $FOLIVOX_EVAL_DIR).
README.md next to this file explains the method and its limits. Standard library and folivox only.
"""

from __future__ import annotations

import argparse
import ast
import collections
import csv
import http.client
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import threading
import time
import unicodedata
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "eval.json"
BOOK_CHARS = 500_000  # cost is reported per book of this many characters
TYPES = ("Explicit", "Anaphoric", "Implicit")


def load_config(path: Path = CONFIG) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# -- fetch and prepare -----------------------------------------------------------------------------------------------

def fetch(work: Path, cfg: dict) -> Path:
    """The configured novels' three PDNC files, from the pinned commit's archive, into WORK/pdnc/<Novel>/."""
    repo, commit = cfg["pdnc"]["repo"], cfg["pdnc"]["commit"]
    url = f"https://codeload.github.com/{repo}/tar.gz/{commit}"
    want = {(n, f) for n in cfg["excerpts"] for f in ("novel_text.txt", "quotation_info.csv", "character_info.csv")}
    out = work / "pdnc"
    req = urllib.request.Request(url, headers={"User-Agent": "folivox-markup-eval"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for m in tar.getmembers():
            parts = m.name.split("/")
            # <repo>-<commit>/data/<Novel>/<file>: only the files named above, never a link or an odd path
            if not m.isfile() or len(parts) != 4 or parts[1] != "data" or (parts[2], parts[3]) not in want:
                continue
            dest = out / parts[2] / parts[3]
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(tar.extractfile(m).read())
            want.discard((parts[2], parts[3]))
    if want:
        raise SystemExit(f"not in the PDNC archive: {sorted(want)}")
    return out


def cut_excerpt(text: str, start: int, end: int, restore_apostrophes: bool = False) -> str:
    """One excerpt as Folivox should read it: hard-wrapped lines joined into paragraphs (a blank line between),
    chapter headings on lines of their own, and (where PDNC's copy lost them) apostrophes put back."""
    ex = re.sub(r"(?m)^[ \t]*CHAPTER[ \t]*\n[ \t]*(\d+)[ \t]*$", r"CHAPTER \1", text[start:end])
    paras: list[str] = []
    for p in re.split(r"\n[ \t]*\n", ex.strip()):
        lines = [x.strip() for x in p.split("\n") if x.strip()]
        if not lines:
            continue
        if re.match(r"^(CHAPTER|Chapter)\b", lines[0]):
            paras.extend(lines)
        else:
            paras.append(" ".join(lines))
    body = "\n\n".join(paras) + "\n"
    if restore_apostrophes:  # "don t" -> "don't"
        body = re.sub(r"(?<=[A-Za-z]) (?=(?:s|t|ll|ve|re|d|m)\b)", "'", body)
    return body


def gold_for(novel_dir: Path, start: int, end: int) -> dict:
    """The quotations that lie inside the excerpt and the novel's characters, from PDNC's CSV files."""
    quotes = []
    with open(novel_dir / "quotation_info.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            spans = ast.literal_eval(r["quoteByteSpans"])
            if spans[0][0] >= start and spans[-1][1] <= end:
                quotes.append({"id": r["quoteID"], "subs": ast.literal_eval(r["subQuotationList"]),
                               "speaker": r["speaker"], "type": r["quoteType"], "text": r["quoteText"]})
    chars = []
    with open(novel_dir / "character_info.csv", encoding="utf-8") as f:
        for c in csv.DictReader(f):
            raw = c["Aliases"].strip()
            try:
                aliases = list(ast.literal_eval(raw))
            except (ValueError, SyntaxError):
                aliases = [raw]
            chars.append({"name": c["Main Name"], "aliases": aliases, "gender": c["Gender"]})
    return {"quotes": quotes, "characters": chars}


def prepare(work: Path, cfg: dict, pdnc: Path | None = None) -> dict:
    """WORK/books/<Novel>.txt, gold.json (quotations, characters) and blocks.json (Folivox's own paragraphs)."""
    from folivox.readers import read_book

    pdnc = pdnc or work / "pdnc"
    books = work / "books"
    books.mkdir(parents=True, exist_ok=True)
    gold, blocks = {}, {}
    for novel, ex in cfg["excerpts"].items():
        text = (pdnc / novel / "novel_text.txt").read_text(encoding="utf-8")
        path = books / f"{novel}.txt"
        path.write_text(cut_excerpt(text, ex["start"], ex["end"], ex.get("restore_apostrophes", False)),
                        encoding="utf-8")
        gold[novel] = {"title": ex.get("title", novel), **gold_for(pdnc / novel, ex["start"], ex["end"])}
        chapters = read_book(path).chapters(2)
        blocks[novel] = [[b.text for b in ch.blocks] for ch in chapters]
        n = len(gold[novel]["quotes"])
        print(f"{novel}: {path.stat().st_size:,} bytes, {len(chapters)} chapters, {n} quotations "
              f"{dict(collections.Counter(q['type'] for q in gold[novel]['quotes']))}")
    (books / "gold.json").write_text(json.dumps(gold, ensure_ascii=False, indent=1), encoding="utf-8")
    (books / "blocks.json").write_text(json.dumps(blocks, ensure_ascii=False), encoding="utf-8")
    return gold


# -- the logging proxy -----------------------------------------------------------------------------------------------

def usage_of(kind: str, reply: dict) -> dict | None:
    """Token usage of one reply in one shape: input (uncached), output (reasoning included), cache_read,
    cache_write, reasoning."""
    if kind == "ollama":
        if "eval_count" not in reply:
            return None
        return {"input": reply.get("prompt_eval_count") or 0, "output": reply.get("eval_count") or 0,
                "cache_read": 0, "cache_write": 0, "reasoning": 0}
    u = reply.get("usage")
    if not isinstance(u, dict):
        return None
    if kind == "anthropic":
        return {"input": u.get("input_tokens") or 0, "output": u.get("output_tokens") or 0,
                "cache_read": u.get("cache_read_input_tokens") or 0,
                "cache_write": u.get("cache_creation_input_tokens") or 0, "reasoning": 0}
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    return {"input": (u.get("prompt_tokens") or 0) - cached, "output": u.get("completion_tokens") or 0,
            "cache_read": cached, "cache_write": 0,
            "reasoning": (u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0}


class Proxy:
    """Forwards /<kind>/<book>/<rest> to the upstream of <kind> (openai, ollama, anthropic) and appends one JSON line
    per reply to the log. Headers pass through untouched and are never written down."""

    HOP = {"host", "content-length", "connection", "accept-encoding", "transfer-encoding"}

    def __init__(self, upstreams: dict[str, str], log: Path):
        self.upstreams, self.log, self.lock = upstreams, log, threading.Lock()
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_GET(self):
                proxy.forward(self, "GET")

            def do_POST(self):
                proxy.forward(self, "POST")

            def do_DELETE(self):
                proxy.forward(self, "DELETE")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def url(self, kind: str, book: str) -> str:
        return f"http://127.0.0.1:{self.port}/{kind}/{book}"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def forward(self, h: BaseHTTPRequestHandler, method: str) -> None:
        parts = h.path.split("/", 3)
        if len(parts) < 4 or parts[1] not in self.upstreams:
            h.send_response(404)
            h.send_header("Content-Length", "0")
            h.end_headers()
            return
        kind, book, rest = parts[1], parts[2], "/" + parts[3]
        n = int(h.headers.get("Content-Length") or 0)
        body = h.rfile.read(n) if n else None
        headers = {k: v for k, v in h.headers.items() if k.lower() not in self.HOP}
        req = urllib.request.Request(self.upstreams[kind].rstrip("/") + rest, data=body, method=method,
                                     headers=headers)
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                status, data, rh = r.status, r.read(), r.headers
        except urllib.error.HTTPError as e:
            status, data, rh = e.code, e.read(), e.headers
        except (OSError, http.client.HTTPException) as e:
            status, data, rh = 502, json.dumps({"error": f"proxy: {type(e).__name__}"}).encode(), None
        if method == "POST" and body:
            rec = {"t": round(t0, 3), "dt": round(time.time() - t0, 3), "kind": kind, "book": book,
                   "path": rest, "status": status}
            try:
                rec["model"] = json.loads(body).get("model")
                rec["usage"] = usage_of(kind, json.loads(data)) if status == 200 else None
            except (ValueError, AttributeError):
                pass
            with self.lock, open(self.log, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        h.send_response(status)
        for k in ("Content-Type", "Retry-After", "request-id"):
            if rh is not None and rh.get(k):
                h.send_header(k, rh.get(k))
        h.send_header("Content-Length", str(len(data)))
        h.end_headers()
        h.wfile.write(data)


# -- run -------------------------------------------------------------------------------------------------------------

def upstreams(base_url: str | None = None) -> dict[str, str]:
    """Where the proxy sends each kind of request: the configured OpenAI-compatible endpoint, Ollama, Anthropic."""
    from folivox import config

    return {"openai": base_url or config.section("llm.openai").get("base_url") or "https://api.openai.com/v1",
            "ollama": os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
            "anthropic": os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")}


def run_config(user: dict, proxy: Proxy, book: str) -> str:
    """config.toml for one excerpt: the user's own, with the OpenAI-compatible endpoint pointed at the proxy."""
    import tomli_w

    data = json.loads(json.dumps(user))  # a deep copy
    data.setdefault("llm", {}).setdefault("openai", {})["base_url"] = proxy.url("openai", book)
    return tomli_w.dumps(data)


def folivox_command(book: Path, llm: str, model: str, best: bool, extra: list[str]) -> list[str]:
    cmd = [sys.executable, "-c", "from folivox.cli import main; main()", "markup", "auto", str(book),
           "--llm", llm, "--model", model]
    return cmd + (["--best"] if best else []) + list(extra)


def run(work: Path, cfg: dict, label: str, llm: str, model: str, best: bool = False, books: list[str] | None = None,
        extra: list[str] | None = None, base_url: str | None = None) -> Path:
    from folivox import config

    rundir = work / "runs" / label
    rundir.mkdir(parents=True, exist_ok=True)
    meta = {"label": label, "llm": llm, "model": model, "best": best, "args": extra or []}
    (rundir / "run.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    proxy = Proxy(upstreams(base_url), rundir / "calls.jsonl")
    user = dict(config.load())
    try:
        for novel in books or list(cfg["excerpts"]):
            src = work / "books" / f"{novel}.txt"
            if not src.is_file():
                raise SystemExit(f"{src} is missing: run prepare first")
            book = rundir / f"{novel}.txt"
            book.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            cfgfile = rundir / f"config.{novel}.toml"
            cfgfile.write_text(run_config(user, proxy, novel), encoding="utf-8")
            env = {**os.environ, "FOLIVOX_CONFIG": str(cfgfile), "OLLAMA_HOST": proxy.url("ollama", novel),
                   "ANTHROPIC_BASE_URL": proxy.url("anthropic", novel), "COLUMNS": "200"}
            t0 = time.time()
            with open(rundir / f"{novel}.log", "w", encoding="utf-8") as log:
                rc = subprocess.run(folivox_command(book, llm, model, best, extra or []), env=env, stdout=log,
                                    stderr=subprocess.STDOUT).returncode
            secs = time.time() - t0
            (rundir / f"{novel}.time.json").write_text(json.dumps({"seconds": round(secs, 1), "rc": rc}),
                                                       encoding="utf-8")
            print(f"{label} {novel}: exit {rc}, {secs:.0f} s")
    finally:
        proxy.close()
    return rundir


# -- score -----------------------------------------------------------------------------------------------------------

HONORIFICS = {"mr", "mrs", "miss", "ms", "dr", "doctor", "lady", "lord", "sir", "madame", "mademoiselle", "monsieur",
              "mme", "mlle", "the", "my", "dear", "dearest", "count", "colonel", "captain", "inspector", "master",
              "mister"}
QMAP = str.maketrans({"“": '"', "”": '"', "„": '"', "‘": "'", "’": "'", "—": "-",
                      "–": "-", " ": " ", "'": " "})


def nname(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.casefold())
    return " ".join("".join(c if c.isalnum() else " " for c in s).split())


def core_tokens(s: str) -> list[str]:
    return [t for t in nname(s).split() if t not in HONORIFICS]


def norm_with_map(s: str) -> tuple[str, list[int]]:
    """The text with quotes, dashes and apostrophes unified and runs of whitespace as one space, and for each of its
    characters the index of the original one."""
    out, idx, prev_space = [], [], True
    for i, c in enumerate(s.translate(QMAP)):
        if c.isspace():
            if not prev_space:
                out.append(" ")
                idx.append(i)
            prev_space = True
        else:
            out.append(c)
            idx.append(i)
            prev_space = False
    return "".join(out), idx


class Mapper:
    """A Folivox speaker name to a PDNC character: the name itself, then the aliases Folivox found, then a first or
    last name that only one character has."""

    def __init__(self, chars: list[dict], same_person: dict | None = None, extra_aliases: dict | None = None):
        extra = extra_aliases or {}
        self.same = same_person or {}
        self.chars = [{**c, "aliases": list(c["aliases"]) + extra.get(c["name"], [])} for c in chars]
        self.forms: dict[str, set[str]] = {}
        for c in self.chars:
            for a in [c["name"], *c["aliases"]]:
                self.forms.setdefault(nname(a), set()).add(c["name"])

    def map(self, name: str, aliases=()) -> tuple[str | None, str]:
        m, how = self._map(name, aliases)
        return self.same.get(m, m), how

    def _map(self, name: str, aliases=()) -> tuple[str | None, str]:
        n = nname(name)
        if n in self.forms and len({self.same.get(x, x) for x in self.forms[n]}) == 1:
            return next(iter(self.forms[n])), "exact"
        votes: collections.Counter = collections.Counter()
        for a in aliases:
            f = self.forms.get(nname(a))
            if f and len(f) == 1:
                votes[next(iter(f))] += 1
        if votes:
            return votes.most_common(1)[0][0], "alias"
        ct = core_tokens(re.sub(r"\(.*?\)", " ", name))
        if ct and "'s" not in name and "’s" not in name:
            for part, how in ((slice(0, 1), "first-token"), (slice(-1, None), "last-token")):
                hits = {c["name"] for c in self.chars for a in [c["name"], *c["aliases"]]
                        if core_tokens(a)[part] == ct[part]}
                if len(hits) == 1:
                    return hits.pop(), how
        return None, "none"


def score_book(blocks: list[list[str]], gold: dict, markup: dict, bible: dict, cfg: dict) -> dict:
    """Each gold quotation found in Folivox's paragraphs (in order); the speaker Folivox gave most of its letters is
    the prediction; a name that maps to nobody, or the narrator, is wrong."""
    parts, spk = [], []
    for ci, ch in enumerate(blocks):
        spans = markup.get("chapters", {}).get(str(ci), {}).get("spans", [])
        for bi, t in enumerate(ch):
            who = [0] * len(t)
            for s in spans:
                if s["b"] == bi:
                    for k in range(s["s"], min(s["e"], len(t))):
                        who[k] = s["sp"]
            parts += [t, "\n"]
            spk += [*who, -1]
    text, idx = norm_with_map("".join(parts))
    mapper = Mapper(gold["characters"], cfg.get("pdnc_same_person"), cfg.get("extra_aliases"))
    by_id = {c["id"]: c for c in bible.get("characters", []) if c.get("id")}
    names = {}
    for s in markup.get("speakers", []):
        if s["id"]:
            c = by_id.get(s["id"])
            names[s["id"]] = (s["name"], *mapper.map(s["name"], (c or s).get("aliases", []) or []))
    cursor, out = 0, []
    for q in gold["quotes"]:
        counts: collections.Counter = collections.Counter()
        found = 0
        for sub in q["subs"]:
            ns = norm_with_map(sub)[0].strip()
            if not ns:
                continue
            p, size = -1, len(ns)
            for probe in (ns, ns[:40]):  # a long quotation may differ near its end (a footnote, a hyphen)
                size = len(probe)
                p = text.find(probe, cursor)
                if p < 0:
                    p = text.find(probe, max(0, cursor - 2000))
                if p >= 0:
                    break
            if p < 0:
                continue
            found += 1
            cursor = p + size
            for k in range(p, p + size):
                if text[k].isalnum():
                    counts[spk[idx[k]]] += 1
        if not found:
            out.append({"id": q["id"], "type": q["type"], "gold": q["speaker"], "aligned": False})
            continue
        sid = counts.most_common(1)[0][0]
        name, pred, _how = ("Narrator", None, "narrator") if sid == 0 else names.get(sid, (f"?{sid}", None, "?"))
        out.append({"id": q["id"], "type": q["type"], "gold": q["speaker"], "aligned": True, "sid": sid,
                    "name": name, "pred": pred, "correct": pred == q["speaker"]})
    speakers = {q["speaker"] for q in gold["quotes"]}
    listed = [c for c in bible.get("characters", []) if c.get("id") and c.get("lines", 0) > 0]
    claimed: set[str] = set()
    for c in listed:
        m, _ = mapper.map(c["name"], c.get("aliases", []))
        if m in speakers and m not in claimed:
            claimed.add(m)
    return {"quotes": out, "listed": len(listed), "true_entries": len(claimed), "speakers": sorted(speakers),
            "missed": sorted(speakers - claimed)}


def summarize(quotes: list[dict], first_person: dict | None = None) -> dict:
    """Accuracy over the aligned quotations, overall and by PDNC quote type. Lenient also counts as right a
    first-person narrator's own lines left to the narrator voice (fine in audio); first_person maps book -> name."""
    al = [r for r in quotes if r["aligned"]]
    fp = first_person or {}

    def acc(rs):
        return sum(r["correct"] for r in rs) / len(rs) if rs else None

    out = {"quotes": len(quotes), "aligned": len(al), "accuracy": acc(al),
           "unattributed": sum(1 for r in al if r["sid"] == 0)}
    for t in TYPES:
        out[t.lower()] = acc([r for r in al if r["type"] == t])
    out["lenient"] = (sum(r["correct"] or (r["sid"] == 0 and r["gold"] == fp.get(r.get("book"))) for r in al)
                      / len(al)) if al else None
    return out


def score(work: Path, cfg: dict, label: str) -> dict:
    rundir = work / "runs" / label
    blocks = json.loads((work / "books" / "blocks.json").read_text(encoding="utf-8"))
    gold = json.loads((work / "books" / "gold.json").read_text(encoding="utf-8"))
    fpn = cfg.get("first_person_narrator", {})
    books, allq = {}, []
    listed = true = speakers = found = 0
    for novel in gold:
        mp, bp = rundir / f"{novel}.markup.json", rundir / f"{novel}.characters.json"
        if not mp.is_file() or not bp.is_file():
            continue
        s = score_book(blocks[novel], gold[novel], json.loads(mp.read_text(encoding="utf-8")),
                       json.loads(bp.read_text(encoding="utf-8")), cfg)
        quotes = [{**q, "book": novel} for q in s["quotes"]]
        s["summary"] = summarize(quotes, fpn)
        books[novel] = s
        allq += quotes
        listed, true = listed + s["listed"], true + s["true_entries"]
        speakers, found = speakers + len(s["speakers"]), found + len(s["speakers"]) - len(s["missed"])
    total = summarize(allq, fpn)
    total["char_precision"] = true / listed if listed else None
    total["char_recall"] = found / speakers if speakers else None
    result = {"total": total, "books": books}
    (rundir / "score.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return result


# -- report ----------------------------------------------------------------------------------------------------------

def spend(rundir: Path, prices: dict) -> dict:
    """Tokens and USD of a run from the proxy's log (USD only for a model with a price in eval.json)."""
    tot: collections.Counter = collections.Counter()
    calls = errors = 0
    model = None
    log = rundir / "calls.jsonl"
    for line in log.read_text(encoding="utf-8").splitlines() if log.is_file() else []:
        r = json.loads(line)
        if not r.get("model"):
            continue
        calls += 1
        model = model or r["model"]
        if r.get("status") != 200:
            errors += 1
        for k, v in (r.get("usage") or {}).items():
            tot[k] += v
    p = prices.get(model or "")
    usd = None
    if p:
        usd = (tot["input"] * p["in"] + tot["output"] * p["out"] + tot["cache_read"] * p.get("cache_read", p["in"])
               + tot["cache_write"] * p.get("cache_write", p["in"])) / 1e6
    return {"calls": calls, "http_errors": errors, "model": model, **dict(tot), "usd": usd}


def report(work: Path, cfg: dict, labels: list[str] | None = None) -> list[dict]:
    gold = json.loads((work / "books" / "gold.json").read_text(encoding="utf-8"))
    chars = sum(len((work / "books" / f"{n}.txt").read_text(encoding="utf-8")) for n in gold)
    rows = []
    for rundir in sorted((work / "runs").iterdir()) if (work / "runs").is_dir() else []:
        if (labels and rundir.name not in labels) or not (rundir / "score.json").is_file():
            continue
        t = json.loads((rundir / "score.json").read_text(encoding="utf-8"))["total"]
        meta = json.loads((rundir / "run.json").read_text(encoding="utf-8")) if (rundir / "run.json").is_file() else {}
        secs = sum(json.loads(f.read_text(encoding="utf-8"))["seconds"] for f in rundir.glob("*.time.json"))
        s = spend(rundir, cfg.get("prices", {}))
        rows.append({"label": rundir.name, "model": meta.get("model", s["model"]), **t, "seconds": secs,
                     "secs_per_10k": secs / chars * 1e4, "calls": s["calls"], "http_errors": s["http_errors"],
                     **{f"tokens_{k}": s.get(k, 0) for k in ("input", "output", "cache_read", "cache_write")},
                     "usd": s["usd"], "usd_per_500k": None if s["usd"] is None else s["usd"] * BOOK_CHARS / chars})
    return rows


def _pct(x) -> str:
    return "-" if x is None else f"{100 * x:.1f}"


def print_report(rows: list[dict]) -> None:
    print("| Run | Model | Accuracy | Lenient | Explicit | Anaphoric | Implicit | Characters P/R | s per 10k chars "
          "| Tokens in / out | Cached in | USD per 500k chars |")
    print("|" + "---|" * 12)
    for r in rows:
        prompt = r["tokens_input"] + r["tokens_cache_read"] + r["tokens_cache_write"]
        usd = "-" if r["usd_per_500k"] is None else "$" + f"{r['usd_per_500k']:.2f}"
        cells = [r["label"], str(r["model"]), _pct(r["accuracy"]), _pct(r["lenient"]), _pct(r["explicit"]),
                 _pct(r["anaphoric"]), _pct(r["implicit"]), f"{_pct(r['char_precision'])} / {_pct(r['char_recall'])}",
                 f"{r['secs_per_10k']:.0f}", f"{prompt:,} / {r['tokens_output']:,}", f"{r['tokens_cache_read']:,}", usd]
        print("| " + " | ".join(cells) + " |")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Speaker attribution of folivox markup auto against PDNC.")
    ap.add_argument("--work", type=Path, default=Path(os.environ.get("FOLIVOX_EVAL_DIR", "markup-eval")),
                    help="work directory (default ./markup-eval or $FOLIVOX_EVAL_DIR)")
    ap.add_argument("--config", type=Path, default=CONFIG, help="excerpts, aliases and prices (eval.json)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch", help="download the PDNC files at the pinned commit")
    p = sub.add_parser("prepare", help="cut the excerpts and the gold quotations")
    p.add_argument("--pdnc", type=Path, help="an existing PDNC data directory (default WORK/pdnc)")
    p = sub.add_parser("run", help="run folivox markup auto over every excerpt")
    p.add_argument("label")
    p.add_argument("--llm", required=True, help="ollama | openai | anthropic")
    p.add_argument("--model", required=True)
    p.add_argument("--best", action="store_true", help="folivox's --best mode")
    p.add_argument("--books", help="comma-separated excerpt names (default: all)")
    p.add_argument("--base-url", help="the OpenAI-compatible endpoint (default: [llm.openai] base_url)")
    p.add_argument("extra", nargs="*", help="more folivox markup auto options, after --")
    p = sub.add_parser("score", help="score runs against the gold quotations")
    p.add_argument("labels", nargs="+")
    p = sub.add_parser("report", help="accuracy, time, tokens and cost of the scored runs")
    p.add_argument("labels", nargs="*")
    p.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    a.work.mkdir(parents=True, exist_ok=True)
    if a.cmd == "fetch":
        print(f"PDNC ({cfg['pdnc']['licence']}) in {fetch(a.work, cfg)}")
    elif a.cmd == "prepare":
        prepare(a.work, cfg, a.pdnc)
    elif a.cmd == "run":
        run(a.work, cfg, a.label, a.llm, a.model, a.best, a.books.split(",") if a.books else None, a.extra,
            a.base_url)
    elif a.cmd == "score":
        for label in a.labels:
            t = score(a.work, cfg, label)["total"]
            print(label, json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in t.items()}))
    else:
        rows = report(a.work, cfg, a.labels or None)
        if a.json:
            print(json.dumps(rows, indent=1))
        else:
            print_report(rows)


if __name__ == "__main__":
    main()

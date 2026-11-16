"""Starlette app behind ``folivox ui``: a thin JSON/SSE layer over the same code the CLI uses."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import ipaddress
import json
import os
import re
import secrets
import shutil
import time
import urllib.parse
import uuid
from pathlib import Path

from platformdirs import user_data_dir
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from starlette.routing import Route

from folivox import config
from folivox import voices as catalog
from folivox.book import BlockKind
from folivox.pipeline import QA_LEVELS, RenderOptions
from folivox.text import pronounce as pronunciations
from folivox.ui import PRIVATE_NETS, fish_routes, lifecycle, markup_routes, runtime_routes
from folivox.ui.jobs import JobManager, chapter_files

STATIC = Path(__file__).parent / "static"
# Bumped when the page starts using a route that servers started before the update do not have: the page asks
# /api/ping and, if the server is older, tells the user to restart it.
API_LEVEL = 5
MAX_UPLOAD = 200 * 1024 * 1024
PREVIEW_CHARS = 600


class ApiError(Exception):
    def __init__(self, message: str, status: int = 400, **extra):
        super().__init__(message)
        self.status, self.extra = status, extra


class State:
    def __init__(self, data_dir: Path, render=None):
        self.data_dir = data_dir
        self.books: dict[str, dict] = {}
        self.manager = JobManager(render)
        self.gen_jobs: dict = {}

    def book(self, book_id: str) -> dict:
        """The uploaded book; re-read from disk after a server restart."""
        if book_id not in self.books:
            from folivox.readers import read_book

            folder = self.data_dir / "uploads" / Path(book_id).name
            files = [p for p in folder.iterdir() if p.is_file()] if folder.is_dir() else []
            if not files:
                raise ApiError("Unknown book; upload it again", 404)
            self.books[book_id] = {"book": read_book(files[0]), "path": files[0]}
        return self.books[book_id]

    @property
    def session_file(self) -> Path:
        return self.data_dir / "session.json"


def _chapters(book, depth: int) -> list[dict]:
    cps = book.chars / book.estimated_seconds if book.estimated_seconds else 14.0
    return [{"n": i, "title": ch.title, "chars": ch.chars, "seconds": ch.chars / cps}
            for i, ch in enumerate(book.chapters(depth), 1)]


def _book_summary(book_id: str, book, depth: int = 2, cover: str = "") -> dict:
    from folivox import lang as languages

    try:
        lang_name = languages.get(book.lang).native
    except ValueError:
        lang_name = book.lang
    return {"book_id": book_id, "title": book.title, "author": book.author, "lang": book.lang,
            "lang_name": lang_name, "format": book.source_format, "year": book.year, "has_cover": bool(book.cover),
            "cover": cover,
            "chars": book.chars, "estimated_seconds": book.estimated_seconds, "chapters": _chapters(book, depth),
            "warnings": list(getattr(book, "warnings", []) or [])}


_PEAKS: dict[tuple[str, float], dict] = {}


def _wave(path: Path, bars: int = 64) -> dict:
    """Waveform thumbnail of a voice sample: ``{peaks: [0..100]*bars, seconds}`` (empty without soundfile)."""
    try:
        key = (str(path), path.stat().st_mtime)
        if key not in _PEAKS:
            import numpy as np
            import soundfile as sf

            data, sr = sf.read(str(path), dtype="float32", always_2d=True)
            mono = np.abs(data).max(axis=1)
            chunks = np.array_split(mono, bars)
            peaks = np.array([c.max() if c.size else 0.0 for c in chunks])
            peaks = peaks / (peaks.max() or 1.0)
            _PEAKS[key] = {"peaks": [int(round(100 * p)) for p in peaks], "seconds": round(len(mono) / sr, 1)}
        return _PEAKS[key]
    except Exception:  # noqa: BLE001 — a thumbnail is a nicety
        return {"peaks": [], "seconds": 0}


def _voice_json(v) -> dict:
    return {**_wave(v.path), "id": v.id, "name": v.name, "gender": v.gender, "lang": v.lang, "license": v.license,
            "bundled": v.bundled, "commercial": v.commercial, "default": v.id == catalog.DEFAULT_VOICE,
            "transcript": v.transcript}


def _registries():
    from folivox.llm import LLMS
    from folivox.qa.asr import ASR_PROVIDERS
    from folivox.tts import ENGINES
    from folivox.voicegen import GENERATORS

    return {r.kind: r for r in (ENGINES, ASR_PROVIDERS, GENERATORS, LLMS)}


def create_app(token: str | None = None, port: int | None = None, data_dir: Path | None = None,
               render=None, services: dict | None = None, network: bool = False,
               allowed_hosts: set[str] | None = None, open_access: bool = False,
               allowed_nets: list | None = None) -> Starlette:
    """``services`` (tests): ``generator(name)``, ``llm(name)`` and ``asr(lang)`` factories replacing the real ones.

    ``network``: the server listens beyond this computer (Docker, a home server). It then refuses to run without a
    token, shows a login page instead of the studio until the token is entered, and accepts any ``Host`` (a LAN
    address or a proxy name) unless ``allowed_hosts`` (bare host names, lower case) lists the ones to accept.

    ``open_access`` (network only, no token): the owner chose no password. Clients are then limited to the private
    and local ``allowed_nets`` (the TCP peer, never X-Forwarded-For) and Host to IP literals and local-style names."""
    if open_access and (token or not network):
        raise ValueError("open access is for network mode without a token")
    if network and not token and not open_access:
        raise ValueError("network mode needs an access token")
    nets = allowed_nets or [ipaddress.ip_network(n) for n in PRIVATE_NETS]
    state = State(data_dir or Path(user_data_dir("folivox", appauthor=False)) / "ui", render)
    fish_install = fish_routes.Install()
    native_install = fish_routes.Install()
    state.installs = [fish_install, native_install]
    presence = lifecycle.Presence()
    services = services or {}
    # what the cookie holds: derived from the token, so it survives restarts but the token itself never sits in it
    session = (hmac.new(token.encode(), b"folivox-ui-cookie", "sha256").hexdigest() if token
               else secrets.token_urlsafe(24))
    cookie_name = f"folivox_session_{port or 0}"
    host_names = {h.lower().strip("[]") for h in allowed_hosts or ()}
    allowed_hosts = None if network or not port else {f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}"}

    # ---- handlers -------------------------------------------------------------------------------
    def same(given: str, expected: str) -> bool:
        """Constant-time comparison that also accepts non-ASCII input (``compare_digest`` raises on it for str)."""
        return secrets.compare_digest(given.encode(), expected.encode())

    guesses = _Throttle()  # network mode: wrong tokens in a row lock the token checks for a while

    def logged_in(request: Request) -> bool:
        return same(request.cookies.get(cookie_name) or "", session)

    def start_session(request: Request) -> RedirectResponse:
        resp = RedirectResponse("/", status_code=303)
        # behind an HTTPS reverse proxy the browser must never send the cookie over plain http
        secure = request.url.scheme == "https" or (network and request.headers.get("x-forwarded-proto") == "https")
        resp.set_cookie(cookie_name, session, httponly=True, samesite="strict", path="/", secure=secure,
                        max_age=(30 if network else 90) * 86400)
        return resp

    def token_ok(request: Request, given: str) -> bool | None:
        """Is ``given`` the token? None: too many wrong guesses, nothing is checked now (network mode only)."""
        if network and guesses.locked():
            return None
        if token and same(given, token):
            return True
        if network:
            guesses.fail()
        return False

    def too_many(html: bool = False):
        headers = {"Retry-After": str(_Throttle.LOCK)}
        if html:
            return HTMLResponse(LOGIN_PAGE.replace("<!--error-->", '<p class="err">Too many wrong tokens. '
                                                   'Wait a few minutes.</p>'), status_code=429, headers=headers)
        return JSONResponse({"error": "Too many wrong tokens, try again later"}, status_code=429, headers=headers)

    async def login(request: Request):
        """Network mode: trade the token typed into the login page for the session cookie."""
        ok = token_ok(request, str((await request.form()).get("token") or ""))
        if ok is None:
            return too_many(html=True)
        if not ok:
            await asyncio.sleep(1.0)  # slows down guessing
            return HTMLResponse(LOGIN_PAGE.replace("<!--error-->", '<p class="err">Wrong token.</p>'), status_code=401)
        return start_session(request)

    async def health(request: Request):
        """For container health checks: no token, and nothing but "the server answers"."""
        return JSONResponse({"ok": True})

    async def index(request: Request):
        given = request.query_params.get("token")
        if token and given is not None:  # the link printed by `folivox ui`: trade the token for a cookie, hide it
            ok = token_ok(request, given)
            if ok is None:
                return too_many()
            if not ok:
                return JSONResponse({"error": "Wrong token"}, status_code=401)
            return start_session(request)
        if network and not open_access and not logged_in(request):
            return HTMLResponse(LOGIN_PAGE.replace("<!--error-->", ""))
        return HTMLResponse((STATIC / "index.html").read_text(encoding="utf-8"))

    async def ping(request: Request):
        """Who is on this port: lets `folivox ui` tell its own studio from another program, and the page tell an
        outdated server from a current one."""
        import os

        from folivox import __version__

        return JSONResponse({"app": "folivox-studio", "api": API_LEVEL, "version": __version__, "pid": os.getpid(),
                             "busy": state.manager.busy() or bool(lifecycle.activity(state)["markup"])})

    async def get_activity(request: Request):
        """The long work the server is doing, so the page can warn before it closes and show it again on reopening."""
        return JSONResponse({**lifecycle.activity(state), "orphan_minutes": int(lifecycle.ORPHAN_SECONDS // 60)})

    async def changelog(request: Request):
        """Version history shown in the footer dialog (``static/changelog.json``, English only)."""
        path = STATIC / "changelog.json"
        entries = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        return JSONResponse(entries, headers={"Cache-Control": "no-cache"})

    async def shutdown(request: Request):
        """Stop this server (`folivox ui --restart` / `--stop`); a running job is not killed without ``force``."""
        from starlette.background import BackgroundTask

        body = await request.json() if request.headers.get("content-length", "0") != "0" else {}
        working = state.manager.busy() or bool(lifecycle.activity(state)["markup"])
        if working and not (isinstance(body, dict) and body.get("force")):
            raise ApiError("A render is running; stopping the studio would interrupt it", 409, busy=True)
        request_exit = getattr(app.state, "request_exit", None)
        if request_exit is None:
            raise ApiError("This server cannot be stopped from outside", 501)
        return JSONResponse({"stopping": True}, background=BackgroundTask(request_exit))

    async def i18n_catalog(request: Request):
        """UI message catalogs (static JSON next to index.html); the code is checked against the shipped files."""
        code = request.path_params["code"]
        path = STATIC / "i18n" / f"{code}.json"
        if not re.fullmatch(r"[a-z]{2}", code) or not path.is_file():
            return JSONResponse({"error": "Unknown language"}, status_code=404)
        return Response(path.read_text(encoding="utf-8"), media_type="application/json",
                        headers={"Cache-Control": "no-cache"})

    # ---- the library: the books uploaded before, so one can go back to them ----------------------------
    def library_file() -> Path:
        return state.data_dir / "library.json"

    def library_read() -> dict:
        try:
            d = json.loads(library_file().read_text(encoding="utf-8"))
            return d if isinstance(d, dict) else {}
        except (OSError, ValueError):
            return {}

    def library_note(book_id: str, book, path: Path) -> None:
        """Remember a book (title, author, language, size) and that it was used just now."""
        lib = library_read()
        lib[book_id] = {"title": book.title, "author": book.author, "lang": book.lang,
                        "chars": sum(len(b.text) for c in book.chapters(2) for b in c.blocks),
                        "name": path.name, "used": time.time(), "warnings": list(getattr(book, "warnings", []) or [])}
        state.data_dir.mkdir(parents=True, exist_ok=True)
        library_file().write_text(json.dumps(lib, ensure_ascii=False), encoding="utf-8")

    def markup_state(book_id: str) -> dict:
        """What the studio's markup file for this book holds, without reading the book."""
        f = state.data_dir / "markup" / f"{Path(book_id).name}.json"
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            spans = sum(len(c.get("spans", [])) for c in d.get("chapters", {}).values())
            return {"characters": sum(1 for x in d.get("speakers", []) if x.get("id")), "spans": spans}
        except (OSError, ValueError, AttributeError):
            return {"characters": 0, "spans": 0}

    async def list_books(request: Request):
        lib = library_read()
        out = []
        for bid in uploads_newest_first()[:60]:
            info = lib.get(bid)
            if info is None:  # an upload from before the library existed: read it once
                try:
                    entry = await asyncio.to_thread(state.book, bid)
                    library_note(bid, entry["book"], entry["path"])
                    lib = library_read()
                    info = lib.get(bid)
                    # keep the order: it was not used just now
                    info["used"] = (state.data_dir / "uploads" / bid).stat().st_mtime
                except Exception:  # noqa: BLE001 — an unreadable upload does not belong in the list
                    continue
            out.append({"book_id": bid, **info, "markup": markup_state(bid)})
        out.sort(key=lambda x: x.get("used", 0), reverse=True)
        return JSONResponse({"books": out})

    async def open_book(request: Request):
        """Go back to a book uploaded earlier."""
        bid = request.path_params["book_id"]
        entry = await asyncio.to_thread(state.book, bid)
        library_note(bid, entry["book"], entry["path"])
        write_session({"book_id": bid})
        return JSONResponse(_book_summary(bid, entry["book"], cover=cover_kind(bid, entry["book"])))

    async def remove_book(request: Request):
        """Forget a book: the uploaded copy and everything the studio keeps for it (markup, characters, pronunciations,
        details, cover choice, cached model replies). Finished audiobooks and the person's voices are not touched."""
        bid = Path(request.path_params["book_id"]).name
        cur = state.manager.current
        if cur is not None and cur.state == "running" and cur.book_id == bid:
            raise ApiError("A job is running for this book", 409)
        folder = state.data_dir / "uploads" / bid
        if not folder.is_dir():
            raise ApiError("Unknown book", 404)
        had = markup_state(bid)
        shutil.rmtree(folder, ignore_errors=True)
        for sub, pattern in (("markup", f"{bid}.json"), ("characters", f"{bid}.json"), ("meta", f"{bid}.json"),
                             ("pronounce", f"{bid}.txt"), ("llm-cache", f"{bid}.sqlite*"), ("covers", f"{bid}.*")):
            d = state.data_dir / sub
            for f in (d.glob(pattern) if d.is_dir() else []):
                f.unlink(missing_ok=True)
        state.books.pop(bid, None)
        lib = library_read()
        if lib.pop(bid, None) is not None:
            library_file().write_text(json.dumps(lib, ensure_ascii=False), encoding="utf-8")
        try:
            if json.loads(state.session_file.read_text(encoding="utf-8")).get("book_id") == bid:
                state.session_file.unlink(missing_ok=True)
        except (OSError, ValueError, AttributeError):
            pass
        return JSONResponse({"removed": bid, "had_markup": bool(had["spans"])})

    async def upload_book(request: Request):
        from folivox.readers import UnsupportedFormat, read_book

        form = await request.form()
        f = form.get("file")
        if f is None or not hasattr(f, "read"):
            raise ApiError("No file in the request")
        data = await f.read()
        if len(data) > MAX_UPLOAD:
            raise ApiError("File is too large", 413)
        # the id comes from the content: the same file again is the same book, with its markup and settings
        book_id = hashlib.sha256(data).hexdigest()[:12]
        folder = state.data_dir / "uploads" / book_id
        known = [p for p in folder.iterdir() if p.is_file()] if folder.is_dir() else []
        fresh = not known
        if fresh:
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / (Path(f.filename or "book").name or "book")
            path.write_bytes(data)
        else:
            path = known[0]
        try:
            book = await asyncio.to_thread(read_book, path)
        except UnsupportedFormat as e:
            if fresh:
                shutil.rmtree(folder, ignore_errors=True)
            raise ApiError(str(e), 415) from None
        except Exception as e:  # noqa: BLE001 — a broken book must not kill the server
            if fresh:
                shutil.rmtree(folder, ignore_errors=True)
            raise ApiError(f"Cannot read the book: {e}") from None
        state.books[book_id] = {"book": book, "path": path}
        library_note(book_id, book, path)
        write_session({"book_id": book_id})  # the last uploaded book survives a reload even before any other change
        return JSONResponse(_book_summary(book_id, book, cover=cover_kind(book_id, book)))

    async def paste_text(request: Request):
        """A book made of pasted text (an article, a note): stored as a .txt so it is read like any upload."""
        from folivox.readers import read_book

        body = await request.json()
        text = str(body.get("text") or "").strip() if isinstance(body, dict) else ""
        if not text:
            raise ApiError("The text is empty")
        if len(text.encode("utf-8")) > MAX_UPLOAD:
            raise ApiError("Text is too large", 413)
        title = " ".join(str(body.get("title") or "").split())
        if not title:
            first = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
            title = first[:60].rstrip()
        # the file name carries the title; " - " would be read as "author - title", and path characters are unsafe
        title = re.sub(r'[\\/:*?"<>|]', " ", title).replace(" - ", " – ").strip(" .") or "Pasted text"
        book_id = hashlib.sha256((title + "\n" + text).encode("utf-8")).hexdigest()[:12]
        folder = state.data_dir / "uploads" / book_id
        known = [p for p in folder.iterdir() if p.is_file()] if folder.is_dir() else []
        fresh = not known
        if fresh:
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{title[:80].strip(' .')}.txt"
            path.write_text(text, encoding="utf-8")
        else:
            path = known[0]
        try:
            book = await asyncio.to_thread(read_book, path)
        except Exception as e:  # noqa: BLE001 — a broken text must not kill the server
            if fresh:
                shutil.rmtree(folder, ignore_errors=True)
            raise ApiError(f"Cannot read the text: {e}") from None
        state.books[book_id] = {"book": book, "path": path}
        library_note(book_id, book, path)
        write_session({"book_id": book_id})
        return JSONResponse(_book_summary(book_id, book, cover=cover_kind(book_id, book)))

    # ---- book details shown in the player (title, author, narrator, series…) -------------------------
    META_FIELDS = {"title": 300, "author": 300, "narrator": 300, "series": 300, "series_part": 20, "year": 20,
                   "description": 2000, "genre": 100}

    def meta_file(book_id: str) -> Path:
        return state.data_dir / "meta" / f"{Path(book_id).name}.json"

    def meta_defaults(book) -> dict:
        return {"title": book.title or "", "author": book.author or "", "narrator": "", "series": "",
                "series_part": "", "year": str(book.year or ""), "description": book.annotation or "",
                "genre": "Audiobook"}

    def book_meta(book_id: str, book) -> dict:
        try:
            saved = json.loads(meta_file(book_id).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            saved = {}
        return {**meta_defaults(book), **{k: v for k, v in saved.items() if k in META_FIELDS and isinstance(v, str)}}

    async def get_meta(request: Request):
        bid = request.path_params["book_id"]
        book = state.book(bid)["book"]
        return JSONResponse({"meta": book_meta(bid, book), "defaults": meta_defaults(book)})

    async def put_meta(request: Request):
        bid = request.path_params["book_id"]
        book = state.book(bid)["book"]
        body = await request.json()
        if not isinstance(body, dict):
            raise ApiError("Book details must be an object")
        if body.get("reset"):
            meta_file(bid).unlink(missing_ok=True)
        else:
            current = book_meta(bid, book)
            for k, limit in META_FIELDS.items():
                if k in body:
                    current[k] = " ".join(str(body[k]).split()) if k != "description" else str(body[k]).strip()
                    if len(current[k]) > limit:
                        raise ApiError(f"{k} is longer than {limit} characters")
            meta_file(bid).parent.mkdir(parents=True, exist_ok=True)
            meta_file(bid).write_text(json.dumps(current, ensure_ascii=False), encoding="utf-8")
        return JSONResponse({"meta": book_meta(bid, book), "defaults": meta_defaults(book)})

    # ---- pronunciations: the lexicon of this book (see folivox.text.pronounce) ---------------------------
    PRONOUNCE_LIMIT = 100_000
    SAMPLE_LIMIT = 600
    SAMPLE_RANGE_LIMIT = 1500

    def markup_file(book_id: str) -> Path:
        return state.data_dir / "markup" / f"{Path(book_id).name}.json"

    async def get_markup(request: Request):
        """The book's markup (speakers, spans, tags) as JSON, or an empty one; spans never carry book text."""
        from folivox import markup as mk

        bid = request.path_params["book_id"]
        state.book(bid)
        f = markup_file(bid)
        try:
            m = mk.Markup.load(f) if f.is_file() else mk.Markup()
        except mk.MarkupError as e:
            raise ApiError(str(e)) from None
        return JSONResponse(m.to_dict())

    async def put_markup(request: Request):
        """Save markup after checking it against the book (stale chapters and bad spans are refused, not repaired)."""
        from folivox import markup as mk

        bid = request.path_params["book_id"]
        entry = state.book(bid)
        body = await request.json()
        try:
            m = mk.Markup.from_dict(body)
            chapters = entry["book"].chapters(m.depth)
            clean, problems = mk.bind(m, chapters)
        except mk.MarkupError as e:
            raise ApiError(str(e)) from None
        if problems:
            raise ApiError("The markup does not fit the book: " + "; ".join(problems[:3]), 400, problems=problems)
        mk.stamp(clean, chapters)
        f = markup_file(bid)
        if clean.is_empty() and len(clean.speakers) == 1:
            f.unlink(missing_ok=True)
        else:
            f.parent.mkdir(parents=True, exist_ok=True)
            clean.save(f)
        return JSONResponse(clean.to_dict())

    lang_cache: dict[tuple, dict] = {}

    async def book_languages(request: Request):
        """The languages of the whole book (letter shares), what each speaker reads and the gaps of their voices.
        ``?voice=<id>`` is the voice of the narrator (the book's main voice)."""
        from folivox import markup as mk
        from folivox.lang import usage
        from folivox.lang.detect import shares

        bid = request.path_params["book_id"]
        entry = state.book(bid)
        book = entry["book"]
        depth = int(request.query_params.get("depth", 2))
        f = markup_file(bid)
        stamp = (bid, book.lang, depth, f.stat().st_mtime_ns if f.is_file() else 0)

        def compute() -> dict:
            if stamp not in lang_cache:
                chapters = book.chapters(depth)
                text = "\n".join(b.text for ch in chapters for b in ch.blocks)
                try:
                    m = mk.Markup.load(f) if f.is_file() else mk.Markup()
                    m, _ = mk.bind(m, chapters)
                except mk.MarkupError:
                    m = mk.Markup()
                lang_cache.clear()
                lang_cache[stamp] = {"shares": shares(text), "used": usage.speaker_languages(chapters, m, book.lang),
                                     "speakers": {s.id: s.voice for s in m.speakers}}
            return lang_cache[stamp]

        data = await asyncio.to_thread(compute)
        narrator = request.query_params.get("voice", "")
        out = {}
        for sid, used in sorted(data["used"].items()):
            ref = narrator if sid == 0 else data["speakers"].get(sid, "") or narrator
            v = catalog.find(ref) if ref else None
            voice_lang = v.lang if v else (book.lang if not ref else "")
            out[str(sid)] = {"letters": dict(used), "voice": ref, "voice_lang": voice_lang,
                             "gaps": usage.voice_gaps(used, voice_lang)}
        return JSONResponse({"lang": book.lang, "shares": data["shares"], "speakers": out})

    async def chapter_text(request: Request):
        """The paragraphs of one chapter, as the markup addresses them (block index, text, and the stretches in
        another language that the script shows)."""
        bid = request.path_params["book_id"]
        entry = state.book(bid)
        depth = int(request.query_params.get("depth", 2))
        n = int(request.query_params.get("chapter", 0))
        chapters = entry["book"].chapters(depth)
        if not 0 <= n < len(chapters):
            raise ApiError("No such chapter", 404)
        ch = chapters[n]
        from folivox.lang.runs import script_runs

        blocks = []
        for i, b in enumerate(ch.blocks):
            item = {"b": i, "kind": b.kind.value, "text": b.text}
            if b.kind != BlockKind.POEM and (runs := script_runs(b.text, entry["book"].lang)):
                item["langs"] = [[r.start, r.end, r.lang] for r in runs]
            blocks.append(item)
        return JSONResponse({"chapter": n, "title": ch.title, "headings": ch.headings, "blocks": blocks})

    async def expression_tags(request: Request):
        from folivox import markup as mk

        return JSONResponse(mk.tag_database())

    def pronounce_file(book_id: str) -> Path:
        return state.data_dir / "pronounce" / f"{Path(book_id).name}.txt"

    def pronounce_state(book_id: str, text: str | None = None) -> dict:
        f = pronounce_file(book_id)
        if text is None:
            text = f.read_text(encoding="utf-8") if f.is_file() else ""
        lex = pronunciations.Lexicon.load([f]) if f.is_file() else pronunciations.Lexicon()
        warnings = [w.replace(f"{f.name}:", "line ", 1) for w in lex.warnings]  # the file is internal: say «line 3»
        return {"text": text, "count": len(lex.entries), "warnings": warnings}

    async def get_pronounce(request: Request):
        bid = request.path_params["book_id"]
        state.book(bid)
        return JSONResponse(pronounce_state(bid))

    async def put_pronounce(request: Request):
        bid = request.path_params["book_id"]
        state.book(bid)
        body = await request.json()
        text = body.get("text") if isinstance(body, dict) else None
        if not isinstance(text, str):
            raise ApiError("The pronunciations must be text")
        if len(text) > PRONOUNCE_LIMIT:
            raise ApiError(f"The pronunciations are longer than {PRONOUNCE_LIMIT} characters")
        f = pronounce_file(bid)
        if text.strip():
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(text, encoding="utf-8")
        else:
            f.unlink(missing_ok=True)
        return JSONResponse(pronounce_state(bid, text))

    def lexicon_of(book_id: str):
        f = pronounce_file(book_id)
        return pronunciations.Lexicon.load([f]) if f.is_file() else pronunciations.Lexicon()

    async def pronounce_sample(request: Request):
        """Sentences of the book that contain the first entries of the lexicon, in the order of the file."""
        from folivox import lang as languages

        bid = request.path_params["book_id"]
        book = state.book(bid)["book"]
        lex = lexicon_of(bid)
        paragraphs = [b.text for ch in book.chapters() for b in ch.blocks]
        text, missing = lex.sample(paragraphs, languages.get(book.lang or "ru").sentences)
        return JSONResponse({"text": text, "missing": missing})

    # ---- cover: the book's own, an image you upload, or none ----------------------------------------
    IMAGE_TYPES = ((b"\xff\xd8\xff", "jpg", "image/jpeg"), (b"\x89PNG\r\n\x1a\n", "png", "image/png"),
                   (b"GIF8", "gif", "image/gif"))

    def sniff_image(data: bytes) -> tuple[str, str] | None:
        for magic, ext, mime in IMAGE_TYPES:
            if data.startswith(magic):
                return ext, mime
        return ("webp", "image/webp") if data[:4] == b"RIFF" and data[8:12] == b"WEBP" else None

    def custom_cover(book_id: str) -> Path | None:
        d = state.data_dir / "covers"
        found = [p for p in d.glob(f"{Path(book_id).name}.*") if p.suffix != ".none"] if d.is_dir() else []
        return found[0] if found else None

    def cover_kind(book_id: str, book) -> str:
        """custom | none | book | '' (no cover at all)"""
        if custom_cover(book_id):
            return "custom"
        if (state.data_dir / "covers" / f"{Path(book_id).name}.none").exists():
            return "none"
        return "book" if book.cover else ""

    async def get_cover(request: Request):
        bid = request.path_params["book_id"]
        book = state.book(bid)["book"]
        mine = custom_cover(bid)
        if mine:
            return FileResponse(mine, headers={"Cache-Control": "no-cache"})
        if cover_kind(bid, book) == "book":
            return Response(book.cover, media_type=book.cover_mime or "image/jpeg")
        raise ApiError("No cover", 404)

    async def put_cover(request: Request):
        bid = request.path_params["book_id"]
        book = state.book(bid)["book"]
        folder = state.data_dir / "covers"
        folder.mkdir(parents=True, exist_ok=True)
        key = Path(bid).name
        if "multipart" in request.headers.get("content-type", ""):
            f = (await request.form()).get("file")
            if f is None or not hasattr(f, "read"):
                raise ApiError("No image in the request")
            data = await f.read()
            if len(data) > 15 * 1024 * 1024:
                raise ApiError("The image is larger than 15 MB", 413)
            kind = sniff_image(data)
            if kind is None:
                raise ApiError("Use a JPEG, PNG, WebP or GIF image", 415)
            for old in folder.glob(f"{key}.*"):
                old.unlink()
            (folder / f"{key}.{kind[0]}").write_bytes(data)
        else:  # {"mode": "book"} back to the book's own cover, {"mode": "none"} no cover
            mode = (await request.json()).get("mode")
            if mode not in ("book", "none"):
                raise ApiError("mode must be book or none")
            for old in folder.glob(f"{key}.*"):
                old.unlink()
            if mode == "none":
                (folder / f"{key}.none").write_bytes(b"")
        return JSONResponse({"cover": cover_kind(bid, book)})

    async def book_chapters(request: Request):
        entry = state.book(request.path_params["book_id"])
        book = entry["book"]
        if lang := request.query_params.get("lang"):
            from folivox import lang as languages

            try:
                book.lang = languages.get(lang).code
            except ValueError as e:
                raise ApiError(str(e)) from None
        depth = int(request.query_params.get("depth", 2))
        bid = request.path_params["book_id"]
        return JSONResponse(_book_summary(bid, book, depth, cover_kind(bid, book)))

    def write_session(body: dict) -> None:
        """Atomic, so a crash or two writers never leave a half-written file that would reset the page."""
        state.data_dir.mkdir(parents=True, exist_ok=True)
        tmp = state.session_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(body), encoding="utf-8")
        os.replace(tmp, state.session_file)

    def uploads_newest_first() -> list[str]:
        root = state.data_dir / "uploads"
        dirs = [d for d in root.iterdir() if d.is_dir() and any(d.iterdir())] if root.is_dir() else []
        return [d.name for d in sorted(dirs, key=lambda d: d.stat().st_mtime, reverse=True)]

    async def get_session(request: Request):
        """What the page had selected, so a reload (or a restart) shows the same screen. If the saved session is
        missing or its book is gone, the most recently uploaded book is selected."""
        try:
            data = json.loads(state.session_file.read_text(encoding="utf-8"))
            state.book(data.get("book_id", ""))  # drop a session whose book is gone
        except (OSError, ValueError, ApiError, AttributeError):
            data = None
        if data is None:
            for bid in uploads_newest_first()[:5]:
                try:
                    state.book(bid)
                except Exception:  # noqa: BLE001 — an unreadable upload must not block the others
                    continue
                data = {"book_id": bid}
                break
        return JSONResponse(data)

    async def put_session(request: Request):
        body = await request.json()
        if not isinstance(body, dict):
            raise ApiError("Session must be an object")
        write_session(body)
        return JSONResponse({"ok": True})

    def read_favorites() -> list[str]:
        try:
            ids = json.loads((state.data_dir / "favorites.json").read_text(encoding="utf-8"))
            return [i for i in ids if isinstance(i, str)]
        except (OSError, ValueError):
            return []

    async def put_favorites(request: Request):
        ids = (await request.json()).get("ids")
        if not isinstance(ids, list) or not all(isinstance(i, str) and len(i) <= 64 for i in ids):
            raise ApiError("ids must be a list of voice ids")
        state.data_dir.mkdir(parents=True, exist_ok=True)
        (state.data_dir / "favorites.json").write_text(json.dumps(sorted(set(ids))), encoding="utf-8")
        return JSONResponse({"ok": True})

    async def list_voices(request: Request):
        from folivox.tts.voice import GOOD_RANGES

        lang = request.query_params.get("lang")
        voices = [v for v in catalog.all_voices() if not lang or v.lang == lang]
        auto = {code: catalog.default_for(code).id for code in sorted({v.lang for v in catalog.all_voices()})}
        return JSONResponse({"voices": [_voice_json(v) for v in voices], "auto": auto, "favorites": read_favorites(),
                             "good_ranges": {k: list(r) for k, r in GOOD_RANGES.items()},
                             "user_dir": str(catalog.user_voices_dir())})

    async def voice_sample(request: Request):
        v = catalog.find(request.path_params["voice_id"])
        if v is None or not v.path.exists():
            raise ApiError("Unknown voice", 404)
        return FileResponse(v.path, media_type="audio/flac")

    async def add_voice(request: Request):
        from folivox import lang as languages
        from folivox.tts.voice import VoiceError, load_voice, transcript_warnings

        form = await request.form()
        f = form.get("file")
        if f is None or not hasattr(f, "read"):
            raise ApiError("No audio file in the request")
        voice_id = str(form.get("id", "")).strip()
        try:
            code = languages.get(str(form.get("lang") or "en")).code
        except ValueError as e:
            raise ApiError(str(e)) from None
        tmp = state.data_dir / "tmp" / uuid.uuid4().hex
        tmp.mkdir(parents=True)
        try:
            audio = tmp / (Path(f.filename or "voice.wav").name or "voice.wav")
            data = await f.read()
            if len(data) > MAX_UPLOAD:
                raise ApiError("File is too large", 413)
            audio.write_bytes(data)
            transcribe = None
            text = str(form.get("transcript") or "").strip()
            if not text:
                from folivox.qa.asr import Transcriber

                transcribe = Transcriber(language=code).transcribe_file
            try:
                v = await asyncio.to_thread(load_voice, audio, text or None, transcribe)
                entry = await asyncio.to_thread(
                    catalog.add, voice_id, audio, v.transcript, name=str(form.get("name") or ""),
                    gender=str(form.get("gender") or ""), lang=code, overwrite=False)
            except (VoiceError, ValueError, FileExistsError) as e:
                raise ApiError(str(e)) from None
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        return JSONResponse({**_voice_json(entry),
                             "warnings": v.warnings_for(str(form.get("engine") or ""))
                             + transcript_warnings(v.transcript, v.duration, code)})

    async def find_voice(request: Request):
        """A video or long recording in, a few clean 6–15 s stretches of speech out (to preview and pick)."""
        from folivox import lang as languages
        from folivox.qa.asr import Transcriber
        from folivox.voices import search

        form = await request.form()
        f = form.get("file")
        if f is None or not hasattr(f, "read"):
            raise ApiError("No file in the request")
        try:
            code = languages.get(str(form.get("lang") or "en")).code
        except ValueError as e:
            raise ApiError(str(e)) from None
        token = uuid.uuid4().hex
        for old in (state.data_dir / "tmp").glob("*"):  # clips of earlier searches, an hour is enough to listen
            if old.is_dir() and old.stat().st_mtime < time.time() - 3600:
                shutil.rmtree(old, ignore_errors=True)
        tmp = state.data_dir / "tmp" / token
        tmp.mkdir(parents=True)
        src = tmp / ("source" + (Path(f.filename or "").suffix.lower()[:8] or ".bin"))
        try:
            with src.open("wb") as out:
                while chunk := await f.read(1024 * 1024):
                    out.write(chunk)
            try:
                found = await asyncio.to_thread(search.find_samples, src, code, Transcriber(language=code), count=3)
                for i, c in enumerate(found):
                    await asyncio.to_thread(search.cut, src, c, tmp / f"{i}.wav")
            except (search.NothingFound, RuntimeError) as e:
                raise ApiError(str(e)) from None
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        src.unlink(missing_ok=True)  # only the short clips stay
        return JSONResponse({"token": token, "found": [
            {"i": i, "start": c.start, "end": c.end, "text": c.text, "warnings": c.warnings}
            for i, c in enumerate(found)]})

    async def found_voice(request: Request):
        token, i = request.path_params["token"], request.path_params["i"]
        path = state.data_dir / "tmp" / token / f"{i}.wav"
        if not re.fullmatch(r"[0-9a-f]{32}", token) or not i.isdigit() or not path.is_file():
            raise ApiError("Unknown sample", 404)
        return FileResponse(path, media_type="audio/wav")

    async def transcribe_voice(request: Request):
        """Recognise the sample's text so the user can check it before the voice is saved."""
        from folivox import lang as languages
        from folivox.qa.asr import Transcriber
        from folivox.tts.voice import audio_duration, transcript_warnings

        form = await request.form()
        f = form.get("file")
        if f is None or not hasattr(f, "read"):
            raise ApiError("No audio file in the request")
        try:
            code = languages.get(str(form.get("lang") or "en")).code
        except ValueError as e:
            raise ApiError(str(e)) from None
        data = await f.read()
        if len(data) > MAX_UPLOAD:
            raise ApiError("File is too large", 413)
        tmp = state.data_dir / "tmp" / uuid.uuid4().hex
        tmp.mkdir(parents=True)
        try:
            audio = tmp / (Path(f.filename or "voice.wav").name or "voice.wav")
            audio.write_bytes(data)
            try:
                duration = await asyncio.to_thread(audio_duration, audio)
                text = await asyncio.to_thread(Transcriber(language=code).transcribe_file, audio)
            except Exception as e:  # noqa: BLE001 — unreadable audio or a missing ASR model is reported, not a crash
                raise ApiError(f"Cannot transcribe the sample: {e}") from None
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        text = " ".join((text or "").split())
        return JSONResponse({"transcript": text, "warnings": transcript_warnings(text, duration, code)})

    async def denoise(request: Request):
        """WAV in → WAV out with steady background noise reduced (for the A/B check before adding a voice)."""
        import io

        import soundfile as sf

        from folivox.audio.denoise import reduce_noise

        form = await request.form()
        f = form.get("file")
        if f is None or not hasattr(f, "read"):
            raise ApiError("No audio file in the request")
        try:
            data, sr = sf.read(io.BytesIO(await f.read()), dtype="float32", always_2d=True)
        except Exception as e:  # noqa: BLE001 — an unreadable upload is the client's error, not a server crash
            raise ApiError(f"Cannot read the audio: {e}", 415) from None
        try:
            cleaned = await asyncio.to_thread(reduce_noise, data.mean(axis=1), sr)
        except RuntimeError as e:
            raise ApiError(str(e), 501) from None
        out = io.BytesIO()
        sf.write(out, cleaned, sr, format="WAV", subtype="PCM_16")
        return Response(out.getvalue(), media_type="audio/wav")

    async def delete_voice(request: Request):
        voice_id = request.path_params["voice_id"]
        v = next((x for x in catalog.user() if x.id == voice_id), None)
        if v is None:
            raise ApiError("Only your own voices can be removed", 404)
        catalog.remove(voice_id)
        return JSONResponse({"ok": True})

    def regs_tts_names() -> list[str]:
        return _registries()["tts"].names()

    def cloud_options(body: dict, o: dict, tts: str, voice) -> dict:
        """Options of a paid cloud engine; refuses to start without the user's explicit agreement."""
        from folivox.tts.cloud import generator_class, voice_modes_of

        cls = generator_class(tts)
        if body.get("acknowledged") is not True:
            raise ApiError("Confirm that the book text is sent to the provider and billed to your account", 400)
        ks = key_state(cls)
        if not (ks["key_in_env"] or ks["key_saved"]):
            raise ApiError(f"No API key for {cls.label}: paste it under Advanced settings → API keys, "
                           f"or set {ks['key_env']}", 412, missing_key=ks["key_env"], provider=tts)
        model = str(o.get("tts_model") or "auto").strip()
        if not re.fullmatch(r"[\w.\-]{1,80}", model):
            raise ApiError("Bad model name")
        from folivox.tts import saved

        speaker = str(o.get("tts_voice") or "").strip()
        mine = saved.find(tts, speaker)  # a voice kept from an earlier book: reused as it is
        mode = mine["mode"] if mine else str(o.get("tts_mode") or "preset")
        if mode not in voice_modes_of(cls):
            raise ApiError(f"{cls.label} cannot use the voice mode {mode!r} for a book")
        description = " ".join(str(o.get("tts_description") or "").split())[:1000]
        ref = None
        if mode == "clone" and not mine:
            if o.get("own_voice") is not True:
                raise ApiError("Confirm that this is your own voice or you have the speaker's permission")
            if voice is None or not any(x.id == voice.id for x in catalog.user()):
                raise ApiError("Choose one of your own voices to clone")
            ref = voice.path
        if mode == "design" and not mine and len(description) < 10:
            raise ApiError("Describe the voice (at least a few words)")
        if mode in ("preset", "localize") and not speaker and tts in ("elevenlabs", "cartesia", "fish"):
            raise ApiError("Pick one of the provider's voices")
        cap = o.get("max_cost")
        try:
            cap = float(cap) if cap not in (None, "") else None
        except (TypeError, ValueError):
            raise ApiError("The spending cap must be a number of dollars") from None
        if cap is not None and cap <= 0:
            raise ApiError("The spending cap must be more than zero")
        style = o.get("tts_style")
        return dict(accept_cloud=True, tts_model=model, tts_voice=speaker, tts_mode=mode, tts_description=description,
                    tts_style=str(style) if style not in (None, "") else None, voice_ref=ref, max_cost=cap,
                    keep_voice=bool(o.get("keep_voice", False)) and not mine,
                    tts_voice_name=" ".join(str(o.get("tts_voice_name") or "").split())[:60])

    def check_fish(tts: str, build_id: str, voice) -> None:
        """Say why a Fish engine cannot start before a job is made (the render checks again)."""
        from folivox.tts import builds as fb
        from folivox.tts import fish_check, fish_setup, licences

        try:
            from folivox.tts import runtime as tts_runtime

            b = fb.build_for(tts, build_id, runtime=tts_runtime.resolve(tts, None))
        except fb.CatalogError as e:
            raise ApiError(str(e)) from None
        if not licences.is_accepted(fb.load().licences[b.licence]):
            raise ApiError(f"Accept the {fb.load().licences[b.licence].name} first (Engine → Licence).", 403,
                           code="fish_licence")
        if not fish_setup.is_ready(b):
            raise ApiError(f"{b.title} is not installed yet ({b.size_gb:.1f} GB download): install it first.", 409,
                           code="fish_install")
        if voice is not None and fish_check.is_bundled_voice(voice.path) and not fish_check.bundled_voices_allowed():
            raise ApiError(fish_check.BUNDLED_VOICES_NOTE, 400, code="fish_voice")

    def build_options(body: dict, book_entry: dict, kind: str) -> RenderOptions:
        from folivox.text.stress import MODES

        o = body.get("options") or {}
        book = book_entry["book"]
        voice_id = body.get("voice") or ""
        from folivox.pipeline import is_cloud
        from folivox.tts import builds as fish_builds

        explicit_tts = config.provider("tts", o.get("tts") or None)
        tts = explicit_tts or fish_builds.default_engine()
        cloud = tts in regs_tts_names() and is_cloud(tts)
        v = catalog.find(voice_id) if voice_id else catalog.default_for(book.lang)
        if not explicit_tts and fish_builds.is_fish_engine(tts) and v is not None:
            from folivox.tts import fish_check

            if fish_check.is_bundled_voice(v.path) and not fish_check.bundled_voices_allowed():
                tts = "cosyvoice3"  # the bundled voices are real readers: Fish engines do not use them
        if v is None and not cloud:
            raise ApiError(f"Unknown voice {voice_id!r}")
        stress = o.get("stress", "yo")
        if stress not in MODES:
            raise ApiError(f"stress must be one of {', '.join(MODES)}")
        regs = _registries()
        asr = config.provider("asr", o.get("asr") or None) or "auto"
        for kind_name, name in (("tts", tts), ("asr", asr)):
            if name not in regs[kind_name].names():
                raise ApiError(f"Unknown {kind_name} provider {name!r}")
        chapters = body.get("chapters") or []
        spec = ",".join(str(int(n)) for n in chapters) or None
        cloud_opts = cloud_options(body, o, tts, v) if cloud else {}
        tts_build = str(o.get("tts_build") or "auto")
        if not cloud and fish_builds.is_fish_engine(tts):
            check_fish(tts, tts_build, v)
        lexicon = pronounce_file(str(body.get("book_id", "")))
        markup = markup_file(str(body.get("book_id", "")))
        common = dict(voice_wav=v.path if v else None,
                      voice_text=body.get("voice_text") or (v.transcript if v else None), lang=book.lang, **cloud_opts,
                      depth=int(o.get("depth", 2)), max_chars=int(o.get("max_chars") or 0) or None, stress=stress,
                      speed=float(o.get("speed", 1.0)), qa=bool(o.get("qa", True)),
                      qa_threshold=QA_LEVELS.get(str(o.get("qa_level")), QA_LEVELS["standard"]),
                      max_takes=int(o.get("takes") or (2 if cloud else 3)),
                      tts=tts, tts_build=tts_build, asr=asr, device=str(o.get("device", "auto")),
                      pronounce=(lexicon,) if lexicon.is_file() else (),
                      markup=markup if markup.is_file() and o.get("use_markup", True)
                      else language_markup(str(body.get("book_id", ""))),
                      expression_tags=bool(o.get("expression_tags", True)),
                      cover=custom_cover(str(body.get("book_id", ""))),
                      no_cover=cover_kind(str(body.get("book_id", "")), book) == "none",
                      meta=book_meta(str(body.get("book_id", "")), book))
        if kind == "preview":
            first = [int(chapters[0])] if chapters else [1]
            work = state.data_dir / "previews" / uuid.uuid4().hex[:8]
            return RenderOptions(output=work / "preview.m4b", work_dir=work / "work", chapters=str(first[0]),
                                 preview=int(o.get("preview_chars", PREVIEW_CHARS)), **common)
        out_dir = Path(str(o.get("output_dir") or config.output_dir())).expanduser()
        mp3 = bool(o.get("mp3", False))
        stem = Path(book_entry["path"]).stem
        return RenderOptions(output=out_dir / stem if mp3 else out_dir / f"{stem}.m4b", chapters=spec, mp3=mp3,
                             bitrate=str(o.get("bitrate", "64k")), **common)

    def language_markup(book_id: str) -> Path | None:
        """For a book read by one voice: the markup reduced to the languages of its fragments, or None."""
        from folivox import markup as mk

        f = markup_file(book_id)
        if not f.is_file():
            return None
        try:
            only = mk.Markup.load(f).languages_only()
        except mk.MarkupError:
            return None
        if only.is_empty():
            return None
        out = state.data_dir / "markup-lang" / f"{Path(book_id).name}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        only.save(out)
        return out

    def sample_from_range(body: dict, entry: dict, kind: str, opts: RenderOptions):
        """A one-chapter book made of the chosen paragraphs, and its markup, for a preview."""
        from folivox import markup as mk
        from folivox.book import Book, Section

        r = body["sample_range"]
        try:
            n, first, last, depth = int(r["chapter"]), int(r["block_from"]), int(r["block_to"]), int(r.get("depth", 2))
        except (KeyError, TypeError, ValueError):
            raise ApiError("sample_range needs chapter, block_from and block_to") from None
        chapters = entry["book"].chapters(depth)
        if kind != "preview" or not 0 <= n < len(chapters):
            raise ApiError("A passage is played as a preview of an existing chapter")
        blocks = chapters[n].blocks[first:last + 1]
        total = sum(len(b.text) for b in blocks)
        if not blocks or first < 0 or total > SAMPLE_RANGE_LIMIT:
            raise ApiError(f"Pick 1–{SAMPLE_RANGE_LIMIT} characters of whole paragraphs to listen to")
        mf = markup_file(str(body.get("book_id", "")))
        opts.markup = None
        if mf.is_file():
            try:
                m = mk.Markup.load(mf)
                if not (body.get("options") or {}).get("use_markup", True):
                    m = m.languages_only()  # one voice: only the languages of the fragments stay
                if m.depth == depth and not m.is_empty():
                    sliced = mk.slice_markup(m, n, first, last)
                    opts.markup = opts.output.parent / "passage.markup.json"
                    opts.output.parent.mkdir(parents=True, exist_ok=True)
                    sliced.save(opts.markup)
            except mk.MarkupError as e:
                raise ApiError(str(e)) from None
        opts.sample_exact, opts.preview, opts.chapters, opts.depth = True, total + 50, "1", 2
        return Book(title=entry["book"].title, lang=entry["book"].lang, sections=[Section(blocks=list(blocks))])

    async def start_job(request: Request):
        body = await request.json()
        kind = body.get("kind", "render")
        if kind not in ("render", "preview"):
            raise ApiError("kind must be render or preview")
        entry = state.book(body.get("book_id", ""))
        if state.manager.busy():
            raise ApiError("Another job is running; cancel it or wait", 409)
        opts = build_options(body, entry, kind)
        book = entry["book"]
        if body.get("sample_text") is not None:  # listen to a short text of your own (or picked from the book)
            from folivox.book import Block, BlockKind, Book, Section

            text = " ".join(str(body["sample_text"]).split())
            if kind != "preview" or not text or len(text) > SAMPLE_LIMIT:
                raise ApiError(f"A test text is 1–{SAMPLE_LIMIT} characters, played as a preview")
            book = Book(title=book.title, lang=book.lang, sections=[Section(blocks=[Block(BlockKind.PARAGRAPH, text)])])
        if body.get("sample_range"):  # listen to some paragraphs with their speakers and tags
            book = sample_from_range(body, entry, kind, opts)
        opts.output.parent.mkdir(parents=True, exist_ok=True)
        try:
            job = state.manager.start(kind, body["book_id"], book, opts)
        except RuntimeError:  # another request won the race between the busy() check and start()
            raise ApiError("Another job is running; cancel it or wait", 409) from None
        return JSONResponse(job.summary(), status_code=202)

    def get_job(request: Request):
        job = state.manager.jobs.get(request.path_params["job_id"])
        if job is None:
            raise ApiError("Unknown job", 404)
        return job

    async def job_info(request: Request):
        job = get_job(request)
        if job.opts is None:
            return JSONResponse({**job.summary(), "takes": getattr(job, "takes", [])})
        chapters = sorted(chapter_files(job.opts.resolved_work_dir()))
        return JSONResponse({**job.summary(), "chapters_built": chapters})

    async def current_job(request: Request):
        job = state.manager.current
        return JSONResponse(job.summary() if job else None)

    async def job_events(request: Request):
        job = get_job(request)

        async def stream():
            idx = int(request.headers.get("last-event-id", -1)) + 1
            while True:
                new, finished = await asyncio.to_thread(job.wait_events, idx)
                for ev in new:
                    yield f"id: {idx}\ndata: {json.dumps(ev)}\n\n"
                    idx += 1
                if finished:
                    return
                if not new:
                    yield ": keep-alive\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    async def job_cancel(request: Request):
        job = get_job(request)
        job.cancel_requested.set()
        return JSONResponse(job.summary())

    async def job_reveal(request: Request):
        """Show the finished file in the system file manager (only the result of a job of this server)."""
        import subprocess
        import sys

        if network:
            raise ApiError("The studio runs on another machine: find the file in the output folder", 400)
        job = get_job(request)
        if job.state != "done" or not job.result or not Path(job.result).exists():
            raise ApiError("Nothing to show yet", 404)
        target = Path(job.result)
        if sys.platform == "darwin":
            cmd = ["open", "-R", str(target)]
        elif sys.platform == "win32":
            cmd = ["explorer", f"/select,{target}"]
        else:
            cmd = ["xdg-open", str(target if target.is_dir() else target.parent)]
        subprocess.Popen(cmd)  # noqa: S603 — fixed argv, path comes from our own job result
        return JSONResponse({"ok": True})

    async def job_chapter(request: Request):
        job = get_job(request)
        path = chapter_files(job.opts.resolved_work_dir()).get(int(request.path_params["n"]))
        if path is None:
            raise ApiError("Chapter is not built yet", 404)
        return FileResponse(path, media_type="audio/flac")

    def review_list(work: Path) -> list[dict]:
        from folivox.journal import Journal

        if not (work / "journal.sqlite").exists():
            return []
        j = Journal(work)
        try:
            return [{"key": c.key, "chapter": c.chapter + 1, "idx": c.idx, "text": c.source, "heard": t.asr or "",
                     "score": round(t.score, 3), "takes": t.takes, "duration": round(t.duration, 1)}
                    for c, t in j.flagged()]
        finally:
            j.close()

    async def job_review(request: Request):
        """Fragments whose kept take still fails QA after every retake: listed for a human to check."""
        job = get_job(request)
        if job.opts is None:
            raise ApiError("This job has no fragments", 404)
        items = await asyncio.to_thread(review_list, job.opts.resolved_work_dir())
        return JSONResponse({"items": items})

    async def job_fragment(request: Request):
        job = get_job(request)
        key = request.path_params["key"]
        if job.opts is None or not re.fullmatch(r"[0-9a-f]{24}", key):
            raise ApiError("Unknown fragment", 404)
        path = job.opts.resolved_work_dir() / "audio" / f"{key}.flac"
        if not path.exists():
            raise ApiError("Unknown fragment", 404)
        return FileResponse(path, media_type="audio/flac")

    def diag_work_dir() -> Path | None:
        job = state.manager.current
        return job.opts.resolved_work_dir() if job is not None and job.opts is not None else None

    async def diagnostics(request: Request):
        from folivox import diag

        data = await asyncio.to_thread(diag.report, diag_work_dir())
        return JSONResponse({"report": data, "text": diag.render_report_text(data)})

    async def support_bundle(request: Request):
        """A redacted zip (no keys, no book text) for a bug report; built in memory-safe temp space."""
        import tempfile

        from folivox import diag

        def build() -> bytes:
            with tempfile.TemporaryDirectory() as tmp:
                return diag.support_bundle(Path(tmp) / "bundle.zip", diag_work_dir()).read_bytes()

        data = await asyncio.to_thread(build)
        return Response(data, media_type="application/zip", headers={
            "Content-Disposition": 'attachment; filename="folivox-support.zip"', "Cache-Control": "no-store"})

    async def get_settings(request: Request):
        kinds = {}
        for kind, reg in _registries().items():
            from folivox.tts import builds as fb

            fallback = fb.default_engine() if kind == "tts" else reg.default
            kinds[kind] = {"default": reg.default, "selected": config.provider(kind) or fallback,
                           "providers": [{"name": p.name, "description": p.description, "local": p.local,
                                          "needs": p.needs} for p in reg.providers()]}
        return JSONResponse({"kinds": kinds, "config_path": str(config.config_path()),
                             "output_dir": str(config.output_dir())})

    def key_names() -> list[dict]:
        """Env var names the cloud providers read: the defaults plus any ``api_key_env`` in the config."""
        owners: dict[str, list[str]] = {}
        for kind, reg in _registries().items():
            for p in reg.providers():
                if not p.local and p.needs.isupper():
                    owners.setdefault(config.section(f"{kind}.{p.name}").get("api_key_env", p.needs), []).append(
                        f"{kind}/{p.name}")
        return [{"name": n, "used_by": u} for n, u in sorted(owners.items())]

    def mask(value: str) -> str:
        return "•" * 8 + value[-4:] if len(value) > 8 else "•" * 8

    async def get_keys(request: Request):
        out = []
        for k in key_names():
            stored = config.keychain_get(k["name"])
            out.append({**k, "env": config.env_key_set(k["name"]), "saved": bool(stored),
                        "masked": mask(stored) if stored else ""})
        return JSONResponse({"keys": out})  # the full value never leaves the server

    async def put_key(request: Request):
        name = request.path_params["name"]
        if name not in {k["name"] for k in key_names()}:
            raise ApiError("Unknown key name", 404)
        value = str((await request.json()).get("value", "")).strip()
        if not value:
            raise ApiError("Empty key")
        try:
            config.keychain_set(name, value)
        except Exception as e:  # noqa: BLE001
            raise ApiError(f"Cannot use the OS keychain: {type(e).__name__}") from None
        return await get_keys(request)

    async def delete_key(request: Request):
        name = request.path_params["name"]
        try:
            config.keychain_delete(name)
        except Exception as e:  # noqa: BLE001
            raise ApiError(f"Cannot use the OS keychain: {type(e).__name__}") from None
        return await get_keys(request)

    async def put_settings(request: Request):
        import tomli_w

        body = await request.json()
        chosen = body.get("providers") or {}
        regs = _registries()
        for kind, name in chosen.items():
            if kind not in regs or name not in regs[kind].names():
                raise ApiError(f"Unknown {kind} provider {name!r}")
        path = config.config_path()
        data = dict(config.load())
        data["providers"] = {**data.get("providers", {}), **chosen}
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))  # comments are lost on rewrite
        path.write_text(tomli_w.dumps(data), encoding="utf-8")
        config.load.cache_clear()
        return await get_settings(request)


    # ---- cloud voice generation -------------------------------------------------------------------
    GEN_TAKE = re.compile(r"^take_t\d+_\d+\.wav$")

    def gen_class(name: str):
        from folivox.voicegen.cloud import cloud_providers

        cls = cloud_providers().get(name)
        if cls is None:
            raise ApiError(f"Unknown cloud voice provider {name!r}", 404)
        return cls

    def key_state(cls) -> dict:
        env = config.section(f"voicegen.{cls.name}").get("api_key_env", cls.key_env)
        stored = config.keychain_get(env)
        return {"key_env": env, "key_in_env": config.env_key_set(env), "key_saved": bool(stored),
                "key_masked": mask(stored) if stored else ""}

    def make_generator(name: str, model: str = ""):
        from folivox.providers import ProviderError

        cls = gen_class(name)
        ks = key_state(cls)
        if not (ks["key_in_env"] or ks["key_saved"]):
            raise ApiError(f"No API key for {cls.label}: paste it below or set {ks['key_env']} in the environment",
                           412, missing_key=ks["key_env"], provider=name)
        try:
            if "generator" in services:
                return services["generator"](name, model=model)
            return GENERATORS_create(name, model)
        except ProviderError as e:
            raise ApiError(str(e), 502) from None
        except ValueError as e:
            raise ApiError(str(e), 412, missing_key=ks["key_env"], provider=name) from None

    def GENERATORS_create(name: str, model: str = ""):
        from folivox.voicegen import GENERATORS

        return GENERATORS.create(name, model=model or None)

    async def gen_call(fn, *args):
        """Run a blocking provider call; a provider failure becomes a clear API error (never the key)."""
        from folivox.providers import ProviderError

        try:
            return await asyncio.to_thread(fn, *args)
        except ProviderError as e:
            raise ApiError(str(e), 429 if getattr(e, "kind", "") == "rate_limit" else 502) from None

    def book_text(book, limit: int = 60000) -> str:
        parts, n = [], 0
        for ch in book.chapters():
            for b in ch.blocks:
                parts.append(b.text)
                n += len(b.text)
                if n > limit:
                    return "\n".join(parts)
        return "\n".join(parts)

    def make_llm(body: dict):
        from folivox.llm import LLMS
        from folivox.providers import ProviderError

        name = config.provider("llm", body.get("llm") or None) or LLMS.default
        try:
            if "llm" in services:
                return services["llm"](name)
            return LLMS.create(name, model=body.get("model") or None)
        except (ProviderError, ValueError) as e:
            raise ApiError(str(e), 412) from None

    async def gen_providers(request: Request):
        from folivox.voicegen.cloud import COMMERCIAL_NOTICE, cloud_providers

        def sample_usd(cls):  # a typical voice sample: ~160 characters, 3 takes
            best = cls.models[0]["id"] if cls.models else ""
            return {m: cls.estimate_for(160, 3, m, best).get("usd") for m in cls.modes}

        out = [{**cls.capabilities(), **key_state(cls), "sample_usd": sample_usd(cls)}
               for cls in cloud_providers().values()]
        order = ["gemini", "cartesia", "fish", "elevenlabs", "openai"]  # by value for money
        tags = {"gemini": "cheap and best", "openai": "low quality for non-English"}
        for p in out:
            p["tag"] = tags.get(p["name"], "")
        out.sort(key=lambda p: order.index(p["name"]) if p["name"] in order else len(order))
        return JSONResponse({"providers": out, "notice": COMMERCIAL_NOTICE})

    async def tts_engines(request: Request):
        """Engines that can read the book: the local one and the paid cloud ones, with prices and limits."""
        from folivox.tts.cloud import CHARS_PER_SECOND, voice_modes_of
        from folivox.voicegen.cloud import COMMERCIAL_NOTICE, cloud_providers
        from folivox.voicegen.cloud_apis import LANGS, FishAudio

        hour = CHARS_PER_SECOND * 3600
        out = [{"name": "cosyvoice3", "label": "CosyVoice 3 (0.5B)", "local": True, "cloud": False,
                "blurb": "Free, private, works offline. Clones a voice from a short recording. Slower and "
                         "the quality depends on your hardware.", "models": [], "per_hour": 0,
                "languages": list(LANGS)}]  # the nine with Folivox rules
        for cls in cloud_providers().values():
            prices = {m["id"]: cls.estimate_for(hour, 1, "preset", m["id"]).get("usd") for m in cls.models}
            out.append({**cls.capabilities(), **key_state(cls), "local": False, "cloud": True,
                        "voice_modes": list(voice_modes_of(cls)), "per_hour": prices.get(cls.models[0]["id"])
                        if cls.models else None, "prices": prices})
        from folivox.tts import builds as fb

        hw = fb.detect_hardware()
        for name in fb.load().engines():  # Fish engines show only where one of their builds can run
            bs = [b for b in fb.load().for_engine(name) if hw.platform_tag in b.platforms]
            if bs:
                out.append({"name": name, "label": bs[0].title.split(",")[0], "local": True, "cloud": False,
                            "fish": True, "blurb": "", "models": [], "per_hour": 0,
                            "voice_langs": list(FishAudio.voice_languages),  # the local models read the same 80+
                            "languages": list(FishAudio.languages),
                            "builds": [b.id for b in bs],
                            # named native builds run under any runtime; "auto" picks them only with runtime=native
                            "native_builds": [b.id for b in bs if b.backend == "native"]})
        order = ["cosyvoice3", "fish-drama3", "fish-s2", "gemini", "cartesia", "fish", "elevenlabs",
                 "openai"]  # free first
        tags = {"gemini": "cheap and best", "openai": "low quality for non-English",
                "fish-drama3": "best"}
        for e in out:
            e["tag"] = tags.get(e["name"], "")
        out.sort(key=lambda e: order.index(e["name"]) if e["name"] in order else len(order))
        return JSONResponse({"engines": out, "notice": COMMERCIAL_NOTICE, "default": fb.default_engine(hw)})

    async def tts_estimate(request: Request):
        from folivox.tts import cost
        from folivox.tts.cloud import generator_class
        from folivox.voicegen.cloud import COMMERCIAL_NOTICE

        body = await request.json()
        entry = state.book(body.get("book_id", ""))
        o = body.get("options") or {}
        try:
            cls = generator_class(str(o.get("tts") or ""))
        except Exception:  # noqa: BLE001
            raise ApiError("Pick a cloud engine", 400) from None
        from folivox.pipeline import parse_selection

        depth = int(o.get("depth", 2))
        total = len(entry["book"].chapters(depth))
        chapters = body.get("chapters") or []
        chosen = [i + 1 for i in parse_selection(",".join(str(int(n)) for n in chapters) or None, total)]
        preview = int(o.get("preview_chars", PREVIEW_CHARS)) if body.get("kind") == "preview" else None
        if preview:
            chosen = chosen[:1]
        chars = cost.book_chars(entry["book"], depth, chosen, preview)
        model = str(o.get("tts_model") or "auto")
        if model in ("auto", "latest"):
            model = cls.models[0]["id"] if cls.models else ""
        est = cost.estimate(cls.name, chars, model, int(o.get("takes") or 2), bool(o.get("qa", True)),
                            str(o.get("tts_mode") or "preset"), str(o.get("lang") or entry["book"].lang or ""))
        return JSONResponse({**est, "summary": cost.describe(est), "notice": COMMERCIAL_NOTICE,
                             "label": cls.label, "price_note": cls.price_note})

    async def tts_saved_list(request: Request):
        from folivox.tts import saved

        return JSONResponse({"voices": saved.listing(request.path_params["provider"])})

    async def tts_saved_rename(request: Request):
        from folivox.tts import saved

        name = " ".join(str((await request.json()).get("name", "")).split())[:60]
        if not name:
            raise ApiError("Give the voice a name")
        if not saved.rename(request.path_params["provider"], request.path_params["voice"], name):
            raise ApiError("Unknown saved voice", 404)
        return JSONResponse({"ok": True})

    async def tts_saved_delete(request: Request):
        """Forget a kept voice and delete it from the provider's account."""
        from folivox.providers import ProviderError
        from folivox.tts import saved

        name, vid = request.path_params["provider"], request.path_params["voice"]
        if saved.find(name, vid) is None:
            raise ApiError("Unknown saved voice", 404)
        gen = make_generator(name)
        try:
            await asyncio.to_thread(gen.teardown_voices, [vid])
        except ProviderError as e:
            if getattr(e, "kind", "") != "bad_request":  # "not found": it is already gone, forget it
                raise ApiError(str(e), 502) from None
        saved.remove(name, vid)
        return JSONResponse({"ok": True})

    async def gen_voices(request: Request):
        name = request.path_params["provider"]
        gen = make_generator(name)
        if "preset" not in gen.modes:
            return JSONResponse({"voices": []})
        scope = request.query_params.get("scope")
        if scope in [x["id"] for x in getattr(gen, "scopes", ())]:
            gen.library = scope  # which part of the provider's voice library to list
        speakers = await gen_call(gen.speakers, request.query_params.get("lang"))
        lang = request.query_params.get("lang")
        names, details = getattr(gen, "names", {}), getattr(gen, "details", {})
        rows = [{"id": k, "lang": v[0], "gender": v[1], "name": names.get(k, k), **details.get(k, {})}
                for k, v in speakers.items() if not lang or v[0] in (lang, "")]
        for r in rows:  # some providers serve samples only to the key holder: the browser plays them through us
            if r.get("preview_url"):
                sample_urls[(name, r["id"])] = r["preview_url"]
                if name in AUTH_SAMPLES:
                    r["preview_url"] = f"/api/tts/{name}/sample/{urllib.parse.quote(r['id'], safe='')}"
        return JSONResponse({"voices": rows, "note": getattr(gen, "voices_note", "")})

    sample_urls: dict[tuple[str, str], str] = {}
    AUTH_SAMPLES = {"cartesia"}  # sample files that answer 403 without the account's key

    async def tts_sample(request: Request):
        """A provider's free voice sample, fetched with the key (never exposed). Only URLs the provider
        itself listed for that voice are fetched."""
        name, voice = request.path_params["provider"], request.path_params["voice"]
        url = sample_urls.get((name, voice))
        host = urllib.parse.urlparse(url or "").hostname or ""
        if name not in AUTH_SAMPLES or not url or not url.startswith("https://") or not host.endswith(".cartesia.ai"):
            raise ApiError("No sample for this voice", 404)
        gen = make_generator(name)

        def fetch():
            resp = gen.transport("GET", url, {**gen.headers(), "User-Agent": "Mozilla/5.0"}, None, 60)
            if resp.status >= 300:
                raise gen.map_error(resp)
            return resp

        resp = await gen_call(fetch)
        return Response(resp.body, media_type=resp.headers.get("content-type", "audio/wav"),
                        headers={"Cache-Control": "private, max-age=3600"})

    async def gen_suggest(request: Request):
        from folivox.voices.suggest import suggest_design_prompt

        body = await request.json()
        entry = state.book(body.get("book_id", ""))
        book = entry["book"]
        llm = make_llm(body)
        try:
            prompt = await asyncio.to_thread(suggest_design_prompt, book_text(book, 12000), book.lang, llm,
                                             title=book.title or "", author=book.author or "")
        except ValueError as e:
            raise ApiError(str(e), 502) from None
        except Exception as e:  # noqa: BLE001 — LLM server down, bad key…
            raise ApiError(f"The text LLM failed: {e}", 502) from None
        return JSONResponse({"prompt": prompt, "llm": getattr(llm, "name", "")})

    async def gen_texts(request: Request):
        from folivox import lang as languages
        from folivox.voices.design import pick_book_texts, safe_texts

        body = await request.json()
        try:
            code = languages.get(str(body.get("lang") or "en")).code
        except ValueError as e:
            raise ApiError(str(e)) from None
        if body.get("source") == "book":
            book = state.book(body.get("book_id", ""))["book"]
            llm = make_llm(body)
            try:
                found = await asyncio.to_thread(pick_book_texts, book_text(book), book.lang, llm, 3)
            except Exception as e:  # noqa: BLE001
                raise ApiError(f"The text LLM failed: {e}", 502) from None
            if not found:
                raise ApiError("No easy passage found in the book; use the built-in text or write your own", 404)
            return JSONResponse({"texts": found, "lang": book.lang})
        texts = safe_texts(code)
        if not texts:
            raise ApiError(f"No built-in text for {code!r}; write your own", 404)
        return JSONResponse({"texts": texts, "lang": code})

    def gen_params(body: dict) -> dict:
        from folivox import lang as languages

        cls = gen_class(str(body.get("provider", "")))
        mode = body.get("mode") or cls.modes[0]
        if mode not in cls.modes:
            raise ApiError(f"{cls.label} supports: {', '.join(cls.modes)}")
        try:
            lang = languages.get(str(body.get("lang") or "en")).code
        except ValueError as e:
            raise ApiError(str(e)) from None
        if lang not in cls.languages:
            raise ApiError(f"{cls.label} does not list {lang!r}; supported: {', '.join(cls.languages)}")
        texts = [" ".join(str(t).split()) for t in (body.get("texts") or [])]
        texts = [t for t in texts if t]
        if not 1 <= len(texts) <= 5:
            raise ApiError("Give one to five texts to be spoken")
        if any(len(t) > cls.max_chars for t in texts):
            raise ApiError(f"A text is longer than {cls.max_chars} characters")
        takes = int(body.get("takes", 3))
        if not 1 <= takes <= 6:
            raise ApiError("takes must be between 1 and 6")
        model = str(body.get("model") or "auto").strip()
        if not re.fullmatch(r"[\w.\-]{1,80}", model):
            raise ApiError("Bad model name")
        description = " ".join(str(body.get("description") or "").split())
        speaker = str(body.get("speaker") or "").strip()
        reference = None
        if mode == "clone":
            mine = next((v for v in catalog.user() if v.id == body.get("reference")), None)
            if mine is None or not mine.path.exists():
                raise ApiError("Choose one of your own voices to clone")
            if body.get("own_voice") is not True:
                raise ApiError("Confirm that this is your own voice or you have the speaker's permission")
            reference = mine.path
        if mode == "localize" and not speaker:
            raise ApiError("Pick the voice whose accent should be localised")
        if mode == "design" and len(description) < 10:
            raise ApiError("Describe the voice (at least a few words), or press Suggest")
        if mode == "preset" and not speaker:
            raise ApiError("Pick one of the provider's voices")
        if len(description) > 1000:
            raise ApiError("The description is too long (1000 characters at most)")
        return {"cls": cls, "mode": mode, "lang": lang, "texts": texts, "takes": takes, "description": description,
                "speaker": speaker, "model": model, "reference": reference, "rank": body.get("rank", True) is not False}

    async def gen_estimate(request: Request):
        from folivox.voicegen.cloud import COMMERCIAL_NOTICE

        prm = gen_params(await request.json())
        cls = prm["cls"]
        model = cls.models[0]["id"] if prm["model"] in ("auto", "latest") and cls.models else prm["model"]
        est = cls.estimate_for(sum(len(t) for t in prm["texts"]), prm["takes"], prm["mode"], model)
        return JSONResponse({**est, "notice": COMMERCIAL_NOTICE, "terms_url": cls.terms_url})

    async def gen_start(request: Request):
        from folivox.ui.gen import GenJob, run_generation

        body = await request.json()
        prm = gen_params(body)
        if body.get("acknowledged") is not True:
            raise ApiError("Confirm that the text is sent to the provider and billed to your account", 400)
        if state.manager.busy():
            raise ApiError("Another job is running; cancel it or wait", 409)
        name = prm["cls"].name
        gen = make_generator(name, prm["model"])
        gen.mode = prm["mode"]
        out_dir = state.data_dir / "gen" / uuid.uuid4().hex[:10]
        out_dir.mkdir(parents=True)
        job = GenJob(name, str(body.get("book_id") or ""), out_dir,
                     {k: v for k, v in prm.items() if k != "cls"})
        texts = prm["texts"]

        def asr_factory(lang):
            if "asr" in services:
                return services["asr"](lang)
            from folivox.qa.asr import Transcriber

            return Transcriber(language=lang)

        try:
            state.manager.start_task(job, lambda j: run_generation(j, gen, texts, asr_factory=asr_factory, wave=_wave))
        except RuntimeError as e:
            raise ApiError(str(e), 409) from None
        state.gen_jobs[job.id] = job
        return JSONResponse(job.summary(), status_code=202)

    def get_gen_job(request: Request):
        job = state.gen_jobs.get(request.path_params["job_id"])
        if job is None:
            raise ApiError("Unknown generation job", 404)
        return job

    async def gen_take(request: Request):
        job = get_gen_job(request)
        name = request.path_params["name"]
        path = job.out_dir / name
        if not GEN_TAKE.match(name) or not path.is_file():
            raise ApiError("Unknown take", 404)
        return FileResponse(path, media_type="audio/wav")

    async def gen_add(request: Request):
        from folivox import lang as languages

        job = get_gen_job(request)
        body = await request.json()
        take = next((t for t in job.takes if t["name"] == body.get("take")), None)
        if take is None:
            raise ApiError("Pick one of the generated takes", 404)
        cls = gen_class(job.provider)
        try:
            code = languages.get(str(body.get("lang") or job.params["lang"])).code
            entry = await asyncio.to_thread(
                catalog.add, str(body.get("id", "")).strip(), job.out_dir / take["name"], take["text"],
                name=str(body.get("name") or ""), gender=str(body.get("gender") or ""), lang=code,
                source=f"{cls.label} {job.params['model']} (commercial API): "
                       f"{job.params['description'] or job.params['speaker'] or job.params['mode']}",
                license=cls.license, url=cls.terms_url, commercial=True)
        except (ValueError, FileExistsError) as e:
            raise ApiError(str(e)) from None
        return JSONResponse(_voice_json(entry))

    # ---- plumbing ---
    def open_access_denied(request: Request, host: str) -> JSONResponse | None:
        try:
            ip = ipaddress.ip_address(request.client.host if request.client else "")
            ip = ip.ipv4_mapped or ip if ip.version == 6 else ip
        except ValueError:
            ip = None
        if ip is None or not any(ip in n for n in nets):
            return JSONResponse({"error": "This studio runs without a password and only answers clients from "
                                          "private networks. To allow yours set FOLIVOX_UI_ALLOWED_NETS "
                                          "(comma-separated CIDR, e.g. 192.168.1.0/24), or set FOLIVOX_UI_TOKEN "
                                          "and remove FOLIVOX_UI_AUTH=off."}, status_code=403)
        if not host_names:  # no names chosen by the owner: refuse public-looking ones (DNS rebinding)
            bare = (host[:host.index("]") + 1] if host.startswith("[") else host.partition(":")[0]).lower()
            try:
                ipaddress.ip_address(bare.strip("[]"))
                return None
            except ValueError:
                pass
            if "." in bare.rstrip(".") and not bare.rstrip(".").endswith(LOCAL_SUFFIXES):
                return JSONResponse({"error": "Bad Host header: without a password only IP addresses and local "
                                              "names (.local, .lan, .home.arpa) are accepted; set "
                                              "FOLIVOX_UI_ALLOWED_HOSTS for another name."}, status_code=403)
        return None

    def check(request: Request) -> JSONResponse | None:
        host = request.headers.get("host") or ""
        if request.url.path == "/healthz":
            return None  # container health check: asks over 127.0.0.1 whatever the allowed names are, tells nothing
        if open_access:
            denied = open_access_denied(request, host)
            if denied is not None:
                return denied
        if allowed_hosts is not None and host not in allowed_hosts:
            return JSONResponse({"error": "Bad Host header"}, status_code=403)  # DNS-rebinding guard
        bare = host[:host.index("]") + 1] if host.startswith("[") else host.partition(":")[0]
        if host_names and bare.lower().strip("[]") not in host_names:
            return JSONResponse({"error": "Bad Host header"}, status_code=403)
        # CSRF guard: a page on another site can still POST a form or multipart body to 127.0.0.1 (a "simple"
        # request needs no CORS preflight), so a browser-sent Origin must be this server itself
        origin = request.headers.get("origin")
        if origin is not None:
            scheme, _, rest = origin.partition("://")
            if scheme not in ("http", "https") or rest.rstrip("/") != request.headers.get("host"):
                return JSONResponse({"error": "Cross-site request refused"}, status_code=403)
        if request.url.path.startswith("/api/") and request.headers.get("sec-fetch-site") == "cross-site":
            return JSONResponse({"error": "Cross-site request refused"}, status_code=403)
        if token and request.url.path.startswith("/api/"):
            given = request.headers.get("x-folivox-token") or request.query_params.get("token") or ""
            if not same(request.cookies.get(cookie_name) or "", session):  # a valid cookie is never throttled
                ok = token_ok(request, given) if given else False
                if ok is None:
                    return too_many()
                if not ok:
                    return JSONResponse({"error": "Missing or wrong token"}, status_code=401)
        return None

    async def api_error(request: Request, exc: ApiError):
        return JSONResponse({"error": str(exc), **exc.extra}, status_code=exc.status)

    async def bad_request(request: Request, exc: Exception):
        return JSONResponse({"error": f"Bad request: {exc}"}, status_code=400)

    routes = [
        Route("/", index),
        Route("/healthz", health),
        Route("/login", login, methods=["POST"]),
        Route("/i18n/{code}.json", i18n_catalog),
        Route("/api/ping", ping),
        Route("/api/activity", get_activity),
        Route("/api/changelog", changelog),
        Route("/api/shutdown", shutdown, methods=["POST"]),
        Route("/api/books", upload_book, methods=["POST"]),
        Route("/api/books", list_books),
        Route("/api/books/{book_id}/open", open_book, methods=["POST"]),
        Route("/api/books/{book_id}", remove_book, methods=["DELETE"]),
        Route("/api/books/text", paste_text, methods=["POST"]),
        Route("/api/books/{book_id}/chapters", book_chapters),
        Route("/api/books/{book_id}/meta", get_meta),
        Route("/api/books/{book_id}/meta", put_meta, methods=["PUT"]),
        Route("/api/books/{book_id}/markup", get_markup),
        Route("/api/books/{book_id}/markup", put_markup, methods=["PUT"]),
        Route("/api/markup/tags", expression_tags),
        Route("/api/books/{book_id}/text", chapter_text),
        Route("/api/books/{book_id}/languages", book_languages),
        Route("/api/books/{book_id}/pronounce", get_pronounce),
        Route("/api/books/{book_id}/pronounce", put_pronounce, methods=["PUT"]),
        Route("/api/books/{book_id}/pronounce/sample", pronounce_sample),
        Route("/api/books/{book_id}/cover", get_cover),
        Route("/api/books/{book_id}/cover", put_cover, methods=["PUT", "POST"]),
        Route("/api/session", get_session),
        Route("/api/session", put_session, methods=["PUT"]),
        Route("/api/favorites", put_favorites, methods=["PUT"]),
        Route("/api/voices", list_voices),
        Route("/api/voices", add_voice, methods=["POST"]),
        Route("/api/voices/transcribe", transcribe_voice, methods=["POST"]),
        Route("/api/voices/find", find_voice, methods=["POST"]),
        Route("/api/voices/found/{token}/{i}", found_voice),
        Route("/api/denoise", denoise, methods=["POST"]),
        Route("/api/voices/{voice_id}/sample", voice_sample),
        Route("/api/voices/{voice_id}", delete_voice, methods=["DELETE"]),
        Route("/api/jobs", start_job, methods=["POST"]),
        Route("/api/jobs/current", current_job),
        Route("/api/jobs/{job_id}", job_info),
        Route("/api/jobs/{job_id}/events", job_events),
        Route("/api/jobs/{job_id}/cancel", job_cancel, methods=["POST"]),
        Route("/api/jobs/{job_id}/reveal", job_reveal, methods=["POST"]),
        Route("/api/jobs/{job_id}/chapters/{n:int}", job_chapter),
        Route("/api/jobs/{job_id}/review", job_review),
        Route("/api/jobs/{job_id}/fragments/{key}", job_fragment),
        Route("/api/diagnostics", diagnostics),
        Route("/api/support-bundle", support_bundle),
        Route("/api/settings", get_settings),
        Route("/api/settings", put_settings, methods=["PUT"]),
        Route("/api/gen/providers", gen_providers),
        Route("/api/tts/engines", tts_engines),
        Route("/api/tts/{provider}/sample/{voice}", tts_sample),
        Route("/api/tts/{provider}/saved", tts_saved_list),
        Route("/api/tts/{provider}/saved/{voice}", tts_saved_delete, methods=["DELETE"]),
        Route("/api/tts/{provider}/saved/{voice}", tts_saved_rename, methods=["PUT"]),
        Route("/api/tts/estimate", tts_estimate, methods=["POST"]),
        Route("/api/gen/suggest", gen_suggest, methods=["POST"]),
        Route("/api/gen/texts", gen_texts, methods=["POST"]),
        Route("/api/gen/estimate", gen_estimate, methods=["POST"]),
        Route("/api/gen/start", gen_start, methods=["POST"]),
        Route("/api/gen/voices/{provider}", gen_voices),
        Route("/api/gen/{job_id}/takes/{name}", gen_take),
        Route("/api/gen/{job_id}/add", gen_add, methods=["POST"]),
        Route("/api/keys", get_keys),
        Route("/api/keys/{name}", put_key, methods=["PUT"]),
        Route("/api/keys/{name}", delete_key, methods=["DELETE"]),
        *fish_routes.routes(ApiError, fish_install, lambda: state.manager.busy()),
        *runtime_routes.routes(ApiError, native_install, lambda: state.manager.busy()),
        *markup_routes.routes(ApiError, state, services),
    ]
    app = Starlette(routes=routes, exception_handlers={ApiError: api_error, ValueError: bad_request,
                                                      TypeError: bad_request, KeyError: bad_request})
    app.add_middleware(_Guard, check=check, presence=presence)
    app.state.ui = state
    app.state.request_exit = None  # set by `folivox ui`: asks uvicorn to stop
    app.state.watchdog = lifecycle.Watchdog(state, presence)

    def cancel_all() -> None:
        """Non-blocking: ask every running job to stop (called from the signal handler)."""
        state.manager.cancel_all()
        for t in getattr(state, "markup_tasks", {}).values():
            if t.state == "running":
                t.release = True
                t.cancel.set()
        pull = getattr(state, "pull", None)
        if pull is not None:
            pull.cancel.set()

    def shutdown() -> None:
        """Stop everything the server started: render children, markup threads (and their model), generation."""
        app.state.watchdog.stop()
        cancel_all()
        state.manager.stop_all()
        end = time.time() + 8
        for t in getattr(state, "markup_tasks", {}).values():
            while t.state == "running" and time.time() < end:
                time.sleep(0.05)
        for t in getattr(state, "markup_tasks", {}).values():  # a finished run leaves its model to the keep-alive
            fn = getattr(t.llm, "unload", None)
            if callable(fn):
                fn()
                t.llm = None

    app.state.cancel_all, app.state.shutdown = cancel_all, shutdown
    return app


class _Throttle:
    """Wrong-token counter for the network mode: ``LIMIT`` misses within ``WINDOW`` seconds lock token checks for
    ``LOCK`` seconds. One counter for everyone (behind a proxy every client has the proxy's address); a request with
    a valid session cookie does not go through it, so the owner is only held up while someone is guessing."""

    LIMIT, WINDOW, LOCK = 10, 60.0, 300

    def __init__(self):
        self.misses: list[float] = []
        self.until = 0.0

    def locked(self) -> bool:
        return time.monotonic() < self.until

    def fail(self) -> None:
        now = time.monotonic()
        self.misses = [t for t in self.misses if now - t < self.WINDOW] + [now]
        if len(self.misses) >= self.LIMIT:
            self.until, self.misses = now + self.LOCK, []


LOGIN_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport"
content="width=device-width"><title>Folivox</title><style>
body{font:16px system-ui,sans-serif;max-width:22rem;margin:18vh auto;padding:0 1rem}
input,button{font:inherit;padding:.5rem;width:100%;box-sizing:border-box;margin:.4rem 0}
.err{color:#b00020}</style></head>
<body><h1>Folivox</h1><p>Enter the access token set for this server (FOLIVOX_UI_TOKEN).</p><!--error-->
<form method="post" action="/login"><input type="password" name="token" autocomplete="current-password" autofocus
required><button>Open the studio</button></form></body></html>"""


LOCAL_SUFFIXES = (".local", ".lan", ".home.arpa", ".internal", ".localdomain", ".localhost")


class _Guard:
    """Pure-ASGI middleware (keeps SSE streaming intact): Host and token checks before routing."""

    def __init__(self, app, check, presence=None):
        self.app, self.check, self.presence = app, check, presence

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            if self.presence is not None and scope.get("path") != "/api/ping":  # `folivox ui` pings; it is no browser
                self.presence.touch()
            denied = self.check(Request(scope, receive))
            if denied is not None:
                return await denied(scope, receive, send)
        await self.app(scope, receive, send)

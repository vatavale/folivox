"""Studio routes for the automatic markup: model choice, character list, background analysis and marking."""

from __future__ import annotations

import shutil
import threading
import time
from pathlib import Path

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from folivox import markup as mk
from folivox import markup_llm as ml


class Task:
    """One background analysis/marking per book; the page polls ``GET /api/books/{id}/llm/status``."""

    def __init__(self):
        self.state = "idle"  # idle | running | done | error | cancelled
        self.task = ""
        self.stage, self.done, self.total, self.message = "", 0, 0, ""
        self.stats: dict = {}
        self.reason = ""  # why a run stopped early: "cost_cap"
        self.budget = None  # the run's spend (folivox.markup_cost.Budget)
        self.cancel = threading.Event()
        self.llm = None
        self.release = False  # unload the model when the run ends (the person stopped it, or nobody is watching)

    def snapshot(self) -> dict:
        out = {"state": self.state, "task": self.task, "stage": self.stage, "done": self.done, "total": self.total,
               "message": self.message, "stats": self.stats, "reason": self.reason}
        b = self.budget
        if b is not None and b.price is not None and not b.price.free:
            out["spent"] = {"usd": round(b.usd or 0.0, 4), "tokens": b.spent.tokens, "calls": b.spent.calls,
                            "cap": b.max_cost}
        return out


PREVIEW_ANALYSE = 6  # windows read for a preview of the characters
PREVIEW_MARK = 3  # windows marked for a preview


class Pull:
    """The one model download at a time; the page polls ``GET /api/llm/pull``."""

    def __init__(self):
        self.state, self.model = "idle", ""  # idle | running | done | error | cancelled
        self.done = self.total = 0
        self.message, self.started = "", 0.0
        self.cancel = threading.Event()

    def snapshot(self) -> dict:
        return {"state": self.state, "model": self.model, "done": self.done, "total": self.total,
                "message": self.message, "elapsed": round(time.time() - self.started, 1) if self.started else 0}


def unload(t: Task) -> None:
    """Free the model a stopped run used (Ollama would keep it for minutes); a hosted endpoint has nothing to free."""
    fn = getattr(t.llm, "unload", None)
    if t.release or t.cancel.is_set():
        t.release = False
        if callable(fn):
            fn()


def routes(api_error, state, services: dict) -> list[Route]:
    tasks: dict[str, Task] = {}
    pull = Pull()
    state.markup_tasks, state.pull = tasks, pull

    def files(bid: str) -> tuple[Path, Path, Path]:
        name = Path(bid).name
        d = state.data_dir
        return d / "characters" / f"{name}.json", d / "markup" / f"{name}.json", d / "llm-cache" / f"{name}.sqlite"

    def task_of(bid: str) -> Task:
        return tasks.setdefault(bid, Task())

    async def models(request: Request):
        from folivox import config, diag
        from folivox import markup_cost as mc
        from folivox.llm import GEMMA_TIERS, LLMS, Ollama, recommend_model

        provider = config.provider("llm") or LLMS.default
        if not mc.is_local(provider):  # a paid API: its configured model and the well-known ones, no downloads
            return JSONResponse({"provider": provider, "local": False, "configured": mc.default_model(provider),
                                 "known": mc.known_models(provider), "installed": [], "tiers": [],
                                 "running": True, "recommended": ""})
        ram = diag.ram_gb()
        ollama = Ollama()
        return JSONResponse({"provider": provider, "local": True, "ram_gb": ram, "recommended": recommend_model(ram),
                             "running": ollama.running(),
                             "free_gb": round(shutil.disk_usage(Path.home()).free / 1e9, 1),
                             "configured": config.section("llm.ollama").get("model", ""),
                             "installed": ollama.installed(),
                             "tiers": [{"min_ram_gb": n, "model": t, "download_gb": g, "note": note}
                                       for n, t, g, note in GEMMA_TIERS]})

    async def get_characters(request: Request):
        bid = request.path_params["book_id"]
        state.book(bid)
        f = files(bid)[0]
        try:
            bible = ml.Bible.load(f) if f.is_file() else ml.Bible()
        except ml.LLMError as e:
            raise api_error(str(e)) from None
        return JSONResponse(bible.to_dict())

    async def put_characters(request: Request):
        bid = request.path_params["book_id"]
        state.book(bid)
        try:
            bible = ml.Bible.from_dict(await request.json())
        except (TypeError, ValueError, AttributeError) as e:
            raise api_error(f"Malformed character list: {e}") from None
        for c in bible.characters:
            if not 0 <= c.id < mk.MAX_SPEAKERS:
                raise api_error(f"{c.name}: the speaker id must be 0..{mk.MAX_SPEAKERS - 1}")
        f = files(bid)[0]
        try:  # what the person changed by hand is not outvoted by the windows later: it is a heavy vote
            before = {(c.id, c.name if c.shared else ""): c for c in ml.Bible.load(f).characters if c.id} \
                if f.is_file() else {}
        except ml.LLMError:
            before = {}
        for c in bible.characters:
            old = before.get((c.id, c.name if c.shared else ""))  # a shared id is held by several
            if old is not None:
                for what in ("gender", "age"):
                    if getattr(c, what) != getattr(old, what):
                        c.vote(what, getattr(c, what), 1000)
        f.parent.mkdir(parents=True, exist_ok=True)
        bible.save(f)
        return JSONResponse(bible.to_dict())

    async def voice_info(request: Request):
        """A speaker's voice-design prompts and the texts the generated voice can read (their own lines)."""
        from folivox.voices import character_voice as cv

        bid = request.path_params["book_id"]
        entry = state.book(bid)
        sid = int(request.path_params["speaker"])
        bp, mp, _ = files(bid)
        try:
            bible = ml.Bible.load(bp) if bp.is_file() else ml.Bible()
        except ml.LLMError as e:
            raise api_error(str(e)) from None
        c = next((x for x in bible.characters if x.id == sid), None)
        markup = mk.Markup.load(mp) if mp.is_file() else None
        depth = markup.depth if markup else 2
        chapters = entry["book"].chapters(depth)
        sp = markup.speaker(sid) if markup else None
        if c is None and sp is not None:  # a speaker made by hand: describe from what the speaker record says
            c = ml.Character(name=sp.name, gender=sp.gender, id=sid, traits=sp.notes)
        prompts = c.voice_prompts if c else {}
        return JSONResponse({"speaker": sid, "name": c.name if c else "", "gender": c.gender if c else "",
                             "prompts": prompts, "lang": entry["book"].lang,
                             "texts": cv.sample_texts(markup, chapters, sid, entry["book"].lang)})

    async def pull_start(request: Request):
        """Download one of the listed models through Ollama; only when the person asks for it."""
        from folivox.llm import GEMMA_TIERS, Ollama, PullError

        body = await request.json()
        model = str(body.get("model") or "")
        tier = next((x for x in GEMMA_TIERS if x[1] == model), None)
        if tier is None:
            raise api_error("Only the models in the list can be downloaded here")
        if pull.state == "running":
            raise api_error("A model is already being downloaded", 409)
        ollama = Ollama(model)
        fake = services.get("pull")
        if not fake and not ollama.running():
            raise api_error("Ollama is not running: install it from ollama.com or start it, then try again", 409)
        free = shutil.disk_usage(Path.home()).free / 1e9
        if free < tier[2] * 1.1 and model not in ollama.installed():
            raise api_error(f"Not enough free disk space: {tier[2] * 1.1:.0f} GB needed, {free:.0f} GB free", 409)
        pull.state, pull.model, pull.done, pull.total, pull.message = "running", model, 0, int(tier[2] * 1e9), ""
        pull.started = time.time()
        pull.cancel.clear()

        def progress(done: int, total: int, status: str) -> None:
            pull.done, pull.message = done, status
            if total:
                pull.total = total

        def work() -> None:
            try:
                (fake or ollama.pull)(model, progress, pull.cancel.is_set)
                pull.state = "cancelled" if pull.cancel.is_set() else "done"
            except PullError as e:
                pull.state, pull.message = "error", str(e)
            except Exception as e:  # noqa: BLE001 — shown to the person
                pull.state, pull.message = "error", str(e)

        threading.Thread(target=work, daemon=True, name="folivox-model-pull").start()
        return JSONResponse(pull.snapshot(), status_code=202)

    async def delete_model(request: Request):
        """Remove a downloaded model from Ollama (only the listed ones; the person asked for it with a button)."""
        from folivox.llm import GEMMA_TIERS, Ollama, PullError

        model = str((await request.json()).get("model") or "")
        if not any(x[1] == model for x in GEMMA_TIERS):
            raise api_error("Only the models in the list can be removed here")
        if pull.state == "running" and pull.model == model:
            raise api_error("This model is being downloaded", 409)
        try:
            (services.get("delete_model") or Ollama(model).delete)(model)
        except PullError as e:
            raise api_error(str(e), 409) from None
        return JSONResponse({"removed": model})

    async def pull_status(request: Request):
        return JSONResponse(pull.snapshot())

    async def pull_cancel(request: Request):
        pull.cancel.set()
        return JSONResponse(pull.snapshot())

    def run_cost(book, chapters, bible, speech, extra: int, llm: str | None, model: str | None) -> dict:
        """Tokens and USD of each mode with the chosen (or configured) language model, as the page shows them."""
        from folivox import config
        from folivox import markup_cost as mc
        from folivox.llm import LLMS

        provider = config.provider("llm", llm) or LLMS.default
        model = model or mc.default_model(provider)
        if mc.is_local(provider):
            return {"provider": provider, "model": model, "local": True}
        try:
            price = mc.price_of(provider, model)
        except ValueError as e:
            return {"provider": provider, "model": model, "error": str(e)}
        runs = {"preview": mc.estimate(chapters, book.lang, bible, price, model=model, provider=provider,
                                       analyse=mc.analyse_windows(chapters, bible, PREVIEW_ANALYSE),
                                       annotate=mc.annotate_windows(chapters, None, PREVIEW_MARK)),
                "best": mc.estimate(chapters, book.lang, bible, price, model=model, provider=provider,
                                    analyse=mc.analyse_windows(chapters, bible), annotate=speech, extra_calls=extra)}
        return {"provider": provider, "model": model, "local": mc.is_local(provider),
                "modes": {k: v.to_dict() for k, v in runs.items()}}

    async def estimate(request: Request):
        """How many model calls an automatic run needs (the page turns them into time for the chosen model)."""
        bid = request.path_params["book_id"]
        entry = state.book(bid)
        depth = int(request.query_params.get("depth", 2))
        chapter = request.query_params.get("chapter")
        picked = {int(x) - 1 for x in request.query_params.get("chapters", "").split(",") if x.strip().isdigit()}
        chapters = entry["book"].chapters(depth)
        bp = files(bid)[0]
        try:
            bible = ml.Bible.load(bp) if bp.is_file() else ml.Bible()
        except ml.LLMError as e:
            raise api_error(str(e)) from None
        every = ml.windows(chapters, ml.WINDOW_CHARS, only_speech=False)
        speech = [w for w in ml.windows(chapters, ml.WINDOW_CHARS, only_speech=True)
                  if (not chapter or w.chapter == int(chapter) - 1) and (not picked or w.chapter in picked)]
        left = max(bible.total_windows - bible.done_windows, 0) if bible.total_windows else len(every)
        every_speech = ml.windows(chapters, ml.WINDOW_CHARS, only_speech=True)
        stale: list[int] = []  # chapters whose text is not the text that was marked (their markup is ignored)
        mp = files(bid)[1]
        if mp.is_file():
            try:
                saved = mk.Markup.load(mp)
                book_chs = entry["book"].chapters(saved.depth)
                stale = [i for i, cm in sorted(saved.chapters.items())
                         if i >= len(book_chs) or (cm.sig and cm.sig != mk.chapter_sig(book_chs[i]))]
            except (mk.MarkupError, OSError, ValueError):
                stale = []
        paragraphs = sum(len(w.blocks) for w in speech)
        modes = {"preview": {"analyse": min(PREVIEW_ANALYSE, len(every)),
                             "annotate": min(PREVIEW_MARK, len(every_speech))},
                 "best": {"analyse": len(every) if bible.sampled else left, "annotate": len(speech),
                          # the second look: the merge and the unsure characters, then the doubtful places (measured:
                          # about 1 in 100 paragraphs after the rule check was tuned)
                          "extra": 8 + round(0.02 * paragraphs)}}
        # the average window of each stage (a short book has short windows, and the page counts time by them)
        read_chars = round(sum(len(w.text()) for w in every) / len(every)) if every else ml.WINDOW_CHARS
        mark_chars = round(sum(len(w.text()) for w in speech) / len(speech)) if speech else ml.WINDOW_CHARS
        cost = run_cost(entry["book"], chapters, bible, speech, modes["best"]["extra"],
                        request.query_params.get("llm") or None, request.query_params.get("model") or None)
        return JSONResponse({"analyse_windows": left, "annotate_windows": len(speech), "window_chars": mark_chars,
                             "read_chars": read_chars, "cost": cost,
                             "modes": modes, "chars": sum(len(w.text()) for w in every),
                             "speech_chapters": sorted({w.chapter for w in every_speech}), "stale_chapters": stale})

    def make_budget(body: dict, model: str):
        """The run's spend counter with the optional cap (``max_cost``, USD), checked as the render's Fish cap is."""
        from folivox import config
        from folivox import markup_cost as mc
        from folivox.llm import LLMS

        cap = body.get("max_cost")
        try:
            cap = float(cap) if cap not in (None, "") else None
        except (TypeError, ValueError):
            raise api_error("The spending cap must be a number of dollars") from None
        if cap is not None and cap <= 0:
            raise api_error("The spending cap must be more than zero")
        provider = config.provider("llm", body.get("llm") or None) or LLMS.default
        try:
            return mc.Budget(mc.price_of(provider, model), cap,
                             hint="raise the spending cap and start again to continue")
        except ValueError as e:
            raise api_error(str(e)) from None

    async def status(request: Request):
        return JSONResponse(task_of(request.path_params["book_id"]).snapshot())

    async def cancel(request: Request):
        t = task_of(request.path_params["book_id"])
        t.cancel.set()
        return JSONResponse(t.snapshot())

    async def start(request: Request):
        bid = request.path_params["book_id"]
        entry = state.book(bid)
        body = await request.json()
        if not isinstance(body, dict):
            raise api_error("Expected a JSON object")
        what = body.get("task")
        if what not in ("analyse", "auto", "prompts", "preview", "best"):
            raise api_error("task must be analyse, auto, prompts, preview or best")
        t = task_of(bid)
        if t.state == "running":
            raise api_error("The language model is already working on this book", 409)
        try:
            if "llm" in services:
                llm, model = services["llm"](body.get("llm") or None), str(body.get("model") or "fake")
            else:
                llm, model = ml.make_llm(body.get("llm") or None, body.get("model") or None)
        except ml.LLMError as e:
            raise api_error(str(e), 409) from None
        except Exception as e:  # noqa: BLE001 — a missing key, an unknown provider
            raise api_error(str(e)) from None
        budget = make_budget(body, model)
        book = entry["book"]
        depth = int(body.get("depth", 2))
        chapters = book.chapters(depth)
        expression = bool(body.get("expression_tags", False))
        sample = int(body["sample"]) if body.get("sample") else None
        only = body.get("chapters")
        if only and not isinstance(only, list):  # a string "12" would be read digit by digit
            raise api_error("chapters must be a list of chapter numbers")
        chapter_filter = {int(n) - 1 for n in only} if only else None
        bp, mp, cp = files(bid)
        for f in (bp, mp, cp):
            f.parent.mkdir(parents=True, exist_ok=True)
        t.llm, t.release, t.budget, t.reason = llm, False, budget, ""
        t.state, t.task, t.message, t.stats = "running", what, "", {}
        t.done = t.total = 0
        t.cancel.clear()

        def progress(stage: str, done: int, total: int, message: str) -> None:
            t.stage, t.done, t.total, t.message = stage, done, total, message

        def work_prompts() -> None:
            from folivox.voices import character_voice as cv

            try:
                bible = ml.Bible.load(bp) if bp.is_file() else ml.Bible()
                markup = mk.Markup.load(mp) if mp.is_file() else None
                ids = {int(i) for i in body.get("speakers") or []} or None
                for sid in ids or ():  # a speaker made by hand has no entry in the character list yet
                    sp = markup.speaker(sid) if markup else None
                    if sp and not any(c.id == sid for c in bible.characters):
                        bible.characters.append(ml.Character(name=sp.name, gender=sp.gender, id=sid, lines=1,
                                                             traits=sp.notes))
                if not bible.characters:
                    raise ml.LLMError("Analyse the characters first")
                n = cv.design_all(bible, book.lang, llm, markup, chapters, ids, bool(body.get("regenerate")), progress)
                bible.save(bp)
                t.state, t.message = "done", f"{n} characters described"
            except Exception as e:  # noqa: BLE001
                t.state, t.message = "error", str(e)
            finally:
                unload(t)

        def work_mode() -> None:
            """The two ways the page offers: a preview (a sample, minutes) or the best that can be done (every
            chapter chosen, the whole book read, the characters looked at again, doubtful places asked about again)."""
            cache = ml.Cache(cp)
            runner = ml.Runner(llm, model, cache, cancelled=t.cancel.is_set, budget=budget)
            try:
                bible = ml.Bible.load(bp) if bp.is_file() else ml.Bible()
                markup = mk.Markup.load(mp) if mp.is_file() else mk.Markup(depth=depth)

                def save_b(b) -> None:
                    b.save(bp)

                def save_m(m, b) -> None:
                    mk.stamp(m, chapters)
                    m.save(mp)
                    b.save(bp)

                preview = what == "preview"
                bible = ml.analyse(chapters, book.lang, runner, bible, save=save_b, progress=progress,
                                   max_windows=PREVIEW_ANALYSE if preview else None, cancelled=t.cancel.is_set)
                if not preview and not t.cancel.is_set():
                    ml.consolidate(chapters, book.lang, bible, runner, save=save_b, progress=progress,
                                   cancelled=t.cancel.is_set)
                if not t.cancel.is_set():
                    markup, stats = ml.annotate(chapters, book.lang, bible, runner, markup,
                                                chapter_filter=None if preview else chapter_filter, save=save_m,
                                                progress=progress, cancelled=t.cancel.is_set, depth=depth,
                                                max_windows=PREVIEW_MARK if preview else None)
                    t.stats = dict(stats.__dict__)
                    if not preview and not t.cancel.is_set():
                        settled, left = ml.second_opinion(chapters, book.lang, bible, runner, markup, save=save_m,
                                                          progress=progress, cancelled=t.cancel.is_set)
                        t.stats.update(settled=settled, to_check=left)
                    mk.stamp(markup, chapters)
                    markup.save(mp)
                    bible.save(bp)
                t.state = "cancelled" if t.cancel.is_set() else "done"
            except ml.CostCapReached as e:  # progress is saved; a higher cap carries on
                t.state, t.reason, t.message = "cancelled", "cost_cap", str(e)
            except Exception as e:  # noqa: BLE001 — shown to the user; progress is saved
                t.state, t.message = "error", str(e)
            finally:
                cache.close()
                unload(t)

        def work() -> None:
            if what == "prompts":
                return work_prompts()
            if what in ("preview", "best"):
                return work_mode()
            cache = ml.Cache(cp)
            runner = ml.Runner(llm, model, cache, cancelled=t.cancel.is_set, budget=budget)
            try:
                bible = ml.Bible.load(bp) if bp.is_file() and body.get("resume", True) else ml.Bible()
                if what == "analyse" or bible.sampled or bible.done_windows < max(bible.total_windows, 1):
                    bible = ml.analyse(chapters, book.lang, runner, bible, save=lambda b: b.save(bp),
                                       max_windows=sample if what == "analyse" else None, progress=progress,
                                       cancelled=t.cancel.is_set)
                if what == "auto" and not t.cancel.is_set():
                    markup = mk.Markup.load(mp) if mp.is_file() else mk.Markup(depth=depth)

                    def save(m, b):
                        mk.stamp(m, chapters)
                        m.save(mp)
                        b.save(bp)

                    markup, stats = ml.annotate(chapters, book.lang, bible, runner, markup,
                                                expression_tags=expression, chapter_filter=chapter_filter, save=save,
                                                progress=progress, cancelled=t.cancel.is_set, depth=depth)
                    mk.stamp(markup, chapters)
                    markup.save(mp)
                    bible.save(bp)
                    t.stats = stats.__dict__
                t.state = "cancelled" if t.cancel.is_set() else "done"
                t.message = f"{bible.voices_needed()} voices needed" if what == "analyse" else t.message
            except ml.CostCapReached as e:  # progress is saved; a higher cap carries on
                t.state, t.reason, t.message = "cancelled", "cost_cap", str(e)
            except Exception as e:  # noqa: BLE001 — shown to the user; progress is saved
                t.state, t.message = "error", str(e)
            finally:
                cache.close()
                unload(t)

        threading.Thread(target=work, daemon=True, name="folivox-llm-markup").start()
        return JSONResponse(t.snapshot(), status_code=202)

    return [Route("/api/llm/models", models),
            Route("/api/llm/models", delete_model, methods=["DELETE"]),
            Route("/api/llm/pull", pull_start, methods=["POST"]),
            Route("/api/llm/pull", pull_status),
            Route("/api/llm/pull/cancel", pull_cancel, methods=["POST"]),
            Route("/api/books/{book_id}/characters", get_characters),
            Route("/api/books/{book_id}/characters", put_characters, methods=["PUT"]),
            Route("/api/books/{book_id}/speakers/{speaker:int}/voice", voice_info),
            Route("/api/books/{book_id}/llm/start", start, methods=["POST"]),
            Route("/api/books/{book_id}/llm/status", status),
            Route("/api/books/{book_id}/llm/estimate", estimate),
            Route("/api/books/{book_id}/llm/cancel", cancel, methods=["POST"])]

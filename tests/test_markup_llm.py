"""Automatic markup: character bible, annotation with verification, resuming and caching, all with a fake LLM."""

import json
import re

import pytest
from typer.testing import CliRunner

from folivox import markup as mk
from folivox import markup_llm as ml
from folivox.book import Block, BlockKind, Chapter
from folivox.llm import LLM, recommend_model

P1 = "Анна вошла в комнату."
P2 = "— Ты куда? — спросила Анна."
P3 = "— Домой, — ответил Борис и вздохнул."
P4 = "Дверь тихо закрылась."
P5 = "— Подожди, — сказала она."  # "she" = Anna, resolved from context
CH = [Chapter("Глава 1", [Block(BlockKind.PARAGRAPH, t) for t in (P1, P2, P3, P4)]),
      Chapter("Глава 2", [Block(BlockKind.PARAGRAPH, P5)])]
CAST = {"Анна": "f", "Борис": "m"}


class FakeLLM(LLM):
    """Plays a model that understands the cast above; modes make it misbehave."""

    name = "fake"

    def __init__(self, mode="good"):
        self.mode, self.calls = mode, []

    def complete(self, prompt, system=None, temperature=0.3):
        return self.complete_json(prompt, system, temperature)

    def complete_json(self, prompt, system=None, temperature=0.2):
        self.calls.append(system[:20])
        if self.mode == "down":
            raise OSError("connection refused")
        if self.mode == "garbage" or (self.mode == "flaky" and len(self.calls) % 2 == 1):
            return "I cannot do that, sorry"
        excerpt = prompt.split("Excerpt:\n")[-1] if "Excerpt:" in prompt else prompt.split("Paragraphs:\n")[-1]
        blocks = re.findall(r"\[B(\d+)\] (.*)", excerpt)
        if system.startswith("You are a careful"):
            chars = []
            for name, g in CAST.items():
                n = sum(1 for _, t in blocks if t.startswith("—") and name in t)
                if any(name in t for _, t in blocks):
                    chars.append({"name": name, "aliases": [], "gender": g, "role": "main", "lines": n,
                                  "traits": "calm"})
            return json.dumps({"narration_person": "third", "characters": chars}, ensure_ascii=False)
        if system.startswith("You know the whole book"):  # the second look at the characters
            if "groups of names" in prompt:
                return json.dumps({"same": [["Анна", "Аня"]]}, ensure_ascii=False)
            who = "m" if "Борис" in prompt else "?"
            return json.dumps({"gender": who, "age": "adult", "traits": "slow"}, ensure_ascii=False)
        speakers = json.loads(re.search(r"\n(\[\{.*\}\])\n", prompt).group(1))
        ids = {s["name"]: s["id"] for s in speakers}
        if "numbered parts" in system:  # the "parts" method: which parts does a character say?
            out = []
            for m in re.finditer(r"\[B(\d+)\]\n((?:  \d+\| .*\n?)+)", prompt):
                parts = re.findall(r"  (\d+)\| (.*)", m.group(2))
                full = " ".join(t for _, t in parts)
                who = next((n for n in CAST if n in full), "Анна" if "она" in full else None)
                if self.mode == "swap" and who in CAST:  # the model gives the speech to the other character
                    who = "Борис" if who == "Анна" else "Анна"
                if not parts[0][1].startswith("—") or who is None or who not in ids:
                    continue
                out.append({"b": int(m.group(1)), "who": {"1": 9 if self.mode == "unknown_speaker" else ids[who]}})
            return json.dumps({"blocks": out, "new_characters": [], "updates": []}, ensure_ascii=False)
        out = []
        for b, t in blocks:
            if not t.startswith("—"):
                continue
            who = next((n for n in CAST if n in t), "Анна" if "она" in t else None)
            if who is None or who not in ids:
                continue
            if self.mode == "rewrite" and "Домой" in t:
                text = t.replace("Домой", "Домой же")
            elif self.mode == "unknown_speaker":
                text = f"<|speaker:9|>{t}"
            elif self.mode == "tags":
                text = f"<|speaker:{ids[who]}|>[sighs] {t} [fakeeffect]"
            else:
                m = re.match(r"(— .*?[?!.,])( ?— .*)$", t)
                text = (f"<|speaker:{ids[who]}|>{m.group(1)} <|speaker:0|>{m.group(2).strip()}" if m
                        else f"<|speaker:{ids[who]}|>{t}")
            out.append({"b": int(b), "text": text})
        return json.dumps({"blocks": out, "new_characters": [], "updates": []}, ensure_ascii=False)


def run_analyse(llm=None, **kw):
    llm = llm or FakeLLM()
    runner = ml.Runner(llm, "fake-model", ml.Cache(None))
    return ml.analyse(CH, "ru", runner, **kw), runner, llm


def test_names_match_across_cases_and_aliases():
    assert ml.same_name("Анна", "Анны") and ml.same_name("Иван", "Ивана") and ml.same_name("Петр", "Петя")
    assert not ml.same_name("Ольга", "Олег") and not ml.same_name("Маша", "Миша")
    b = ml.Bible()
    anna = b.merge({"name": "Анна", "gender": "f", "lines": 2})
    again = b.merge({"name": "Анну", "aliases": ["Аня"], "lines": 3, "traits": "quiet"})
    assert again is anna and anna.lines == 5 and "Аня" in anna.aliases and "Анну" in anna.aliases
    assert b.merge({"name": "Аня"}) is anna  # found through the alias
    assert b.merge({"name": ""}) is None


def test_analyse_builds_the_bible_and_counts_voices():
    bible, runner, _ = run_analyse()
    names = {c.name: c for c in bible.characters}
    assert set(names) == {"Анна", "Борис"} and names["Анна"].gender == "f" and names["Борис"].lines == 1
    assert names["Анна"].lines >= 1 and bible.narration == "third"
    assert {c.id for c in bible.characters} == {1, 2}
    assert bible.voices_needed(min_lines=1) == 3 and bible.voices_needed() == 1  # nobody reaches 3 lines here


def test_bible_round_trip(tmp_path):
    bible, _, _ = run_analyse()
    f = tmp_path / "b.characters.json"
    bible.save(f)
    again = ml.Bible.load(f)
    assert again.to_dict() == bible.to_dict()
    f.write_text("[1]")
    with pytest.raises(ml.LLMError):
        ml.Bible.load(f)


def test_analyse_is_resumable_and_cached():
    cache = ml.Cache(None)
    llm = FakeLLM()
    runner = ml.Runner(llm, "m", cache)
    bible = ml.analyse(CH, "ru", runner)
    calls = len(llm.calls)
    assert calls == 2  # one window per chapter
    # same book, same cache, fresh bible: nothing is asked again
    again = ml.analyse(CH, "ru", ml.Runner(llm, "m", cache))
    assert len(llm.calls) == calls and {c.name for c in again.characters} == {c.name for c in bible.characters}
    # a bible that already did the first window continues with the second
    part = ml.Bible()
    part.done_windows = 1
    llm2 = FakeLLM()
    ml.analyse(CH, "ru", ml.Runner(llm2, "m", ml.Cache(None)), part)
    assert len(llm2.calls) == 1


def test_analyse_survives_a_garbage_model_and_reports_a_dead_one():
    bible, runner, _ = run_analyse(FakeLLM("garbage"))
    assert bible.characters == [] and runner.failures == 2
    flaky, runner, _ = run_analyse(FakeLLM("flaky"))
    assert {c.name for c in flaky.characters} == {"Анна", "Борис"}  # the retry worked
    with pytest.raises(ml.LLMError, match="did not answer"):
        run_analyse(FakeLLM("down"))


def test_more_than_fifteen_speakers_share_the_last_ids():
    b = ml.Bible()
    names = "Альфа Бета Гамма Дельта Эпсилон Зита Тета Йота Каппа Лямбда Сигма Омега Ро Хи Пси Тау Фи Эта Мю Ню".split()
    for i, n in enumerate(names):
        b.merge({"name": n, "gender": "f" if i % 2 else "m", "lines": 40 - i})
    b.assign_ids()
    ids = [c.id for c in sorted(b.characters, key=lambda c: -c.lines)]
    assert ids[:13] == list(range(1, 14)) and set(ids[13:]) <= {14, 15} and max(ids) <= 15


class ShortLLM(FakeLLM):
    """Answers the marking request for the first paragraph only and stops (seen with a small cloud model), for the
    first ``times`` marking calls; the other passes answer as FakeLLM does."""

    def __init__(self, times):
        super().__init__()
        self.times, self.marking = times, 0

    def complete_json(self, prompt, system=None, temperature=0.2):
        reply = super().complete_json(prompt, system, temperature)
        if not system.startswith("You mark up books"):
            return reply
        self.marking += 1
        if self.marking > self.times:
            return reply
        d = json.loads(reply)
        d["blocks"] = d["blocks"][:1]
        return json.dumps(d, ensure_ascii=False)


def test_a_reply_that_leaves_out_paragraphs_with_speech_is_asked_again():
    llm = ShortLLM(times=1)
    markup, stats, bible = annotate(llm=llm)
    spans = {(s.block, s.speaker) for s in markup.chapters[0].spans}
    assert spans == {(1, bible.find("Анна").id), (2, bible.find("Борис").id)}  # P3 is marked after the retry
    assert markup.chapters[0].review == [] and stats.short_windows == 0
    assert llm.marking == 3  # two windows, one asked twice


def test_a_reply_that_stays_short_keeps_what_it_answered_and_reports_the_rest(tmp_path):
    cache = ml.Cache(tmp_path / "c.sqlite")
    bible = run_analyse()[0]
    runner = ml.Runner(ShortLLM(times=99), "fake-model", cache)
    markup, stats = ml.annotate(CH, "ru", bible, runner)
    assert {s.block for s in markup.chapters[0].spans} == {1}  # what it did answer is kept
    assert markup.chapters[0].review == [{"b": 2, "code": "skipped"}]  # the rest is on the review list
    assert stats.short_windows == 1 and stats.skipped == 1 and stats.failed_windows == 0
    # the short reply was not cached: the next run asks again, and a full answer marks the paragraph
    markup, stats = ml.annotate(CH, "ru", bible, ml.Runner(FakeLLM(), "fake-model", cache), markup)
    assert {s.block for s in markup.chapters[0].spans} == {1, 2} and markup.chapters[0].review == []
    assert stats.short_windows == 0 and stats.skipped == 0


def test_cli_auto_reports_paragraphs_the_model_left_out(tmp_path, monkeypatch):
    from folivox import cli_markup
    from folivox.cli import app

    book = tmp_path / "b.txt"
    book.write_text("Глава 1\n\n" + "\n\n".join((P1, P2, P3, P4)) + "\n\nГлава 2\n\n" + P5 + "\n", encoding="utf-8")
    monkeypatch.setattr(cli_markup, "_llm", lambda provider, model: (ShortLLM(times=99), "fake-model"))
    r = CliRunner()
    assert r.invoke(app, ["markup", "analyse", str(book)]).exit_code == 0
    res = r.invoke(app, ["markup", "auto", str(book)])
    assert res.exit_code == 0, res.output
    out = " ".join(res.output.split())  # rich wraps long lines
    assert "The model left out 1 paragraph(s) with speech (asked again, 1 window(s) stayed short)" in out


def test_a_late_speaker_never_takes_an_id_one_character_already_holds():
    """15 speakers fill ids 1..15 one each; a 16th met later (while marking) must not be read in another
    character's voice (here a woman would land on id 15, held by a man): the narrator reads them instead."""
    b = ml.Bible()
    names = "Альфа Бета Гамма Дельта Эпсилон Зита Тета Йота Каппа Лямбда Сигма Омега Ро Хи Пси".split()
    for i, n in enumerate(names):
        b.merge({"name": n, "gender": "m" if i == 14 else "f", "lines": 40 - i})
    b.assign_ids()
    before = {c.name: c.id for c in b.characters}
    assert sorted(before.values()) == list(range(1, 16)) and not any(c.shared for c in b.characters)
    b.merge({"name": "Тау", "gender": "f", "lines": 1})
    b.merge({"name": "Фи", "gender": "m", "lines": 1})
    b.assign_ids()
    assert {c.name: c.id for c in b.characters if c.name in before} == before  # nobody is renumbered
    assert b.find("Тау").id == 0 and b.find("Фи").id == 0
    assert [c.name for c in b.unassigned()] == ["Тау", "Фи"]


def test_shared_ids_are_marked_and_survive_a_saved_list_from_before():
    b = ml.Bible()
    names = "Альфа Бета Гамма Дельта Эпсилон Зита Тета Йота Каппа Лямбда Сигма Омега Ро Хи Пси Тау Фи".split()
    for i, n in enumerate(names):
        b.merge({"name": n, "gender": "f" if i % 2 else "m", "lines": 40 - i})
    b.assign_ids()
    shared = {c.name: c.id for c in b.characters if c.shared}
    assert set(shared) == set(names[13:]) and all(c.id <= 13 for c in b.characters if not c.shared)
    assert all(i == (15 if b.find(n).gender == "f" else 14) for n, i in shared.items())
    # a list saved before the flag existed: two characters on one id are read as a shared id
    old = b.to_dict()
    for c in old["characters"]:
        c.pop("shared")
    again = ml.Bible.from_dict(old)
    again.merge({"name": "Эта", "gender": "f", "lines": 1})
    again.assign_ids()
    assert again.find("Эта").id == 15 and again.find("Эта").shared
    assert {c.name for c in again.characters if c.shared} == set(names[13:]) | {"Эта"}
    assert again.unassigned() == []


def test_the_markup_names_a_shared_speaker_as_shared():
    bible = run_analyse()[0]
    for c in bible.characters:  # Anna and Boris speak least of 16: they go to the shared ids
        c.id = 0
    for i, n in enumerate("Альфа Бета Гамма Дельта Эпсилон Зита Тета Йота Каппа Лямбда Сигма Омега Ро Хи".split()):
        bible.merge({"name": n, "gender": "m", "lines": 50 - i})
    old = mk.Markup()
    old.ensure_speaker(15, "Анна")  # what the markup said before: the first woman on the shared id
    markup, stats, bible = annotate(bible=bible, markup=old)
    anna, boris = bible.find("Анна"), bible.find("Борис")
    assert (anna.id, boris.id) == (15, 14) and anna.shared and boris.shared
    sp15, sp14 = markup.speaker(15), markup.speaker(14)
    assert sp15.name == ml.SHARED_NAMES[15] and "Анна" in sp15.aliases and sp15.gender == "f"
    assert sp14.name == ml.SHARED_NAMES[14] and set(sp14.aliases) >= {"Борис", "Хи"}
    assert {s.speaker for c in markup.chapters.values() for s in c.spans} == {14, 15}  # the speech is still marked
    sp14.name = "Прохожие"  # a name the person chose stays
    markup, _, _ = annotate(bible=bible, markup=markup)
    assert markup.speaker(14).name == "Прохожие"


class NewcomerLLM(FakeLLM):
    """While marking, lists a first-person narrator as a character called "Narrator" (as a benchmark run did) and
    answers his speech with the id the list then gives him, as the model there did."""

    def complete_json(self, prompt, system=None, temperature=0.2):
        reply = json.loads(super().complete_json(prompt, system, temperature))
        if system.startswith("You mark up books"):
            reply["updates"] = [{"name": "Narrator", "aliases": ["Jake"], "gender": "m"}]
            reply["new_characters"] = [{"name": "Первая жена", "gender": "f"}]
        return json.dumps(reply, ensure_ascii=False)


def test_a_narrator_listed_while_marking_never_lands_on_another_characters_voice():
    """The benchmark case: 14 speakers from the reading, then the marking model lists "Narrator" (the first-person
    narrator, the book's busiest speaker) and a minor woman. The narrator went to id 14, which one man held, so his
    lines were read in that man's voice."""
    bible = run_analyse()[0]
    for i, n in enumerate("Альфа Бета Гамма Дельта Эпсилон Зита Тета Йота Каппа Лямбда Сигма Ми".split()):
        bible.merge({"name": n, "gender": "m", "lines": 50 - i})
    bible.assign_ids()
    held = {c.name: c.id for c in bible.characters}
    assert len(held) == 14 and max(held.values()) == 14
    markup, stats, bible = annotate(llm=NewcomerLLM(), bible=bible)
    assert {c.name: c.id for c in bible.characters if c.name in held} == held
    assert bible.find("Narrator").id == 0 and bible.find("Первая жена").id == 15
    assert not any(c.shared for c in bible.characters) and bible.unassigned() == []
    assert markup.speaker(14).name == bible.characters[[c.id for c in bible.characters].index(14)].name
    assert all(s.name != "Narrator" for s in markup.speakers if s.id)


def annotate(llm=None, bible=None, **kw):
    bible = bible or run_analyse()[0]
    llm = llm or FakeLLM()
    runner = ml.Runner(llm, "fake-model", ml.Cache(None))
    markup, stats = ml.annotate(CH, "ru", bible, runner, **kw)
    return markup, stats, bible


def test_annotate_marks_speech_and_leaves_the_text_alone():
    markup, stats, bible = annotate()
    ids = {c.name: c.id for c in bible.characters}
    spans = markup.chapters[0].spans
    q = [s for s in spans if s.block == 1]
    assert len(q) == 1 and q[0].speaker == ids["Анна"] and P2[q[0].start:q[0].end].startswith("— Ты куда?")
    assert [s.speaker for s in spans if s.block == 2] == [ids["Борис"]]
    assert stats.marked == 3 and stats.fell_back == 0
    assert markup.chapters[1].spans[0].speaker == ids["Анна"]  # "she" resolved by the model with the bible
    segs = markup.segments(0, 1, P2)
    assert "".join(s.text for s in segs) == P2
    assert {s.name for s in markup.speakers} >= {"Narrator", "Анна", "Борис"}
    # the result is valid for the book
    clean, problems = mk.bind(markup, CH)
    assert not problems and clean.chapters.keys() == markup.chapters.keys()


def test_annotate_never_accepts_changed_text_or_unknown_speakers():
    markup, stats, _ = annotate(FakeLLM("rewrite"), method="text")
    assert not [s for s in markup.chapters[0].spans if s.block == 2]  # the rewritten paragraph stays narrator
    assert stats.fell_back == 1 and stats.marked == 2
    markup, stats, _ = annotate(FakeLLM("unknown_speaker"))
    assert stats.unknown_speakers >= 1 and not any(s.speaker == 9 for c in markup.chapters.values() for s in c.spans)
    markup, stats, _ = annotate(FakeLLM("garbage"))
    assert stats.failed_windows == 2 and markup.is_empty()


def test_expression_tags_are_validated_against_the_database():
    markup, stats, _ = annotate(FakeLLM("tags"), expression_tags=True)
    used = {t for c in markup.chapters.values() for s in c.spans for t in s.tags}
    used |= {m.tag for c in markup.chapters.values() for m in c.marks}
    assert used == {"sighs"} and stats.dropped_tags >= 1
    off, stats2, _ = annotate(FakeLLM("tags"), expression_tags=False, method="text")
    assert not any(s.tags for c in off.chapters.values() for s in c.spans) and not any(
        c.marks for c in off.chapters.values())


def test_annotation_is_cached_and_chapters_can_be_filtered():
    cache = ml.Cache(None)
    bible = run_analyse()[0]
    llm = FakeLLM()
    ml.annotate(CH, "ru", bible, ml.Runner(llm, "m", cache))
    n = len(llm.calls)
    ml.annotate(CH, "ru", bible, ml.Runner(llm, "m", cache))
    assert len(llm.calls) == n
    markup, _ = ml.annotate(CH, "ru", bible, ml.Runner(FakeLLM(), "m", ml.Cache(None)), chapter_filter={1})
    assert set(markup.chapters) == {1}


def test_model_recommendation_follows_memory():
    assert recommend_model(64) == "gemma4:31b-it-qat" and recommend_model(36) == "gemma4:26b-a4b-it-qat"
    assert recommend_model(16) == "gemma4:12b-it-qat" and recommend_model(8) == "gemma4:12b-it-qat"
    assert recommend_model(None) == "gemma4:12b-it-qat"


def test_json_extraction_is_forgiving():
    assert ml.extract_json('Sure!\n```json\n{"a": 1,}\n```') == {"a": 1}
    with pytest.raises(ValueError):
        ml.extract_json("no json")


def test_cli_analyse_and_auto(tmp_path, monkeypatch):
    from folivox import cli_markup
    from folivox.cli import app

    book = tmp_path / "b.txt"
    book.write_text("Глава 1\n\n" + "\n\n".join((P1, P2, P3, P4)) + "\n\nГлава 2\n\n" + P5 + "\n", encoding="utf-8")
    monkeypatch.setattr(cli_markup, "_llm", lambda provider, model: (FakeLLM(), "fake-model"))
    r = CliRunner()
    res = r.invoke(app, ["markup", "analyse", str(book)])
    assert res.exit_code == 0, res.output
    assert "Voices needed" in res.output and (tmp_path / "b.characters.json").is_file()
    assert r.invoke(app, ["markup", "characters", str(book)]).exit_code == 0
    res = r.invoke(app, ["markup", "auto", str(book)])
    assert res.exit_code == 0, res.output
    saved = json.loads((tmp_path / "b.markup.json").read_text(encoding="utf-8"))
    assert saved["chapters"] and any(s["name"] == "Анна" for s in saved["speakers"])
    assert r.invoke(app, ["markup", "check", str(book)]).exit_code == 0
    assert r.invoke(app, ["markup", "models"]).exit_code == 0


def test_cli_says_who_shares_an_id_and_who_has_none(tmp_path):
    from folivox.cli import app

    book = tmp_path / "b.txt"
    book.write_text("Глава 1\n\n" + P2 + "\n", encoding="utf-8")
    b = ml.Bible("ru")
    for i, n in enumerate("Альфа Бета Гамма Дельта Эпсилон Зита Тета Йота Каппа Лямбда Сигма Омега Ро".split()):
        b.characters.append(ml.Character(n, gender="m", lines=40 - i, id=i + 1))
    b.characters += [ml.Character("Хи", gender="m", lines=5, id=14, shared=True),
                     ml.Character("Пси", gender="m", lines=4, id=14, shared=True),
                     ml.Character("Тау", gender="m", lines=6, id=15)]  # id 15 is one man's own: no woman joins him
    b.merge({"name": "Фи", "gender": "f", "lines": 1})
    b.assign_ids()
    b.save(ml.bible_path(book))
    res = CliRunner().invoke(app, ["markup", "characters", str(book)])
    assert res.exit_code == 0, res.output
    assert "14 shared" in res.output and "share speaker 14" in res.output
    assert "No speaker id left for 1" in res.output and "Фи" in res.output


# -- voice-design prompts for the characters -----------------------------------------------------------------

class VoiceLLM(FakeLLM):
    def __init__(self, reply=None):
        super().__init__()
        self.reply, self.prompts = reply, []

    def complete_json(self, prompt, system=None, temperature=0.2):
        self.prompts.append(prompt)
        if self.reply == "down":
            raise OSError("no server")
        return self.reply if self.reply is not None else json.dumps({
            "simple": "A woman in her thirties, calm and warm, speaking slowly.",
            "standard": "A woman in her thirties with a soft, warm mid-range voice, a native Russian speaker, "
                        "unhurried and gentle.",
            "detailed": "A woman in her thirties with a soft, warm mid-range voice, a native Russian speaker, "
                        "unhurried, with small pauses and a trace of tiredness that lifts when she is amused."})


def test_voice_ideas_are_english_tiered_and_never_use_the_name():
    from folivox.voices import character_voice as cv

    anna = ml.Character("Анна", gender="f", age="adult", traits="quiet", id=1, lines=5)
    ideas = cv.ideas_for(anna, "ru", VoiceLLM(), others=["A gruff old man"], lines=["— Ты куда?"])
    assert ideas.from_llm and set(ideas.prompts) == set(cv.TIERS)
    assert len(ideas.prompts["simple"]) <= cv.LIMITS["simple"] < len(ideas.prompts["detailed"]) <= cv.LIMITS["detailed"]
    # a tier that names the character or imitates a person is replaced by the plain fallback
    bad = json.dumps({"simple": "Анна, a woman with a warm voice", "standard": "She sounds like a famous actress today",
                      "detailed": "A calm adult woman, native Russian speaker, even pace, soft low timbre."})
    ideas = cv.ideas_for(anna, "ru", VoiceLLM(bad))
    base = cv.fallback_prompts(anna, "Russian")
    assert ideas.prompts["simple"] == base["simple"] and ideas.prompts["standard"] == base["standard"]
    assert ideas.prompts["detailed"].startswith("A calm adult woman")
    assert all(ord(ch) < 128 for p in base.values() for ch in p)  # plain English even for a Russian book


def test_voice_ideas_without_a_usable_model_fall_back_to_plain_text():
    from folivox.voices import character_voice as cv

    boris = ml.Character("Борис", gender="m", age="old", traits="ворчливый")
    for llm in (None, VoiceLLM("garbage"), VoiceLLM("down")):
        ideas = cv.ideas_for(boris, "ru", llm)
        assert not ideas.from_llm and "elderly man" in ideas.prompts["simple"]
        assert "ворчливый" not in ideas.prompts["standard"]  # a non-English trait is not pasted into the prompt


def test_design_all_tells_later_characters_how_to_differ_and_can_regenerate():
    from folivox.voices import character_voice as cv

    bible, _, _ = run_analyse()
    llm = VoiceLLM()
    assert cv.design_all(bible, "ru", llm) == 2
    assert all(c.voice_prompts for c in bible.characters)
    assert "none" in llm.prompts[0].split("already described")[1][:120] and "A woman in her thirties" in llm.prompts[1]
    assert cv.design_all(bible, "ru", llm) == 0  # nothing new: no calls
    assert len(llm.prompts) == 2
    first = bible.characters[0].id
    assert cv.design_all(bible, "ru", llm, speakers={first}) == 1  # one speaker again
    assert cv.design_all(bible, "ru", llm, regenerate=True) == 2


def test_sample_texts_are_the_characters_own_lines():
    from folivox.voices import character_voice as cv

    markup, _, bible = annotate()
    anna = next(c.id for c in bible.characters if c.name == "Анна")
    lines = cv.speaker_lines(markup, CH, anna, max_chars=300)
    assert lines and all(len(x) >= 25 for x in lines) or lines == []
    assert cv.sample_texts(markup, CH, anna, "ru")
    assert cv.sample_texts(None, None, 3, "ru") == cv.sample_texts(None, None, 4, "ru")  # the built-in sample


def test_loose_speaker_tags_are_read_as_the_tag_they_mean():
    from folivox.markup_llm import _fix_speaker_tags

    assert _fix_speaker_tags('<|speaker:1>"Hi" <|speaker:0|>said <speaker:2|>x <|speaker: 3 >y') == \
        '<|speaker:1|>"Hi" <|speaker:0|>said <|speaker:2|>x <|speaker:3|>y'


def test_spanish_dialogue_dash_without_space_counts_as_speech():
    from folivox.markup_llm import SPEECH_MARKS

    assert SPEECH_MARKS.search("—¿Vas a alguna parte? —preguntó Daniel.")
    assert SPEECH_MARKS.search("— Ты куда? — спросил он.")
    assert not SPEECH_MARKS.search("Plain narration without any speech.")


def test_split_parts_and_spans_from_parts():
    from folivox import markup as mk
    from folivox.markup_llm import spans_from_parts, split_parts

    t = "—Perdone —graznó el anciano—, busco al señor Harrow."
    assert [t[a:b] for a, b in split_parts(t)] == ["—Perdone", "—graznó el anciano", "—, busco al señor Harrow."]
    t2 = '"To Mum\'s," Clara said. "I promised to help her."'
    assert [t2[a:b] for a, b in split_parts(t2)] == ['"To Mum\'s,"', "Clara said.", '"I promised to help her."']
    assert [(s.start, s.end, s.speaker) for s in spans_from_parts(1, t, {"1": 4, "3": 4})] == [(0, 8, 4), (27, 52, 4)]
    assert spans_from_parts(1, t, [4, 0, 4]) == spans_from_parts(1, t, {"1": 4, "3": 4})
    assert spans_from_parts(1, t, {"9": 4}) == []  # a part that does not exist is ignored
    for bad in ("x", [1], {"a": 1}):
        with pytest.raises(mk.MarkupError):
            spans_from_parts(1, t, bad)


def test_gender_and_age_are_the_majority_of_what_the_windows_said():
    b = ml.Bible()
    for g in ("f", "m", "m"):
        b.merge({"name": "Саша", "gender": g, "age": "young", "lines": 1})
    c = b.characters[0]
    assert c.gender == "m" and c.age == "young" and c.votes["gender"] == {"f": 1, "m": 2}
    assert c.unsure() == ["gender", "age"] or c.unsure() == ["gender"]  # 2 against 1 is still shaky for the gender
    b.merge({"name": "Саша", "gender": "m", "lines": 1})
    b.merge({"name": "Саша", "gender": "m", "age": "young", "lines": 1})
    assert c.unsure() == []
    again = ml.Bible.from_dict(b.to_dict()).characters[0]
    assert again.votes == c.votes and again.gender == "m"


def test_consolidate_joins_names_and_looks_again_at_the_unsure_ones():
    b = ml.Bible()
    b.merge({"name": "Анна", "gender": "f", "age": "adult", "lines": 3})
    b.merge({"name": "Аня", "gender": "f", "lines": 1})
    b.merge({"name": "Борис", "gender": "?", "lines": 2})
    b.merge({"name": "Вера", "gender": "f", "age": "old", "lines": 2})
    runner = ml.Runner(FakeLLM(), "fake-model", ml.Cache(None))
    seen = []
    ml.consolidate(CH, "ru", b, runner, progress=lambda st, d, n, m: seen.append((st, m)))
    assert sorted(c.name for c in b.characters) == ["Анна", "Борис", "Вера"]
    anna = next(c for c in b.characters if c.name == "Анна")
    assert "Аня" in anna.aliases and anna.lines == 4
    boris = next(c for c in b.characters if c.name == "Борис")
    assert boris.gender == "m" and boris.age == "adult" and "slow" in boris.traits
    assert ("consolidate", "Борис") in seen


def test_a_good_run_marks_every_chapter_done_and_flags_nothing():
    markup, stats, _ = annotate()
    assert all(c.done for c in markup.chapters.values()) and len(markup.chapters) == 2
    assert stats.flagged == 0 and all(not c.review for c in markup.chapters.values())


def test_the_rule_check_flags_a_speaker_that_contradicts_the_attribution():
    markup, stats, bible = annotate(FakeLLM("swap"))
    assert stats.flagged >= 2
    codes = [(r["code"], r.get("named"), r.get("marked")) for c in markup.chapters.values() for r in c.review]
    assert ("attribution", "Анна", "Борис") in codes  # "— спросила Анна" but the speech went to Борис
    again = mk.Markup.from_dict(markup.to_dict())  # the review list and the done flags are part of the file
    assert [c.review for c in again.chapters.values()] == [c.review for c in markup.chapters.values()]
    assert all(c.done for c in again.chapters.values())


def test_check_paragraph_reads_pronouns_and_attributions():
    from folivox.markup_llm import check_paragraph, split_parts

    b = ml.Bible()
    for _ in range(3):
        b.merge({"name": "Анна", "gender": "f", "age": "adult", "lines": 2})
        b.merge({"name": "Борис", "gender": "m", "age": "adult", "lines": 2})
    b.assign_ids()
    anna, boris = b.find("Анна").id, b.find("Борис").id
    t = "— Домой, — ответил он."
    parts = split_parts(t)
    assert check_paragraph(t, parts, [boris, 0], b, "ru") == []
    wrong = check_paragraph(t, parts, [anna, 0], b, "ru")
    assert wrong and wrong[0]["code"] == "gender"
    t2 = "— Ты куда? — спросила Анна."
    assert check_paragraph(t2, split_parts(t2), [anna, 0], b, "ru") == []
    assert check_paragraph(t2, split_parts(t2), [boris, 0], b, "ru")[0]["code"] == "attribution"


def test_check_paragraph_does_not_cry_wolf_on_names_and_pronouns_that_are_not_the_speaker():
    from folivox.markup_llm import check_paragraph, split_parts

    b = ml.Bible()
    for _ in range(3):
        b.merge({"name": "Анна", "gender": "f", "age": "adult", "lines": 2})
        b.merge({"name": "Борис", "gender": "m", "age": "adult", "lines": 2})
        b.merge({"name": "Вера Петровна", "aliases": ["мать"], "gender": "f", "age": "old", "lines": 2})
    b.assign_ids()
    anna, boris, vera = b.find("Анна").id, b.find("Борис").id, b.find("Вера Петровна").id
    def ids_of(t, who):  # the first part is the speech, the rest is narration
        return [who] + [0] * (len(split_parts(t)) - 1)

    later = "— Не открывай ему больше, — сказала мать так тихо, что Анна едва расслышала."  # a name in a later clause
    for t, who in ((later, vera),
                   ("— Ты куда? — спросил Борис Анну.", boris),  # the listener in the accusative
                   ("— Домой, — ответил Борис, глядя на неё.", boris),  # an object pronoun
                   ("— Дурак, — выдохнула Анна, и он засмеялся.", anna),  # "он" is not the speaker here
                   ("— Спасибо, — сказала ему Анна.", anna)):
        assert check_paragraph(t, split_parts(t), ids_of(t, who), b, "ru") == [], t
    t = "— Домой, — сказала она."  # a real contradiction is still found
    assert check_paragraph(t, split_parts(t), ids_of(t, boris), b, "ru")[0]["code"] == "gender"
    t = "— Домой, — сказал Борис. Анна кивнула."  # narration that is not next to the speech says nothing
    assert check_paragraph(t, split_parts(t), ids_of(t, boris), b, "ru") == []


def test_second_opinion_settles_a_doubtful_place_with_a_better_answer():
    markup, stats, bible = annotate(FakeLLM("swap"))
    flagged = sum(len(c.review) for c in markup.chapters.values())
    assert flagged >= 2
    runner = ml.Runner(FakeLLM(), "fake-model", ml.Cache(None))  # a model that gets it right this time
    settled, left = ml.second_opinion(CH, "ru", bible, runner, markup)
    assert settled == flagged and left == 0
    anna = bible.find("Анна").id
    spans = [(s.block, s.speaker) for s in markup.chapters[0].spans]
    assert (1, anna) in spans  # "— Ты куда? — спросила Анна" belongs to Анна again


def test_a_preview_reads_a_sample_and_a_full_run_starts_over():
    b = ml.Bible()
    runner = ml.Runner(FakeLLM(), "fake-model", ml.Cache(None))
    ml.analyse(CH, "ru", runner, b, max_windows=1)
    assert b.sampled and b.total_windows == 1
    ml.analyse(CH, "ru", runner, b)  # the full reading after the preview does not think it is already done
    assert not b.sampled and b.done_windows == b.total_windows >= 2
    markup, _ = ml.annotate(CH, "ru", b, runner, max_windows=1)
    assert not any(c.done for c in markup.chapters.values())  # a sample does not mark a chapter as processed


def test_a_shared_first_name_does_not_join_two_people_with_different_full_names():
    bible = ml.Bible("en")
    for e in ({"name": "Robert Cohn", "aliases": ["Robert", "Cohn"], "gender": "m", "lines": 30},
              {"name": "Robert Prentiss", "aliases": ["Robert"], "gender": "m", "lines": 6},  # met without Cohn
              {"name": "Darcy", "gender": "m", "lines": 4},
              {"name": "Mr. Darcy", "aliases": ["Darcy"], "gender": "m", "lines": 2},
              {"name": "Elizabeth Bennet", "aliases": ["Elizabeth"], "gender": "f", "lines": 5},
              {"name": "Miss Elizabeth Bennet", "aliases": ["Elizabeth"], "gender": "f", "lines": 1}):
        bible.merge(e)
    assert [(c.name, c.lines) for c in bible.characters] == [
        ("Robert Cohn", 30), ("Robert Prentiss", 6), ("Darcy", 6), ("Elizabeth Bennet", 6)]
    bible.assign_ids()
    assert bible.find("Robert Cohn").id != bible.find("Robert Prentiss").id


class CastLLM(LLM):
    """Reports the characters each excerpt names; asked which names are one person, gives ``same``."""

    name = "fake"

    def __init__(self, same=()):
        self.same = [list(g) for g in same]

    def complete(self, prompt, system=None, temperature=0.3):
        return self.complete_json(prompt, system, temperature)

    def complete_json(self, prompt, system=None, temperature=0.2):
        if "groups of names" in prompt:
            return json.dumps({"same": self.same})
        if system.startswith("You know the whole book"):
            return json.dumps({"gender": "m", "age": "adult", "traits": "dry"})
        excerpt = prompt.split("Excerpt:\n")[-1]
        chars = []
        if "Cohn" in excerpt:
            chars.append({"name": "Robert Cohn", "aliases": ["Robert", "Cohn"], "gender": "m", "lines": 2})
        if "Prentiss" in excerpt:  # the model calls him "Robert" too
            chars.append({"name": "Robert Prentiss", "aliases": ["Robert"], "gender": "m", "lines": 1})
        return json.dumps({"narration_person": "first", "characters": chars})


def test_reading_keeps_robert_prentiss_apart_from_robert_cohn():
    text = ['"Hello," said Robert Cohn. "Robert is my name."', '"Good evening," said Robert Prentiss.']
    chapters = [Chapter(f"Chapter {i}", [Block(BlockKind.PARAGRAPH, t * 3)]) for i, t in enumerate(text)]
    bible = ml.analyse(chapters, "en", ml.Runner(CastLLM(), "m", ml.Cache(None)))
    assert bible.total_windows == 2
    assert [(c.name, c.lines) for c in bible.speakers()] == [("Robert Cohn", 2), ("Robert Prentiss", 1)]
    assert len({c.id for c in bible.speakers()}) == 2


def test_consolidate_joins_names_in_full_but_not_through_a_shared_first_name():
    def bible():
        return ml.Bible("en", characters=[
            ml.Character("Robert Cohn", aliases=["Robert", "Cohn"], gender="m", lines=30),
            ml.Character("Robert Prentiss", gender="m", lines=6),
            ml.Character("Bob Cohn", gender="m", lines=3),
            ml.Character("Fitzwilliam Darcy", gender="m", lines=8),
            ml.Character("Darcy", gender="m", lines=2)])

    same = [["Robert Prentiss", "Robert"], ["Robert Cohn", "Bob Cohn"], ["Fitzwilliam Darcy", "Darcy"]]
    b = bible()
    ml.consolidate([], "en", b, ml.Runner(CastLLM(same), "m", ml.Cache(None)), max_look=0)
    assert [(c.name, c.lines) for c in b.characters] == [
        ("Robert Cohn", 33), ("Robert Prentiss", 6), ("Fitzwilliam Darcy", 10)]
    b = bible()
    b.characters[1].aliases.append("Robert")  # now "Robert" is both of theirs: it points at neither
    ml.consolidate([], "en", b, ml.Runner(CastLLM([["Robert", "Robert Prentiss"]]), "m", ml.Cache(None)), max_look=0)
    assert len(b.characters) == 5

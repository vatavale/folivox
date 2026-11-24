# Speakers and expression tags (markup)

A book can have **several voices** (a narrator and characters) and **expression tags** (`[sighs]`,
`[whispering]`…). The markup is a separate layer: it never contains or changes the book text. It is stored in
`BOOK.markup.json` next to the book (the studio keeps its own copy per uploaded book) and `folivox render`
picks it up by itself; `--no-markup` ignores it, `--markup FILE` names another file.

Books without markup are read exactly as before.

## What it says

- **Speakers**: id 0 is the narrator and reads everything not given to someone else; ids 1–15 are characters
  (the range of Fish's `<|speaker:N|>` tags). Each speaker has a name and optionally a **voice**: an id from
  `folivox voices` or a path to a recording. A speaker without a voice is read with the book's main voice.
- **Spans**: a range of characters inside a paragraph belongs to a speaker, optionally with delivery tags put
  in front of it. A paragraph can be split between narrator and characters («— Ты куда? — спросила соседка»).
- **Marks**: an inline expression tag at an offset, where the effect should happen.
- **Language**: a span may carry the language of its text (`"lang": "en"`, ISO 639-1) when that is not the book's
  language; without it the fragment is in the book's language, so files made before this existed read the same.
  See [Books with several languages](#books-with-several-languages).
- The markup remembers a signature of each chapter's text. If you replace the book with another edition,
  chapters whose text changed are ignored (the narrator reads them) and `folivox markup check` says so.

Every fragment is read by one speaker with that speaker's voice; the voice, the speaker and the tags are part
of the fragment's cache key, so changing one re-renders only what it touches.

Expression tags are written into the text only for the Fish engines (`fish-s2`, `fish-drama3`); other engines
read the text without them. `--no-expression-tags` turns them off for a render, and the markup can switch them
off for the book (`@expression_tags off`). Characters may use different voices with every local engine; a
cloud engine reads the whole book in one voice.

With a Fish engine each speaker's voice must be your own recording, a generated voice, or the voice of someone
who agreed: the bundled voices are not used (see [Fish Audio engines](fish-engines.md#voices-of-real-people)).

## Books with several languages

A Russian textbook with English phrases, a German novel with French dialogue: every fragment is read in its own
language, with the same voice. The engines are multilingual and a cloned voice speaks every language of the
engine, but a text mixing two languages read without a hint is pronounced with the wrong rules (English words
spelled out or read with a Russian accent). So each fragment carries its language:

- **By script, always on, no model**: Latin words and phrases inside Cyrillic text, Cyrillic inside Latin,
  Han/kana/Hangul inside either are found by Unicode script. A run is whole words; the spaces, numbers and light
  punctuation between its words belong to it, and punctuation attached to its last word goes with it («How are
  you?»). Quotes, brackets and numbers at the edges stay with the surrounding text. Roman numerals,
  abbreviations such as USB, web addresses and one- or two-letter words are left to the book's language. Latin
  is told between English, German, French, Spanish and Italian by function words and accents (English when
  nothing says). These are not stored in the markup file.
- **Same script (English inside German, French inside English)** cannot be seen by script. When a book has
  more than one language (a language with at least 2% of its letters, found over the whole book), the automatic
  markup also asks the model for the language of each part it already looks at and stores it as `"lang"` in
  the span. The answer is checked: the code must be on the allow-list of ISO 639-1 codes, differ from the book's
  language, and fit the part's letters (a Latin language needs Latin letters; function words must not say the
  book's own language). It costs no extra requests, and a one-language book gets exactly the old prompt. Only
  paragraphs with speech are sent to the model, so a same-script phrase in plain narration is not marked.
- **By hand**: the text form has `<|lang:xx|>` (`<|lang:-|>` goes back to the book's language), so
  `folivox markup export --format text` / `import` round-trip languages; the studio editor shows a small
  language label on every stretch that will be read in another language, found by script or marked.

Each such fragment is planned with the text rules of its language (numbers, abbreviations, sentence splitting;
Russian stress marks only for Russian; a language without Folivox rules, such as Polish, is only cleaned and split
into sentences and its numbers are left to the voice, so it needs an online engine that lists the language), tells the engine its language, and is checked by speech recognition in
that language. The language is part of the fragment's cache key (only when it is not the book's), so a book in
one language keeps its cache. A word covered by your [pronunciation list](pronunciation.md) stays in the book's
language, as the entry says how to read it there.

| Engine | What the language does |
|---|---|
| CosyVoice 3, Fish Audio S2 Pro, Drama 3 | No language parameter: the model reads the language of the text. The fragment is planned and cut in its language and checked in it. |
| ElevenLabs | Reads the language of the text. |
| OpenAI, Gemini | No language parameter; the "native speaker, no English accent" note names the fragment's language. |
| Cartesia, Fish Audio (cloud) | The fragment's language is sent in the request. |

An engine that does not list a language (or a language Folivox has no text rules for) reads such fragments in
the book's language, and the render log says how many. One-voice books use this too: the studio keeps the languages
of the fragments even when the speakers are not used.

## Working with the markup in the console

```bash
folivox markup export book.epub -o book.txt           # text form (an empty template if there is no markup yet)
folivox markup import book.epub book.txt              # read it back, check it, save BOOK.markup.json
folivox markup speaker book.epub 1 --name Anna --voice anna.wav --gender f
folivox markup speakers book.epub                     # who reads how much, which voice exists
folivox markup check book.epub                        # stale chapters, bad spans, unknown tags, missing voices
folivox markup tags --category emotion                # the tag list
folivox markup clear book.epub
```

The **text form** is one line per paragraph with Fish-style tags inserted:

```
@speaker 0 | Narrator |  |
@speaker 1 | Anna | anna.wav | f
@expression_tags on

## chapter 1: Home
B1.0  <|speaker:1|> [whispering] Ты куда? <|speaker:0|> — спросила соседка.
```

`<|speaker:N|>` switches the speaker until the next tag; `[tag]` is an expression tag at that spot. When you
import, Folivox removes every tag and checks that what is left is exactly the book's paragraph: an edit that
changes the text is refused and names the line. (Paragraphs with line breaks, such as poems, are not in the
text form; use the JSON.) `--format json` exports the stored form.

## Expression tags

`src/folivox/tts/expression_tags.json` lists the tags the vendors document, by category (emotion, tone, sound,
effect, pause), with the model they were seen for and the source page (the S2 Pro model card, the Drama 3 model
card and Fish Audio's emotion documentation, retrieved 2026-10-15). It holds only the short tag names. The
models take **free-form** tags, so a tag that is not in the list is kept (and flagged by `markup check`);
the list feeds the editor's picker and the prompts of the automatic markup. Fish advises at most three tags
per sentence, emotions at the start of a sentence, `[emphasis]` right before the word. Drama 3's own card shows
only `[sighs]` and `[laughs]`; the others were documented for S2 and are expected to work, listen first.

Refresh the list with `python tools/fish_tags.py` (prints tags found in the vendors' pages that the database
does not have yet).

## Automatic markup with a language model

A local model (or an API you choose) can find the characters and mark who speaks each part. It only
*annotates*: the text is never taken from the model.

```bash
folivox markup models                        # which Gemma 4 suits this computer, what is installed
folivox markup analyse book.epub             # read the book once: who is in it, how many voices it needs
folivox markup characters book.epub          # show the saved list again
folivox markup auto book.epub                # mark the speakers (--chapters 1-3, --expression-tags)
```

**Models.** Gemma 4 is multilingual and comes in sizes for different memory; Folivox proposes the largest that
fits (`gemma4:31b-it-qat` from 48 GB, `26b-a4b-it-qat` from 32 GB, `12b-it-qat` for everything smaller) and
you can use any other Ollama model, or an OpenAI-compatible API (`--llm openai --model NAME`, your key, the text
goes to that service). Set a default in `config.toml`: `[llm.ollama] model = "gemma4:12b-it-qat"`. The analysis
runs before the audio is made, never together with the voice model. Small models miss subtle speakers: check
the result.

A small test on an Apple-silicon Mac with 36 GB (made-up texts of 20 paragraphs, 15 with speech, the same story in
English, Russian, German and Spanish; counted: paragraphs whose quoted speech went to the right character):

| Model (Ollama) | Method | English | Russian | German | Spanish | Time per text |
|---|---|---|---|---|---|---|
| `gemma4:31b-it-qat` | repeat text | 15/15 | 15/15 | 15/15 | 3/15 | 4-5 min |
| `gemma4:31b-it-qat` | **parts** | - | 15/15 | 15/15 | 15/15 | 2.2-3 min |
| `Qwen3.8-27B` (unsloth GGUF, UD-Q6_K) | repeat text | 15/15 | 15/15 | - | 15/15 | 4-5 min |
| `gemma4:12b-it-qat` | repeat text | 15/15 | 15/15 | 3/15 | 12/15 | 1.2-1.6 min |
| `gemma4:12b-it-qat` | **parts** | 15/15 | 15/15 | 15/15 | 14/15 | 0.8-1.2 min |

One run each, so read it as a rough picture. The "parts" method (the default now) is both more accurate and faster,
and it is what lets the 12B model handle German and Spanish. The first version, which asked for a list with one
speaker per part, did badly with 12B (7/15 on English): it miscounted the parts and forgot that a speech continues
after the attribution; asking for a map of part number to speaker, with one example, fixed that.

Models that were tried with the repeat-the-text method and dropped: Ministral-3-3B (0-11/15), Cohere tiny-aya-global
3.3B (0-7/15), Liquid LFM2.5-2.6B (0-5/15), EuroLLM-9B (0-4/15), and the two smallest Gemma 4 builds (below). They
were not retried with the parts method, which may suit small models better.

The two smallest Gemma 4 builds (`e4b` 6 GB, `e2b` 4 GB) were tried and **dropped from the list**: they are about ten
times faster, but unstable. `e2b` worked only on English (10/15) and returned unmarked text for Russian, German
and Spanish (0/15); `e4b` handled all four languages (9-12/15) but the same English text gave 0/15 once and
12/15 on the next run, and an unreliable result is worse than a slow one. A speaker tag with a missing bar
(`<|speaker:1>`) is understood, and a paragraph that starts with a dialogue dash with no space after it (Spanish «—¿Vas?») counts as speech.

**Memory of the book ("bible").** `analyse` goes through the book in windows of about 5000 characters. For each
window the model gets the characters found so far (name, nicknames, gender, age and manner of speech of those named
in the window, active in the last windows or the busiest) and reports who speaks, thinks or matters. What it says
about gender and age is a **vote**: the answer is the majority over the windows, so one wrong first impression does
not stick, and a character whose gender or age is still a guess (one vote, or nearly tied) is marked as unsure. A
correction made by hand in the studio counts as a heavy vote and is not outvoted later. Names are merged across
cases and nicknames. `BOOK.characters.json` holds the list with the number of **voices the book needs**.

**More than 15 speakers.** Speaker ids stop at 15, so each character keeps its own id only while 15 or fewer speak.
With more, the 13 who speak most keep theirs and the rest share two: 14 (men and unknown) and 15 (women). Those
characters are marked `"shared": true` in the list, and the markup names the two speakers *Other men* and *Other
women* (with the names of everyone on them), so a voice given to one of them is plainly a voice for all of them. An
id that one character already holds is never shared with another: a character met later while marking, when no id
is free, keeps none and the narrator reads them; `folivox markup characters` and `auto` list them. A character
the model lists as "Narrator" (a first-person narrator met while marking) gets no id at all: the narrator's own
voice, speaker 0, reads them.

**Looking again (the best quality).** When the whole book has been read, `consolidate` asks the model which of the
names are one person and joins them, then looks again, with passages from all over the book (pronouns, forms of
address, who is called a child or a grandfather), at every character whose gender or age is still unsure. This is
what the voice ideas rely on. After the marking, a **rule check** reads only the text: the name in an attribution
("said Boris", the clause right next to the speech, the name in its base form) must be the speaker of that speech,
and a subject pronoun (or the ending of a Russian speech verb) must not contradict the speaker's gender; a name or an
object pronoun ("said Anna to him", "so softly that Anna barely heard") is never taken as the speaker. On 268
paragraphs of a real Russian novel this left 3 doubts, all of them real mistakes of the model, where the first
version of the rules raised 13 (5 of them false). A paragraph that fails, that the model skipped or that it answered unusably goes on
the **review list** in the markup file; `second_opinion` asks about each of them again with six paragraphs before and
two after and keeps the new answer only if the rule check accepts it. Whatever is left is for the person.

**Two ways to run it in the studio.** *Try on a sample* reads a spread sample of the book (a few windows) and marks a
few windows, to show how it works in minutes. *Mark the whole book* does everything above for the chapters chosen in step 2:
the whole book is read, the characters are looked at again, the doubtful places are asked about again; it can take
hours, is saved as it goes and carries on where it stopped. In the console: `folivox markup auto BOOK --best`.

**Marking.** `auto` takes the paragraphs that contain speech. Folivox cuts each one into numbered parts (at dashes,
quotes and sentence ends) and the model only answers *which parts which character says*; it never has to repeat
the text, so punctuation it would "correct" (Spanish dialogue dashes) cannot ruin a paragraph. With
`--expression-tags` the older method is used instead: the model repeats each paragraph with `<|speaker:N|>`
markers and `[tag]`s from the tag list (at most one per sentence).
The model gets the relevant bible entries and the last few marked paragraphs, so "he said" and pronouns resolve
across chapters, and may add or correct entries as it reads. The answer is accepted for a paragraph only if
removing the markers gives exactly the book's paragraph (whitespace differences are tolerated), the speaker
exists and the tags are in the tag list. Otherwise the narrator reads that paragraph. The summary at the end
counts how many fell back. Expression tags are a separate switch, off by default: the models already read
emotion reasonably well on their own.
A reply must answer every paragraph of the window that opens with speech. One that leaves some out (a model that
answers the first paragraph and stops) is asked again like any other bad reply; if every try stays short, the most
complete one is used, the paragraphs it left out are read by the narrator and go on the review list, the summary
says how many, and the reply is not cached, so the next run asks about that window again.

Everything is **cached and resumable**: the replies are kept in `BOOK.llm-cache.sqlite` and the list and the
markup are saved after every window, so an interrupted run continues where it stopped and an unchanged window
is never asked twice (`--fresh` forgets the replies). The result is the ordinary markup file: export it as
text, fix what the model got wrong, import it back, assign voices with `folivox markup speaker`.

## Studio: the markup editor

Several voices are an **optional extra**: most books sound fine read by one voice. Step *4 · Voice* starts with a
choice of two cards, **One voice** (the default: one voice reads the whole book) and **Several voices** (a narrator
and a separate voice for each character, like a radio play). The voice list below it then means the book's voice,
or, with several voices on, the **narrator's voice** (everything the characters do not say). Pressing *Render* with
several voices on but no characters marked yet says so instead of silently reading with one voice. The choice is
remembered per book (a book that already has markup opens in *Several voices*); switching back to one voice keeps the
markup. The engine is chosen first, in step *3 · Reading engine*: online engines read the whole book in one voice,
so with one of them chosen *Several voices* is greyed out (marked *not supported by engine*) and the mode switches back to
*One voice*.

With *Several voices* on, step 4 shows the **Characters** box. It does not run anything: it shows the **state** and
has one button, **Characters and voices…**, which opens the editor, where the automatic marking and every manual
tool live. The state says what is found ("Found 3 characters, 12 marked passages"), which chapters with dialogue are
processed and which are not (only the chapters chosen in step 2 count: one you left out needs no markup, and a not-yet-processed chosen chapter is shown in red), and who has which
voice (the characters without their own voice are read by the narrator). With nothing set up it says so and points to
the editor. A chapter counts as processed once the automatic marking (or an edit) has touched it.

**Time, measured** (36 GB Mac, Russian prose, 5000-character windows, the estimate within 2% of the real run): reading a
window takes about 40 s with gemma4 12b and 100 s with 31b; marking a window 90 s and 240 s; a short second look
(one character or one doubtful paragraph) 6-9 s and 12-27 s. Two chapters (59k characters) in *Mark the whole book*: 19 min
with 12b, 47 min with 31b; *Try on a sample* of a whole novel: 7 min with 12b, about 20 min with 31b.

**Time.** The editor shows an estimate *before* you press the button and the progress line shows the time left while it
works. The estimate counts the model calls the run needs (the server's `/llm/estimate`) and turns them into time by
the model's size; it is a rough guess. During a run the speed is measured and the estimate corrected, and the speed
of the last run is remembered in the browser for that model. A long estimate suggests trying one chapter first.

**Language models are downloaded from the studio.** If the chosen model is not installed, the *Mark automatically* button asks once (name and size), checks
that Ollama is running and that the disk has room, downloads the recommended model with a progress bar, a time left
and *Stop* (a stopped download resumes later), and goes on with the marking. *Options* in the editor also has a
*Download* button for the model chosen in the list and *Remove from this computer* for an installed one. Only the
models in the list can be downloaded or removed here; Ollama itself is installed by the person (ollama.com).


In the editor the top of the right panel has two buttons, **Mark the whole book** and **Try on a sample** (each with its
estimated time), a progress bar with the time left and a plain result. *Options* (collapsed) holds the model (the
largest installed one is chosen by itself), the model download, and *Analyse characters* alone. After a run the panel
lists the characters, offers the next step (*Choose voices*), and, when the rule check or the second look left some,
shows the **places to check**: the places are marked in the text with an orange bar, and the panel steps through them
(*Show*, ‹ ›, *Looks fine*); a place leaves the list when you say it is fine or give that passage to a speaker
yourself. Running again asks first, because passages you changed by hand may be replaced. The top bar has the chapter
picker, undo and redo (the arrows) and a close cross; edits are saved by themselves.

Each speaker's card has **gender** and **age** lists: they are what the automatic reading found (an amber frame means
"not sure, please check"), you can change them, and the voice ideas use them.

Below the button the manual tools are always visible:

- The **text** of the chapter is shown with every passage coloured by its speaker; the narrator's text has no
  colour. Pick the chapter in the list at the top (or with the arrows).
- **Assign**: select text (it may span paragraphs); a bar under the text then says *The selected text will be read by:*
  and shows a chip for every speaker (the narrator is 0): click one, or press its number key (0–9). The bar is always
  there, dimmed until something is selected. *Clear selection* in the speaker panel gives the selected text back to
  the narrator and removes its tags.
- **Speakers**: rename, choose the **voice** of each (from the voice list; "main voice" = the book's voice),
  add (up to 16 in all) or remove a speaker (their passages return to the narrator). With a Fish engine the
  bundled voices are marked with `*` and are refused (see [Fish Audio engines](fish-engines.md#voices-of-real-people)).
- **Expression tags** (a nested disclosure): put the cursor where the effect should happen and click a tag in the list (search by name
  or category). A tag at the very start of a passage is a delivery tag of that passage. Click a tag in the text to
  remove it. *Use expression tags* is the per-book switch.
- **Listen**: select a passage (up to 1500 characters, whole paragraphs are read) and press *Listen* in the bar under
  the text, to the right of the speaker chips. It is read with the speakers' voices and tags without touching the
  audiobook. The list next to the button chooses the **engine** to listen with (the local ones; it starts with the
  engine of step 3), so a passage can be tried with, say, CosyVoice 3 and Fish S2 Pro in turn.
- **Automatic markup**: see above; progress is saved and the editor reloads the result so you can correct it.
- **Voice ideas**: the *Voice ideas* button of a speaker writes three **English** voice descriptions from the
  character list (simple, standard, detailed: the simple one works with any voice-design model, the detailed one
  helps strong ones; a weak model ignores words it does not know). They never contain the character's name or a
  real person, and each character is asked to differ from the voices described before. Edit them, choose one,
  pick a generator (cloud providers that can design a voice in the book's language; the estimate and the
  "text is sent and billed" confirmation are shown) and make 3 takes that read one of the character's own
  lines. **You listen and choose**: no take is preferred automatically (speech recognition is not used), takes
  from several descriptions stay in the list, *Use for this speaker* adds the take to your voices and gives it
  to the speaker. In the console: `folivox markup voices BOOK` (add `-g qwen3-voicedesign` or a cloud provider to
  make takes into `BOOK.voices/`).
- **Saving and Undo / Redo**: every edit is **saved by itself** a second after you stop (the top bar says *Saving…*,
  then *Saved.*, or shows the reason in red if the markup does not fit the book); there is no Save button. Closing
  saves what is pending. Undo/Redo (the arrows, or Ctrl/Cmd+Z, with Shift to redo) cover every edit since the editor
  was opened; Ctrl/Cmd+S saves at once.

Paragraphs that contain line breaks (poems) are shown but cannot be marked in the editor; the JSON can address
them. The same operations exist in the console (`folivox markup export` / `import`).

## Studio routes

The studio keeps the markup per book (`GET/PUT /api/books/{id}/markup`, `GET /api/markup/tags`) and uses it in
renders and previews (options `use_markup`, `expression_tags`). Automatic markup runs as a background task per
book: `POST /api/books/{id}/llm/start` (`{"task": "analyse" | "auto", "model", "llm", "chapters",
"expression_tags", "sample"}`), `GET .../llm/status`, `GET .../llm/estimate?chapter=N` (windows of a run), `POST .../llm/cancel`; models: `POST /api/llm/pull` `{"model"}`, `GET /api/llm/pull` (progress), `POST /api/llm/pull/cancel`, `DELETE /api/llm/models` `{"model"}`; the character list is
`GET/PUT /api/books/{id}/characters` and the model proposals `GET /api/llm/models`. `GET /api/books/{id}/languages?voice=ID` gives the language shares of the whole book, the letters each speaker reads per language and the languages their voice does not speak; `GET /api/books/{id}/text` lists the stretches found by script in each paragraph (`langs`).

# How it works

1. **Read** — a reader per format produces a common `Book` model: metadata, cover, nested sections,
   typed blocks (paragraph, epigraph, poem, quote…). Footnotes are skipped; PDF headers, footers and
   page numbers are removed and hyphenation is undone.
2. **Normalise** — a language module turns numbers and abbreviations into words (and restores ё
   for Russian), splits sentences ([razdel](https://github.com/natasha/razdel) for Russian, a
   punctuation splitter elsewhere, including `。！？` for Chinese and Japanese) and packs them into
   fragments.
3. **Synthesise** — CosyVoice 3 clones the voice zero-shot (or an optional Fish Audio engine, which runs
   in its own worker process and Python environment: [Fish Audio engines](fish-engines.md)). Each accepted
   take is stored in the work dir and recorded in an SQLite journal; its key hashes the text, the voice and
   the synthesis settings (engine, build and revision, plus speaker and expression tags when used), so a
   changed engine or build never reuses old audio. The optional [native runtime](native-runtime.md) (audio.cpp)
   has its own key (engine release, weights file and precision), so switching runtime never mixes takes silently. With [markup](markup.md) a paragraph is split by speaker first,
   and every fragment is read with its speaker's voice.
4. **Check** — Whisper transcribes the take; character error rate and speaking rate decide whether
   to retake.
5. **Assemble** — fragments are joined with structural pauses (sentence < paragraph < heading <
   chapter), loudness-normalised (EBU R128, −18 LUFS) and encoded to AAC in an `.m4b` with chapters.

## What the normaliser reads out, per language

Every language gets the same pipeline (abbreviations → roman numerals → dates → times → money → phone
numbers → units → ordinals → gender agreement → symbols → plain numbers); the data and the agreement
rules differ.

| | Roman numerals | Dates / times | Money, units | Agreement and other |
|---|---|---|---|---|
| ru | yes | yes | yes | cases, `ё`, stress marks |
| en | `Henry the eighth`, `Chapter four` | `July fourth`, `seven oh five` | `$5.50` → five dollars fifty cents | `Mr.` → Mister |
| de | `Ludwig der Vierzehnte`, `XVII. Jahrhundert` | `am dritten Mai`, `3.5.2020`, `7:30 Uhr` | `5,50 €`, units | `ein Mann` / `eine Frau` |
| fr | `Louis quatorze`, `XXe siècle` | `1er mai`, `7h30` | `5,50 €`, units | `une fille`, `vingt et une filles` |
| es | `Felipe segundo`, `siglo diecinueve` | `1.º de mayo`, `7:30` | `5,50 €`, units | `veintiuna mujeres`, `doscientas casas` |
| it | `Luigi quattordicesimo`, `XIX secolo` | `1º maggio`, `7:30` | `5,50 €`, units | `un'amica`, `uno zaino`, `3ª` |
| zh | | `7:30`, `2024年` digit by digit | `¥5.5`, units | `两个` instead of `二个` |
| ja | | `7:30` → `七時三十分` | `$5`, `¥500`, units | |
| ko | | `7:30`, months `유월`/`시월` | `₩5,000`, units | native counters: `세 명`, `두 시간`, `스무 살` |

Not covered yet: Japanese kanji readings (needs a morphological analyser such as `pyopenjtalk`),
heteronyms in English, and noun gender in French/German beyond common words and endings.

## Languages without rules

The nine languages above have their own rules, and so does Dutch (`lang/nl.py`: numbers with the hundreds written
together, dates such as 3-5-2024, times, euros, units, Roman numerals and ordinals such as 19e and 1ste, common
abbreviations such as bijv., d.w.z., dhr.). CosyVoice does not read Dutch; the Fish engines (local and online) and
the other online engines do. Any other language that an online engine can read (Fish Audio
reads 84 languages in all: Polish, Portuguese, Arabic, Hindi, Thai and so on) is handled by a rule-less
language: only whitespace, quotes and dashes are cleaned, and numbers, dates, units and abbreviations stay as they
are, because the voice model reads them itself. Sentences are split on the terminators of every script
(`. ! ? … 。 ؟ ۔ । ॥ ։ ።`, the Greek `;`), and Thai, Lao, Khmer and Burmese on runs of spaces. Fish's own
normalisation stays off and the request always names the language, since Fish's auto-detection mislabels chunks.
The book language comes from the file's metadata when it is a known code, otherwise it is guessed from the script
(Arabic, Devanagari, Thai, Greek, Hebrew … are never taken for Russian). Check amounts, Roman numerals and
abbreviations by ear in such books. The local CosyVoice engine still reads only the nine.

# Cloud reading engines

Besides the local CosyVoice 3 model, the whole book can be read by a paid online service. **The book
text is sent to the provider and billed to your account** — a full audiobook (≈ 10 h, 500,000 characters)
costs roughly $8–40 depending on the provider. Folivox therefore shows an estimate first and asks you to
confirm (checkbox in the studio, a question or `--yes` in the CLI), and `--max-cost` stops a run cleanly
at a spending cap (progress is saved; raise the cap and run the same command to continue — fragments already
rendered are never paid for twice).

| Engine (`--tts`) | Voice quality / languages | ≈ price per hour of audio¹ | Request limit | Notes and terms |
|---|---|---|---|---|
| `elevenlabs` | most natural; 90+ languages (v4), 29 (multilingual v2) | $4.0 (v4, v3, multilingual v2), $2.0 (flash) | 10,000 chars (v3: 5,000) | Free plan: personal, non-commercial only; selling audiobooks needs a paid plan. Audio tags (`[calm]`) on v3/v4. |
| `cartesia` | clean, fast; 40+ languages | $2.5 (Pro plan credits) | 5,000 chars | Commercial use needs a paid plan; output must not train competing models. Voice cloning from your own recording. |
| `fish` | Drama 3: directed in plain words; library voices, voice from a description, clone; 80+ languages | ≈ $2.5 English, ≈ $4.7 Russian (Drama 3, unconfirmed list price), $0.8–1.5 (S2) | 3,000 chars (not documented; fragments are 300) | Billed per UTF-8 byte; needs an API balance (free-plan credits do not cover the API). Fish Audio Community License: free below USD 500k revenue, "Built with Fish Audio", no voice imitation without consent. |
| `openai` | good, simple; 13 voices, style note on `gpt-4o-mini-tts` | $0.9 (`gpt-4o-mini-tts`), $0.8 (`tts-1`), $1.5 (`tts-1-hd`) | 4,096 chars | Cheapest. Disclose that the voice is AI-generated (OpenAI usage policy). |
| `gemini` | natural; 30 voices, style note, voice from a description | $0.8 (`gemini-3.8-flash-tts`) | 4,000 chars | Free tier exists, but Google may then use your text; paid tier does not. Output must not train competing models. |

¹ List prices seen in October 2026 at 14 characters per second of speech; approximate, check your dashboard.
The cloud voices are not clones of a reference recording: pick one of the provider's voices (with sex and
traits, as in the voice gallery), or have a voice made for the book: Gemini and ElevenLabs *design* (from a description) and
Cartesia *localize* (take any voice, e.g. an English one, and adapt it to the book's language with a native accent — the default for non-English books) / *clone* (your own voice only, with a confirmation). Such a voice lives in your account
for the whole run, so every fragment sounds the same. Tick *Keep this voice* in the studio (CLI: `--keep-voice
--tts-voice-name "Calm narrator"`) and it stays in your account and is listed under *My created voices* for
every later book (`--tts-voice "Calm narrator"`, `folivox tts-voices gemini`, delete with `--delete`); otherwise it
is deleted when the book is finished. A kept voice uses a slot of your provider plan.

### Fish Audio (Drama 3)

`FISH_API_KEY` (an API key from fish.audio/app/developers; add credit there, the free plan's credits do not apply to
the API). The same Drama 3 model also runs [on this computer](how-it-works.md) (`fish-drama3`); the cloud engine
needs no 32 GB Mac and is faster, the local one is free.

- **Voices.** *Pick a voice*: your own voices first, then the library voices Fish Audio has licensed (its own
  "Fish Official" voices first). A switch above the list adds the whole community library, a menu picks the voice
  language (the book's language by default; a voice of another language can read the book too) and a search box
  filters the list. Listings are cached on disk for 12 hours. Never use a
  voice that imitates a real person without consent. *Describe a voice*: `/v1/voice-design` ($0.01 per request) makes one candidate, which is saved as a
  private model so the whole book has one voice. *Clone*: one of your own voices becomes a private model
  (`enhance_audio_quality` on, Fish transcribes the sample itself; 10–30 s of clean speech works best, not measured yet). Made voices are
  deleted when the book is finished unless you keep them (`--keep-voice`).
- **Direction.** The *style note* is sent as a bracketed direction before every fragment, e.g.
  `[calm, warm audiobook narration] …`. Drama 3 and S2 read it as direction, not as words, in any language. No
  note is sent by default.
- **Several voices.** Fish Audio is the one online engine that takes a [markup](markup.md) with several speakers.
  Each fragment is sent with its speaker's own voice, so a character is a Fish voice: pick one in the studio's speaker
  list (your own voices, the library list shown under *Voice*, a clone of a catalog voice, or *Describe a new voice…*),
  or write it into the markup file as `fish:<voice id>`, `fish-design:<description>` or `fish-clone:<catalog voice or
  recording>`. Voices made from a description or a clone live in your account only for the run (deleted at the end
  unless `--keep-voice`; an interrupted run goes on with the same ones). The markup's expression tags are sent as
  bracketed directions in front of the fragment (`[whispering] …`). Cost follows the text sent, as for one voice.
- **Fragment size.** Fish cuts text into chunks of at most 300 characters itself (`chunk_length`, 100–300) and
  conditions each chunk on the previous one. Folivox already sends sentence-packed fragments of 300 characters, so
  one request is one chunk and the voice stays steady; longer fragments (`--max-chars`) only save requests.
- **Languages and normalisation.** The book can be in any language Fish has voices for (84 in all: the nine below plus 75 in
  the studio's *Other languages* list, for example Polish, Portuguese, Dutch, Turkish, Arabic, Hindi, Thai, Welsh, Kazakh). For the nine languages
  with Folivox rules (ru, en, de, fr, es, it, zh, ja, ko) and for Dutch (nl) numbers and abbreviations are expanded
  before sending; for every other language the text is sent as it is. Fish's own text normalisation (`normalize`) is **off on purpose**
  and the fragment's `language` is always sent: in a test of Drama 3 on the live API, Fish's normaliser added nothing
  in pt, pl, nl, tr, ar, cs, sv, uk and hi (it has no rules for most of them), made Russian worse ("1250" came
  out as "тысяч двести пятьдесят") and rewrote a 24-hour time in English; with `language` left out its automatic
  detection mislabelled Turkish and Czech as Spanish and Ukrainian as Russian. Without any help the model reads
  years, ordinals, percentages, times, km/kg and common abbreviations correctly in pt, pl, nl, tr, ar and mostly
  uk. It is weaker in cs, sv and hi (numbers such as 1250 failed in about half of the draws), with amounts with
  decimals ("1250,50 zł") and with Roman numerals in some languages: listen to those before rendering a whole
  book. The studio shows a hint under the language field. Quality control (the retake gate) is more tolerant in
  such languages, since Whisper writes digits its own way.
- **Concurrency.** Starter accounts (< $100 prepaid) run 5 simultaneous requests; Folivox uses 3.
- **Not used yet.** Fish can pin pronunciation: phoneme tags in the text (`<|phoneme_start|>…<|phoneme_end|>`;
  CMU Arpabet for English, tone-numbered pinyin for Chinese, pitch-annotated romaji for Japanese) and
  pronunciation dictionaries (`pronunciation_dictionary`, up to 3 per request, rules `key` → phonemes). They need
  phonemes, not respellings, so Folivox's [pronunciation list](pronunciation.md) is not mapped onto them yet; Russian
  has no phoneme format (stress marks are in the text).

```bash
export OPENAI_API_KEY=…                       # or save the key in `folivox ui`; keys: GEMINI/OPENAI/ELEVENLABS/CARTESIA_API_KEY
# in a container: OPENAI_API_KEY_FILE=/run/secrets/openai_api_key reads it from a file (Docker secret); the plain variable wins if both are set
folivox tts-voices openai                     # voices with sex and traits (Cartesia/ElevenLabs: ids and names)
folivox render book.epub --tts openai --tts-voice coral --preview 600            # first step: ~1 minute for a few cents
folivox render book.epub --tts elevenlabs --tts-voice Rachel --tts-model auto --max-cost 10 --yes
folivox render book.epub --tts gemini --tts-mode design --tts-describe "calm warm woman, unhurried" --tts-style "audiobook narration"
folivox render book.epub --tts cartesia --tts-mode clone --voice my-voice --own-voice   # my-voice = your own catalog voice
```

Quality control still works (Whisper QA with retakes), but every retake is billed, so cloud runs default to
2 takes (`--takes`) and never re-split a failed fragment; the estimate includes a retake allowance. Fragments
are limited to the provider's request size and sent in a few parallel requests (Retry-After is honoured; see
[Request limits](#request-limits)).
Reading speed (`--speed`) and Russian stress marks (`--stress all`) do not apply; the narration style is
`--tts-style` (OpenAI and Gemini get a neutral audiobook note by default). Audio from commercial APIs is
yours to keep under the provider's terms and is never bundled with Folivox.

## Request limits

Every provider caps how many requests an account may send per minute (RPM) and per day (RPD); the caps depend on
the model and on your tier (free, or paid tier 1, 2, 3 by spend). Hitting one gives an HTTP 429 even when your
balance is fine. Google does not publish per-minute numbers for its Gemini TTS models: check your project in
[AI Studio](https://aistudio.google.com/) (Rate limits), and see the
[Gemini rate-limit guide](https://ai.google.dev/gemini-api/docs/rate-limits). Limits are per Google project, not per key.

What Folivox does:

- Gemini starts with 1 request at a time and at most 10 requests a minute. Other engines have no limit of their own.
- On a "too many requests" reply it waits as long as the provider asks (Gemini's `retryDelay`, or `Retry-After`),
  halves its own rate for the rest of the run, and retries. Nothing is lost: finished fragments are kept and are
  not sent again.
- A **per-day** limit is not retried (waiting a minute cannot help). The run stops with a message; the quota resets
  at midnight Pacific time. Run the same command again then, pick another model, or enable billing for the project.

To change the rate, put this in `config.toml` (or set `FOLIVOX_GEMINI_RPM` / `FOLIVOX_GEMINI_PARALLEL`; the section
name is the engine, `0` removes the rate limit):

```toml
[cloud.gemini]
requests_per_minute = 4   # lower this if you still see rate-limit errors on a free tier
parallel = 1
```

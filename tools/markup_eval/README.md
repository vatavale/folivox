# Markup evaluation against PDNC

Measures how well `folivox markup auto` attributes speakers, with any provider and model, against the
[Project Dialogism Novel Corpus](https://github.com/Priya22/project-dialogism-novel-corpus) (PDNC: every quotation of
28 English novels annotated with its speaker), and what each run costs in time and tokens. A developer tool, not part
of the package.

```bash
python tools/markup_eval/markup_eval.py fetch      # PDNC files for the excerpts, from a pinned commit
python tools/markup_eval/markup_eval.py prepare    # cut the excerpts; gold quotations; Folivox's own paragraphs
python tools/markup_eval/markup_eval.py run gemma12b --llm ollama --model gemma4:12b-it-qat
python tools/markup_eval/markup_eval.py run sonnet --llm openai --model some/model -- --best   # folivox options after --
python tools/markup_eval/markup_eval.py score gemma12b sonnet
python tools/markup_eval/markup_eval.py report     # a Markdown table of every scored run (--json for the numbers)
```

Work files go to `./markup-eval` (`--work DIR` or `$FOLIVOX_EVAL_DIR`): `pdnc/`, `books/` (excerpts, `gold.json`,
`blocks.json`) and `runs/<label>/` (Folivox's markup, character list, reply cache and log for each excerpt, the
proxy's `calls.jsonl`, `score.json`). Run it from a checkout with Folivox installed (`uv run python ...`); no Docker.

## The corpus and its licence

PDNC is released under **CC BY 4.0** (Vishnubhotla, Hammond and Hirst, *The Project Dialogism Novel Corpus*, LREC
2022, https://aclanthology.org/2022.lrec-1.628/). The corpus is not copied into this repository: `fetch` downloads the
three files of each configured novel from the commit pinned in `eval.json`, and every result stays in the work
directory. One of the novels (*The Sun Also Rises*, 1926) is still under copyright in some countries, another reason
not to redistribute the texts.

## What is measured

- **Excerpts** (`eval.json`): *Pride and Prejudice* ch. VIII-X, *The Sign of the Four* ch. VIII-IX, *The Sun Also Rises*
  ch. 2-4: 108,190 characters, 610 quotations, two of them with a first-person narrator. Hard-wrapped lines are joined
  into paragraphs (otherwise Folivox reads the book line by line) and the apostrophes PDNC's copy of *The Sign of the
  Four* lost are put back.
- **Accuracy**: each gold quotation is found, in order, in the paragraphs Folivox made (quotes, dashes, apostrophes and
  whitespace normalised); the speaker Folivox gave most of its letters is the prediction, mapped to a PDNC character
  through the PDNC aliases (the name, then the aliases Folivox found, then a first or last name only one character
  has). The narrator, or a name that maps to nobody, is wrong. Reported overall and by PDNC quote type (explicit,
  anaphoric, implicit). **Lenient** also counts a first-person narrator's own lines left to the narrator voice as
  right: in an audiobook that is fine.
- **Character list**: precision and recall of the speaking characters in Folivox's list against the PDNC characters who
  speak in the excerpts.
- **Time** per 10,000 characters, and **tokens and USD** from the usage each reply reported, priced with `eval.json`
  (list prices with the date they were read) and scaled to a 500,000-character book.

`run` starts the real `folivox markup auto` once per excerpt, with `FOLIVOX_CONFIG` set to a copy of your
`config.toml` whose OpenAI-compatible `base_url`, `OLLAMA_HOST` and `ANTHROPIC_BASE_URL` point at a small logging
proxy on 127.0.0.1. The proxy passes every request on unchanged (your key included) and writes one line per reply:
model, status, time and token usage, never the request headers.

## Caveats

- Folivox sends no temperature, so runs vary: run each model at least twice. Gpt-style models swung up to 16 points
  between runs, mostly on the first-person book.
- About six PDNC gold labels look wrong, which caps the top score near 99%.
- Three excerpts are a small sample, all English and all nineteenth- or early twentieth-century prose.

## Results, 2026-11-20 (Folivox at commit 87029f5, default depth, two runs per model)

| Model | Accuracy, mean (range) | Lenient | Explicit / non-explicit | Characters P/R | s per 10k chars | USD per 500k-char book | `--best` |
|---|---|---|---|---|---|---|---|
| claude-opus-5-5 | 97.4 (97.2-97.5) | 97.4 | 100 / 96.8 | 70 / 100 | 37 | $7.02 | 98.7 |
| gpt-6.2-astra | 93.5 (88.5-98.5) | 98.6 | 100 / 92.1 | 57 / 97 | 44 | $9.67 | 98.5 |
| claude-sonnet-5-5 | 93.6 (91.1-96.1) | 94.6 | 98.1 / 92.6 | 83 / 100 | 22 | $2.98 | 94.8 |
| gpt-6.2-sol | 84.8 (76.6-93.1) | 90.2 | 99.5 / 81.7 | 54 / 95 | 51 | $2.62 | - |
| claude-haiku-5-5 | 90.7 (89.7-91.6) | 90.9 | 95.8 / 89.5 | 80 / 97 | 41 | $0.31 | - |
| gpt-6-luna | 89.3 (84.6-94.1) | 93.8 | 96.8 / 87.7 | 60 / 100 | 64 | $0.22 | - |
| gemma4:12b-it-qat (local) | 75.2 (73.9-76.4) | 75.7 | 91.7 / 71.6 | 61 / 97 | 44 | free | 76.2 |
| gemma4:26b-a4b-it-qat (local) | 71.8 (67.5-76.1) | 72.8 | 88.4 / 68.2 | 57 / 95 | 91 | free | - |

The API models ran through OpenAI-compatible endpoints (Anthropic's included, which does no prompt caching). About 46
calls per run, each about 2,100-2,900 prompt tokens and 700-2,300 output tokens (reasoning included): output is
56-81% of the cost. Local timings are from one consumer GPU and overlapped model downloads.

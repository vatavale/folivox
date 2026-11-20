# Providers and configuration

```bash
folivox providers        # what is available and selected
```

| Kind | Local (default) | Optional |
|---|---|---|
| `tts` | `cosyvoice3` | `fish-s2`, `fish-drama3` ([Fish Audio engines](fish-engines.md): optional download, own licence), and the paid cloud engines |
| `asr` | `auto` (mlx-whisper / faster-whisper) | `openai` (any OpenAI-compatible transcription API) |
| `voicegen` | `qwen3-voicedesign`, `qwen3-customvoice` | `gemini`, `openai`, `elevenlabs`, `cartesia`, `fish` |
| `llm` | `ollama` | `openai` (OpenAI, OpenRouter, llama.cpp server… any OpenAI-compatible API), `anthropic` (Claude through Anthropic's API, with prompt caching) |

Choose with CLI options, `FOLIVOX_TTS` / `FOLIVOX_ASR` / `FOLIVOX_VOICEGEN` / `FOLIVOX_LLM`, or a
`config.toml` in the user config directory (`$FOLIVOX_CONFIG` overrides the path):

```toml
[providers]
llm = "ollama"

[llm.ollama]
model = "gemma4:31b-it-qat"

[voicegen.qwen3-voicedesign]
python = "~/qwen3tts/.venv/bin/python"   # a separate venv with `pip install qwen-tts`
model = "~/qwen3tts/models/VoiceDesign"   # Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign

[llm.openai]
base_url = "https://openrouter.ai/api/v1"
model = "some/model"
api_key_env = "OPENROUTER_API_KEY"

[llm.anthropic]                       # pip install 'folivox[anthropic]'; key: ANTHROPIC_API_KEY
model = "claude-opus-5-5"             # or claude-sonnet-5-5, claude-haiku-5-5
effort = "medium"                     # optional: low | medium | high | xhigh | max (unset: the model's default)
```

**Inference runtime** is a separate, general switch (not a provider): `--runtime auto|pytorch|native`,
`FOLIVOX_RUNTIME`, or

```toml
[runtime]
inference = "auto"        # auto | pytorch | native

[runtime.native]          # all optional
backend = "auto"          # auto | metal | cuda | vulkan
weights = "q8_0"          # q8_0 | f32 (GGUF file of the CosyVoice 3 model)
weight_type = "f16"       # precision the engine converts the weights to on load: native | f32 | f16 | bf16 | q8_0
```

It changes how `cosyvoice3` runs; the other engines ignore it. See [Native runtime](native-runtime.md).

API keys are read from environment variables, or from your **OS keychain** when you save them in
the web interface (`folivox ui` → Providers → API keys). The config file refuses `api_key` fields, so
keys never end up in a file you might share. Cloud providers run at your own cost, and your text
or audio leaves your computer.

> **Commercial voice output.** Some providers' terms restrict using their outputs to build
> competing models or services (for example the Gemini API Additional Terms). Using such audio as a
> cloning prompt is a grey zone; that decision and its consequences are yours. Folivox never
> bundles audio produced by commercial APIs.

Third-party packages can register providers through the `folivox.providers` entry-point group
(see [CONTRIBUTING.md](../CONTRIBUTING.md)).

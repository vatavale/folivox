"""Claude through Anthropic's own API (the ``anthropic`` package: ``pip install 'folivox[anthropic]'``).

What it does beyond an OpenAI-compatible endpoint:

- **Prompt caching.** The markup's prompts (:class:`folivox.markup_llm.Prompt`) arrive with the same text in cache
  order: the instructions, then the character list, then the window. Cache breakpoints after the instructions and
  after the list let every later window of a pass read them at a tenth of the input price.
- **Structured outputs** where the markup gives the JSON shape it wants (``schema``).
- Adaptive thinking with an optional ``effort``, refusals (and, on models that have them, the server-side fallbacks)
  handled as an unusable answer, typed errors mapped onto the markup's retry rules.

Config (``[llm.anthropic]`` in config.toml): ``model``, ``effort`` (low | medium | high | xhigh | max; ``high`` unless
set, for models with adaptive thinking: on the markup benchmark it scored 0.7 points better than ``medium`` for about
7% more cost, folivox#16; the older models get none),
``price_in`` / ``price_out`` (USD per million tokens, for cost estimates). The key comes from
``ANTHROPIC_API_KEY`` (or ``ANTHROPIC_API_KEY_FILE``) or the OS keychain, never from the file.
"""

from __future__ import annotations

from folivox.llm import LLM

DEFAULT_MODEL = "claude-opus-5-5"
EFFORTS = ("low", "medium", "high", "xhigh", "max")
_UNSET = object()  # "no effort given" (None means "send none"), so the default can depend on the model
DEFAULT_EFFORT = "high"  # +0.7 lenient points over "medium" for about 7% more cost (folivox#16)
MAX_TOKENS = 16000  # the markup's answers are a few thousand tokens; reasoning comes out of the same budget
# models that reject adaptive thinking (they take a fixed thinking budget, which the markup does not need)
NO_ADAPTIVE = ("claude-haiku-4-5", "claude-sonnet-4-5", "claude-opus-4-5", "claude-opus-4-1", "claude-opus-4-0",
               "claude-sonnet-4-0", "claude-3")
# models whose refusals the API can hand to another model on the server (beta "fallbacks"; Haiku has none)
FALLBACK_MODELS = ("claude-opus-5-5", "claude-sonnet-5-5", "claude-opus-5", "claude-fable-5-1", "claude-fable-5-2")
FALLBACK_BETA = "server-side-fallback-2026-07-01"
CACHE = {"type": "ephemeral"}


class AnthropicLLM(LLM):
    name = "anthropic"
    caches_prompts = True  # the markup sends the whole character list, which stays the same from window to window
    structured = True  # complete_json takes a JSON schema

    def __init__(self, model: str = DEFAULT_MODEL, effort: str | None | object = _UNSET,
                 api_key_env: str = "ANTHROPIC_API_KEY", price_in: float | None = None,
                 price_out: float | None = None, client=None, max_tokens: int = MAX_TOKENS):
        from folivox.config import api_key

        if effort is _UNSET:  # nothing asked for: high, but only where the model has adaptive thinking
            effort = None if model.startswith(NO_ADAPTIVE) else DEFAULT_EFFORT
        if effort is not None and str(effort) not in EFFORTS:
            raise ValueError(f"[llm.anthropic] effort must be one of {', '.join(EFFORTS)}, got {effort!r}")
        self.model, self.effort, self.max_tokens = model, effort, max_tokens
        self.price_in, self.price_out = price_in, price_out  # USD per million tokens (config), for estimates
        if client is None:
            try:
                import anthropic
            except ImportError:
                raise ValueError("The anthropic provider needs the anthropic package: "
                                 "pip install 'folivox[anthropic]'") from None
            client = anthropic.Anthropic(api_key=api_key(api_key_env, "Anthropic"))
        self.client = client

    # -- requests ------------------------------------------------------------------------------------------------
    def _request(self, prompt: str, system: str | None, schema: dict | None) -> dict:
        blocks = getattr(prompt, "blocks", ()) or ((str(prompt), False),)
        content = [{"type": "text", "text": text, **({"cache_control": CACHE} if mark else {})}
                   for text, mark in blocks if text]
        req: dict = {"model": self.model, "max_tokens": self.max_tokens,
                     "messages": [{"role": "user", "content": content}]}
        if system:
            req["system"] = system
        if not self.model.startswith(NO_ADAPTIVE):
            req["thinking"] = {"type": "adaptive"}
        output: dict = {}
        if self.effort:
            output["effort"] = self.effort
        if schema is not None:
            output["format"] = {"type": "json_schema", "schema": schema}
        if output:
            req["output_config"] = output
        return req

    def _send(self, req: dict):
        """One request; the SDK retries 429, 5xx and dropped connections itself. What is left is mapped onto the
        markup's rules: busy (ConnectionError: wait and try again), a bad request (ValueError: an unusable answer)
        or a setup problem (LLMError: stop and say what to fix)."""
        import anthropic

        from folivox.markup_llm import LLMError

        try:
            if self.model in FALLBACK_MODELS:  # "default": the API picks the fallback model by the refusal's kind
                return self.client.beta.messages.create(**req, betas=[FALLBACK_BETA], fallbacks="default")
            return self.client.messages.create(**req)
        except anthropic.AuthenticationError:
            raise LLMError("Anthropic refused the API key (HTTP 401): check ANTHROPIC_API_KEY.") from None
        except anthropic.PermissionDeniedError as e:
            raise LLMError(f"Anthropic refused the request (HTTP 403): {e.message}") from None
        except anthropic.NotFoundError:
            raise LLMError(f"Anthropic does not know the model {self.model!r} (HTTP 404): check [llm.anthropic] "
                           "model or --model.") from None
        except (anthropic.BadRequestError, anthropic.UnprocessableEntityError) as e:
            raise ValueError(f"Anthropic refused this request (HTTP {e.status_code}): {e.message}") from None
        except anthropic.RateLimitError as e:
            raise ConnectionError(f"Anthropic is busy (HTTP 429): {e.message}") from None
        except anthropic.APIStatusError as e:
            if e.status_code >= 500 or e.status_code == 413:
                raise (ConnectionError if e.status_code >= 500 else ValueError)(
                    f"Anthropic did not answer (HTTP {e.status_code}): {e.message}") from None
            raise LLMError(f"Anthropic did not answer (HTTP {e.status_code}): {e.message}") from None
        except (anthropic.APIConnectionError, anthropic.APITimeoutError) as e:
            raise ConnectionError(f"Anthropic could not be reached: {e}") from None

    def _read(self, msg) -> str:
        u = getattr(msg, "usage", None)
        self.last_usage = None if u is None else {
            "input": int(getattr(u, "input_tokens", 0) or 0), "output": int(getattr(u, "output_tokens", 0) or 0),
            "cache_read": int(getattr(u, "cache_read_input_tokens", 0) or 0),
            "cache_write": int(getattr(u, "cache_creation_input_tokens", 0) or 0)}
        if msg.stop_reason == "refusal":  # also after a fallback model declined: the narrator reads these paragraphs
            details = getattr(msg, "stop_details", None)
            raise ValueError(f"the model declined ({getattr(details, 'category', None) or 'refusal'})")
        if msg.stop_reason == "max_tokens":
            raise ValueError("the answer was cut off at the token limit")
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()

    def complete(self, prompt, system=None, temperature=0.3):
        # no temperature: current Claude models accept only the default
        return self._read(self._send(self._request(prompt, system, None)))

    def complete_json(self, prompt, system=None, temperature=0.2, schema=None):
        return self._read(self._send(self._request(prompt, system, schema)))

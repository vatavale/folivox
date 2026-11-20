"""The markup's cost with a paid language model: prices, the estimate before a run, the spending cap during one."""

import json

import pytest
from test_markup_llm import CH, P1, P2, P3, P4, P5, FakeLLM
from typer.testing import CliRunner

from folivox import config
from folivox import llm as llms
from folivox import markup_cost as mc
from folivox import markup_llm as ml
from folivox.book import Block, BlockKind, Chapter


class PaidLLM(FakeLLM):
    """The fake model of the markup tests, billing like an API: every reply reports its usage, and every prompt is
    kept so a test can compare what was sent with what was estimated."""

    def __init__(self, mode="good", output=500):
        super().__init__(mode)
        self.output, self.sent = output, []

    def complete_json(self, prompt, system=None, temperature=0.2):
        self.sent.append(len(system or "") + len(prompt))
        reply = super().complete_json(prompt, system, temperature)
        self.last_usage = {"input": self.sent[-1] // 3, "output": self.output, "cache_read": 0, "cache_write": 0}
        return reply


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    """Write config.toml for one test."""
    path = tmp_path / "config.toml"
    monkeypatch.setenv("FOLIVOX_CONFIG", str(path))

    def write(text: str):
        path.write_text(text, encoding="utf-8")
        config.load.cache_clear()

    write("")
    yield write
    config.load.cache_clear()


def test_replies_report_their_token_usage(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    answers = {"usage": {"prompt_tokens": 1000, "completion_tokens": 300,
                         "prompt_tokens_details": {"cached_tokens": 400}},
               "choices": [{"message": {"content": "{}"}}]}
    monkeypatch.setattr(llms, "_post", lambda url, payload, headers=None, timeout=600: answers)
    api = llms.OpenAICompatible(model="gpt-6-luna")
    api.complete_json("hi")
    assert api.last_usage == {"input": 600, "output": 300, "cache_read": 400, "cache_write": 0}
    answers.pop("usage")  # a server that says nothing: no stale numbers
    api.complete_json("hi")
    assert api.last_usage is None
    local = llms.Ollama("gemma4:12b-it-qat")
    monkeypatch.setattr(llms, "_post", lambda *a, **k: {"message": {"content": "{}"}, "prompt_eval_count": 900,
                                                        "eval_count": 120})
    local.complete_json("hi")
    assert local.last_usage == {"input": 900, "output": 120, "cache_read": 0, "cache_write": 0}


def test_prices_come_from_the_config_the_list_or_nowhere(cfg):
    assert mc.price_of("ollama", "gemma4:12b-it-qat").free
    sonnet = mc.price_of("openai", "claude-sonnet-5-5")
    assert (sonnet.input, sonnet.output, sonnet.source) == (2.0, 10.0, "list")
    assert mc.price_of("openai", "anthropic/claude-sonnet-5.5") == sonnet  # a router's name for it
    assert mc.price_of("openai", "some/new-model") is None  # never a guess
    cfg('[llm.openai]\nmodel = "some/new-model"\nprice_in = 0.5\nprice_out = 1.5\n')
    p = mc.price_of("openai", "some/new-model")
    assert (p.input, p.output, p.cache_read, p.source) == (0.5, 1.5, 0.5, "config")
    assert mc.default_model("openai") == "some/new-model"
    cfg('[llm.openai]\nprice_in = 0.5\n')
    with pytest.raises(ValueError, match="both"):
        mc.price_of("openai", "x")
    cfg('[llm.openai]\nprice_in = "cheap"\nprice_out = 1\n')
    with pytest.raises(ValueError, match="number"):
        mc.price_of("openai", "x")


def test_a_cap_needs_a_price_and_no_cap_is_the_default():
    with pytest.raises(ValueError, match="price"):
        mc.Budget(None, 5)
    b = mc.Budget(None)
    b.check(10**9)  # no cap: never stops
    b.record({"input": 10, "output": 5}, 30, 10)
    assert b.usd is None and b.spent.tokens == 15


def test_the_cap_stops_before_the_call_that_would_pass_it_and_a_higher_cap_carries_on():
    price = mc.Price(100.0, 100.0, 100.0, 100.0)  # $0.0001 a token
    cache = ml.Cache(None)
    llm = PaidLLM(output=1000)  # the first guess of a reply is smaller: the second window is judged by the first
    budget = mc.Budget(price, max_cost=0.2)  # room for one window of this book, not two
    with pytest.raises(ml.CostCapReached, match=r"Spending cap \$0.20 reached"):
        ml.analyse(CH, "ru", ml.Runner(llm, "m", cache, budget=budget), ml.Bible())
    assert len(llm.calls) == 1 and budget.spent.calls == 1
    assert budget.usd == pytest.approx((llm.sent[0] // 3 + 1000) / 1e4)
    assert budget.usd <= 0.2
    # the same run with a higher cap: the window already paid for comes from the reply cache
    again = mc.Budget(price, max_cost=10)
    bible = ml.analyse(CH, "ru", ml.Runner(llm, "m", cache, budget=again), ml.Bible())
    assert len(llm.calls) == 2 and again.spent.calls == 1 and bible.done_windows == 2


def test_a_reply_without_usage_is_counted_by_its_size():
    b = mc.Budget(mc.Price(1.0, 1.0, 0.1, 1.25))
    b.record(None, 2500, 500)
    assert b.spent.estimated == 1 and b.spent.input == 1000 and b.spent.output == 400


def test_the_estimate_counts_the_calls_and_prompts_a_run_sends():
    chapters = [Chapter(f"Глава {i}", [Block(BlockKind.PARAGRAPH, f"{b.text} ({i})") for b in ch.blocks])
                for i in range(3) for ch in CH]  # a few windows per pass, none the same as another
    bible = ml.Bible()
    est = mc.estimate(chapters, "ru", bible, mc.Price(*mc.PRICES["claude-haiku-5-5"], "list"),
                      analyse=mc.analyse_windows(chapters, bible), annotate=mc.annotate_windows(chapters))
    llm = PaidLLM()
    runner = ml.Runner(llm, "m", ml.Cache(None))
    bible = ml.analyse(chapters, "ru", runner, bible)
    reading = len(llm.sent)
    ml.annotate(chapters, "ru", bible, runner)
    assert (est.analyse_calls, est.annotate_calls) == (reading, len(llm.sent) - reading)
    sent_tokens = sum(llm.sent) / mc.CHARS_PER_TOKEN
    assert 0.8 * sent_tokens <= est.input_tokens <= 1.6 * sent_tokens  # a fresh book is sized with stand-in characters
    assert est.output_tokens < est.output_tokens_max and est.usd < est.usd_max
    # a run that already read the book only marks
    assert mc.analyse_windows(chapters, bible) == []
    assert "≈ $" in mc.describe(est) and "list prices" in mc.describe(est)
    est.price = None
    assert "not known" in mc.describe(est) and "price_in" in mc.describe(est)


def write_book(tmp_path):
    book = tmp_path / "b.txt"
    book.write_text("Глава 1\n\n" + "\n\n".join((P1, P2, P3, P4)) + "\n\nГлава 2\n\n" + P5 + "\n", encoding="utf-8")
    return book


def test_cli_estimate_sends_nothing_and_the_cap_stops_cleanly(tmp_path, monkeypatch, cfg):
    from folivox import cli_markup
    from folivox.cli import app

    cfg('[providers]\nllm = "openai"\n[llm.openai]\nmodel = "claude-haiku-5-5"\n')
    book = write_book(tmp_path)

    def no_model(provider, model):
        raise AssertionError("an estimate must not reach the model")

    monkeypatch.setattr(cli_markup, "_llm", no_model)
    r = CliRunner()
    res = r.invoke(app, ["markup", "auto", str(book), "--estimate"])
    assert res.exit_code == 0, res.output
    assert "claude-haiku-5-5" in res.output and "tokens in" in res.output and "≈ $" in res.output
    assert not (tmp_path / "b.llm-cache.sqlite").exists()
    llm = PaidLLM(output=10**7)  # every reply costs $5 on Haiku: the first one already passes a $1 cap
    monkeypatch.setattr(cli_markup, "_llm", lambda provider, model: (llm, "claude-haiku-5-5"))
    res = r.invoke(app, ["markup", "auto", str(book), "--max-cost", "1"])
    assert res.exit_code == 3, res.output
    assert "Spending cap $1.00 reached" in res.output and "Spent ≈ $5.00" in res.output
    assert len(llm.calls) == 1
    res = r.invoke(app, ["markup", "auto", str(book), "--max-cost", "100"])
    assert res.exit_code == 0, res.output
    assert (tmp_path / "b.markup.json").is_file()
    # a model with no known price: tokens only, and a cap is refused rather than guessed
    cfg('[providers]\nllm = "openai"\n[llm.openai]\nmodel = "some/new-model"\n')
    res = r.invoke(app, ["markup", "auto", str(book), "--estimate"])
    assert res.exit_code == 0 and "not known" in res.output
    monkeypatch.setattr(cli_markup, "_llm", lambda provider, model: (PaidLLM(), "some/new-model"))
    res = r.invoke(app, ["markup", "auto", str(book), "--max-cost", "1"])
    assert res.exit_code == 2 and "needs the model's price" in res.output


def test_a_local_model_is_free_and_shows_no_bill(tmp_path, monkeypatch, cfg):
    from folivox import cli_markup
    from folivox.cli import app

    book = write_book(tmp_path)
    monkeypatch.setattr(cli_markup, "_llm", lambda provider, model: (PaidLLM(), "gemma4:12b-it-qat"))
    res = CliRunner().invoke(app, ["markup", "auto", str(book), "--model", "gemma4:12b-it-qat", "--estimate"])
    assert res.exit_code == 0 and "free" in res.output
    res = CliRunner().invoke(app, ["markup", "auto", str(book)])
    assert res.exit_code == 0 and "Spent" not in res.output and "Paid" not in res.output
    saved = json.loads((tmp_path / "b.markup.json").read_text(encoding="utf-8"))
    assert saved["chapters"]

"""Studio routes of the automatic markup, with a fake language model."""

import time

import pytest

pytest.importorskip("starlette")
pytest.importorskip("httpx")

from starlette.testclient import TestClient  # noqa: E402
from test_markup_llm import P1, P2, P3, P4, P5, FakeLLM  # noqa: E402

from folivox import config  # noqa: E402
from folivox.ui.app import create_app  # noqa: E402

TOKEN = "t0ken"
BOOK = ("Глава 1\n\n" + "\n\n".join((P1, P2, P3, P4)) + "\n\nГлава 2\n\n" + P5 + "\n").encode()


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("FOLIVOX_CONFIG", str(tmp_path / "config.toml"))
    config.load.cache_clear()
    holder = {"llm": FakeLLM(), "pulled": [], "removed": []}

    def fake_pull(model, progress, cancelled):
        holder["pulled"].append(model)
        progress(5, 10, "pulling")
        progress(10, 10, "success")
    app = create_app(token=TOKEN, port=8765, data_dir=tmp_path / "ui", render=lambda b, o, e: o.output,
                     services={"llm": lambda name: holder["llm"], "pull": fake_pull,
                               "delete_model": holder["removed"].append})
    c = TestClient(app, base_url="http://127.0.0.1:8765", headers={"X-Folivox-Token": TOKEN})
    c.holder = holder
    c.data_dir = tmp_path / "ui"
    yield c
    config.load.cache_clear()


def upload(client):
    r = client.post("/api/books", files={"file": ("tale.txt", BOOK, "text/plain")})
    return r.json()["book_id"]


def wait(client, bid):
    for _ in range(200):
        st = client.get(f"/api/books/{bid}/llm/status").json()
        if st["state"] != "running":
            return st
        time.sleep(0.05)
    raise AssertionError("the task did not finish")


def test_models_route_lists_gemma_tiers(client):
    m = client.get("/api/llm/models").json()
    assert m["recommended"].startswith("gemma4:") and len(m["tiers"]) == 3


def test_analyse_then_auto_then_edit_characters(client):
    bid = upload(client)
    assert client.get(f"/api/books/{bid}/characters").json()["characters"] == []
    assert client.post(f"/api/books/{bid}/llm/start", json={"task": "nope"}).status_code == 400
    assert client.post(f"/api/books/{bid}/llm/start", json={"task": "analyse"}).status_code == 202
    st = wait(client, bid)
    assert st["state"] == "done" and "voices needed" in st["message"]
    bible = client.get(f"/api/books/{bid}/characters").json()
    assert {c["name"] for c in bible["characters"]} == {"Анна", "Борис"} and bible["voices_needed"] >= 1
    assert client.post(f"/api/books/{bid}/llm/start", json={"task": "auto"}).status_code == 202
    st = wait(client, bid)
    assert st["state"] == "done" and st["stats"]["marked"] == 3
    mk = client.get(f"/api/books/{bid}/markup").json()
    assert mk["chapters"] and {s["name"] for s in mk["speakers"]} >= {"Анна", "Борис"}
    bible["characters"][0]["name"] = "Анна Сергеевна"
    saved = client.put(f"/api/books/{bid}/characters", json=bible).json()
    assert saved["characters"][0]["name"] == "Анна Сергеевна"
    bible["characters"][0]["id"] = 40
    assert client.put(f"/api/books/{bid}/characters", json=bible).status_code == 400


def test_model_failure_is_reported_not_raised(client):
    bid = upload(client)
    client.holder["llm"] = FakeLLM("down")
    client.post(f"/api/books/{bid}/llm/start", json={"task": "analyse"})
    st = wait(client, bid)
    assert st["state"] == "error" and "did not answer" in st["message"]


def test_chapter_text_route_gives_the_paragraphs_the_markup_addresses(client):
    bid = upload(client)
    r = client.get(f"/api/books/{bid}/text?chapter=0").json()
    assert r["title"] == "Глава 1" and [b["text"] for b in r["blocks"]] == [P1, P2, P3, P4]
    assert client.get(f"/api/books/{bid}/text?chapter=9").status_code == 404


def test_listening_to_a_passage_uses_its_own_markup(tmp_path, monkeypatch):
    from folivox import markup as mk

    seen = {}

    def capture(book, opts, events):
        seen["text"] = [b.text for s in book.sections for b in s.blocks]
        seen["markup"] = mk.Markup.load(opts.markup) if opts.markup else None
        seen["opts"] = opts  # last: the test waits for it
        return opts.output

    monkeypatch.setenv("FOLIVOX_CONFIG", str(tmp_path / "config.toml"))
    config.load.cache_clear()
    app = create_app(token=TOKEN, port=8765, data_dir=tmp_path / "ui", render=capture)
    c = TestClient(app, base_url="http://127.0.0.1:8765", headers={"X-Folivox-Token": TOKEN})
    bid = upload(c)
    m = mk.Markup()
    m.ensure_speaker(1, "Анна")
    m.assign(0, 2, 0, 5, 1)
    m.assign(0, 0, 0, 3, 1)  # outside the chosen passage
    mk.stamp(m, [])
    body = {"version": 1, "depth": 2, **{k: v for k, v in m.to_dict().items() if k in ("speakers", "chapters")}}
    assert c.put(f"/api/books/{bid}/markup", json=body).status_code == 200
    job = {"kind": "preview", "book_id": bid, "chapters": [1], "voice": "", "options": {},
           "sample_range": {"chapter": 0, "block_from": 1, "block_to": 2}}
    r = c.post("/api/jobs", json=job)
    assert r.status_code == 202, r.text
    for _ in range(100):
        if "opts" in seen:
            break
        time.sleep(0.05)
    assert seen["text"] == [P2, P3] and seen["opts"].sample_exact
    spans = seen["markup"].chapters[0].spans
    assert [(s.block, s.speaker) for s in spans] == [(1, 1)]  # the old block 2 is block 1 of the passage
    time.sleep(0.3)  # the first job has finished: another may start
    wrong = dict(job, sample_range={"chapter": 7, "block_from": 0, "block_to": 1})
    assert c.post("/api/jobs", json=wrong).status_code == 400


def test_voice_prompts_task_and_the_speakers_voice_info(client):
    from test_markup_llm import VoiceLLM

    bid = upload(client)
    client.post(f"/api/books/{bid}/llm/start", json={"task": "analyse"})
    wait(client, bid)
    client.post(f"/api/books/{bid}/llm/start", json={"task": "auto"})
    wait(client, bid)
    client.holder["llm"] = VoiceLLM()
    assert client.post(f"/api/books/{bid}/llm/start", json={"task": "prompts"}).status_code == 202
    st = wait(client, bid)
    assert st["state"] == "done" and "2 characters" in st["message"]
    bible = client.get(f"/api/books/{bid}/characters").json()
    assert all(set(c["voice_prompts"]) == {"simple", "standard", "detailed"} for c in bible["characters"])
    anna = next(c for c in bible["characters"] if c["name"] == "Анна")
    info = client.get(f"/api/books/{bid}/speakers/{anna['id']}/voice").json()
    assert info["name"] == "Анна" and info["prompts"]["simple"].startswith("A woman") and info["texts"]
    # a speaker made by hand gets described from its own record
    mk = client.get(f"/api/books/{bid}/markup").json()
    mk["speakers"].append({"id": 9, "name": "Хозяйка", "voice": "", "gender": "f", "aliases": [], "notes": ""})
    assert client.put(f"/api/books/{bid}/markup", json=mk).status_code == 200
    client.post(f"/api/books/{bid}/llm/start", json={"task": "prompts", "speakers": [9]})
    assert wait(client, bid)["state"] == "done"
    assert client.get(f"/api/books/{bid}/speakers/9/voice").json()["prompts"]["standard"]


def test_estimate_counts_the_windows_of_a_run(client):
    bid = upload(client)
    e = client.get(f"/api/books/{bid}/llm/estimate").json()
    assert e["analyse_windows"] >= 1 and e["annotate_windows"] >= 1 and e["window_chars"] > 0
    one = client.get(f"/api/books/{bid}/llm/estimate?chapter=2").json()
    assert 1 <= one["annotate_windows"] <= e["annotate_windows"]
    client.post(f"/api/books/{bid}/llm/start", json={"task": "analyse"})
    wait(client, bid)
    assert client.get(f"/api/books/{bid}/llm/estimate").json()["analyse_windows"] == 0



def test_model_download_and_removal(client):
    assert client.post("/api/llm/pull", json={"model": "evil/model"}).status_code == 400  # only the listed models
    assert client.request("DELETE", "/api/llm/models", json={"model": "evil/model"}).status_code == 400
    r = client.post("/api/llm/pull", json={"model": "gemma4:12b-it-qat"})
    assert r.status_code == 202
    for _ in range(100):
        st = client.get("/api/llm/pull").json()
        if st["state"] != "running":
            break
        time.sleep(0.05)
    assert st["state"] == "done" and st["done"] == st["total"] == 10 and st["model"] == "gemma4:12b-it-qat"
    assert client.holder["pulled"] == ["gemma4:12b-it-qat"]
    gone = client.request("DELETE", "/api/llm/models", json={"model": "gemma4:12b-it-qat"})
    assert gone.json() == {"removed": "gemma4:12b-it-qat"}
    assert client.holder["removed"] == ["gemma4:12b-it-qat"]


def test_estimate_lists_the_chapters_with_dialogue(client):
    bid = upload(client)
    assert client.get(f"/api/books/{bid}/llm/estimate").json()["speech_chapters"] == [0, 1]


def test_estimate_names_the_chapters_whose_text_changed(client):
    import json

    bid = upload(client)
    client.post(f"/api/books/{bid}/llm/start", json={"task": "auto"})
    wait(client, bid)
    assert client.get(f"/api/books/{bid}/llm/estimate").json()["stale_chapters"] == []
    mp = client.data_dir / "markup" / f"{bid}.json"
    d = json.loads(mp.read_text(encoding="utf-8"))
    d["chapters"]["0"]["sig"] = "another-text"  # as if the first chapter had been edited since it was marked
    mp.write_text(json.dumps(d), encoding="utf-8")
    assert client.get(f"/api/books/{bid}/llm/estimate").json()["stale_chapters"] == [0]


def test_preview_and_best_modes(client):
    bid = upload(client)
    est = client.get(f"/api/books/{bid}/llm/estimate").json()["modes"]
    assert est["preview"]["analyse"] >= 1 and est["best"]["annotate"] >= 1 and est["best"]["extra"] >= 8
    assert client.post(f"/api/books/{bid}/llm/start", json={"task": "preview"}).status_code == 202
    st = wait(client, bid)
    assert st["state"] == "done" and "flagged" in st["stats"]
    assert client.get(f"/api/books/{bid}/characters").json()["sampled"] is False  # a small book fits in the sample
    assert client.post(f"/api/books/{bid}/llm/start", json={"task": "best", "chapters": [1, 2]}).status_code == 202
    st = wait(client, bid)
    assert st["state"] == "done" and "settled" in st["stats"] and "to_check" in st["stats"]
    assert client.get(f"/api/books/{bid}/characters").json()["sampled"] is False
    m = client.get(f"/api/books/{bid}/markup").json()
    assert all(c["done"] for c in m["chapters"].values())


def test_a_hand_made_change_of_gender_is_not_outvoted(client):
    bid = upload(client)
    client.post(f"/api/books/{bid}/llm/start", json={"task": "analyse"})
    wait(client, bid)
    bible = client.get(f"/api/books/{bid}/characters").json()
    anna = next(c for c in bible["characters"] if c["name"] == "Анна")
    assert "unsure" in anna
    anna["gender"] = "m"
    saved = client.put(f"/api/books/{bid}/characters", json=bible).json()
    again = next(c for c in saved["characters"] if c["name"] == "Анна")
    assert again["gender"] == "m" and again["votes"]["gender"]["m"] >= 1000 and "gender" not in again["unsure"]


def test_characters_on_a_shared_id_are_compared_with_themselves(client):
    bid = upload(client)
    client.post(f"/api/books/{bid}/llm/start", json={"task": "analyse"})
    wait(client, bid)
    bible = client.get(f"/api/books/{bid}/characters").json()
    for c in bible["characters"]:  # both on the shared id 14 (more than 15 speakers in a real book)
        c["id"], c["shared"] = 14, True
    bible = client.put(f"/api/books/{bid}/characters", json=bible).json()
    saved = client.put(f"/api/books/{bid}/characters", json=bible).json()  # saved again, nothing changed
    assert all(max(c["votes"]["gender"].values()) < 1000 for c in saved["characters"])


def test_a_paid_api_offers_its_own_model_and_nothing_to_download(client):
    (client.data_dir.parent / "config.toml").write_text(
        '[providers]\nllm = "anthropic"\n[llm.anthropic]\nmodel = "claude-haiku-5-5"\n', encoding="utf-8")
    config.load.cache_clear()
    m = client.get("/api/llm/models").json()
    assert (m["provider"], m["local"], m["configured"]) == ("anthropic", False, "claude-haiku-5-5")
    assert m["tiers"] == [] and m["installed"] == []
    s = client.get("/api/settings").json()["kinds"]["llm"]
    assert any(p["name"] == "anthropic" and not p["local"] for p in s["providers"])
    assert any(k["name"] == "ANTHROPIC_API_KEY" for k in client.get("/api/keys").json()["keys"])

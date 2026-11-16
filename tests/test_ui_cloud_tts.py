"""Web UI endpoints of the cloud reading engines: estimate, consent gate, options reaching the pipeline."""
import pytest

pytest.importorskip("starlette")
pytest.importorskip("httpx")

from starlette.testclient import TestClient  # noqa: E402
from test_ui import TOKEN, upload, wait_done  # noqa: E402

from folivox.ui.app import create_app  # noqa: E402


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("FOLIVOX_CONFIG", str(tmp_path / "config.toml"))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-SECRET-1234")
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.setattr("folivox.config.keychain_get", lambda name: None)
    from folivox import config

    config.load.cache_clear()
    seen = {}

    def fake_render(book, opts, events):
        seen["opts"] = opts
        return opts.output

    app = create_app(token=TOKEN, port=8765, data_dir=tmp_path / "ui", render=fake_render)
    c = TestClient(app, base_url="http://127.0.0.1:8765", headers={"X-Folivox-Token": TOKEN})
    yield c, seen
    config.load.cache_clear()


def test_engine_list_has_prices_limits_and_badges(env):
    c, _ = env
    data = c.get("/api/tts/engines").json()
    by = {e["name"]: e for e in data["engines"]}
    # the optional Fish engines appear only where a build can run (Apple Silicon): not part of this check
    assert [e["name"] for e in data["engines"] if not e.get("fish")] == [
        "cosyvoice3", "gemini", "cartesia", "fish", "elevenlabs", "openai"]
    assert by["gemini"]["tag"] == "cheap and best" and "non-English" in by["openai"]["tag"]
    assert by["cosyvoice3"]["local"] and not by["openai"]["local"]
    assert set(by) >= {"elevenlabs", "openai", "gemini", "cartesia", "fish"}
    assert by["fish"]["models"][0]["id"] == "drama-3" and by["fish"]["voice_modes"] == ["preset", "design", "clone"]
    assert by["fish"]["per_hour"] == 2.52 and "UTF-8" in by["fish"]["price_note"]
    assert by["openai"]["per_hour"] > 0 and by["openai"]["max_chars"] == 4096 and by["openai"]["key_in_env"]
    # fewer requests for the engines that bill and limit per request
    assert [by[n]["fragment_chars"] for n in ("gemini", "openai", "elevenlabs", "cartesia")] == [2000, 2000, 2000, 300]
    assert by["gemini"]["max_chars"] == 4000 and by["elevenlabs"]["model_max_chars"]["eleven_v3"] == 5000
    assert "SECRET" not in c.get("/api/tts/engines").text
    assert by["gemini"]["voice_modes"] == by["elevenlabs"]["voice_modes"] == ["preset", "design"]


def test_estimate_counts_the_book(env):
    c, _ = env
    b = upload(c)
    body = {"book_id": b["book_id"], "options": {"tts": "openai", "takes": 2}}
    e = c.post("/api/tts/estimate", json=body).json()
    assert e["chars"] > 500 and e["usd"] > 0 and e["usd_max"] >= e["usd"] and e["approximate"]
    prev = c.post("/api/tts/estimate", json={**body, "kind": "preview"}).json()
    assert prev["chars"] <= 600 and prev["usd"] < e["usd"]
    assert c.post("/api/tts/estimate", json={**body, "options": {"tts": "cosyvoice3"}}).status_code == 400


def test_cloud_job_needs_acknowledgement_and_key(env):
    c, seen = env
    b = upload(c)
    job = {"kind": "preview", "book_id": b["book_id"], "options": {"tts": "openai", "tts_voice": "coral"}}
    assert c.post("/api/jobs", json=job).status_code == 400  # no acknowledgement
    ack = {**job, "acknowledged": True, "options": {**job["options"], "tts": "elevenlabs", "tts_voice": "x"}}
    r = c.post("/api/jobs", json=ack)
    assert r.status_code == 412 and r.json().get("missing_key") == "ELEVENLABS_API_KEY"
    r = c.post("/api/jobs", json={**job, "acknowledged": True,
                                  "options": {**job["options"], "max_cost": "7.5", "tts_style": "calm"}})
    assert r.status_code == 202, r.text
    wait_done(c, r.json()["id"])
    o = seen["opts"]
    assert (o.tts, o.tts_voice, o.max_cost, o.tts_style, o.accept_cloud) == ("openai", "coral", 7.5, "calm", True)
    assert o.max_takes == 2  # cloud default: retakes are billed


def test_bad_cap_and_mode_are_rejected(env):
    c, _ = env
    b = upload(c)
    base = {"kind": "preview", "book_id": b["book_id"], "acknowledged": True}
    for extra in ({"max_cost": "abc"}, {"max_cost": 0}, {"tts_mode": "clone"}, {"tts_mode": "design"}):
        r = c.post("/api/jobs", json={**base, "options": {"tts": "openai", **extra}})
        assert r.status_code == 400, extra


def test_cartesia_samples_are_proxied_with_the_key(tmp_path, monkeypatch):
    """Cartesia sample files answer 403 without the key, so the browser plays them through the server."""
    from folivox.voicegen.cloud import Response

    monkeypatch.setenv("FOLIVOX_CONFIG", str(tmp_path / "config.toml"))
    monkeypatch.setenv("CARTESIA_API_KEY", "sk-test-SECRET-1234")
    monkeypatch.setattr("folivox.config.keychain_get", lambda name: None)
    seen = []

    class FakeCartesia:
        modes, label, name = ("preset",), "Cartesia", "cartesia"
        names = {"v1": "Skylar", "v2": "Daniel"}
        details = {"v1": {"traits": "Guide", "preview_url": "https://files.cartesia.ai/files/f1/download?format=playback"},
                   "v2": {"traits": "Assistant", "preview_url": "https://evil.example/x.wav"}}

        def speakers(self, lang=None):
            return {"v1": ("ru", "f"), "v2": ("ru", "m")}

        def headers(self):
            return {"Authorization": "Bearer sk-test-SECRET-1234"}

        def transport(self, method, url, headers, body, timeout):
            seen.append((url, headers))
            return Response(200, b"RIFFfake", {"content-type": "audio/wav"})

    app = create_app(token=TOKEN, port=8765, data_dir=tmp_path / "ui", render=lambda *a: None,
                     services={"generator": lambda n, model="": FakeCartesia()})
    c = TestClient(app, base_url="http://127.0.0.1:8765", headers={"X-Folivox-Token": TOKEN})
    assert c.get("/api/tts/cartesia/sample/v1").status_code == 404  # unknown until the voices were listed
    voices = c.get("/api/gen/voices/cartesia?lang=ru").json()["voices"]
    assert voices[0]["preview_url"] == "/api/tts/cartesia/sample/v1"
    assert "SECRET" not in c.get("/api/gen/voices/cartesia?lang=ru").text
    r = c.get("/api/tts/cartesia/sample/v1")
    assert r.status_code == 200 and r.content == b"RIFFfake" and r.headers["content-type"] == "audio/wav"
    assert seen[0][1]["Authorization"].startswith("Bearer ") and "files.cartesia.ai" in seen[0][0]
    assert c.get("/api/tts/cartesia/sample/v2").status_code == 404  # a foreign host is never fetched


def test_saved_voices_list_delete_and_reuse(env, tmp_path, monkeypatch):
    from folivox.tts import saved

    monkeypatch.setattr("folivox.tts.saved._path", lambda: tmp_path / "cloud_voices.json")
    monkeypatch.setenv("GEMINI_API_KEY", "k-test")
    calls = []

    class FakeGen:
        label, name, modes = "Gemini API", "gemini", ("preset", "design")

        def teardown_voices(self, ids):
            calls.append(ids)

    saved.add("gemini", "v-1", "Calm narrator", "design", "a calm woman", "en")
    _, seen = env
    app = create_app(token=TOKEN, port=8765, data_dir=tmp_path / "ui2", render=lambda b, o, e: seen.update(opts=o),
                     services={"generator": lambda n, model="": FakeGen()})
    c = TestClient(app, base_url="http://127.0.0.1:8765", headers={"X-Folivox-Token": TOKEN})
    assert [v["name"] for v in c.get("/api/tts/gemini/saved").json()["voices"]] == ["Calm narrator"]
    b = upload(c)
    job = {"kind": "preview", "book_id": b["book_id"], "acknowledged": True,
           "options": {"tts": "gemini", "tts_voice": "v-1", "tts_mode": "preset", "keep_voice": True}}
    r = c.post("/api/jobs", json=job)
    assert r.status_code == 202, r.text
    wait_done(c, r.json()["id"])
    o = seen["opts"]
    assert o.tts_voice == "v-1" and o.keep_voice is False  # a kept voice is reused, never re-created
    assert c.put("/api/tts/gemini/saved/v-1", json={"name": "Warm narrator"}).status_code == 200
    assert c.put("/api/tts/gemini/saved/v-1", json={"name": " "}).status_code == 400
    assert c.put("/api/tts/gemini/saved/nope", json={"name": "x"}).status_code == 404
    assert saved.find("gemini", "warm narrator")["id"] == "v-1"
    assert c.delete("/api/tts/gemini/saved/nope").status_code == 404
    assert c.delete("/api/tts/gemini/saved/v-1").status_code == 200
    assert calls == [["v-1"]] and saved.listing("gemini") == []


def test_only_online_engines_carry_voice_modes(env):
    # The studio draws voice kinds from `voice_modes`; it must only do so for engines that have them
    # (a local entry has none, so the UI has to check `cloud` before reading the list).
    c, _ = env
    engines = c.get("/api/tts/engines").json()["engines"]
    assert any(not e["cloud"] for e in engines) and any(e["cloud"] for e in engines)
    for e in engines:
        assert bool(e.get("voice_modes")) == bool(e["cloud"]), e["name"]

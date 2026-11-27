"""Studio routes of the Fish Audio engines: state, licence consent, install progress, job pre-checks."""

import time

import pytest

pytest.importorskip("starlette")
pytest.importorskip("httpx")

from starlette.testclient import TestClient  # noqa: E402

from folivox import config  # noqa: E402
from folivox.tts import builds, fish_setup, licences  # noqa: E402
from folivox.ui.app import create_app  # noqa: E402

TOKEN = "t0ken"
MAC = builds.Hardware("darwin", "arm64", 64.0, 50.0)
BOOK = ("Chapter 1\n\n" + "It was a quiet morning in the village. " * 12 + "\n").encode()


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("FOLIVOX_CONFIG", str(tmp_path / "config.toml"))
    monkeypatch.setattr(builds, "engines_root", lambda: tmp_path / "engines")
    monkeypatch.setattr(builds, "detect_hardware", lambda: MAC)
    config.load.cache_clear()
    app = create_app(token=TOKEN, port=8765, data_dir=tmp_path / "ui", render=lambda b, o, e: o.output)
    yield TestClient(app, base_url="http://127.0.0.1:8765", headers={"X-Folivox-Token": TOKEN})
    config.load.cache_clear()


def post(client, url, **body):
    return client.post(url, json=body)


def test_state_lists_builds_with_fit_and_licence(client):
    s = client.get("/api/engines/fish").json()
    ids = [b["id"] for b in s["builds"]]
    assert ids[0] == "drama3-mlx-4bit" and "s2pro-mlx-8bit" in ids
    d4 = s["builds"][0]
    assert d4["fit"] == "ok" and not d4["accepted"] and not d4["installed"] and d4["commercial"]
    s2 = next(b for b in s["builds"] if b["id"] == "s2pro-mlx-8bit")
    assert s2["commercial"] is False and s2["licence"] == "fish-research"
    assert s["recommended"] is None and s["bundled_voices_allowed"] is False
    assert s["licences"]["fish-research"]["points"][1]["id"] == "no_selling"


def test_accept_needs_an_explicit_confirmation(client):
    assert post(client, "/api/engines/fish/accept", licence="fish-research").status_code == 400
    assert post(client, "/api/engines/fish/accept", licence="fish-research", confirm="yes").status_code == 400
    r = post(client, "/api/engines/fish/accept", licence="fish-research", confirm=True)
    assert r.status_code == 200
    assert next(b for b in r.json()["builds"] if b["id"] == "s2pro-mlx-8bit")["accepted"] is True
    assert post(client, "/api/engines/fish/accept", licence="nope", confirm=True).status_code == 404
    r = post(client, "/api/engines/fish/revoke", licence="fish-research")
    assert next(b for b in r.json()["builds"] if b["id"] == "s2pro-mlx-8bit")["accepted"] is False


def test_licence_text_route_falls_back_to_the_page(client, monkeypatch):
    monkeypatch.setattr(licences, "fetch_text", lambda lic: "FULL TEXT")
    assert client.get("/api/engines/fish/licence/fish-community").json()["text"] == "FULL TEXT"

    def offline(lic):
        raise licences.LicenceError("no network")

    monkeypatch.setattr(licences, "fetch_text", offline)
    r = client.get("/api/engines/fish/licence/fish-community").json()
    assert r["text"] == "" and "no network" in r["error"] and r["page"].startswith("https://")
    assert client.get("/api/engines/fish/licence/other").status_code == 404


def test_install_requires_licence_and_reports_progress(client, monkeypatch):
    assert post(client, "/api/engines/fish/install", build="s2pro-mlx-8bit").status_code == 403
    assert post(client, "/api/engines/fish/install", build="nope").status_code == 404
    post(client, "/api/engines/fish/accept", licence="fish-research", confirm=True)

    def fake_install(b, progress=None, log=print, phase=None):
        progress(b.title, 5, 10)
        log("working")
        time.sleep(0.2)

    monkeypatch.setattr(fish_setup, "install", fake_install)
    assert post(client, "/api/engines/fish/install", build="s2pro-mlx-8bit").status_code == 202
    assert post(client, "/api/engines/fish/install", build="s2pro-mlx-8bit").status_code == 409
    seen = client.get("/api/engines/fish/install").json()
    assert seen["state"] == "running" and seen["build"] == "s2pro-mlx-8bit"
    for _ in range(50):
        if client.get("/api/engines/fish/install").json()["state"] != "running":
            break
        time.sleep(0.05)
    assert client.get("/api/engines/fish/install").json()["state"] == "done"


def test_install_failure_is_reported(client, monkeypatch):
    post(client, "/api/engines/fish/accept", licence="fish-research", confirm=True)

    def broken(b, progress=None, log=print, phase=None):
        raise fish_setup.InstallError("disk full")

    monkeypatch.setattr(fish_setup, "install", broken)
    post(client, "/api/engines/fish/install", build="s2pro-mlx-8bit")
    for _ in range(50):
        st = client.get("/api/engines/fish/install").json()
        if st["state"] != "running":
            break
        time.sleep(0.05)
    assert st["state"] == "error" and st["message"] == "disk full"


def test_engine_list_has_fish_engines_on_this_platform(client):
    names = [e["name"] for e in client.get("/api/tts/engines").json()["engines"]]
    assert names[0] == "cosyvoice3" and {"fish-s2", "fish-drama3"} <= set(names)
    e = next(e for e in client.get("/api/tts/engines").json()["engines"] if e["name"] == "fish-s2")
    assert e["fish"] is True and e["local"] and not e["cloud"] and e["builds"][0] == "s2pro-mlx-8bit"
    # native GGUF builds are listed (a named build runs under any runtime) but marked as native
    assert set(e["builds"]) - set(e["native_builds"]) == {"s2pro-mlx-8bit"}
    assert all(builds.load().build(i).backend == "native" for i in e["native_builds"])


def upload(client):
    r = client.post("/api/books", files={"file": ("tale.txt", BOOK, "text/plain")})
    return r.json()["book_id"]


def start(client, book_id, **options):
    return client.post("/api/jobs", json={"kind": "preview", "book_id": book_id, "chapters": [1],
                                          "voice": options.pop("voice", ""), "options": options})


def test_job_is_refused_with_a_reason_before_it_starts(client, monkeypatch, tmp_path):
    bid = upload(client)
    r = start(client, bid, tts="fish-s2")
    assert r.status_code == 403 and "Accept the Fish Audio Research License" in r.json()["error"]
    post(client, "/api/engines/fish/accept", licence="fish-research", confirm=True)
    r = start(client, bid, tts="fish-s2")
    assert r.status_code == 409 and "install it first" in r.json()["error"]
    monkeypatch.setattr(fish_setup, "is_ready", lambda b: True)
    r = start(client, bid, tts="fish-s2")  # the default voice is a bundled one
    assert r.status_code == 400 and "real LibriVox readers" in r.json()["error"]
    mine = tmp_path / "mine.flac"
    mine.write_bytes(b"x")
    monkeypatch.setattr("folivox.voices.find", lambda vid: type("V", (), {
        "path": mine, "transcript": "t", "id": "mine", "lang": "en"})() if vid == "mine" else None)
    r = start(client, bid, tts="fish-s2", voice="mine")
    assert r.status_code == 202, r.text


def test_default_engine_falls_back_to_cosyvoice_for_bundled_voices(client, monkeypatch):
    monkeypatch.setattr(builds, "default_engine", lambda hw=None: "fish-drama3")
    bid = upload(client)
    r = start(client, bid)  # no engine named, default voice is bundled → CosyVoice, so no Fish refusal
    assert r.status_code == 202, r.text


def test_a_named_local_engine_wins_over_the_configured_fish_one(client, monkeypatch, tmp_path):
    # what the studio sends when CosyVoice 3 is picked: that engine runs, not FOLIVOX_TTS, and a bundled voice is fine
    monkeypatch.setenv("FOLIVOX_TTS", "fish-s2")
    engines = []
    app = create_app(token=TOKEN, port=8765, data_dir=tmp_path / "ui2",
                     render=lambda b, o, e: engines.append(o.tts) or o.output)
    c = TestClient(app, base_url="http://127.0.0.1:8765", headers={"X-Folivox-Token": TOKEN})
    bid = upload(c)
    r = start(c, bid, tts="cosyvoice3")
    assert r.status_code == 202, r.text
    for _ in range(100):
        if c.get(f"/api/jobs/{r.json()['id']}").json()["state"] != "running":
            break
        time.sleep(0.05)
    assert engines == ["cosyvoice3"]


def test_markup_routes_store_checked_markup_and_tags(client):
    bid = upload(client)
    empty = client.get(f"/api/books/{bid}/markup").json()
    assert empty["speakers"][0]["id"] == 0 and empty["chapters"] == {}
    assert any(t["tag"] == "sighs" for t in client.get("/api/markup/tags").json()["tags"])
    mk = {"version": 1, "depth": 2, "speakers": [{"id": 0, "name": "Narrator"}, {"id": 1, "name": "Anna"}],
          "chapters": {"0": {"spans": [{"b": 0, "s": 0, "e": 10, "sp": 1, "tags": []}], "marks": []}}}
    r = client.put(f"/api/books/{bid}/markup", json=mk)
    assert r.status_code == 200 and r.json()["chapters"]["0"]["sig"]
    assert client.get(f"/api/books/{bid}/markup").json()["chapters"]["0"]["spans"][0]["sp"] == 1
    mk["chapters"]["0"]["spans"][0]["e"] = 10**6  # outside the block: refused, nothing is stored
    assert client.put(f"/api/books/{bid}/markup", json=mk).status_code == 400
    assert client.get(f"/api/books/{bid}/markup").json()["chapters"]["0"]["spans"][0]["e"] == 10
    job = client.post("/api/jobs", json={"kind": "preview", "book_id": bid, "chapters": [1], "voice": "",
                                         "options": {}}).json()
    assert job["state"] in ("running", "done", "queued")

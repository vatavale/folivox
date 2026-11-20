"""A Fish render (local Drama 3 / S2 Pro, or Fish Audio online) ends the comment tag with "Built with Fish Audio"."""

import json
import shutil
import subprocess
import wave

import numpy as np
import pytest

from folivox import config, pipeline
from folivox import voices as catalog
from folivox.book import Block, BlockKind, Book, Section
from folivox.pipeline import RenderOptions, render
from folivox.tts import builds, fish_check
from folivox.tts.base import TTSEngine
from folivox.tts.builds import Hardware
from folivox.tts.licences import ATTRIBUTION


@pytest.fixture(autouse=True)
def private(tmp_path, monkeypatch):
    monkeypatch.setenv("FOLIVOX_CONFIG", str(tmp_path / "config.toml"))
    monkeypatch.setattr(catalog, "user_voices_dir", lambda: tmp_path / "voices")
    config.load.cache_clear()
    yield
    config.load.cache_clear()


class Tone(TTSEngine):
    """A local engine that reads every fragment as half a second of tone."""

    name = "tone"

    def synthesize_audio(self, text, voice, seed, speed):
        return (0.3 * np.sin(np.linspace(0, 1000, self.sample_rate // 2))).astype(np.float32), 0.0


@pytest.fixture
def local_fish(monkeypatch):
    """A Fish render without a Fish build: the build checks pass and the engine is a tone."""
    monkeypatch.setattr(fish_check, "check", lambda opts, events: None)
    monkeypatch.setattr(builds, "detect_hardware", lambda: Hardware("darwin", "arm64", 64.0, 50.0))
    monkeypatch.setattr(pipeline, "load_engine", lambda opts, events: Tone())


def write_wav(path, seconds=8.0, sr=16000):
    pcm = (3000 * np.sin(np.linspace(0, 3000, int(sr * seconds)))).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return path


TEXT = "Hello to everyone. Then the door closed."


def book():
    return Book(title="T", lang="en", annotation="A short book.",
                sections=[Section(title="One", blocks=[Block(BlockKind.PARAGRAPH, TEXT)])])


def options(tmp_path, tts):
    return RenderOptions(output=tmp_path / "b.m4b", tts=tts, voice_wav=write_wav(tmp_path / "m.wav"),
                         voice_text="Hello there.", qa=False, stress="none", work_dir=tmp_path / "w")


def comment(m4b):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags=comment", "-of", "json", str(m4b)],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)["format"].get("tags", {}).get("comment", "")


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffmpeg is needed to build the audiobook")
def test_fish_render_carries_attribution(tmp_path, local_fish):
    (tmp_path / "fish").mkdir()
    (tmp_path / "cosy").mkdir()
    fish = render(book(), options(tmp_path / "fish", "fish-drama3"))
    cosy = render(book(), options(tmp_path / "cosy", "cosyvoice3"))
    assert comment(fish).startswith("A short book.") and comment(fish).endswith(ATTRIBUTION)
    assert comment(cosy) == "A short book."


def test_attribution_survives_a_long_description():
    long = pipeline.attributed("x" * 1200, "fish")
    assert long.endswith(ATTRIBUTION) and len(long) <= pipeline.COMMENT_LIMIT
    assert pipeline.attributed("", "fish-s2") == ATTRIBUTION
    assert pipeline.attributed("A book.", "cosyvoice3") == "A book."

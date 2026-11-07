"""Dutch text rules (lang/nl.py): read by the Fish and online engines, not by CosyVoice."""

import pytest

from folivox import lang


@pytest.mark.parametrize("text,expect", [
    ("Op 3-5-2024 om 14:30.", "Op drie mei tweeduizendvierentwintig om veertien uur dertig."),
    ("Het kost €12,50.", "Het kost twaalf euro vijftig cent."),
    ("bijv. 10.000 km", "bijvoorbeeld tienduizend kilometer"),
    ("In de 19e eeuw", "In de negentiende eeuw"),
    ("dhr. Jansen en Willem III", "de heer Jansen en Willem de derde"),
    ("In 1984 waren er 1.250 mensen", "In negentienhonderdvierentachtig waren er twaalfhonderdvijftig mensen"),
    ("2.100.000 vogels", "twee miljoen honderdduizend vogels"),
    ("3,5% op blz. 42", "drie komma vijf procent op bladzijde tweeënveertig"),
    ("als 1ste, d.w.z. snel", "als eerste, dat wil zeggen snel"),
])
def test_dutch_normalisation(text, expect):
    assert lang.get("nl").normalize(text) == expect


def test_dutch_has_rules_but_is_not_a_cosyvoice_language():
    assert lang.has_rules("nl") and lang.has_rules("nld")
    assert "nl" not in lang.SUPPORTED
    assert lang.get("nl").sentences("Dhr. Jansen kwam. Hij bleef.") == ["Dhr. Jansen kwam.", "Hij bleef."]

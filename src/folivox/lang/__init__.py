"""Per-language text processing: normalisation, sentence splitting, QA canonical form, headings.

Languages register themselves on import; :func:`get` returns the module for a book language.
"""

from __future__ import annotations

import importlib

from folivox.lang.base import CJKLanguage, Language, LatinLanguage, has_speech

__all__ = ["CJKLanguage", "Language", "LatinLanguage", "available", "get", "has_rules", "has_speech", "register"]

# Languages with normalisation rules (the CosyVoice 3 set). Any other known code gets a rule-less
# GenericLanguage from get(). Each code maps to the module folivox.lang.<code>.
SUPPORTED = ("ru", "en", "de", "fr", "es", "it", "zh", "ja", "ko")
# Languages with text rules: the CosyVoice set plus those only the Fish and online engines read.
RULES = (*SUPPORTED, "nl")

_registry: dict[str, Language] = {}
_generic: dict[str, Language] = {}  # rule-less languages, built on demand


def register(lang: Language) -> Language:
    _registry[lang.code] = lang
    return lang


def _load_all() -> None:
    for code in RULES:
        if code not in _registry:
            importlib.import_module(f"folivox.lang.{code}")


def available() -> list[Language]:
    _load_all()
    return [_registry[c] for c in RULES if c in _registry] + [
        v for k, v in _registry.items() if k not in RULES]


def normalize_code(code: str | None) -> str:
    """'ru-RU' / 'RUS' / 'zh_Hans' → 'ru' / 'ru' / 'zh'."""
    code = (code or "").strip().lower().replace("_", "-").split("-")[0]
    return {"rus": "ru", "eng": "en", "deu": "de", "ger": "de", "fra": "fr", "fre": "fr", "spa": "es",
            "ita": "it", "zho": "zh", "chi": "zh", "cmn": "zh", "jpn": "ja", "kor": "ko",
            "ukr": "uk", "bel": "be", "jw": "jv", "nld": "nl", "dut": "nl"}.get(code, code)


def get(code: str | None) -> Language:
    code = normalize_code(code)
    if code not in _registry:
        try:
            importlib.import_module(f"folivox.lang.{code}")
        except ImportError:
            pass
    if code in _registry:
        return _registry[code]
    from folivox.lang.generic import NAMES, GenericLanguage

    if code not in NAMES:
        raise ValueError(f"Unknown language {code!r}; languages with rules: {', '.join(SUPPORTED)}")
    if code not in _generic:
        _generic[code] = GenericLanguage(code)
    return _generic[code]


def has_rules(code: str | None) -> bool:
    """True for the languages with their own normalisation rules (:data:`RULES`)."""
    return normalize_code(code) in RULES

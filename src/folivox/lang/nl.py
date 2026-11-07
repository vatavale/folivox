"""Dutch: numbers, dates, times, money, units and the common abbreviations.

CosyVoice does not read Dutch; these rules serve the Fish engines (local and online) and the other online voices,
which read Dutch digits less reliably than words.
"""

from __future__ import annotations

import re

from num2words import num2words

from folivox.lang import LatinLanguage, register
from folivox.lang.rules import parse_units

_UNITS = ("twee|drie|vier|vijf|zes|zeven|acht|negen|tien|elf|twaalf|dertien|veertien|vijftien|zestien|zeventien|"
          "achttien|negentien")


class Dutch(LatinLanguage):
    code, name, native = "nl", "Dutch", "Nederlands"
    chapter_label, intro_label = "Hoofdstuk {n}", "Inleiding"
    part_words = ("deel", "boek")
    chapter_words = ("hoofdstuk",)
    standalone_words = ("proloog", "epiloog", "voorwoord", "nawoord", "inleiding")
    abbreviations = ("bijv", "d.w.z", "o.a", "enz", "ca", "dr", "prof", "nr", "dhr", "mevr", "blz", "jl", "mr", "ir",
                     "drs", "st", "e.d", "m.b.t", "i.p.v", "t.o.v", "a.u.b", "z.g", "vs")
    hallucinations = (r"ondertitel\w* (?:door|ingediend)[^.!?]*[.!?]?", r"bedankt voor het kijken\W*")
    SYMBOLS = {"&": " en ", "%": " procent", "€": " euro", "+": " plus ", "=": " is "}
    thousands_re = r"(?<=\d)\.(?=\d{3}(?!\d))"
    abbreviation_map = (
        (r"\bbijv\.", "bijvoorbeeld"), (r"\bd\.\s?w\.\s?z\.", "dat wil zeggen"), (r"\bo\.\s?a\.", "onder andere"),
        (r"\benz\.", "enzovoort"), (r"\bca\.", "circa"), (r"\be\.\s?d\.", "en dergelijke"),
        (r"\bm\.\s?b\.\s?t\.", "met betrekking tot"), (r"\bi\.\s?p\.\s?v\.", "in plaats van"),
        (r"\bt\.\s?o\.\s?v\.", "ten opzichte van"), (r"\ba\.\s?u\.\s?b\.", "alstublieft"),
        (r"\bblz\.(?=\s*\d)", "bladzijde"), (r"\bnr\.(?=\s*\d)", "nummer"), (r"\bNr\.(?=\s*\d)", "nummer"),
        (r"\b[Dd]hr\.(?=\s+[A-Z])", "de heer"), (r"\b[Mm]evr\.(?=\s+[A-Z])", "mevrouw"),
        (r"\bDr\.(?=\s+[A-Z])", "dokter"), (r"\bProf\.(?=\s+[A-Z])", "professor"),
        (r"\bSt\.(?=\s+[A-Z])", "Sint"), (r"\bjl\.", "jongstleden"), (r"\bv\.\s?Chr\.", "voor Christus"),
        (r"\bn\.\s?Chr\.", "na Christus"),
    )
    months = ("januari", "februari", "maart", "april", "mei", "juni", "juli", "augustus", "september", "oktober",
              "november", "december")
    roman_words = ("akte", "scène", "artikel", "bijlage", "tabel", "figuur", "fig")
    century_after = ("eeuw", "eeuwen")
    century_suffix = r"(?:e|de|ste)?"
    time_preps = ("om", "tegen", "tot", "vanaf", "van", "sinds", "na", "voor", "rond", "tussen", "en", "ca")
    currencies = {
        "usd": ("dollar", "dollar", "cent", "cent", False),
        "eur": ("euro", "euro", "cent", "cent", False),
        "gbp": ("pond", "pond", "penny", "pence", False),
        "jpy": ("yen", "yen", "sen", "sen", False),
    }
    units = parse_units("""
        km/u=kilometer per uur
        km/h=kilometer per uur
        m/s=meter per seconde
        km²=vierkante kilometer
        m²=vierkante meter
        cm²=vierkante centimeter
        m³=kubieke meter
        km=kilometer
        kg=kilogram
        mg=milligram
        cm=centimeter
        mm=millimeter
        ml=milliliter
        kHz=kilohertz
        MHz=megahertz
        GHz=gigahertz
        Hz=hertz
        kW=kilowatt
        GB=gigabyte
        MB=megabyte
        KB=kilobyte
        TB=terabyte
        °C=graden Celsius
        °F=graden Fahrenheit
        °=graden
        g=gram
        m=meter
        l=liter
        u=uur
    """)

    def number(self, s: str) -> str:
        # Dutch writes the hundreds together: «negentienhonderdvierentachtig», «twaalfhonderdvijftig»
        words = re.sub(rf"\b({_UNITS}) honderd", r"\1honderd", super().number(s))
        return re.sub(r"honderd (?!miljoen|miljard|biljoen)(?=\w)", "honderd", words)

    def ordinal_word(self, n: int, feminine: bool = False) -> str:
        return num2words(n, lang="nl", to="ordinal")

    def roman_regnal(self, n: int, name: str) -> str:  # «Willem III» → «Willem de derde»
        return "de " + self.ordinal_word(n)

    def roman_century(self, n: int, ctx: str = "") -> str:
        return self.ordinal_word(n)

    def time_words(self, h: int, m: int, meridiem: str | None) -> str:
        hw = self.number(str(h))
        return f"{hw} uur {self.number(str(m))}" if m else f"{hw} uur"

    def dates(self, text: str) -> str:
        """«3-5-2024», «3/5/2024» → «3 mei 2024» (days are cardinal in Dutch, the numbers are read later)."""
        def numeric(m: re.Match) -> str:
            d, mth = int(m.group(1)), int(m.group(2))
            if not (1 <= d <= 31 and 1 <= mth <= 12):
                return m.group(0)
            return f"{d} {self.months[mth - 1]} {m.group(3)}"

        return re.sub(r"(?<![\d./-])(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})(?!\d)", numeric, text)

    def ordinals(self, text: str) -> str:
        """«3e», «3de», «1ste», «21ste» → «derde», «eerste», «eenentwintigste»."""
        return re.sub(r"(?<![\w.,])(\d+)(ste|de|e)(?![\w])", lambda m: self.ordinal_word(int(m.group(1))), text)

    def agree(self, n: int, word: str) -> str | None:
        return None  # Dutch numbers do not change with the noun


register(Dutch())

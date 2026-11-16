import pytest

from folivox.book import Block, BlockKind, Chapter
from folivox.text import stress
from folivox.text.chunk import ChunkOptions, pack, plan_chapter, sentences, split_long
from folivox.text.normalize import has_speech, normalize, plural_form, roman_to_int


@pytest.mark.parametrize("src,expected", [
    ("В 7:30 он вышел.", "В семь тридцать он вышел."),
    ("Встреча в 07:05, ужин в 18:00.", "Встреча в семь ноль пять, ужин в восемнадцать ноль-ноль."),
    ("Дом № 12.", "Дом номер двенадцать."),
    ("В 1999 году всё было иначе.", "В тысяча девятьсот девяносто девятом году всё было иначе."),
    ("К 2000 году.", "К двухтысячному году."),
    ("Война 1812 г. закончилась.", "Война тысяча восемьсот двенадцатого года закончилась."),
    ("В 2024 г. он уехал.", "В две тысячи двадцать четвёртом году он уехал."),
    ("В 1945 он вернулся.", "В тысяча девятьсот сорок пятом он вернулся."),
    ("в 1990-х годах", "В тысяча девятьсот девяностых годах"),
    ("Родился 12.03.1985 в Москве.", "Родился двенадцатое марта тысяча девятьсот восемьдесят пятого года в Москве."),
    ("До 5 мая.", "До пятого мая."),
    ("Это 31 декабря 2023 г.", "Это тридцать первое декабря две тысячи двадцать третьего года."),
    ("В XIX веке.", "В девятнадцатом веке."),
    ("XX в.", "Двадцатый век."),
    ("Пётр I и Екатерина II.", "Пётр Первый и Екатерина Вторая."),
    ("Глава 3. Утро", "Глава третья. Утро"),
    ("Глава XII", "Глава двенадцатая"),
    ("Часть 2", "Часть вторая"),
    ("На 5-м этаже.", "На пятом этаже."),
    ("1-го числа.", "Первого числа."),
    ("Цены выросли на 15%.", "Цены выросли на пятнадцать процентов."),
    ("Он купил 3 кг яблок и т. д. Вечером ушёл.", "Он купил три килограмма яблок и так далее. Вечером ушёл."),
    ("т.е. и т.п.", "То есть и тому подобное."),
    ("3,5 кг", "Три целых пять десятых килограмма"),
    ("$100 и 10 000 рублей", "Сто долларов и десять тысяч рублей"),
    ("Температура -5°C.", "Температура минус пять градусов Цельсия."),
    ("Скорость 60 км/ч.", "Скорость шестьдесят километров в час."),
    ("Ему 1 год, ей 22 года.", "Ему один год, ей двадцать два года."),
    ("Он прожил 21 год.", "Он прожил двадцать один год."),
    ("Была 1 книга и 2 машины.", "Была одна книга и две машины."),
    ("Съел 21 грушу и 1 яблоко.", "Съел двадцать одну грушу и одно яблоко."),
    ("С 5 до 7 часов.", "С пяти до семи часов."),
    ("Около 300 человек.", "Около трёхсот человек."),
    ("Он сказал... «Нет!»", "Он сказал… Нет!"),
    ("Слово[1] здесь.", "Слово здесь."),
    ("I love you.", "I love you."),
])
def test_normalize(src, expected):
    assert normalize(src) == expected


def test_helpers():
    assert plural_form(1, "a", "b", "c") == "a"
    assert plural_form(3, "a", "b", "c") == "b"
    assert plural_form(11, "a", "b", "c") == "c"
    assert plural_form(25, "a", "b", "c") == "c"
    assert roman_to_int("XIV") == 14
    assert roman_to_int("MCMXCIX") == 1999
    assert not has_speech("* * *")
    assert has_speech("Да")


def test_sentences_keep_dialogue_attribution():
    s = sentences("Ты куда? — спросила соседка. Он молчал.")
    assert s == ["Ты куда? — спросила соседка.", "Он молчал."]


def test_split_long_at_clauses():
    long = ", ".join(["это довольно длинная часть предложения"] * 10) + "."
    parts = split_long(long, 120)
    assert all(len(p) <= 120 for p in parts)
    assert " ".join(parts) == long


def test_split_long_without_punctuation():
    long = " ".join(["слово"] * 100)
    parts = split_long(long, 100)
    assert all(len(p) <= 100 for p in parts)
    assert " ".join(parts) == long


def test_pack_merges_tiny():
    out = pack(["Да.", "Нет.", "Это обычное предложение средней длины, довольно спокойное."], 220, 30)
    assert len(out) == 1


def test_plan_chapter_pauses():
    ch = Chapter("Глава 1", headings=["Глава 1"], blocks=[
        Block(BlockKind.PARAGRAPH, "— Привет! — сказал он."),
        Block(BlockKind.PARAGRAPH, "— Привет, — ответила она."),
        Block(BlockKind.PARAGRAPH, "Помолчали."),
        Block(BlockKind.SUBTITLE, "* * *"),
        Block(BlockKind.PARAGRAPH, "Утром в 7:00 он ушёл."),
    ])
    o = ChunkOptions(stress="none")
    chunks = plan_chapter(ch, o)
    assert [c.text for c in chunks] == ["Глава первая.", "Привет! — сказал он.", "Привет, — ответила она.",
                                       "Помолчали.", "Утром в семь ноль-ноль он ушёл."]
    p = o.pauses
    assert [c.pause_after for c in chunks] == [p.heading, p.dialogue, p.paragraph, p.scene, p.chapter_end]
    assert [c.index for c in chunks] == list(range(5))


def test_stress_conversion():
    marked = "В+ечером над рек+ой подн+ялся тум+ан, и ст+арый пар+омщик смотр+ел на огн+и далёкого г+орода."
    assert stress.from_ruaccent(marked, "yo") == marked.replace("+", "")
    acc = stress.from_ruaccent("м+ёд и з+амок", "all")
    assert acc == "мёд и за́мок"
    assert stress.strip_marks(acc) == "мёд и замок"
    assert stress.apply("Ёлка", "none") == "Ёлка"


def test_russian_clause_numbers_are_not_read_as_decimals():
    from folivox.text.normalize import normalize

    assert normalize("2.3. Арендатор обязан вовремя платить.").startswith("Два точка три. Арендатор")
    assert normalize("2.4 Стороны несут ответственность.").startswith("Два точка четыре. Стороны")
    assert "пунктами два точка три и два точка четыре" in normalize("В соответствии с пунктами 2.3 и 2.4 договора.")
    assert "пункт один точка два точка три" in normalize("См. пункт 1.2.3 договора.").lower()
    assert "пункту два точка три" in normalize("Согласно пункту 2.3 договора.")
    assert "пункт два точка три договора" in normalize("В силу п. 2.3 договора.")
    assert normalize("1.2.3 Прочие условия.").startswith("Один точка два точка три")
    # real decimals and dates are untouched
    assert "целых" in normalize("Цена составила 2.5 рубля.")
    assert "точка" not in normalize("Договор от 12.05.2020 действует.")


def test_online_engines_can_join_paragraphs_into_one_request():
    from folivox.book import Block, BlockKind, Chapter
    from folivox.text.chunk import ChunkOptions, plan_chapter

    para = lambda t: Block(BlockKind.PARAGRAPH, t)  # noqa: E731
    chapter = Chapter("c", [para("Первый абзац книги."), para("Второй абзац книги."),
                            Block(BlockKind.SUBTITLE, "Часть"), para("Третий абзац книги."), para("* * *"),
                            para("Четвёртый абзац книги."), para("Пятый абзац книги.")])
    base = dict(stress="none", speak_headings=False, local_engine=False, min_chars=5)
    alone = plan_chapter(chapter, ChunkOptions(max_chars=300, **base))
    joined = plan_chapter(chapter, ChunkOptions(max_chars=3000, join_paragraphs=True, **base))
    assert len(alone) > len(joined)
    assert joined[0].text == "Первый абзац книги.\n\nВторой абзац книги."
    assert joined[0].source == joined[0].text and joined[0].pause_after == alone[1].pause_after
    # a heading and a scene break keep their own request and pause; indexes follow the new order
    assert [c.kind for c in joined] == ["p", "heading", "p", "p"]
    assert joined[3].text == "Четвёртый абзац книги.\n\nПятый абзац книги."
    assert [c.index for c in joined] == [0, 1, 2, 3] and joined[-1].pause_after == ChunkOptions().pauses.chapter_end
    # the size limit still holds
    tight = plan_chapter(chapter, ChunkOptions(max_chars=30, join_paragraphs=True, **base))
    assert all(len(c.text) <= 30 for c in tight)

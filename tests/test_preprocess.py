"""Тесты предобработки данных и текста (ЛР3, раздел 2).

Проверяется поведение на маленьких фиксированных примерах:
форма результата, обработка пропусков/пустых значений и преобразования.
"""

import pandas as pd
import pytest

from src.data import normalize_key, prepare
from src.predict import preprocess


@pytest.fixture
def raw_small():
    # 6 строк: пустой текст, текст из пробелов, два дубликата с разным регистром/пробелами
    return pd.DataFrame({
        "label": ["ham", "spam", "ham", "ham", "spam", "ham"],
        "text": ["See you  at lunch", "WIN a FREE prize", "", "   ", "win a free   prize", "Call me later"],
    })


def test_prepare_shape_and_columns(raw_small):
    out = prepare(raw_small)
    # пустые и дубликаты удалены: остаются 3 строки; добавлены row_id и y
    assert out.shape == (3, 4)
    assert list(out.columns) == ["label", "text", "row_id", "y"]


def test_prepare_removes_empty_values_instead_of_filling(raw_small):
    out = prepare(raw_small)
    # пропуски не заполняются выдуманным текстом: пустые строки исключаются
    assert (out["text"].str.strip() != "").all()
    assert 2 not in set(out["row_id"]) and 3 not in set(out["row_id"])


def test_prepare_keeps_first_duplicate_and_maps_labels(raw_small):
    out = prepare(raw_small)
    assert out["row_id"].tolist() == [0, 1, 5]          # дубликат row 4 удалён, row 1 сохранён
    assert out["y"].tolist() == [0, 1, 0]               # ham -> 0, spam -> 1


def test_prepare_rejects_unknown_labels():
    with pytest.raises(ValueError):
        prepare(pd.DataFrame({"label": ["ham", "phishing"], "text": ["a", "b"]}))


def test_normalize_key_ignores_case_and_spaces():
    keys = normalize_key(pd.Series(["Sorry, I'll call  later", " sorry, i'll call later\t"]))
    assert keys.iloc[0] == keys.iloc[1] == "sorry, i'll call later"


@pytest.mark.parametrize("raw, expected", [
    ("  FREE   prize\tnow ", "FREE prize now"),        # схлопывание пробельных символов
    ("ＦＲＥＥ ①", "FREE 1"),                           # NFKC: полноширинные буквы, обведённая цифра
    ("line1\r\nline2", "line1 line2"),                # переводы строк -> пробел
])
def test_preprocess_transformations(raw, expected):
    assert preprocess([raw]) == [expected]


def test_preprocess_preserves_length_and_order():
    texts = ["b", "a  a", "c"]
    assert preprocess(texts) == ["b", "a a", "c"]

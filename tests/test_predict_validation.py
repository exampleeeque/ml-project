"""Негативные и положительные тесты единой функции предсказания src.predict.predict.

Во всех негативных тестах вместо модели передаётся заглушка (Mock):
тест проверяет и контролируемое отклонение входа, и то, что модель
при этом НЕ вызывалась. Для проверки лимитов используются маленькие
искусственные пределы и вход чуть больше предела.

Запуск из корня проекта:  python -m pytest tests/ -v
"""

import math
from unittest.mock import Mock

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src import config
from src.predict import InputValidationError, Limits, ModelIntegrityError, load_model, predict

SMALL = Limits(max_batch=3, max_bytes=40, max_chars=10)


@pytest.fixture
def mock_model():
    m = Mock()
    m.predict_proba.side_effect = lambda X: np.tile([0.3, 0.7], (len(X), 1))
    return m


@pytest.fixture(scope="module")
def tiny_model():
    """Небольшая настоящая модель того же вида, что и рабочая (не требует артефакта)."""
    texts = ["win free prize now call", "free cash prize claim now", "urgent you won free entry",
             "see you at lunch", "are you home tonight", "call me when you are free"]
    y = [1, 1, 1, 0, 0, 0]
    return Pipeline([("tfidf", TfidfVectorizer()),
                     ("clf", LogisticRegression(random_state=0))]).fit(texts, y)


def expect_rejected(data, code, model, limits=Limits()):
    with pytest.raises(InputValidationError) as exc:
        predict(data, model=model, limits=limits)
    assert exc.value.code == code
    model.predict_proba.assert_not_called()
    return exc.value


# ---------------------------------------------------------------- 1. неверный тип / структура
@pytest.mark.parametrize("data, code", [
    (12345, "wrong_type"),                        # число вместо текста
    (["ok text", 42], "wrong_type"),              # неверный тип внутри пакета
    ({"message": "hi"}, "missing_field"),         # нет обязательного поля text
    ({"text": "hi", "user": "x"}, "unknown_field"),  # лишнее поле
    ([["nested", "list"]], "wrong_type"),         # неверная размерность (вложенный список)
])
def test_wrong_type_or_structure(data, code, mock_model):
    expect_rejected(data, code, mock_model)


# ---------------------------------------------------------------- 2. пустой вход, None, NaN, Inf
@pytest.mark.parametrize("data, code", [
    ("", "empty"),
    ("   \n\t ", "empty"),
    ([], "empty_batch"),
    (None, "missing_value"),
    ({"text": None}, "missing_value"),
    (float("nan"), "wrong_type"),
    (math.inf, "wrong_type"),
])
def test_empty_none_nan_inf(data, code, mock_model):
    expect_rejected(data, code, mock_model)


# ---------------------------------------------------------------- 3. превышение лимитов
def test_text_too_long(mock_model):
    expect_rejected("a" * (SMALL.max_chars + 1), "too_long", mock_model, SMALL)


def test_batch_too_large(mock_model):
    expect_rejected(["ok"] * (SMALL.max_batch + 1), "batch_too_large", mock_model, SMALL)


def test_bytes_too_large_checked_before_decoding(mock_model):
    # 41 байт при пределе 40: отклоняется до декодирования
    expect_rejected(b"x" * (SMALL.max_bytes + 1), "too_large", mock_model, SMALL)


def test_limits_boundary_accepted(mock_model):
    # Ровно на пределе вход принимается
    res = predict(["a" * SMALL.max_chars] * SMALL.max_batch, model=mock_model, limits=SMALL)
    assert len(res) == SMALL.max_batch


# ---------------------------------------------------------------- 4. повреждённое кодирование / символы
@pytest.mark.parametrize("data, code", [
    (b"\xff\xfe\xfa broken", "bad_encoding"),     # невалидный UTF-8
    ("hello\x00world", "bad_char"),               # нулевой байт
    ("text\x1b[31m", "bad_char"),                 # ESC-последовательность терминала
    ("bad \ud800 surrogate", "bad_char"),         # непарный суррогат
])
def test_corrupted_input(data, code, mock_model):
    expect_rejected(data, code, mock_model)


# ---------------------------------------------------------------- 5. повреждённый или подменённый файл модели
def test_tampered_model_file_rejected(tmp_path):
    fake = tmp_path / "model.skops"
    fake.write_bytes(b"not a model")
    ref = tmp_path / "SHA256SUMS"
    ref.write_text("0" * 64 + "  " + fake.as_posix() + "\n", encoding="utf-8")
    with pytest.raises(ModelIntegrityError):
        load_model(fake, ref)


# ---------------------------------------------------------------- положительные тесты
def test_valid_single_and_batch(tiny_model):
    single = predict("Win a FREE prize now, call!", model=tiny_model)
    assert single["label"] == "spam" and 0.5 <= single["spam_probability"] <= 1.0
    batch = predict(["see you at lunch", {"text": "free cash prize"}, "hello".encode()], model=tiny_model)
    assert [r["label"] for r in batch][:2] == ["ham", "spam"]
    assert all(0.0 <= r["spam_probability"] <= 1.0 for r in batch)


def test_preprocessing_applied(mock_model):
    predict("  FREE   prize\tnow ", model=mock_model)
    mock_model.predict_proba.assert_called_once_with(["FREE prize now"])


@pytest.mark.skipif(not config.MODEL_PATH.exists(), reason="модель не обучена (python -m src.train)")
def test_saved_model_loads_and_predicts():
    model = load_model()
    res = predict("Congratulations! You have won a FREE prize. Call 09061701461 to claim", model=model)
    assert res["label"] == "spam"
    assert predict("Are we still meeting for lunch tomorrow?", model=model)["label"] == "ham"

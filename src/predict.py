"""Единая функция предсказания с валидацией входов и безопасной загрузкой модели.

Порядок обработки: проверка входа -> предобработка -> вызов модели.
При отклонении входа выбрасывается InputValidationError, модель не вызывается.

Политика валидации:
  * принимается одна строка (str), bytes в UTF-8, объект {"text": str}
    либо список/кортеж таких элементов (пакет);
  * пакет: от 1 до MAX_BATCH элементов;
  * bytes: не более MAX_BYTES байт, проверяется ДО декодирования;
    декодирование строгое (невалидный UTF-8 отклоняется);
  * текст: от 1 до MAX_CHARS символов, не пустой и не только из пробелов;
  * None, NaN, Inf, числа, bool и другие типы отклоняются (числовых признаков
    у текстовой модели нет, поэтому NaN/Inf трактуются как неверный тип);
  * управляющие символы (кроме \\t, \\n, \\r) и непарные суррогаты отклоняются;
  * у объекта-словаря обязательно поле "text", лишние поля запрещены.

Запуск из корня проекта:
    python -m src.predict "Congratulations! You won a FREE prize"
"""

from __future__ import annotations

import sys
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src import config
from src.data import read_reference, sha256_of

ALLOWED_CONTROL = {"\t", "\n", "\r"}


class InputValidationError(ValueError):
    """Контролируемое отклонение входа. code — машинно-читаемый код причины."""

    def __init__(self, code: str, message: str, index: int | None = None):
        self.code = code
        self.index = index
        where = f" (элемент {index})" if index is not None else ""
        super().__init__(f"[{code}]{where} {message}")


class ModelIntegrityError(RuntimeError):
    """Файл модели не прошёл проверку хэша или содержит недоверенные типы."""


@dataclass(frozen=True)
class Limits:
    max_batch: int = config.MAX_BATCH
    max_bytes: int = config.MAX_BYTES
    max_chars: int = config.MAX_CHARS


DEFAULT_LIMITS = Limits()


# ---------------------------------------------------------------- валидация
def _validate_text(item: Any, limits: Limits, index: int | None) -> str:
    if isinstance(item, Mapping):
        if "text" not in item:
            raise InputValidationError("missing_field", "отсутствует обязательное поле 'text'", index)
        extra = set(item) - {"text"}
        if extra:
            raise InputValidationError("unknown_field", f"недопустимые поля: {sorted(map(str, extra))}", index)
        item = item["text"]

    if isinstance(item, (bytes, bytearray)):
        if len(item) > limits.max_bytes:
            raise InputValidationError(
                "too_large", f"размер {len(item)} байт превышает предел {limits.max_bytes}", index)
        try:
            item = bytes(item).decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            raise InputValidationError("bad_encoding", "текст не является корректным UTF-8", index) from None

    if item is None:
        raise InputValidationError("missing_value", "значение отсутствует (None)", index)
    if not isinstance(item, str):
        # float('nan'), float('inf'), int, bool, list и т. п.
        raise InputValidationError(
            "wrong_type", f"ожидалась строка, получен {type(item).__name__}: {item!r:.40}", index)

    if len(item) > limits.max_chars:
        raise InputValidationError(
            "too_long", f"длина {len(item)} символов превышает предел {limits.max_chars}", index)
    if len(item.encode("utf-8", errors="surrogatepass")) > limits.max_bytes:
        raise InputValidationError("too_large", f"размер превышает предел {limits.max_bytes} байт", index)
    if not item.strip():
        raise InputValidationError("empty", "пустой текст", index)
    for ch in item:
        cat = unicodedata.category(ch)
        if (cat == "Cc" and ch not in ALLOWED_CONTROL) or cat == "Cs":
            raise InputValidationError(
                "bad_char", f"недопустимый символ U+{ord(ch):04X}", index)
    return item


def validate_input(data: Any, limits: Limits = DEFAULT_LIMITS) -> tuple[list[str], bool]:
    """Проверяет вход. Возвращает (список текстов, был ли вход одиночным)."""
    if isinstance(data, (list, tuple)):
        if len(data) == 0:
            raise InputValidationError("empty_batch", "пустой пакет")
        if len(data) > limits.max_batch:
            raise InputValidationError(
                "batch_too_large", f"пакет из {len(data)} элементов превышает предел {limits.max_batch}")
        return [_validate_text(x, limits, i) for i, x in enumerate(data)], False
    return [_validate_text(data, limits, None)], True


# ---------------------------------------------------------------- предобработка
def preprocess(texts: list[str]) -> list[str]:
    """Нормализация перед моделью: Unicode NFKC и схлопывание пробелов.

    Приведение к нижнему регистру и токенизацию выполняет TfidfVectorizer
    внутри сохранённого Pipeline, поэтому при обучении, оценке и предсказании
    применяются одни и те же преобразования.
    """
    return [" ".join(unicodedata.normalize("NFKC", t).split()) for t in texts]


# ---------------------------------------------------------------- модель
def trusted_types() -> list[str]:
    """Типы, которые разрешено восстанавливать из файла skops."""
    return [
        "numpy.dtype",
        "numpy.float64",
        "sklearn.feature_extraction.text.TfidfTransformer",
        "sklearn.feature_extraction.text.TfidfVectorizer",
        "sklearn.linear_model._logistic.LogisticRegression",
        "sklearn.pipeline.Pipeline",
        "builtins.dict",
    ]


def verify_model_file(path: Path = config.MODEL_PATH,
                      reference: Path = config.MODEL_REFERENCE) -> str:
    if not path.exists():
        raise ModelIntegrityError(f"Файл модели {path} не найден")
    if not reference.exists():
        raise ModelIntegrityError(f"Эталон {reference} не найден")
    path = path.resolve()
    key = path.relative_to(config.ROOT).as_posix() if path.is_relative_to(config.ROOT) else path.as_posix()
    expected = read_reference(reference).get(key)
    actual = sha256_of(path)
    if expected is None or actual != expected:
        raise ModelIntegrityError(f"SHA-256 модели не совпадает с эталоном: {actual} != {expected}")
    return actual


def load_model(path: Path = config.MODEL_PATH, reference: Path = config.MODEL_REFERENCE):
    """Загрузка модели: сначала хэш, затем проверка типов, затем skops.io.load.

    Формат skops не выполняет произвольный код при загрузке (в отличие от pickle);
    типы, не входящие в trusted_types(), приводят к отказу.
    """
    import skops.io as sio

    verify_model_file(path, reference)
    untrusted = set(sio.get_untrusted_types(file=path))
    unexpected = untrusted - set(trusted_types())
    if unexpected:
        raise ModelIntegrityError(f"В файле модели недоверенные типы: {sorted(unexpected)}")
    return sio.load(path, trusted=sorted(untrusted))


_MODEL = None


def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = load_model()
    return _MODEL


# ---------------------------------------------------------------- предсказание
def predict(data: Any, model=None, limits: Limits = DEFAULT_LIMITS,
            threshold: float = config.DECISION_THRESHOLD):
    """Возвращает {"label": "ham"|"spam", "spam_probability": float}
    для одиночного входа или список таких словарей для пакета."""
    texts, single = validate_input(data, limits)          # 1. проверка
    prepared = preprocess(texts)                          # 2. предобработка
    model = model if model is not None else get_model()
    proba = model.predict_proba(prepared)[:, 1]           # 3. модель
    results = [{"label": config.POSITIVE_LABEL if p >= threshold else "ham",
                "spam_probability": round(float(p), 4)} for p in proba]
    return results[0] if single else results


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print('Использование: python -m src.predict "текст сообщения"', file=sys.stderr)
        return 2
    try:
        print(predict(argv[0]))
    except InputValidationError as e:
        print(f"Вход отклонён: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

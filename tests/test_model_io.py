"""Интеграционные тесты: сохранение -> проверка хэша -> загрузка -> предсказание (ЛР3).

Используется небольшая модель той же конструкции, что и рабочая
(src.train.build_pipeline), обученная за доли секунды на фиксированных
примерах. Рабочий артефакт для этих тестов не нужен, поэтому они
выполняются в CI до обучения кандидата.
"""

import operator

import pytest
import skops.io as sio
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.data import sha256_of
from src.predict import ModelIntegrityError, load_model, predict
from src.train import build_pipeline

SPAM = ["win a free prize now", "free prize claim now call", "urgent free prize call now",
        "claim your free cash prize", "you won a free prize call"]
HAM = ["see you at lunch tomorrow", "are you at home now", "lunch at home tomorrow",
       "see you at home tonight", "are you free for lunch tomorrow"]


def _save(model, tmp_path):
    path = tmp_path / "model.skops"
    sio.dump(model, path)
    ref = tmp_path / "MODEL_SHA256SUMS"
    ref.write_text(f"{sha256_of(path)}  {path.resolve().as_posix()}\n", encoding="utf-8")
    return path, ref


@pytest.fixture(scope="module")
def trained():
    return build_pipeline().fit(SPAM + HAM, [1] * len(SPAM) + [0] * len(HAM))


def test_saved_model_loads_and_predicts_valid_input(trained, tmp_path):
    path, ref = _save(trained, tmp_path)
    model = load_model(path, ref)
    assert predict("claim your free prize now", model=model)["label"] == "spam"
    assert predict("see you at lunch tomorrow", model=model)["label"] == "ham"
    res = predict(["free prize", "at home"], model=model)
    assert len(res) == 2 and all(0.0 <= r["spam_probability"] <= 1.0 for r in res)


def test_modified_file_is_not_loaded(trained, tmp_path):
    path, ref = _save(trained, tmp_path)
    path.write_bytes(path.read_bytes() + b"\x00")       # изменение после фиксации хэша
    with pytest.raises(ModelIntegrityError):
        load_model(path, ref)


def test_untrusted_type_is_not_loaded(tmp_path):
    # файл содержит ссылку на функцию (_operator.neg): хэш верный, но тип не из доверенного списка
    other = Pipeline([("f", FunctionTransformer(operator.neg))])
    path, ref = _save(other, tmp_path)
    with pytest.raises(ModelIntegrityError, match="недоверенные типы"):
        load_model(path, ref)

"""Тест расчёта метрик на маленьком примере с заранее известным результатом (ЛР3).

y_true = [1, 1, 1, 0, 0, 0, 0, 0]
y_pred = [1, 1, 0, 1, 0, 0, 0, 0]
TP = 2, FN = 1, FP = 1, TN = 4 (посчитано вручную):
  precision(spam) = 2 / 3, recall(spam) = 2 / 3, F1(spam) = 2 / 3 ≈ 0,6667
  precision(ham) = 4 / 5,  recall(ham) = 4 / 5,  F1(ham) = 0,8
  accuracy = 6 / 8 = 0,75
"""

import pytest

from src.metrics import evaluate, primary_value

Y_TRUE = [1, 1, 1, 0, 0, 0, 0, 0]
Y_PRED = [1, 1, 0, 1, 0, 0, 0, 0]


def test_confusion_matrix_known_example():
    assert evaluate(Y_TRUE, Y_PRED)["confusion_matrix"] == {"tn": 4, "fp": 1, "fn": 1, "tp": 2}


def test_per_class_and_primary_metric_known_example():
    m = evaluate(Y_TRUE, Y_PRED)
    assert m["accuracy"] == pytest.approx(0.75)
    assert m["per_class"]["spam"]["precision"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["per_class"]["spam"]["recall"] == pytest.approx(2 / 3, abs=1e-4)
    assert primary_value(m) == pytest.approx(2 / 3, abs=1e-4)
    assert m["per_class"]["ham"]["f1"] == pytest.approx(0.8)
    assert m["average"]["macro"]["f1"] == pytest.approx((2 / 3 + 0.8) / 2, abs=1e-4)


def test_constant_predictor_has_zero_f1_spam():
    # константный ответ ham: accuracy высокая, F1(spam) = 0 — поэтому accuracy не основная метрика
    m = evaluate(Y_TRUE, [0] * len(Y_TRUE))
    assert m["accuracy"] == pytest.approx(5 / 8)
    assert primary_value(m) == 0.0

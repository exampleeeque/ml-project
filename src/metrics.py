"""Единое определение метрик качества (ЛР3).

Одна и та же функция используется при обучении (src.train), при оценке
кандидатной модели в CI (src.evaluate) и в тестах, поэтому определение
основной метрики F1(spam) одинаково на всех этапах.
"""

from __future__ import annotations

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from src import config

PRIMARY_METRIC = "f1_spam"   # основная метрика проекта (ЛР2): F1 для класса spam


def evaluate(y_true, y_pred) -> dict:
    """Accuracy, метрики по классам, macro/weighted и матрица ошибок. 1 = spam."""
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    per_class = {lab: {"precision": round(float(p[i]), 4), "recall": round(float(r[i]), 4),
                       "f1": round(float(f[i]), 4), "support": int(s[i])}
                 for i, lab in enumerate(config.LABELS)}
    avg = {}
    for mode in ("macro", "weighted"):
        pm, rm, fm, _ = precision_recall_fscore_support(y_true, y_pred, average=mode, zero_division=0)
        avg[mode] = {"precision": round(float(pm), 4), "recall": round(float(rm), 4), "f1": round(float(fm), 4)}
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "per_class": per_class, "average": avg,
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}


def primary_value(metrics: dict) -> float:
    """Значение основной метрики из результата evaluate()."""
    return metrics["per_class"][config.POSITIVE_LABEL]["f1"]

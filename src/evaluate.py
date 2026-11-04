"""Оценка кандидатной модели на фиксированной контрольной выборке (ЛР3).

Модель загружается той же функцией load_model() (проверка SHA-256 и списка
типов skops). Предсказания получаются так же, как при расчёте baseline в ЛР2
(src.train): та же предобработка preprocess(), тот же порог, та же функция
метрик src.metrics.evaluate(). Валидация входа predict() здесь не применяется:
данные уже проверены data gate, а 8 сообщений контрольной выборки содержат
известные C1-символы источника, которые политика входа API отклоняет.

Результат: reports/candidate_metrics.json — метрики кандидата, версия
данных, SHA-256 контрольной выборки и модели, commit. Baseline не меняется.

Запуск из корня проекта (после python -m src.control_set и data gate):
    python -m src.evaluate
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime

import numpy as np

from src import config
from src.data import sha256_of
from src.metrics import PRIMARY_METRIC, evaluate, primary_value
from src.predict import load_model, preprocess, verify_model_file


def read_control(path=config.CONTROL_PATH) -> tuple[list[str], np.ndarray]:
    lines = path.read_text(encoding="utf-8").rstrip("\n").split("\n")[1:]   # без заголовка
    texts, y = [], []
    for line in lines:
        _, label, text = line.split("\t")
        texts.append(text)
        y.append(1 if label == config.POSITIVE_LABEL else 0)
    return texts, np.array(y)


def main() -> int:
    model_sha = verify_model_file()
    model = load_model()
    texts, y_true = read_control()

    proba = model.predict_proba(preprocess(texts))[:, 1]          # как в src.train (ЛР2)
    y_pred = (proba >= config.DECISION_THRESHOLD).astype(int)

    metrics = evaluate(y_true, y_pred)
    manifest = json.loads((config.REPORTS_DIR / "control_set.json").read_text(encoding="utf-8"))
    result = {
        "metric": PRIMARY_METRIC,
        "value": primary_value(metrics),
        "n": int(len(y_true)),
        "data_id": manifest["data_id"],
        "control_sha256": sha256_of(config.CONTROL_PATH),
        "model_file": config.MODEL_PATH.relative_to(config.ROOT).as_posix(),
        "model_sha256": model_sha,
        "commit": os.environ.get("GITHUB_SHA", "local"),
        "evaluated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "threshold": config.DECISION_THRESHOLD,
        "metrics": metrics,
    }
    config.REPORTS_DIR.mkdir(exist_ok=True)
    (config.REPORTS_DIR / "candidate_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    cm = metrics["confusion_matrix"]
    print(f"Кандидат: {PRIMARY_METRIC} = {result['value']} на {result['n']} объектах "
          f"(TP {cm['tp']}, FP {cm['fp']}, FN {cm['fn']}, TN {cm['tn']})")
    print(f"SHA-256 модели: {model_sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Обучение и оценка базовой модели TF-IDF + логистическая регрессия.

Шаги:
  1) проверка SHA-256 датасета относительно эталона ЛР1;
  2) удаление дубликатов, стратифицированное разбиение train/test 80/20, seed 42;
  3) подбор C 5-кратной стратифицированной кросс-валидацией ТОЛЬКО на train;
  4) оценка на отложенном test, сравнение с DummyClassifier;
  5) сохранение модели (skops) и её SHA-256.

ЛР3: по умолчанию train.py обучает КАНДИДАТНУЮ модель и пишет параметры
и метрики обучения в reports/. Baseline (metrics/baseline.json) больше
не перезаписывается автоматически: только явным ключом --update-baseline.

Запуск из корня проекта:
    python -m src.train                    # кандидатная модель (так запускает CI)
    python -m src.train --force            # перезаписать существующий файл модели
    python -m src.train --force --update-baseline   # осознанное обновление baseline
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from src import config
from src.data import check_raw, data_version, load_raw, prepare, sha256_of, verify_raw_data
from src.metrics import PRIMARY_METRIC, evaluate, primary_value
from src.predict import preprocess


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(**config.TFIDF_PARAMS)),
        ("clf", LogisticRegression(**config.LOGREG_PARAMS)),
    ])


def split(df: pd.DataFrame):
    return train_test_split(df, test_size=config.TEST_SIZE,
                            random_state=config.SEED, stratify=df["y"])


def ids_hash(ids) -> str:
    return hashlib.sha256(",".join(map(str, sorted(ids))).encode()).hexdigest()


def save_confusion_figure(y_test, y_dummy, y_pred, best_c) -> None:
    """Матрицы ошибок для отчёта (только при обновлении baseline)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, yp, title in ((axes[0], y_dummy, "DummyClassifier (всегда ham)"),
                          (axes[1], y_pred, f"TF-IDF + LogReg (C={best_c})")):
        ConfusionMatrixDisplay(confusion_matrix(y_test, yp, labels=[0, 1]),
                               display_labels=list(config.LABELS)).plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(title)
        ax.set_xlabel("Предсказанный класс")
        ax.set_ylabel("Истинный класс")
    plt.tight_layout()
    plt.savefig(config.FIG_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def _parse(argv):
    parser = argparse.ArgumentParser(description="Обучение модели")
    parser.add_argument("--force", action="store_true",
                        help="перезаписать существующую модель и её эталон SHA-256")
    parser.add_argument("--update-baseline", action="store_true",
                        help="записать metrics/baseline.json (только осознанно, вручную)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    import skops
    import skops.io as sio

    args = _parse(argv)
    if config.MODEL_PATH.exists() and not args.force:
        print(f"Модель {config.MODEL_PATH.name} уже существует; её SHA-256 зафиксирован в "
              f"{config.MODEL_REFERENCE.name}. Для переобучения запустите с --force.")
        return 1

    out_dir = config.METRICS_DIR if args.update_baseline else config.REPORTS_DIR
    for d in (config.MODELS_DIR, out_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Данные
    raw_sha = verify_raw_data()
    raw = load_raw()
    checks = check_raw(raw)
    df = prepare(raw)
    version = data_version(raw_sha, df)
    print("Проверка данных:", json.dumps(checks, ensure_ascii=False))
    print("Версия данных:", version["data_id"])

    # 2. Разбиение
    train_df, test_df = split(df)
    X_train, y_train = preprocess(train_df["text"].tolist()), train_df["y"].to_numpy()
    X_test, y_test = preprocess(test_df["text"].tolist()), test_df["y"].to_numpy()
    print(f"train={len(train_df)} (spam {y_train.mean():.2%}), test={len(test_df)} (spam {y_test.mean():.2%})")

    # 3. Подбор C на train (кросс-валидация), обучение
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.SEED)
    search = GridSearchCV(build_pipeline(), {"clf__C": list(config.C_GRID)},
                          scoring="f1", cv=cv, n_jobs=1, refit=True)
    t0 = time.perf_counter()
    search.fit(X_train, y_train)
    search_time = time.perf_counter() - t0
    best_c = search.best_params_["clf__C"]

    model = build_pipeline().set_params(clf__C=best_c)
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0
    cv_scores = {str(c): round(float(s), 4) for c, s in
                 zip(search.cv_results_["param_clf__C"], search.cv_results_["mean_test_score"], strict=True)}
    print(f"CV F1(spam) по C: {cv_scores}; выбрано C={best_c}")
    print(f"Время подбора: {search_time:.2f} c, время обучения итоговой модели: {fit_time:.3f} c")

    # 4. Оценка на test и ориентир
    dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    y_dummy = dummy.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= config.DECISION_THRESHOLD).astype(int)
    m_model, m_dummy = evaluate(y_test, y_pred), evaluate(y_test, y_dummy)
    print(classification_report(y_test, y_pred, target_names=list(config.LABELS), digits=4, zero_division=0))

    # 5. Сохранение модели и хэша
    sio.dump(model, config.MODEL_PATH)
    model_sha = sha256_of(config.MODEL_PATH)
    rel = config.MODEL_PATH.relative_to(config.ROOT).as_posix()
    config.MODEL_REFERENCE.write_text(f"{model_sha}  {rel}\n", encoding="utf-8")
    print(f"Модель: {rel}, SHA-256 {model_sha}")

    params = {
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": "Pipeline(TfidfVectorizer -> LogisticRegression)",
        "tfidf": {k: list(v) if isinstance(v, tuple) else v for k, v in config.TFIDF_PARAMS.items()},
        "logreg": {**config.LOGREG_PARAMS, "C": best_c},
        "c_grid": list(config.C_GRID), "cv": f"StratifiedKFold(n_splits={config.CV_FOLDS}, shuffle=True)",
        "cv_scoring": "f1 (класс spam)", "cv_f1_by_C": cv_scores,
        "seed": config.SEED, "test_size": config.TEST_SIZE, "split": "train_test_split, stratify=y",
        "decision_threshold": config.DECISION_THRESHOLD,
        "train_size": int(len(train_df)), "test_size_rows": int(len(test_df)),
        "train_ids_sha256": ids_hash(train_df["row_id"]), "test_ids_sha256": ids_hash(test_df["row_id"]),
        "data": version, "data_checks": checks,
        "time_seconds": {"grid_search": round(search_time, 2), "final_fit": round(fit_time, 3)},
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__, "skops": skops.__version__},
        "model_file": rel, "model_sha256": model_sha,
        "vocabulary_size": int(len(model.named_steps["tfidf"].vocabulary_)),
        "test_metrics": m_model,
    }
    (out_dir / "train_params.json").write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"F1(spam) на test = {primary_value(m_model)} (Dummy: {primary_value(m_dummy)})")

    if args.update_baseline:
        # Осознанное обновление ориентира: только вручную, не в CI.
        errors = test_df.assign(spam_probability=proba.round(4), pred=y_pred)
        errors = errors[errors["pred"] != errors["y"]][["row_id", "label", "spam_probability", "text"]]
        errors.to_csv(config.METRICS_DIR / "test_errors.csv", index=False)
        config.FIG_DIR.mkdir(parents=True, exist_ok=True)
        save_confusion_figure(y_test, y_dummy, y_pred, best_c)
        baseline = {
            "primary_metric": PRIMARY_METRIC,
            "value": primary_value(m_model),
            "test_size": int(len(test_df)),
            "data_id": version["data_id"],
            "model_sha256": model_sha,
            "threshold": config.DECISION_THRESHOLD,
            "model": m_model,
            "reference_dummy_most_frequent": m_dummy,
        }
        config.BASELINE_PATH.write_text(json.dumps(baseline, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Baseline обновлён: {config.BASELINE_PATH.relative_to(config.ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Общие параметры проекта: пути, seed, гиперпараметры и пределы входных данных.

Все значения, влияющие на воспроизводимость и на политику валидации,
собраны в одном месте, чтобы их можно было сослаться в отчёте и в ЛР3/ЛР4.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# --- Данные ---
RAW_DATA = ROOT / "data" / "raw" / "SMSSpamCollection"
DATA_REFERENCE = ROOT / "integrity" / "SHA256SUMS"        # эталон из ЛР1
RAW_DATA_KEY = "data/raw/SMSSpamCollection"               # путь в эталоне
LABELS = ("ham", "spam")
POSITIVE_LABEL = "spam"
DEDUP_RULE = "lower+collapse-whitespace"                  # правило удаления дубликатов

# --- Модель и артефакты ---
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "spam_tfidf_logreg.skops"
MODEL_REFERENCE = ROOT / "integrity" / "MODEL_SHA256SUMS"  # эталон хэша модели (в Git)
METRICS_DIR = ROOT / "metrics"
FIG_DIR = ROOT / "notebooks" / "figures"

# --- Воспроизводимость ---
SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
C_GRID = (0.1, 1.0, 10.0, 100.0)   # единственный подбираемый параметр

TFIDF_PARAMS = {
    "lowercase": True,
    "ngram_range": (1, 2),
    "min_df": 2,
    "sublinear_tf": True,
}
LOGREG_PARAMS = {
    "class_weight": "balanced",
    "solver": "liblinear",
    "max_iter": 1000,
    "random_state": SEED,
}
DECISION_THRESHOLD = 0.5

# --- Пределы входных данных (политика валидации, раздел 5 ЛР2) ---
MAX_BATCH = 100        # не более 100 сообщений в одном вызове
MAX_BYTES = 4000       # не более 4000 байт UTF-8 на сообщение (проверка ДО декодирования)
MAX_CHARS = 1000       # не более 1000 символов на сообщение (после декодирования)

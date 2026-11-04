"""Загрузка, проверка целостности и подготовка датасета SMS Spam Collection."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pandas as pd

from src import config


class DataIntegrityError(RuntimeError):
    """SHA-256 файла не совпадает с эталоном или эталон отсутствует."""


def sha256_of(path: Path, chunk_size: int = 65536) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_reference(path: Path) -> dict[str, str]:
    """Читает файл формата sha256sum: '<hash>  <путь>' -> {путь: hash}."""
    entries: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            digest, name = line.split(maxsplit=1)
            entries[name.lstrip("*")] = digest.lower()
    return entries


def verify_raw_data(path: Path = config.RAW_DATA,
                    reference: Path = config.DATA_REFERENCE,
                    key: str = config.RAW_DATA_KEY) -> str:
    """Сверяет SHA-256 сырого файла с эталоном ЛР1. Возвращает фактический хэш."""
    if not reference.exists():
        raise DataIntegrityError(f"Эталон {reference} не найден")
    expected = read_reference(reference).get(key)
    if expected is None:
        raise DataIntegrityError(f"В эталоне нет записи для {key}")
    actual = sha256_of(path)
    if actual != expected:
        raise DataIntegrityError(f"SHA-256 {key} не совпадает с эталоном: {actual} != {expected}")
    return actual


def load_raw(path: Path = config.RAW_DATA) -> pd.DataFrame:
    """Читает TSV без заголовка. QUOTE_NONE обязателен: в текстах есть непарные кавычки."""
    df = pd.read_csv(path, sep="\t", header=None, names=["label", "text"],
                     quoting=csv.QUOTE_NONE, encoding="utf-8", dtype=str,
                     keep_default_na=False)
    return df


def check_raw(df: pd.DataFrame) -> dict:
    """Проверки качества данных: схема, пропуски, метки, пустые тексты, дубликаты."""
    key = normalize_key(df["text"])
    conflicts = df.assign(key=key).groupby("key")["label"].nunique()
    return {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "missing": {c: int((df[c].isna() | (df[c] == "")).sum()) for c in df.columns},
        "labels": sorted(df["label"].unique().tolist()),
        "labels_valid": bool(set(df["label"]) <= set(config.LABELS)),
        "blank_texts": int((df["text"].str.strip() == "").sum()),
        "exact_duplicates": int(df.duplicated().sum()),
        "normalized_duplicates": int(key.duplicated().sum()),
        "conflicting_labels": int((conflicts > 1).sum()),
        "class_counts": df["label"].value_counts().to_dict(),
    }


def normalize_key(texts: pd.Series) -> pd.Series:
    """Ключ для поиска дубликатов: нижний регистр и схлопывание пробелов.

    TF-IDF приводит текст к нижнему регистру, поэтому такие сообщения
    для модели неразличимы и при попадании в train и test дают утечку.
    """
    return texts.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Удаляет пустые тексты и дубликаты (первое вхождение сохраняется).

    Добавляет столбец y (1 = spam). Индекс исходного файла сохраняется в row_id,
    чтобы разбиение можно было воспроизвести и проверить.
    """
    if not set(df["label"]) <= set(config.LABELS):
        raise ValueError(f"Неожиданные метки: {set(df['label']) - set(config.LABELS)}")
    out = df[df["text"].str.strip() != ""].copy()
    out["row_id"] = out.index
    out = out[~normalize_key(out["text"]).duplicated(keep="first")]
    out["y"] = (out["label"] == config.POSITIVE_LABEL).astype(int)
    return out.reset_index(drop=True)


def data_version(raw_sha256: str, prepared: pd.DataFrame) -> dict:
    """Идентификатор версии данных: хэш сырого файла + хэш подготовленного набора."""
    canon = "\n".join(f"{r}\t{lab}\t{t}" for r, lab, t in
                      zip(prepared["row_id"], prepared["label"], prepared["text"], strict=True))
    prepared_sha = hashlib.sha256(canon.encode("utf-8")).hexdigest()
    return {
        "data_id": f"smsspam-raw-{raw_sha256[:12]}-dedup-{prepared_sha[:12]}",
        "raw_sha256": raw_sha256,
        "prepared_sha256": prepared_sha,
        "dedup_rule": config.DEDUP_RULE,
        "rows_prepared": int(len(prepared)),
    }

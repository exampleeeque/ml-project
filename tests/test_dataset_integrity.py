"""Автоматические проверки датасета.

1) Целостность: SHA256 файлов в data/raw/ совпадает с эталоном из integrity/SHA256SUMS.
2) Формат: файл читается как TSV из двух столбцов, метки только ham/spam.

Запуск из корня проекта:  python -m pytest -v
"""

import csv
import hashlib
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "integrity" / "SHA256SUMS"
DATA_FILE = ROOT / "data" / "raw" / "SMSSpamCollection"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_reference():
    """Читает файл формата sha256sum: '<hash>  <путь относительно корня проекта>'."""
    entries = []
    for line in REFERENCE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            digest, name = line.split(maxsplit=1)
            entries.append((digest.lower(), name.lstrip("*")))
    return entries


@pytest.mark.skipif(not REFERENCE.exists(), reason="эталон integrity/SHA256SUMS ещё не создан")
def test_sha256_matches_reference():
    entries = read_reference()
    assert entries, "файл эталона пуст"
    for expected, name in entries:
        path = ROOT / name
        assert path.exists(), f"файл {name} отсутствует — загрузите датасет"
        assert sha256_of(path) == expected, f"SHA256 файла {name} не совпадает с эталоном"


@pytest.mark.skipif(not DATA_FILE.exists(), reason="датасет не загружен")
def test_dataset_format():
    df = pd.read_csv(DATA_FILE, sep="\t", header=None, names=["label", "text"],
                     quoting=csv.QUOTE_NONE, encoding="utf-8")
    assert len(df) > 0
    assert set(df["label"].unique()) == {"ham", "spam"}
    assert df["text"].notna().all()

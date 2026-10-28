"""Загрузка датасета SMS Spam Collection (UCI) в каталог data/raw/.

Скрипт:
  1) скачивает официальный архив с UCI Machine Learning Repository по HTTPS;
  2) безопасно извлекает из архива только файл SMSSpamCollection
     (без распаковки произвольных путей — защита от path traversal);
  3) выводит SHA256 архива и файла данных в формате утилиты sha256sum.

Запуск из корня проекта:
    python src/download_dataset.py
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

DATASET_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
MEMBER_NAME = "SMSSpamCollection"  # основной файл данных внутри архива

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
ARCHIVE_PATH = RAW_DIR / "sms_spam_collection.zip"
DATA_PATH = RAW_DIR / MEMBER_NAME


def sha256_of(path: Path, chunk_size: int = 65536) -> str:
    """Возвращает SHA256 файла (чтение блоками, чтобы не загружать файл целиком)."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest: Path) -> None:
    """Скачивает файл по HTTPS (сертификат сервера проверяется по умолчанию)."""
    if not url.startswith("https://"):
        raise ValueError("Разрешена загрузка только по HTTPS")
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Загрузка {url}")
    with urllib.request.urlopen(url, timeout=60) as response, dest.open("wb") as out:
        out.write(response.read())
    print(f"Сохранено: {dest.relative_to(ROOT)} ({dest.stat().st_size} байт)")


def extract_member(archive: Path, member: str, dest: Path) -> None:
    """Извлекает один файл из архива, записывая его по заранее известному пути."""
    with zipfile.ZipFile(archive) as zf:
        if member not in zf.namelist():
            raise FileNotFoundError(f"В архиве нет файла {member}: {zf.namelist()}")
        dest.write_bytes(zf.read(member))
    print(f"Извлечено: {dest.relative_to(ROOT)} ({dest.stat().st_size} байт)")


def main() -> int:
    if ARCHIVE_PATH.exists():
        print(f"Архив уже существует: {ARCHIVE_PATH.relative_to(ROOT)} — повторная загрузка не выполняется")
    else:
        download(DATASET_URL, ARCHIVE_PATH)
    extract_member(ARCHIVE_PATH, MEMBER_NAME, DATA_PATH)

    print("\nSHA256 (формат sha256sum):")
    for path in (ARCHIVE_PATH, DATA_PATH):
        print(f"{sha256_of(path)}  {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

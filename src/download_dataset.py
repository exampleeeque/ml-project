"""Загрузка датасета SMS Spam Collection (UCI) в каталог data/raw/ (из ЛР1)."""
from __future__ import annotations

import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import urlparse

DATASET_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
MEMBER_NAME = "SMSSpamCollection"
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
ARCHIVE_PATH = RAW_DIR / "sms_spam_collection.zip"
DATA_PATH = RAW_DIR / MEMBER_NAME
ALLOWED_HOSTS = {"archive.ics.uci.edu"}
MAX_ARCHIVE_BYTES = 5 * 1024 * 1024      # архив ~200 КБ; предел с запасом
MAX_MEMBER_BYTES = 5 * 1024 * 1024       # файл данных ~470 КБ; защита от zip-бомбы


class _HttpsOnlyRedirect(urllib.request.HTTPRedirectHandler):
    """Запрещает перенаправление на другую схему или узел (понижение до http, file:)."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _check_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _check_url(url: str) -> None:
    parts = urlparse(url)
    if parts.scheme != "https" or parts.hostname not in ALLOWED_HOSTS:
        raise ValueError(f"Разрешена загрузка только по HTTPS с {sorted(ALLOWED_HOSTS)}: {url}")


def sha256_of(path: Path, chunk_size: int = 65536) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest: Path) -> None:
    """Скачивает файл по HTTPS с проверкой схемы, узла, перенаправлений и размера."""
    _check_url(url)
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Только HTTPS-обработчик: схемы file:, ftp: и data: недоступны.
    opener = urllib.request.OpenerDirector()
    for handler in (urllib.request.HTTPSHandler(), _HttpsOnlyRedirect(),
                    urllib.request.HTTPErrorProcessor(), urllib.request.ProxyHandler()):
        opener.add_handler(handler)
    # Схема и узел проверены в _check_url выше, перенаправления — в _HttpsOnlyRedirect.
    with opener.open(url, timeout=60) as response:
        data = response.read(MAX_ARCHIVE_BYTES + 1)
    if len(data) > MAX_ARCHIVE_BYTES:
        raise ValueError(f"Архив больше {MAX_ARCHIVE_BYTES} байт")
    dest.write_bytes(data)


def extract_member(archive: Path, member: str, dest: Path) -> None:
    with zipfile.ZipFile(archive) as zf:
        if member not in zf.namelist():
            raise FileNotFoundError(f"В архиве нет файла {member}: {zf.namelist()}")
        if zf.getinfo(member).file_size > MAX_MEMBER_BYTES:
            raise ValueError(f"Файл {member} после распаковки больше {MAX_MEMBER_BYTES} байт")
        dest.write_bytes(zf.read(member))


def main() -> int:
    if not ARCHIVE_PATH.exists():
        download(DATASET_URL, ARCHIVE_PATH)
    extract_member(ARCHIVE_PATH, MEMBER_NAME, DATA_PATH)
    for path in (ARCHIVE_PATH, DATA_PATH):
        print(f"{sha256_of(path)}  {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

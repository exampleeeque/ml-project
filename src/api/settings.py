"""Настройки сервиса предсказаний (ЛР4).

Значения по умолчанию — рабочие пределы, обоснованные в README (раздел «Лимиты»).
Любое значение можно переопределить переменной окружения при запуске; так
задаются малые пороги для демонстрации, не меняя код и образ.

Секреты (API-токен и токен сборщика метрик) в коде и образе не хранятся:
они читаются при старте из файлов, путь к которым передаётся переменными
API_TOKEN_FILE и METRICS_TOKEN_FILE (в контейнере — смонтированный только
для чтения каталог /run/secrets).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from src import config

MIN_TOKEN_LENGTH = 32      # не короче 32 символов: 128 бит и более при hex-кодировании


@dataclass(frozen=True)
class Settings:
    model_path: Path = config.MODEL_PATH
    model_reference: Path = config.MODEL_REFERENCE      # эталон SHA-256, хранится отдельно от модели
    static_dir: Path = config.ROOT / "static"

    # Секреты: либо файл (рабочий режим), либо значение (только тесты)
    api_token_file: Path | None = None
    metrics_token_file: Path | None = None
    api_token: str | None = None
    metrics_token: str | None = None

    # Пределы входа (пояснения — README, раздел «Лимиты»)
    max_body_bytes: int = 512 * 1024                    # тело запроса, читается потоково
    max_batch: int = 32                                 # сообщений в одном запросе
    max_chars: int = config.MAX_CHARS                   # 1000 символов на сообщение (ЛР2)
    max_text_bytes: int = config.MAX_BYTES              # 4000 байт UTF-8 на сообщение (ЛР2)

    # Ограничение частоты: rate_limit_requests запросов за rate_limit_window_seconds на ключ
    rate_limit_requests: int = 60
    rate_limit_window_seconds: float = 60.0

    # Параллелизм и время выполнения
    max_concurrency: int = 4                            # одновременных вычислений модели
    inference_timeout_seconds: float = 2.0

    # Доля известных модели слов, ниже которой ответ — undetermined (мера из ЛР2, раздел 4.4)
    min_known_token_share: float = 0.1

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if env is None else env

        def path(name: str, default: Path | None) -> Path | None:
            value = env.get(name)
            if not value:
                return default
            p = Path(value)
            return p if p.is_absolute() else config.ROOT / p

        def number(name: str, default, kind):
            value = env.get(name)
            result = default if value in (None, "") else kind(value)
            if result <= 0:
                raise ValueError(f"{name} должно быть положительным, получено {result}")
            return result

        defaults = cls()
        return cls(
            model_path=path("MODEL_PATH", defaults.model_path),
            model_reference=path("MODEL_REFERENCE", defaults.model_reference),
            api_token_file=path("API_TOKEN_FILE", None),
            metrics_token_file=path("METRICS_TOKEN_FILE", None),
            max_body_bytes=number("MAX_BODY_BYTES", defaults.max_body_bytes, int),
            max_batch=number("MAX_BATCH", defaults.max_batch, int),
            rate_limit_requests=number("RATE_LIMIT_REQUESTS", defaults.rate_limit_requests, int),
            rate_limit_window_seconds=number("RATE_LIMIT_WINDOW_SECONDS", defaults.rate_limit_window_seconds, float),
            max_concurrency=number("MAX_CONCURRENCY", defaults.max_concurrency, int),
            inference_timeout_seconds=number("INFERENCE_TIMEOUT_SECONDS", defaults.inference_timeout_seconds, float),
        )


def read_secret(value: str | None, file: Path | None) -> str | None:
    """Секрет из значения или файла; пробельные символы по краям удаляются."""
    if value is not None:
        return value.strip()
    if file is None or not file.is_file():
        return None
    return file.read_text(encoding="utf-8").strip()

"""Состояние сервиса и вычисление предсказаний (ЛР4, разделы 1 и 3).

Модель загружается один раз при старте функцией load_model() из ЛР2:
SHA-256 файла сверяется с эталоном integrity/MODEL_SHA256SUMS (хранится
отдельно от каталога models/), затем проверяется список типов skops.
При любой ошибке сервис остаётся неготовым: /health -> 503, /predict -> 503.

Предсказание выполняет та же функция predict() из ЛР2 (валидация ->
preprocess() -> predict_proba), что и при оценке модели в ЛР2–ЛР3.

Параллелизм и время выполнения:
  * не более max_concurrency вычислений одновременно (семафор); при занятых
    слотах новый запрос сразу получает 503 busy, а не встаёт в очередь;
  * клиент получает 504 через inference_timeout_seconds, но поток Python
    принудительно остановить нельзя: вычисление продолжается, и слот
    освобождается только по его фактическому завершению (done-callback).
    Поэтому зависшие вычисления не накапливаются сверх max_concurrency, а
    длительность одного вычисления ограничена пределами входа (пакет до
    max_batch сообщений по 1000 символов).
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from src.api.observability import Metrics, log_event
from src.api.security import ROLE_METRICS, ROLE_PREDICT, RateLimiter, TokenAuth
from src.api.settings import MIN_TOKEN_LENGTH, Settings, read_secret
from src.predict import Limits, load_model, predict, preprocess, verify_model_file

MODEL_NAME = "spam-tfidf-logreg"


class ServiceBusy(RuntimeError):
    """Все слоты вычислений заняты."""


class InferenceTimeout(RuntimeError):
    """Вычисление не уложилось во время ожидания клиента."""


class ModelService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.metrics = Metrics()
        self.limits = Limits(max_batch=settings.max_batch, max_bytes=settings.max_text_bytes,
                             max_chars=settings.max_chars)
        self.model = None
        self.model_version: str | None = None
        self.analyzer = None
        self.vocabulary: dict = {}
        self.auth: TokenAuth | None = None
        self.limiter = RateLimiter(settings.rate_limit_requests, settings.rate_limit_window_seconds)
        self.ready = False
        self._slots: asyncio.Semaphore | None = None
        self._executor: ThreadPoolExecutor | None = None

    # ------------------------------------------------------------ старт и остановка
    def startup(self) -> None:
        self._slots = asyncio.Semaphore(self.settings.max_concurrency)
        self._executor = ThreadPoolExecutor(max_workers=self.settings.max_concurrency,
                                            thread_name_prefix="inference")
        auth_ok = self._load_auth()
        model_ok = self._load_model()
        self.ready = auth_ok and model_ok
        self.metrics.model_ready.set(1 if self.ready else 0)
        log_event("startup", ready=self.ready, model_version=self.model_version,
                  max_body_bytes=self.settings.max_body_bytes, max_batch=self.settings.max_batch,
                  rate_limit=f"{self.settings.rate_limit_requests}/{self.settings.rate_limit_window_seconds:g}s",
                  max_concurrency=self.settings.max_concurrency,
                  inference_timeout_s=self.settings.inference_timeout_seconds)

    def shutdown(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)

    def _load_auth(self) -> bool:
        s = self.settings
        api = read_secret(s.api_token, s.api_token_file)
        metrics = read_secret(s.metrics_token, s.metrics_token_file)
        problems = []
        if not api or len(api) < MIN_TOKEN_LENGTH:
            problems.append("api_token")
        if not metrics or len(metrics) < MIN_TOKEN_LENGTH:
            problems.append("metrics_token")
        if api and api == metrics:
            problems.append("tokens_equal")
        if problems:
            log_event("auth_config_invalid", logging.ERROR, problems=problems)   # значения токенов не логируются
            return False
        self.auth = TokenAuth({ROLE_PREDICT: api, ROLE_METRICS: metrics})
        return True

    def _load_model(self) -> bool:
        s = self.settings
        try:
            sha = verify_model_file(s.model_path, s.model_reference)    # хэш ДО десериализации
            model = load_model(s.model_path, s.model_reference)          # повторная проверка + типы skops
            vectorizer = model.named_steps["tfidf"]
            self.analyzer, self.vocabulary = vectorizer.build_analyzer(), vectorizer.vocabulary_
        except Exception as exc:  # любая ошибка загрузки = сервис не готов
            log_event("model_load_failed", logging.ERROR, error_type=type(exc).__name__, detail=str(exc)[:300])
            return False
        self.model, self.model_version = model, f"{MODEL_NAME}-{sha[:12]}"
        log_event("model_loaded", model_version=self.model_version, model_sha256=sha)
        return True

    # ------------------------------------------------------------ вычисление
    async def run_inference(self, texts: list[str]) -> list[dict]:
        if self._slots.locked():
            raise ServiceBusy
        await self._slots.acquire()
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(self._executor, infer, self, texts)

        def release(f: asyncio.Future) -> None:
            self._slots.release()                       # слот освобождается по завершении вычисления
            if not f.cancelled():
                f.exception()                           # результат после 504 больше никому не нужен

        future.add_done_callback(release)
        try:
            # shield: таймаут ожидания не отменяет future, иначе слот освободился бы раньше времени
            return await asyncio.wait_for(asyncio.shield(future), self.settings.inference_timeout_seconds)
        except TimeoutError:
            raise InferenceTimeout from None


def infer(service: ModelService, texts: list[str]) -> list[dict]:
    """Выполняется в рабочем потоке. predict() из ЛР2 валидирует вход до вызова модели."""
    results = predict(list(texts), model=service.model, limits=service.limits)
    out = []
    for text, res in zip(preprocess(texts), results, strict=True):
        tokens = [t for t in service.analyzer(text) if " " not in t]      # только униграммы, как в ЛР2
        known = sum(t in service.vocabulary for t in tokens) / len(tokens) if tokens else 0.0
        if known < service.settings.min_known_token_share:
            out.append({"prediction": "undetermined", "spam_score": None, "known_token_share": round(known, 3)})
        else:
            out.append({"prediction": res["label"], "spam_score": res["spam_probability"],
                        "known_token_share": round(known, 3)})
    return out

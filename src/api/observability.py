"""Структурированные логи, метрики Prometheus и ASGI-middleware сервиса (ЛР4, разделы 3 и 7).

ObservabilityMiddleware (внешний слой): присваивает request_id, измеряет
длительность, добавляет заголовки безопасности, пишет одну JSON-строку лога
на запрос и обновляет метрики. Необработанное исключение превращается в
ответ 500 без traceback.

BodyLimitMiddleware: ограничивает тело запроса ДО его полной загрузки в
память. Объявленный Content-Length больше предела -> 413 без чтения тела;
при потоковой передаче (в том числе chunked, без Content-Length) байты
считаются по мере поступления, и чтение прекращается, как только сумма
превысила предел.

В логи не попадают тело запроса, текст сообщений, токены и адрес клиента.
Метки метрик берутся только из фиксированных наборов (шаблон маршрута,
метод, код ответа, причина отказа), поэтому число временных рядов ограничено.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from datetime import UTC, datetime

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram
from starlette.datastructures import MutableHeaders

LOGGER_NAME = "ml_api"
ROUTES = ("/predict", "/health", "/metrics", "/")
METHODS = {"GET", "POST", "HEAD", "OPTIONS", "PUT", "PATCH", "DELETE"}
REJECT_REASONS = (
    "auth_missing", "auth_invalid", "forbidden", "rate_limited", "not_ready",
    "unsupported_media_type", "body_too_large", "invalid_json", "schema_error",
    "validation_error", "busy", "timeout", "not_found", "method_not_allowed", "internal_error",
)
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
    "Content-Security-Policy": (
        "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; "
        "img-src 'self'; form-action 'none'; base-uri 'none'; frame-ancestors 'none'"
    ),
}


# ---------------------------------------------------------------- логи
class _StdoutHandler(logging.StreamHandler):
    """Пишет в текущий sys.stdout (а не в поток на момент создания)."""

    @property
    def stream(self):
        return sys.stdout

    @stream.setter
    def stream(self, value):
        pass


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname.lower(),
            "event": record.getMessage(),
        }
        payload.update(getattr(record, "fields", {}))
        return json.dumps(payload, ensure_ascii=False)


def get_logger() -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if not any(isinstance(h, _StdoutHandler) for h in logger.handlers):
        handler = _StdoutHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def log_event(event: str, level: int = logging.INFO, **fields) -> None:
    get_logger().log(level, event, extra={"fields": fields})


# ---------------------------------------------------------------- метрики
class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter("ml_api_requests", "HTTP-запросы по маршруту, методу и коду ответа",
                                ["route", "method", "status"], registry=self.registry)
        self.duration = Histogram("ml_api_request_duration_seconds", "Время ответа, с", ["route"],
                                  buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
                                  registry=self.registry)
        self.rejections = Counter("ml_api_rejections", "Отказы приложения по причине",
                                  ["reason"], registry=self.registry)
        self.predictions = Counter("ml_api_predictions", "Выданные предсказания по классу",
                                   ["prediction"], registry=self.registry)
        self.model_ready = Gauge("ml_api_model_ready", "1 — модель загружена и сервис готов",
                                 registry=self.registry)
        for reason in REJECT_REASONS:           # нулевые значения видны до первого отказа
            self.rejections.labels(reason=reason)


def route_label(path: str) -> str:
    if path in ROUTES:
        return path
    if path.startswith("/static/"):
        return "/static"
    return "other"


# ---------------------------------------------------------------- ответы
def error_payload(code: str, message: str, request_id: str) -> dict:
    return {"error": {"code": code, "message": message}, "request_id": request_id}


async def send_json(send, status: int, payload: dict, headers: list[tuple[bytes, bytes]] | None = None) -> None:
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    raw = [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]
    await send({"type": "http.response.start", "status": status, "headers": raw + (headers or [])})
    await send({"type": "http.response.body", "body": body})


# ---------------------------------------------------------------- middleware
class ObservabilityMiddleware:
    def __init__(self, app, service):
        self.app, self.service = app, service

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_id = uuid.uuid4().hex
        state = scope.setdefault("state", {})
        state["request_id"] = request_id
        status = {"code": 500, "started": False}
        t0 = time.perf_counter()

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status["code"], status["started"] = message["status"], True
                headers = MutableHeaders(scope=message)
                headers["X-Request-ID"] = request_id
                for name, value in SECURITY_HEADERS.items():
                    headers[name] = value
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as exc:  # ответ 500 без traceback; тип ошибки — только в лог
            state["reject_reason"] = "internal_error"
            state["error_code"] = type(exc).__name__
            if not status["started"]:
                await send_json(send_wrapper, 500, error_payload(
                    "internal_error", "Внутренняя ошибка сервиса", request_id))
        finally:
            self._observe(scope, state, status["code"], time.perf_counter() - t0)

    def _observe(self, scope, state: dict, code: int, duration: float) -> None:
        route = route_label(scope["path"])
        method = scope["method"] if scope["method"] in METHODS else "OTHER"
        reason = state.get("reject_reason")
        m = self.service.metrics
        m.requests.labels(route=route, method=method, status=str(code)).inc()
        m.duration.labels(route=route).observe(duration)
        if reason:
            m.rejections.labels(reason=reason if reason in REJECT_REASONS else "internal_error").inc()
        outcome = reason or state.get("outcome") or ("ok" if code < 400 else "http_error")
        fields = {
            "request_id": state["request_id"], "method": method, "route": route, "status": code,
            "duration_ms": round(duration * 1000, 2), "model_version": self.service.model_version,
            "outcome": outcome,
        }
        if state.get("error_code"):
            fields["error_code"] = state["error_code"]
        if state.get("items"):
            fields["items"] = state["items"]
        log_event("request", logging.WARNING if code >= 400 else logging.INFO, **fields)


class BodyLimitMiddleware:
    def __init__(self, app, max_body_bytes: int):
        self.app, self.limit = app, max_body_bytes

    async def _reject(self, scope, send, status: int, code: str, message: str) -> None:
        state = scope.setdefault("state", {})
        state["reject_reason"] = "body_too_large" if status == 413 else "invalid_json"
        state["error_code"] = code
        await send_json(send, status, error_payload(code, message, state.get("request_id", "")),
                        headers=[(b"connection", b"close")])

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        too_large = f"Тело запроса больше {self.limit} байт"
        declared = dict(scope["headers"]).get(b"content-length")
        if declared is not None:
            try:
                size = int(declared)
            except ValueError:
                await self._reject(scope, send, 400, "bad_content_length", "Некорректный Content-Length")
                return
            if size > self.limit:                       # отказ без чтения тела
                await self._reject(scope, send, 413, "body_too_large", too_large)
                return

        chunks, total = [], 0
        while True:                                     # потоковое чтение с подсчётом байтов
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            total += len(chunk)
            if total > self.limit:                      # Content-Length отсутствует или неверен
                await self._reject(scope, send, 413, "body_too_large", too_large)
                return
            chunks.append(chunk)
            if not message.get("more_body", False):
                break

        body, replayed = b"".join(chunks), False

        async def replay():
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()                      # далее — только http.disconnect

        await self.app(scope, replay, send)

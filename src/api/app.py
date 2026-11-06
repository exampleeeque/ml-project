"""Защищённый API модели классификации SMS (ЛР4).

Эндпоинты:
  POST /predict  — предсказание; Bearer-токен роли predict; JSON {"text": str} или {"texts": [str, ...]}
  GET  /health   — 200 healthy только при загруженной модели и настроенной аутентификации, иначе 503
  GET  /metrics  — метрики Prometheus; Bearer-токен роли metrics
  GET  /         — демонстрационная страница (static/index.html), статика — /static/*

Порядок проверок /predict: размер тела (middleware, 413) -> токен (401/403)
-> частота (429) -> готовность (503) -> Content-Type (415) -> разбор JSON (400)
-> схема (422) -> политика текста ЛР2 (422) -> вычисление (503 busy, 504 timeout).

Запуск из корня проекта (секреты — файлы вне Git, см. README):
    API_TOKEN_FILE=.secrets/api_token METRICS_TOKEN_FILE=.secrets/metrics_token \\
    python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --workers 1 \\
        --no-access-log --no-proxy-headers --no-server-header
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api import inference
from src.api.observability import BodyLimitMiddleware, ObservabilityMiddleware, error_payload
from src.api.security import ROLE_METRICS, ROLE_PREDICT
from src.api.settings import Settings
from src.predict import InputValidationError


class SchemaError(ValueError):
    """Корректный JSON, не соответствующий схеме запроса."""


# ---------------------------------------------------------------- разбор и схема
def _reject_constant(name: str):
    raise ValueError(f"недопустимое значение {name}")            # NaN, Infinity, -Infinity


def _unique_keys(pairs: list[tuple]) -> dict:
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError("повторяющийся ключ")
        obj[key] = value
    return obj


def parse_json(body: bytes):
    """Строгий разбор: UTF-8, без NaN/Infinity и повторяющихся ключей."""
    return json.loads(body.decode("utf-8"), parse_constant=_reject_constant, object_pairs_hook=_unique_keys)


def parse_schema(data, max_batch: int) -> tuple[list[str], bool]:
    """Схема: ровно одно из полей text (строка) или texts (непустой список строк <= max_batch)."""
    if not isinstance(data, dict):
        raise SchemaError("ожидался JSON-объект")
    keys = set(data)
    unknown = keys - {"text", "texts"}
    if unknown:
        raise SchemaError(f"недопустимые поля: {sorted(unknown)[:5]}")
    if len(keys) != 1:
        raise SchemaError("требуется ровно одно поле: text или texts")
    if "text" in data:
        if not isinstance(data["text"], str):
            raise SchemaError("поле text должно быть строкой")
        return [data["text"]], True
    texts = data["texts"]
    if not isinstance(texts, list):
        raise SchemaError("поле texts должно быть списком строк")
    if not texts:
        raise SchemaError("список texts пуст")
    if len(texts) > max_batch:
        raise SchemaError(f"в texts {len(texts)} элементов, предел {max_batch}")
    if not all(isinstance(t, str) for t in texts):
        raise SchemaError("все элементы texts должны быть строками")
    return texts, False


def _is_json(content_type: str | None) -> bool:
    if not content_type:
        return False
    media, *params = [p.strip().lower() for p in content_type.split(";")]
    charset = next((p.split("=", 1)[1].strip('"') for p in params if p.startswith("charset=")), "utf-8")
    return media == "application/json" and charset == "utf-8"


# ---------------------------------------------------------------- приложение
def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    service = inference.ModelService(settings)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        service.startup()                    # модель загружается один раз при старте
        yield
        service.shutdown()

    app = FastAPI(title="ml-project API", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.service = service
    app.add_middleware(BodyLimitMiddleware, max_body_bytes=settings.max_body_bytes)
    app.add_middleware(ObservabilityMiddleware, service=service)      # добавлен последним = внешний слой

    def reject(request: Request, status: int, reason: str, code: str, message: str,
               headers: dict | None = None) -> JSONResponse:
        request.state.reject_reason, request.state.error_code = reason, code
        return JSONResponse(error_payload(code, message, request.state.request_id),
                            status_code=status, headers=headers)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        reason = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "internal_error")
        message = {404: "Ресурс не найден", 405: "Метод не поддерживается"}.get(exc.status_code, "Ошибка запроса")
        return reject(request, exc.status_code, reason, reason, message)

    @app.get("/health")
    async def health(request: Request):
        if service.ready:
            return {"status": "healthy", "model_version": service.model_version}
        request.state.outcome = "not_ready"
        return JSONResponse({"status": "unavailable"}, status_code=503)   # без путей и подробностей

    @app.post("/predict")
    async def predict_endpoint(request: Request):
        if service.auth is None:
            return reject(request, 503, "not_ready", "not_ready", "Сервис не готов")
        auth = service.auth.check(request.headers.get("authorization"), ROLE_PREDICT)
        if auth.status in ("missing", "invalid"):
            # Неудачные попытки ограничиваются по адресу клиента (forwarded-заголовки не используются)
            client = request.client.host if request.client else "unknown"
            allowed, retry = service.limiter.hit(f"addr:{client}")
            if not allowed:
                return reject(request, 429, "rate_limited", "rate_limited", "Превышен лимит запросов",
                              {"Retry-After": str(retry)})
            return reject(request, 401, f"auth_{auth.status}", f"{auth.status}_token",
                          "Требуется действительный API-токен", {"WWW-Authenticate": 'Bearer realm="ml-api"'})
        if auth.status == "forbidden":
            return reject(request, 403, "forbidden", "forbidden", "Недостаточно прав для этой операции")
        allowed, retry = service.limiter.hit(f"token:{auth.fingerprint}")
        if not allowed:
            return reject(request, 429, "rate_limited", "rate_limited", "Превышен лимит запросов",
                          {"Retry-After": str(retry)})
        if not service.ready:
            return reject(request, 503, "not_ready", "not_ready", "Сервис не готов")
        if not _is_json(request.headers.get("content-type")):
            return reject(request, 415, "unsupported_media_type", "unsupported_media_type",
                          "Поддерживается только application/json в UTF-8")

        body = await request.body()          # тело уже ограничено BodyLimitMiddleware
        try:
            data = parse_json(body)
        except (ValueError, RecursionError):  # UnicodeDecodeError и JSONDecodeError — подклассы ValueError
            return reject(request, 400, "invalid_json", "invalid_json", "Тело запроса не является корректным JSON")
        try:
            texts, single = parse_schema(data, settings.max_batch)
        except SchemaError as e:
            return reject(request, 422, "schema_error", "schema_error", f"Нарушена схема запроса: {e}")

        request.state.items = len(texts)
        try:
            items = await service.run_inference(texts)
        except InputValidationError as e:     # политика текста ЛР2; модель не вызывалась
            return reject(request, 422, "validation_error", e.code, str(e))
        except inference.ServiceBusy:
            return reject(request, 503, "busy", "busy", "Сервис занят, повторите запрос",
                          {"Retry-After": "1"})
        except inference.InferenceTimeout:
            return reject(request, 504, "timeout", "timeout", "Превышено время обработки")

        for item in items:
            service.metrics.predictions.labels(prediction=item["prediction"]).inc()
        meta = {"model_version": service.model_version, "request_id": request.state.request_id}
        return {**items[0], **meta} if single else {"predictions": items, **meta}

    @app.get("/metrics")
    async def metrics(request: Request):
        if service.auth is None:
            return reject(request, 503, "not_ready", "not_ready", "Сервис не готов")
        auth = service.auth.check(request.headers.get("authorization"), ROLE_METRICS)
        if auth.status in ("missing", "invalid"):
            return reject(request, 401, f"auth_{auth.status}", f"{auth.status}_token",
                          "Требуется токен сборщика метрик", {"WWW-Authenticate": 'Bearer realm="ml-api-metrics"'})
        if auth.status == "forbidden":
            return reject(request, 403, "forbidden", "forbidden", "Недостаточно прав для этой операции")
        return Response(generate_latest(service.metrics.registry), media_type=CONTENT_TYPE_LATEST)

    @app.get("/")
    async def index():
        return FileResponse(settings.static_dir / "index.html", media_type="text/html")

    app.mount("/static", StaticFiles(directory=settings.static_dir), name="static")
    return app


app = create_app()

"""Положительные и негативные тесты API (ЛР4, раздел 6).

Используется клиент приложения (fastapi.testclient) и небольшая модель той
же конструкции, что и рабочая (src.train.build_pipeline), сохранённая в
skops с эталоном SHA-256 во временном каталоге. Токены генерируются при
каждом запуске. После старта модель сервиса заменяется обёрткой Mock(wraps=...),
поэтому каждый негативный тест проверяет, что predict_proba не вызывался.
"""

import json
import secrets
import time
from unittest.mock import Mock

import pytest
import skops.io as sio
from fastapi.testclient import TestClient

from src.api import inference
from src.api.app import create_app
from src.api.settings import Settings
from src.data import sha256_of
from src.train import build_pipeline

SPAM = ["win a free prize now", "free prize claim now call", "urgent free prize call now",
        "claim your free cash prize", "you won a free prize call"]
HAM = ["see you at lunch tomorrow", "are you at home now", "lunch at home tomorrow",
       "see you at home tonight", "are you free for lunch tomorrow"]
API_TOKEN = secrets.token_hex(32)
METRICS_TOKEN = secrets.token_hex(32)
AUTH = {"Authorization": f"Bearer {API_TOKEN}"}


@pytest.fixture(scope="module")
def model_files(tmp_path_factory):
    d = tmp_path_factory.mktemp("model")
    path = d / "model.skops"
    sio.dump(build_pipeline().fit(SPAM + HAM, [1] * 5 + [0] * 5), path)
    ref = d / "MODEL_SHA256SUMS"
    ref.write_text(f"{sha256_of(path)}  {path.resolve().as_posix()}\n", encoding="utf-8")
    return path, ref


def make_settings(model_files, **over) -> Settings:
    path, ref = model_files
    values = {"model_path": path, "model_reference": ref, "api_token": API_TOKEN,
              "metrics_token": METRICS_TOKEN, "rate_limit_requests": 1000}
    values.update(over)
    return Settings(**values)


@pytest.fixture
def make_client(model_files):
    clients = []

    def factory(**over):
        client = TestClient(create_app(make_settings(model_files, **over)))
        client.__enter__()                                   # lifespan: загрузка модели
        service = client.app.state.service
        if service.model is not None:
            service.model = Mock(wraps=service.model)        # шпион: был ли вызов модели
        clients.append(client)
        return client, service

    yield factory
    for c in clients:
        c.__exit__(None, None, None)


def model_called(service) -> bool:
    return service.model.predict_proba.called


def post(client, payload=None, headers=AUTH, **kw):
    if payload is not None:
        kw["json"] = payload
    return client.post("/predict", headers=headers, **kw)


# ================================================================ положительные
def test_health_ready(make_client):
    client, service = make_client()
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "healthy", "model_version": service.model_version}
    assert service.model_version.startswith("spam-tfidf-logreg-")


def test_predict_single_ok(make_client):
    client, service = make_client()
    r = post(client, {"text": "claim your free prize now"})
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] == "spam"
    assert 0.5 <= body["spam_score"] <= 1.0
    assert body["model_version"] == service.model_version
    assert body["request_id"] == r.headers["X-Request-ID"]
    assert model_called(service)


def test_predict_batch_ok(make_client):
    client, _ = make_client()
    r = post(client, {"texts": ["see you at lunch tomorrow", "win a free prize now"]})
    assert r.status_code == 200
    assert [p["prediction"] for p in r.json()["predictions"]] == ["ham", "spam"]


def test_unknown_vocabulary_is_undetermined(make_client):
    client, _ = make_client()
    r = post(client, {"text": "Поздравляем! Вы выиграли приз"})
    assert r.status_code == 200
    assert r.json()["prediction"] == "undetermined" and r.json()["spam_score"] is None


def test_page_and_static_served_with_security_headers(make_client):
    client, _ = make_client()
    page = client.get("/")
    assert page.status_code == 200 and "text/html" in page.headers["content-type"]
    assert 'id="predict-form"' in page.text
    assert "default-src 'none'" in page.headers["Content-Security-Policy"]
    js = client.get("/static/app.js")
    assert js.status_code == 200 and ".innerHTML" not in js.text and "localStorage." not in js.text


# ================================================================ 1. токен
def test_missing_token_401(make_client):
    client, service = make_client()
    r = post(client, {"text": "free prize"}, headers={})
    assert r.status_code == 401 and r.json()["error"]["code"] == "missing_token"
    assert r.headers["WWW-Authenticate"].startswith("Bearer")
    assert not model_called(service)


@pytest.mark.parametrize("header", ["Bearer wrong-token", "Basic abc", "Bearer", API_TOKEN],
                         ids=["wrong_token", "basic_scheme", "empty_bearer", "no_scheme"])   # без токена в id
def test_invalid_token_401(make_client, header):
    client, service = make_client()
    r = post(client, {"text": "free prize"}, headers={"Authorization": header})
    assert r.status_code == 401
    assert not model_called(service)


def test_roles_403_and_metrics_access(make_client):
    client, service = make_client()
    metrics_auth = {"Authorization": f"Bearer {METRICS_TOKEN}"}
    assert post(client, {"text": "free prize"}, headers=metrics_auth).status_code == 403
    assert client.get("/metrics", headers=AUTH).status_code == 403
    assert client.get("/metrics").status_code == 401
    r = client.get("/metrics", headers=metrics_auth)
    assert r.status_code == 200 and "ml_api_requests_total" in r.text
    assert not model_called(service)


# ================================================================ 2. пустой вход и схема
@pytest.mark.parametrize("payload", [
    {}, [], "text", {"text": None}, {"text": 123}, {"text": ["a"]}, {"texts": []}, {"texts": "a"},
    {"texts": ["ok", 1]}, {"text": "a", "texts": ["b"]}, {"text": "a", "user": "x"},
], ids=["empty_object", "array", "string", "text_null", "text_number", "text_list", "texts_empty", "texts_string",
        "texts_number_item", "both_fields", "extra_field"])
def test_schema_violation_422(make_client, payload):
    client, service = make_client()
    r = post(client, payload)
    assert r.status_code == 422 and r.json()["error"]["code"] == "schema_error"
    assert not model_called(service)


@pytest.mark.parametrize("text, code", [("", "empty"), ("   \n\t ", "empty"), ("a" * 1001, "too_long"),
                                        ("bad\x1b[31m", "bad_char"), ("it\x92s", "bad_char")],
                         ids=["empty", "whitespace", "too_long", "esc_sequence", "c1_artifact"])
def test_text_policy_lr2_422(make_client, text, code):
    client, service = make_client()
    r = post(client, {"text": text})
    assert r.status_code == 422 and r.json()["error"]["code"] == code
    assert not model_called(service)


def test_batch_too_large_422(make_client):
    client, service = make_client(max_batch=3)
    assert post(client, {"texts": ["ok"] * 4}).status_code == 422
    assert not model_called(service)


# ================================================================ 3. неверный тип / повреждённый вход
@pytest.mark.parametrize("raw", [b"{bad json", b'{"text": NaN}', b'{"text": Infinity}', b'{"text": -Infinity}',
                                 b'{"text": "a", "text": "b"}', b'{"text": "\xff\xfe"}', b"[" * 100_000],
                         ids=["syntax", "nan", "infinity", "minus_infinity", "duplicate_key", "bad_utf8",
                              "deep_nesting"])
def test_malformed_json_400(make_client, raw):
    client, service = make_client()
    r = post(client, content=raw, headers={**AUTH, "Content-Type": "application/json"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_json"
    assert "Traceback" not in r.text
    assert not model_called(service)


@pytest.mark.parametrize("ctype", ["text/plain", "multipart/form-data; boundary=x",
                                   "application/x-www-form-urlencoded", "application/json; charset=latin-1"])
def test_unsupported_media_type_415(make_client, ctype):
    client, service = make_client()
    r = post(client, content=b'{"text": "free prize"}', headers={**AUTH, "Content-Type": ctype})
    assert r.status_code == 415
    assert not model_called(service)


# ================================================================ 4. размер тела
def test_body_too_large_by_content_length_413(make_client):
    client, service = make_client(max_body_bytes=1024)
    r = post(client, {"text": "a" * 2000})
    assert r.status_code == 413 and r.json()["error"]["code"] == "body_too_large"
    assert not model_called(service)


def test_body_too_large_streamed_without_content_length_413(make_client):
    client, service = make_client(max_body_bytes=1024)

    def chunks():                        # chunked transfer: заголовка Content-Length нет
        yield b'{"text": "'
        for _ in range(10):
            yield b"a" * 200
        yield b'"}'

    r = post(client, content=chunks(), headers={**AUTH, "Content-Type": "application/json"})
    assert r.status_code == 413
    assert not model_called(service)


# ================================================================ 5. частота запросов
def test_rate_limit_429_for_valid_token(make_client):
    client, service = make_client(rate_limit_requests=3, rate_limit_window_seconds=60)
    codes = [post(client, {"text": "free prize"}).status_code for _ in range(4)]
    assert codes == [200, 200, 200, 429]
    r = post(client, {"text": "free prize"})
    assert r.status_code == 429 and int(r.headers["Retry-After"]) >= 1
    assert service.model.predict_proba.call_count == 3


def test_invalid_token_flood_limited_but_valid_client_works(make_client):
    client, _ = make_client(rate_limit_requests=3, rate_limit_window_seconds=60)
    bad = {"Authorization": "Bearer " + "x" * 64}
    codes = [post(client, {"text": "hi"}, headers=bad).status_code for _ in range(5)]
    assert codes == [401, 401, 401, 429, 429]
    assert post(client, {"text": "free prize"}).status_code == 200       # другой ключ лимитера


# ================================================================ целостность модели и готовность
def test_hash_mismatch_not_ready(make_client, model_files, tmp_path):
    path, _ = model_files
    bad_ref = tmp_path / "MODEL_SHA256SUMS"
    bad_ref.write_text(f"{'0' * 64}  {path.resolve().as_posix()}\n", encoding="utf-8")
    client, service = make_client(model_reference=bad_ref)
    r = client.get("/health")
    assert r.status_code == 503 and r.json() == {"status": "unavailable"}
    assert service.model is None
    p = post(client, {"text": "free prize"})
    assert p.status_code == 503 and p.json()["error"]["code"] == "not_ready"


@pytest.mark.parametrize("missing", ["model", "reference"])
def test_missing_model_or_reference_not_ready(make_client, tmp_path, missing):
    over = {"model_path": tmp_path / "nope.skops"} if missing == "model" else {"model_reference": tmp_path / "nope"}
    client, _ = make_client(**over)
    assert client.get("/health").status_code == 503


def test_tampered_model_file_not_ready(make_client, model_files, tmp_path):
    path, ref = model_files
    copy = tmp_path / "model.skops"
    copy.write_bytes(path.read_bytes() + b"\x00")
    ref2 = tmp_path / "REF"
    ref2.write_text(ref.read_text().split()[0] + f"  {copy.resolve().as_posix()}\n", encoding="utf-8")
    client, _ = make_client(model_path=copy, model_reference=ref2)
    assert client.get("/health").status_code == 503


@pytest.mark.parametrize("over", [{"api_token": None}, {"api_token": "short"}, {"metrics_token": API_TOKEN}],
                         ids=["no_api_token", "short_token", "equal_tokens"])
def test_auth_misconfiguration_not_ready(make_client, over):
    client, _ = make_client(**over)
    assert client.get("/health").status_code == 503


# ================================================================ параллелизм и время выполнения
def test_timeout_504_and_slot_released_after_completion(make_client, monkeypatch):
    client, _ = make_client(max_concurrency=1, inference_timeout_seconds=0.1)
    original = inference.infer

    def slow(service, texts):
        time.sleep(0.6)
        return original(service, texts)

    monkeypatch.setattr(inference, "infer", slow)
    assert post(client, {"text": "free prize"}).status_code == 504
    busy = post(client, {"text": "free prize"})              # вычисление ещё идёт, слот занят
    assert busy.status_code == 503 and busy.json()["error"]["code"] == "busy"
    monkeypatch.setattr(inference, "infer", original)
    time.sleep(0.8)                                          # вычисление завершилось, слот свободен
    assert post(client, {"text": "free prize"}).status_code == 200


# ================================================================ ошибки, логи, метрики
def test_internal_error_without_traceback(make_client, monkeypatch):
    client, _ = make_client()

    def boom(service, texts):
        raise RuntimeError("секретная подробность /app/models")

    monkeypatch.setattr(inference, "infer", boom)
    r = post(client, {"text": "free prize"})
    assert r.status_code == 500 and r.json()["error"]["code"] == "internal_error"
    assert "секретная" not in r.text and "Traceback" not in r.text


def test_unknown_route_404_json(make_client):
    client, _ = make_client()
    r = client.get("/admin/../../etc/passwd")
    assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"
    assert client.get("/predict").status_code == 405


def test_logs_are_structured_and_contain_no_secrets(make_client, capsys):
    client, service = make_client()
    private_text = "my private message 12345"
    post(client, {"text": private_text})
    post(client, {"text": "x"}, headers={"Authorization": "Bearer leaked-wrong-token-value"})
    out = capsys.readouterr().out
    records = [json.loads(line) for line in out.splitlines() if line.startswith("{")]
    requests = [r for r in records if r["event"] == "request"]
    assert [r["status"] for r in requests][-2:] == [200, 401]
    for r in requests:
        assert {"ts", "request_id", "route", "status", "duration_ms", "model_version", "outcome"} <= set(r)
    assert requests[-1]["outcome"] == "auth_invalid"
    assert API_TOKEN not in out and "leaked-wrong-token-value" not in out and private_text not in out


def test_metrics_counters_and_bounded_labels(make_client):
    client, _ = make_client()
    metrics_auth = {"Authorization": f"Bearer {METRICS_TOKEN}"}
    rid = post(client, {"text": "free prize"}).json()["request_id"]
    post(client, {"text": ""})
    post(client, {"text": "free prize"}, headers={})
    client.get("/static/does-not-exist-12345.js")
    text = client.get("/metrics", headers=metrics_auth).text
    assert 'ml_api_rejections_total{reason="validation_error"} 1.0' in text
    assert 'ml_api_rejections_total{reason="auth_missing"} 1.0' in text
    assert 'ml_api_requests_total{method="POST",route="/predict",status="200"} 1.0' in text
    assert "ml_api_request_duration_seconds_bucket" in text
    assert rid not in text and "free prize" not in text and "does-not-exist" not in text

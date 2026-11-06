"""Модульные тесты токенов, лимитера и правила обнаружения (ЛР4)."""

import json
import secrets
from datetime import UTC, datetime, timedelta

from src.api.security import ROLE_METRICS, ROLE_PREDICT, RateLimiter, TokenAuth
from src.detect import evaluate, read_events

PREDICT_TOKEN = secrets.token_hex(32)
METRICS_TOKEN = secrets.token_hex(32)


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_token_auth_statuses():
    auth = TokenAuth({ROLE_PREDICT: PREDICT_TOKEN, ROLE_METRICS: METRICS_TOKEN})
    assert auth.check(None, ROLE_PREDICT).status == "missing"
    assert auth.check("Bearer nope", ROLE_PREDICT).status == "invalid"
    assert auth.check(f"Token {PREDICT_TOKEN}", ROLE_PREDICT).status == "invalid"
    assert auth.check(f"Bearer {METRICS_TOKEN}", ROLE_PREDICT).status == "forbidden"
    ok = auth.check(f"bearer {PREDICT_TOKEN}", ROLE_PREDICT)
    assert ok.status == "ok" and PREDICT_TOKEN not in ok.fingerprint and len(ok.fingerprint) == 16


def test_rate_limiter_window_and_retry_after():
    clock = FakeClock()
    lim = RateLimiter(limit=2, window=10, clock=clock)
    assert lim.hit("a") == (True, 0)
    clock.now += 4
    assert lim.hit("a") == (True, 0)
    assert lim.hit("a") == (False, 6)            # первое попадание выйдет из окна через 6 с
    assert lim.hit("b") == (True, 0)             # другой ключ независим
    clock.now += 6
    assert lim.hit("a")[0] is True


def test_rejected_hits_do_not_extend_block():
    clock = FakeClock()
    lim = RateLimiter(limit=1, window=10, clock=clock)
    lim.hit("a")
    for _ in range(5):
        clock.now += 1
        assert lim.hit("a")[0] is False
    clock.now += 5
    assert lim.hit("a")[0] is True


def test_rate_limiter_bounded_number_of_keys():
    clock = FakeClock()
    lim = RateLimiter(limit=1, window=10, max_keys=3, clock=clock)
    for k in "abcdef":
        lim.hit(k)
    assert len(lim._hits) <= 3


def _log(ts: datetime, status: int, route: str = "/predict") -> str:
    return json.dumps({"ts": ts.isoformat(), "event": "request", "route": route, "status": status})


def test_detection_rule_fires_and_resolves():
    t0 = datetime(2026, 11, 5, 12, 0, tzinfo=UTC)
    lines = [_log(t0 + timedelta(seconds=i), 401 if i < 5 else 429) for i in range(12)]
    lines += [_log(t0, 200), _log(t0, 401, "/metrics"), "INFO: uvicorn text line"]
    times = read_events(lines, "/predict", {401, 429})
    assert len(times) == 12
    tr = evaluate(times, window=60, threshold=10, now=t0 + timedelta(minutes=5))
    assert [x["state"] for x in tr] == ["FIRING", "RESOLVED"]
    assert tr[0]["at"] == t0 + timedelta(seconds=9)             # 10-й ответ
    assert tr[1]["at"] == t0 + timedelta(seconds=62)            # в окне осталось 9


def test_detection_rule_below_threshold_and_still_firing():
    t0 = datetime(2026, 11, 5, 12, 0, tzinfo=UTC)
    few = [t0 + timedelta(seconds=i) for i in range(9)]
    assert evaluate(few, 60, 10, t0 + timedelta(minutes=5)) == []
    many = [t0 + timedelta(seconds=i) for i in range(10)]
    assert [x["state"] for x in evaluate(many, 60, 10, t0 + timedelta(seconds=30))] == ["FIRING"]

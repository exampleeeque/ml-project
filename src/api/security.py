"""Аутентификация по API-токену и ограничение частоты запросов (ЛР4, раздел 4).

Роли: токен predict даёт доступ только к /predict, токен metrics — только к
/metrics. Отсутствующий или неверный токен -> 401, токен другой роли -> 403.
Сравнение выполняется по SHA-256 токенов функцией hmac.compare_digest
(время сравнения не зависит от совпадающего префикса). Сами токены не
логируются; ключ лимитера — короткий отпечаток SHA-256, а не токен.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

ROLE_PREDICT = "predict"
ROLE_METRICS = "metrics"


def _digest(token: str) -> bytes:
    return hashlib.sha256(token.encode("utf-8")).digest()


@dataclass(frozen=True)
class AuthResult:
    status: str                  # ok | missing | invalid | forbidden
    fingerprint: str = ""        # отпечаток токена для ключа лимитера (не сам токен)


class TokenAuth:
    def __init__(self, tokens: dict[str, str]):
        """tokens: {роль: токен}. Пустые роли не допускаются (проверяется при старте)."""
        self._digests = {role: _digest(tok) for role, tok in tokens.items()}

    def check(self, authorization: str | None, required_role: str) -> AuthResult:
        if not authorization:
            return AuthResult("missing")
        scheme, _, token = authorization.partition(" ")
        token = token.strip()
        if scheme.lower() != "bearer" or not token:
            return AuthResult("invalid")
        given = _digest(token)
        role = None
        for name, expected in self._digests.items():   # перебор всех ролей без раннего выхода
            if hmac.compare_digest(given, expected):
                role = name
        if role is None:
            return AuthResult("invalid")
        if role != required_role:
            return AuthResult("forbidden")
        return AuthResult("ok", hashlib.sha256(given).hexdigest()[:16])


class RateLimiter:
    """Скользящее окно в памяти процесса: не более limit запросов за window секунд на ключ.

    Отклонённые запросы в окно не записываются, поэтому повторные попытки во
    время блокировки не продлевают её. Число ключей ограничено max_keys.
    Ограничения: счётчики не разделяются между процессами (при N workers
    фактический лимит до N раз выше) и обнуляются при перезапуске.
    Вызывается только из потока event loop, поэтому блокировки не нужны.
    """

    def __init__(self, limit: int, window: float, max_keys: int = 10_000,
                 clock: Callable[[], float] = time.monotonic):
        self.limit, self.window, self.max_keys, self._clock = limit, window, max_keys, clock
        self._hits: dict[str, deque[float]] = {}

    def hit(self, key: str) -> tuple[bool, int]:
        """Возвращает (разрешено, через сколько секунд повторить)."""
        now = self._clock()
        q = self._hits.get(key)
        if q is None:
            if len(self._hits) >= self.max_keys:
                self._evict(now)
            q = self._hits[key] = deque()
        while q and q[0] <= now - self.window:
            q.popleft()
        if len(q) >= self.limit:
            return False, max(1, math.ceil(q[0] + self.window - now))
        q.append(now)
        return True, 0

    def _evict(self, now: float) -> None:
        for key in [k for k, q in self._hits.items() if not q or q[-1] <= now - self.window]:
            del self._hits[key]
        while len(self._hits) >= self.max_keys:          # всё ещё полно — удаляем самые старые ключи
            del self._hits[next(iter(self._hits))]

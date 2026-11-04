"""Общие настройки тестов (ЛР3).

Тесты изолированы от сети: любая попытка открыть сетевое соединение
во время теста завершается ошибкой. Данные для CI загружаются отдельным
шагом pipeline до запуска тестов, а не из тестов.
"""

import socket

import pytest


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def guard(*args, **kwargs):
        raise RuntimeError("Сетевые обращения в тестах запрещены")

    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.setattr(socket, "create_connection", guard)

# Образ сервиса предсказаний (ЛР4): FastAPI + uvicorn поверх модели ЛР2.
#
# Базовый образ: официальный python, версия зафиксирована тегом и digest (как в ЛР3).
FROM python:3.14.10-slim-trixie@sha256:930e1abba7c4e8f96ed93e8012c35d174d7732752731bbd5e8e757d7ec6f3966

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Только runtime-зависимости (scikit-learn, skops, numpy, pandas, fastapi, uvicorn, prometheus-client
# и их транзитивные зависимости); без Jupyter, pytest, httpx, линтеров и сканеров.
COPY requirements-runtime.txt .
# После установки pip, setuptools и колёса ensurepip удаляются: в работе образа они не нужны.
RUN pip install --no-cache-dir --root-user-action=ignore -r requirements-runtime.txt \
    && pip uninstall -y --root-user-action=ignore pip setuptools wheel \
    && find /usr/local/lib -path '*ensurepip/_bundled*' -name '*.whl' -delete

# Пользователь без root-привилегий и без оболочки входа.
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app

# Явно перечисленные файлы. Принадлежат root, пользователю app доступны только на чтение.
# Секреты в образ не копируются: токены монтируются при запуске в /run/secrets (только чтение).
COPY src/config.py src/data.py src/predict.py src/
COPY src/api/ src/api/
COPY static/ static/
COPY integrity/MODEL_SHA256SUMS integrity/
COPY models/spam_tfidf_logreg.skops models/

USER 10001:10001

# Внутри контейнера сервис слушает интерфейс контейнера; на хосте порт публикуется
# только на loopback: docker run -p 127.0.0.1:8000:8000 ...
EXPOSE 8000

# 200 только при загруженной модели и настроенной аутентификации; 503 -> исключение -> unhealthy.
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; r = urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2); sys.exit(0 if r.status == 200 else 1)"]

ENTRYPOINT ["python", "-m", "uvicorn", "src.api.app:app"]
CMD ["--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log", "--no-proxy-headers", \
     "--no-server-header", "--limit-concurrency", "32", "--timeout-keep-alive", "5", "--timeout-graceful-shutdown", "10"]

# syntax=docker/dockerfile:1
# Образ для командной smoke-проверки модели (ЛР3). API будет добавлен в ЛР4.
#
# Базовый образ: официальный python, версия зафиксирована тегом и digest
# (digest — индекс multi-arch, подходит для arm64 и amd64).
FROM python:3.14.N-slim-trixie@sha256:python@sha256:930e1abba7c4e8f96ed93e8012c35d174d7732752731bbd5e8e757d7ec6f3966

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Только runtime-зависимости (без Jupyter, pytest, линтеров и сканеров).
COPY requirements-runtime.txt .
RUN pip install --no-cache-dir --root-user-action=ignore -r requirements-runtime.txt

# Пользователь без root-привилегий и без оболочки входа.
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app

# В образ явно передаются только нужные файлы; модель — ровно один файл
# и её эталон SHA-256 (проверяется функцией load_model() при каждой загрузке).
# Файлы принадлежат root и доступны пользователю app только на чтение.
COPY src/config.py src/data.py src/predict.py src/
COPY integrity/MODEL_SHA256SUMS integrity/
COPY models/spam_tfidf_logreg.skops models/

USER 10001:10001

# Контейнер выполняет одно предсказание и завершается, поэтому проверка
# состояния не нужна. HEALTHCHECK для сервиса будет задан в ЛР4.
HEALTHCHECK NONE

ENTRYPOINT ["python", "-m", "src.predict"]
CMD ["Are we still meeting for lunch tomorrow?"]

# ml-project — фильтр SMS-спама

Учебный проект дисциплины «Киберустойчивость систем искусственного интеллекта» (ИТМО).
Задача — бинарная классификация SMS-сообщений на `spam` и `ham` (датасет SMS Spam Collection, UCI, CC BY 4.0).

| Работа | Что сделано |
| --- | --- |
| ЛР1 | Защищённое окружение (Ubuntu Server 26.04.1 LTS, SSH по ключу, UFW), репозиторий, датасет, эталон SHA-256, модель угроз |
| ЛР2 | Базовая модель TF-IDF + логистическая регрессия, метрики, единая функция предсказания с валидацией входов, негативные тесты, эксперимент с устойчивостью, Bandit |

## Структура

```
data/raw/                 датасет (не в Git), загружается src/download_dataset.py
integrity/SHA256SUMS      эталон SHA-256 датасета (ЛР1)
integrity/MODEL_SHA256SUMS эталон SHA-256 модели (ЛР2)
models/                   модель spam_tfidf_logreg.skops (не в Git, см. «Артефакты»)
metrics/baseline.json     метрики базовой модели — ориентир для ЛР3
metrics/train_params.json параметры обучения, seed, версии, время, идентификатор данных
metrics/robustness.json   результаты эксперимента с изменёнными и нетипичными входами
notebooks/                01_eda_sms_spam.ipynb (ЛР1), 02_baseline_model.ipynb (ЛР2)
src/config.py             пути, seed, гиперпараметры, пределы входов
src/data.py               загрузка, проверка SHA-256, проверка качества, удаление дубликатов
src/train.py              обучение, оценка, сохранение модели и метрик
src/predict.py            валидация входа -> предобработка -> модель; безопасная загрузка модели
src/robustness.py         эксперимент с устойчивостью
tests/                    тесты целостности датасета и валидации входов
```

## Воспроизведение

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/download_dataset.py           # загрузка датасета
sha256sum -c integrity/SHA256SUMS        # проверка целостности данных
```

## Команды

| Действие | Команда |
| --- | --- |
| Обучение и оценка | `python -m src.train` (повторно — `python -m src.train --force`) |
| Предсказание | `python -m src.predict "текст сообщения"` |
| Эксперимент с устойчивостью | `python -m src.robustness` |
| Тесты | `python -m pytest tests/ -v` |
| Статический анализ | `python -m bandit -r src/` |
| Проверка модели | `sha256sum -c integrity/MODEL_SHA256SUMS` |

Использование из кода:

```python
from src.predict import predict, InputValidationError
predict("WINNER! Claim your prize now")          # {'label': 'spam', 'spam_probability': 0.98}
predict(["see you at 6", {"text": "Call now!"}])  # пакет до 100 сообщений
```

## Базовая модель

- `TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, sublinear_tf=True)` +
  `LogisticRegression(class_weight="balanced", solver="liblinear", C подобран CV)`.
- Дубликаты (без учёта регистра и пробелов) удаляются до разбиения: 5574 → 5159 сообщений.
- Разбиение train/test 80/20, `stratify=y`, `random_state=42`; C выбирается 5-кратной
  стратифицированной кросс-валидацией только на train по F1 класса spam.
- Основная метрика — F1 класса `spam` на тестовой выборке; значение и идентификатор данных — в `metrics/baseline.json`.

## Политика валидации входа

| Проверка | Предел / поведение |
| --- | --- |
| Тип | `str`, `bytes` (UTF-8), `{"text": str}` или список таких элементов; иначе `wrong_type` |
| Обязательное поле | у словаря только поле `text`; иначе `missing_field` / `unknown_field` |
| Пакет | 1–100 элементов (`empty_batch`, `batch_too_large`) |
| Размер в байтах | ≤ 4000 байт на сообщение, для `bytes` — до декодирования (`too_large`) |
| Кодировка | строгий UTF-8 (`bad_encoding`) |
| Длина | 1–1000 символов (`too_long`); пустая строка или только пробелы — `empty` |
| None / NaN / Inf | отклоняются (`missing_value`, `wrong_type`) |
| Символы | управляющие (кроме `\t \n \r`) и непарные суррогаты — `bad_char` |

При отклонении выбрасывается `InputValidationError` с кодом причины; модель не вызывается.

## Защита артефактов

- Модель сохраняется в формате **skops** (не выполняет код при загрузке, в отличие от pickle).
- `load_model()` проверяет SHA-256 по `integrity/MODEL_SHA256SUMS`, затем список типов в файле,
  и только после этого загружает модель.
- Данные, модель, `.env` и ключи в Git не помещаются (`.gitignore`).

## Артефакты

Репозиторий публичный, поэтому файл модели в нём не публикуется. Преподавателю модель
передаётся по ссылке с ограниченным доступом. Эталон хэша `integrity/MODEL_SHA256SUMS` хранится в Git.

Получить модель можно двумя способами:
1. положить переданный файл в `models/` и проверить `sha256sum -c integrity/MODEL_SHA256SUMS`;
2. обучить заново: `python -m src.train --force`. Метрики совпадут с `metrics/baseline.json`,
   но SHA-256 файла будет другим (архив skops содержит время записи), и эталон обновится.

## Jupyter

```bash
# на ВМ, от пользователя mluser
jupyter lab --ip=127.0.0.1 --port=8888 --no-browser
# на хосте
ssh -N -L 8888:127.0.0.1:8888 mlvm
```

Порт 8888 привязан к loopback и в UFW не открыт; вход по токену.

## Security-проверки

| Проверка | Результат |
| --- | --- |
| pip-audit (ЛР1) | 1 уязвимость pip, исправлена обновлением |
| Bandit (ЛР2) | B310 в `src/download_dataset.py` исправлена; повторный запуск — нет находок |
| pytest (ЛР2) | 26 тестов пройдены |
## ЛР3. Безопасный CI/CD

Платформа: GitHub Actions, файл `.github/workflows/ci.yml`. Запуск: push в `main`,
каждый pull request, ручной запуск (`workflow_dispatch`). Раннер `ubuntu-24.04-arm`
(та же архитектура ARM64, что и у ВМ разработки).

### Последовательность pipeline

| № | Этап | Команда | Job |
|---|------|---------|-----|
| 1 | Код, датасет с UCI и проверка SHA-256 | `python -m src.download_dataset`, `sha256sum -c integrity/SHA256SUMS` | quality |
| 2 | Зависимости (зафиксированные версии) | `pip install -r requirements-dev.txt`, `pip check` | quality |
| 3 | Линтер и статический анализ | `ruff check src/ tests/`, `python -m bandit -r src/` | quality |
| 4 | Тесты | `python -m pytest tests/ -q` | quality |
| 5 | Аудит зависимостей | `pip-audit --strict --no-deps -r requirements-runtime.txt -r requirements-dev.txt` | quality |
| 6 | Data gate | `python -m src.data_gate --dataset raw`, `python -m src.control_set`, `python -m src.data_gate --dataset control` | quality |
| 7 | Обучение и оценка кандидата, model gate | `python -m src.train`, `python -m src.evaluate`, `python -m src.model_gate` | quality |
| 8 | Проверка Dockerfile | `trivy config --exit-code 1 Dockerfile` | image |
| 9 | Сборка образа | `docker build -t ml-project:lab3 .` | image |
| 10 | Сканирование образа и security gate | `trivy image ...`, `python -m src.security_gate --trivy ...` | image |
| 11 | UID и smoke-проверка | `docker run --rm --entrypoint id ml-project:lab3 -u`, запуск предсказания | image |
| 12 | Отчёты | артефакты `reports-quality`, `reports-image` (сохраняются и при сбое) | оба |

Job `image` запускается только после успешного job `quality` (`needs`). Любой непройденный
шаг завершает job с ошибкой. `continue-on-error` не используется. Все команды выполняются
в `bash -eo pipefail`, поэтому `команда | tee отчёт` возвращает код команды.

### Политики безопасности

* **Зависимости.** Строгая политика: любая находка pip-audit блокирует сборку. Пакет, который
  не удалось проверить, тоже блокирует. pip-audit не сообщает severity.
  `requirements-runtime.txt` — зависимости образа, `requirements-dev.txt` — окружение CI
  с инструментами проверок. Все версии зафиксированы.
* **Образ.** Trivy (версия 0.76.1, официальный релиз GitHub, SHA-256 архива проверяется)
  создаёт полный JSON-отчёт. `src.security_gate` блокирует сборку на уровнях **CRITICAL и UNKNOWN**:
  неизвестная severity не считается безопасной. Отсутствующий или некорректный отчёт, ошибка
  загрузки базы или сканирования также дают отказ. Исключения допускаются только по точному
  идентификатору в `config/security_policy.toml` с обоснованием. Сейчас исключений нет.
* **Dockerfile.** `trivy config --exit-code 1` блокирует сборку при любой найденной ошибке конфигурации.
* **Секреты.** Для работы pipeline секреты не нужны и не создавались. Токен `GITHUB_TOKEN`
  имеет только право `contents: read`. `checkout` выполняется с `persist-credentials: false`.
  Событие `pull_request_target` не используется: код из внешних PR выполняется без секретов.
  Build arguments и переменные образа секретов не содержат.
* **Действия CI** зафиксированы по SHA коммита, версия указана в комментарии. Таймауты jobs
  составляют 20 и 25 минут.

### Данные и модель в CI

* Датасет загружается с UCI по HTTPS и сверяется с эталоном `integrity/SHA256SUMS` (ЛР1).
* **Контрольная выборка CI** `data/ci/control.tsv` совпадает с тестовой выборкой ЛР2: 1 032 сообщения,
  seed 42, стратификация. Файл детерминированно воспроизводится командой `python -m src.control_set`.
  Источник: UCI SMS Spam Collection (CC BY 4.0). Схема: `row_id<TAB>label<TAB>text`, UTF-8,
  с заголовком. Ожидаемые SHA-256 и версия данных записаны в `config/data_gate.toml`.
  Сам файл в Git не хранится, поскольку тексты сообщений не публикуются.
* Кандидатная модель обучается в CI (около 1 с) и передаётся в job `image` как артефакт.
  Там её SHA-256 сверяется с выходом job `quality`.

### Gates

* **Data gate.** Правила хранятся в `config/data_gate.toml`, проверка выполняется модулем `src/data_gate.py`, отчёт сохраняется в `reports/data_gate_*.json`.
  В датасете есть известные артефакты кодировки источника: C1-символы U+0091–U+0096 на месте
  типографских кавычек и тире cp1252. Они встречаются в 35 сообщениях сырого файла и в 8 сообщениях
  контрольной выборки. Gate допускает только эти символы и только в зафиксированном числе текстов.
  Любой другой управляющий символ или рост числа таких текстов приводит к отказу.
* **Model gate.** Политика хранится в `config/model_gate.toml`, baseline — в `metrics/baseline.json` (ЛР2, F1(spam) = 0,949). Правило прохождения: `candidate ≥ baseline × 0,95`, то есть ≥ 0,90155. Baseline не перезаписывается: `train.py` меняет его только с ключом `--update-baseline` вручную. Результат сохраняется в `reports/model_gate.json`.
* **Security gate.** Политика хранится в `config/security_policy.toml`, проверка выполняется модулем `src/security_gate.py`.

> Контрольная выборка совпадает с тестовой выборкой ЛР2. Повторная оценка на ней в каждом
> запуске CI — это регрессионный контроль, а не независимая оценка качества. Финальная
> оценка выполняется на новой отложенной выборке.

### Локальная проверка (ВМ, пользователь mluser, из корня проекта)

```bash
source .venv/bin/activate
ruff check src/ tests/
python -m bandit -r src/
python -m pytest tests/ -q
pip-audit --strict --no-deps -r requirements-runtime.txt -r requirements-dev.txt
python -m src.data_gate --dataset raw
python -m src.control_set && python -m src.data_gate --dataset control
python -m src.evaluate && python -m src.model_gate      # оценка сохранённой модели ЛР2
```

Docker (пользователь sysadmin, так как mluser не входит в группу docker):

```bash
sudo docker build -t ml-project:lab3 .
sudo docker run --rm --entrypoint id ml-project:lab3 -u          # ожидается 10001
sudo docker run --rm --network none --read-only --cap-drop ALL \
  --security-opt no-new-privileges:true --tmpfs /tmp ml-project:lab3 "Win a FREE prize, call now"
```

## ЛР4. Защищённый API модели, веб-интерфейс и мониторинг

Сервис `src/api/` публикует модель ЛР2 (TF-IDF + логистическая регрессия) через FastAPI.
Предсказание выполняет та же функция `src.predict.predict()` (валидация → `preprocess()` → модель),
что использовалась при оценке модели в ЛР2–ЛР3.

### Эндпоинты

| Метод и путь | Доступ | Назначение |
|---|---|---|
| `POST /predict` | `Authorization: Bearer <API-токен>` (роль predict) | Предсказание |
| `GET /health` | без токена | `200 {"status":"healthy","model_version":...}` только при загруженной модели и настроенной аутентификации, иначе `503 {"status":"unavailable"}` |
| `GET /metrics` | `Authorization: Bearer <токен метрик>` (роль metrics) | Метрики Prometheus |
| `GET /`, `/static/*` | без токена | Демонстрационная страница |

Swagger/OpenAPI (`/docs`, `/openapi.json`) отключены. Токен роли metrics на `/predict` и токен роли predict на `/metrics` → 403.

### Схема запроса `POST /predict`

`Content-Type: application/json` (UTF-8). Ровно одно поле:

```json
{"text": "Are we still meeting for lunch tomorrow?"}
{"texts": ["first message", "second message"]}
```

Ответ для `text`:

```json
{"prediction": "ham", "spam_score": 0.0159, "known_token_share": 1.0,
 "model_version": "spam-tfidf-logreg-f8a60e7a786e", "request_id": "3f2b…"}
```

Для `texts` — `{"predictions": [{...}, ...], "model_version": ..., "request_id": ...}`.

* `prediction` — `spam`, `ham` или `undetermined` (доля известных модели слов < 0,1: модель ничего не знает о тексте — мера из ЛР2, раздел 4.4).
* `spam_score` — выход `predict_proba` логистической регрессии; модель обучена с `class_weight="balanced"`, поэтому значение **не является откалиброванной вероятностью** спама. Порог решения 0,5. Для `undetermined` — `null`.
* `model_version` — `spam-tfidf-logreg-` + первые 12 символов SHA-256 файла модели.
* `request_id` — генерируется сервером, дублируется в заголовке `X-Request-ID` и в логе.

### Лимиты и их обоснование

| Параметр (переменная окружения) | Значение | Обоснование |
|---|---|---|
| Тело запроса (`MAX_BODY_BYTES`) | 512 КиБ | 32 сообщения × 1000 символов даже при полном `\uXXXX`-экранировании (до 12 байт на символ) ≈ 384 КиБ + служебная часть. Читается потоково, при превышении чтение прекращается (413) |
| Сообщений в запросе (`MAX_BATCH`) | 32 | Ограничивает время одного вычисления; ЛР2 допускала 100 для локального вызова |
| Длина сообщения | 1000 символов, 4000 байт UTF-8 | Политика ЛР2: максимум в датасете 910 символов |
| Частота (`RATE_LIMIT_REQUESTS` / `RATE_LIMIT_WINDOW_SECONDS`) | 60 запросов / 60 с | Ключ — отпечаток токена; для запросов без токена или с неверным токеном — адрес клиента |
| Параллельные вычисления (`MAX_CONCURRENCY`) | 4 | Число vCPU ВМ; при занятых слотах — 503 busy |
| Время ожидания вычисления (`INFERENCE_TIMEOUT_SECONDS`) | 2 с | Обычное время ответа — миллисекунды; при превышении — 504 |
| Соединений uvicorn (`--limit-concurrency`) | 32 | Сверх предела uvicorn сам отвечает 503, приложение запрос не видит |

Клиентский или серверный таймаут не останавливает поток вычисления. Слот освобождается только по фактическому
завершению вычисления, поэтому зависшие вычисления не накапливаются сверх `MAX_CONCURRENCY`, а длительность
одного вычисления ограничена пределами входа.

Лимитер хранит счётчики в памяти одного процесса: при нескольких workers каждый считает отдельно (фактический
лимит выше в N раз), после перезапуска счётчики обнуляются. Поэтому сервис запускается с `--workers 1`;
для нескольких процессов нужен общий счётчик (Redis) или лимит на reverse proxy.
Forwarded-заголовки не используются (`--no-proxy-headers`): ключ по адресу нельзя подменить заголовком.

### Коды ошибок

Формат: `{"error": {"code": "...", "message": "..."}, "request_id": "..."}` без traceback.

| HTTP | code | Причина |
|---|---|---|
| 400 | `invalid_json` | Не UTF-8, не JSON, `NaN`/`Infinity`, повторяющиеся ключи, чрезмерная вложенность |
| 401 | `missing_token`, `invalid_token` | Нет токена или токен неверен |
| 403 | `forbidden` | Токен другой роли |
| 413 | `body_too_large` | Тело больше `MAX_BODY_BYTES` (по Content-Length или при потоковом чтении) |
| 415 | `unsupported_media_type` | Не `application/json` в UTF-8 |
| 422 | `schema_error` | Нарушена схема: не объект, лишние поля, нет `text`/`texts`, неверные типы, пакет пуст или больше предела |
| 422 | `empty`, `too_long`, `too_large`, `bad_char` | Политика текста ЛР2 (пустой текст, длина, управляющие символы, включая C1-артефакты U+0091–U+0096) |
| 429 | `rate_limited` | Превышена частота; заголовок `Retry-After` |
| 503 | `not_ready`, `busy` | Модель не загружена / все слоты вычислений заняты |
| 504 | `timeout` | Вычисление не уложилось в `INFERENCE_TIMEOUT_SECONDS` |

### Секреты

Токены не хранятся в Git и образе. Локально — файлы в `.secrets/` (каталог в `.gitignore`, права 600):

```bash
(umask 077; mkdir -p .secrets
 python -c 'import secrets; print(secrets.token_hex(32))' > .secrets/api_token
 python -c 'import secrets; print(secrets.token_hex(32))' > .secrets/metrics_token
 printf 'Authorization: Bearer %s\n' "$(cat .secrets/api_token)" > .secrets/api_auth.header
 printf 'Authorization: Bearer %s\n' "$(cat .secrets/metrics_token)" > .secrets/metrics_auth.header)
```

Файлы `*.header` используются в `curl -H @файл`, чтобы токен не попадал в аргументы команд и историю.
В контейнер токены монтируются только для чтения в `/run/secrets` (владелец root, группа 10001, права 0440).

### Запуск

Локально (ВМ, пользователь mluser), доступ с хоста — SSH-туннель `ssh -N -L 8000:127.0.0.1:8000 mlvm`:

```bash
API_TOKEN_FILE=.secrets/api_token METRICS_TOKEN_FILE=.secrets/metrics_token \
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --workers 1 \
  --no-access-log --no-proxy-headers --no-server-header --limit-concurrency 32 --timeout-keep-alive 5 \
  > logs/api.jsonl
```

Контейнер (sysadmin), порт опубликован только на loopback:

```bash
sudo docker run -d --name ml-api -p 127.0.0.1:8000:8000 \
  --read-only --tmpfs /tmp:rw,noexec,nosuid,size=16m --cap-drop ALL \
  --security-opt no-new-privileges:true --memory 512m --cpus 1.0 --pids-limit 128 \
  -v /etc/ml-api:/run/secrets:ro ml-project:lab4
```

### Логи и метрики

Одна JSON-строка на запрос (stdout): `ts`, `request_id`, `method`, `route`, `status`, `duration_ms`,
`model_version`, `outcome` (тип отказа или `ok`), `error_code`, `items`. Тексты, токены и адреса клиентов не логируются.

Метрики: `ml_api_requests_total{route,method,status}`, `ml_api_request_duration_seconds{route}`,
`ml_api_rejections_total{reason}`, `ml_api_predictions_total{prediction}`, `ml_api_model_ready`.
Метки — только из фиксированных наборов (маршрут — шаблон, а не путь; неизвестный метод — `OTHER`).

Отказы uvicorn (некорректный HTTP-запрос — 400 «Invalid HTTP request received», превышение
`--limit-concurrency` — 503) происходят до приложения: их нет в JSON-логе и метриках приложения,
они видны только в stderr uvicorn (`logs/uvicorn.log`, `docker logs`).

### Сценарий обнаружения

```bash
# сервис запущен с RATE_LIMIT_REQUESTS=5 RATE_LIMIT_WINDOW_SECONDS=60
bash scripts/scenario.sh                                   # 3 корректных запроса, 12 неверных токенов, корректный запрос
python -m src.detect logs/api.jsonl --window 60 --threshold 10   # FIRING
sleep 60; python -m src.detect logs/api.jsonl --window 60 --threshold 10   # FIRING -> RESOLVED, состояние OK
```

Правило: не менее 10 ответов 401/429 на `/predict` за скользящие 60 с. Метрики и логи API показывают поведение
сервиса, но сами по себе не доказывают атаку на модель или ухудшение её точности.

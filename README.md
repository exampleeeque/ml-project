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

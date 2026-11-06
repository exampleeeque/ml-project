#!/usr/bin/env bash
# Контролируемый сценарий обнаружения (ЛР4, раздел 8).
#
# 1) нормальная работа: 3 корректных запроса;
# 2) снимок метрик ДО;
# 3) серия из BAD_REQUESTS (по умолчанию 12) запросов с неверным токеном:
#    ожидается 401, а после исчерпания лимита по адресу — 429;
# 4) снимок метрик ПОСЛЕ;
# 5) корректный запрос после сценария: ожидается 200.
#
# Токены читаются из файлов-заголовков .secrets/*.header (права 600): значение
# токена не попадает в аргументы команд, историю оболочки и вывод скрипта.
# Сервис должен быть запущен с малым лимитом, например RATE_LIMIT_REQUESTS=5.
#
# Запуск из корня проекта:  bash scripts/scenario.sh [URL]
set -euo pipefail

URL="${1:-http://127.0.0.1:8000}"
BAD_REQUESTS="${BAD_REQUESTS:-12}"
API_HEADER=".secrets/api_auth.header"
METRICS_HEADER=".secrets/metrics_auth.header"
OUT="reports/scenario"
mkdir -p "$OUT"

for f in "$API_HEADER" "$METRICS_HEADER"; do
  [[ -r "$f" ]] || { echo "Нет файла $f — см. README, раздел «Секреты»" >&2; exit 2; }
done

ts() { date -u +%Y-%m-%dT%H:%M:%S.%3NZ; }
predict() {   # predict <файл заголовка или -> <текст>
  local auth=()
  [[ "$1" != "-" ]] && auth=(-H "@$1")
  curl -sS -o /dev/null -w '%{http_code}' -X POST "$URL/predict" "${auth[@]}" \
       -H 'Content-Type: application/json' --data "{\"text\": \"$2\"}"
}
snapshot() {
  curl -sS -H "@$METRICS_HEADER" "$URL/metrics" \
    | grep -E '^ml_api_(requests_total\{[^}]*route="/predict"|rejections_total\{reason="(auth_invalid|auth_missing|rate_limited|validation_error)"\})' \
    > "$1"
}

echo "== 1. Нормальная работа ($(ts))"
curl -sS "$URL/health"; echo
for i in 1 2 3; do echo "корректный запрос $i: HTTP $(predict "$API_HEADER" "Are we still meeting for lunch tomorrow?")"; done

echo "== 2. Метрики ДО"
snapshot "$OUT/metrics_before.txt"; cat "$OUT/metrics_before.txt"

echo "== 3. Сценарий: $BAD_REQUESTS запросов с неверным токеном (начало $(ts))"
printf 'Authorization: Bearer %s\n' "scenario-wrong-token-0000000000000000000000000000" > "$OUT/wrong.header"
codes=()
for ((i = 1; i <= BAD_REQUESTS; i++)); do codes+=("$(predict "$OUT/wrong.header" "hello")"); done
rm -f "$OUT/wrong.header"
echo "коды ответов: ${codes[*]}  (окончание $(ts))"

echo "== 4. Метрики ПОСЛЕ"
snapshot "$OUT/metrics_after.txt"; cat "$OUT/metrics_after.txt"
echo "-- разница (ДО -> ПОСЛЕ):"
diff "$OUT/metrics_before.txt" "$OUT/metrics_after.txt" || true

echo "== 5. Корректный запрос после сценария ($(ts))"
echo "HTTP $(predict "$API_HEADER" "Congratulations! You have won a FREE prize. Call now to claim")"

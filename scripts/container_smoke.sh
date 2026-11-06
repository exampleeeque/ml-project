#!/usr/bin/env bash
# Smoke-проверка образа сервиса (ЛР4, раздел 10). Используется в CI (job image).
#
#  1) временные токены генерируются на время проверки и монтируются в контейнер
#     как файлы только для чтения (в образе и в репозитории секретов нет);
#  2) контейнер запускается без root, без capabilities, с файловой системой
#     только для чтения, порт публикуется только на 127.0.0.1;
#  3) проверяются /health (200 healthy), /predict (200 spam; 401 без токена),
#     /metrics (200 с токеном сборщика), UID процесса != 0;
#  4) второй контейнер запускается с неверным эталоном SHA-256 модели:
#     /health должен вернуть 503, /predict — 503.
#
# Запуск:  bash scripts/container_smoke.sh ИМЯ_ОБРАЗА [КАТАЛОГ_ОТЧЁТОВ]
set -euo pipefail

IMAGE="$1"
OUT="${2:-reports}"
PORT=8000
SUDO=""
[[ "$(id -u)" -ne 0 ]] && SUDO="sudo"
mkdir -p "$OUT"
WORK="$(mktemp -d)"
SECRETS="$WORK/secrets"
mkdir -p "$SECRETS"

cleanup() {
  docker rm -f lab4-smoke lab4-badhash >/dev/null 2>&1 || true
  $SUDO rm -rf "$WORK"
}
trap cleanup EXIT

api_token="$(openssl rand -hex 32)"
metrics_token="$(openssl rand -hex 32)"
if [[ -n "${GITHUB_ACTIONS:-}" ]]; then        # маскирование значений в логе GitHub Actions
  echo "::add-mask::$api_token"
  echo "::add-mask::$metrics_token"
fi
printf '%s\n' "$api_token" > "$SECRETS/api_token"
printf '%s\n' "$metrics_token" > "$SECRETS/metrics_token"
chmod 0440 "$SECRETS/api_token" "$SECRETS/metrics_token"
$SUDO chown -R 0:10001 "$SECRETS"              # читает только группа пользователя app (GID 10001)
$SUDO chmod 0750 "$SECRETS"

run() {   # run ИМЯ [доп. аргументы docker run]
  local name="$1"; shift
  docker run -d --name "$name" -p "127.0.0.1:${PORT}:8000" \
    --read-only --tmpfs /tmp:rw,noexec,nosuid,size=16m \
    --cap-drop ALL --security-opt no-new-privileges:true \
    --memory 512m --pids-limit 128 \
    -v "$SECRETS:/run/secrets:ro" "$@" "$IMAGE" >/dev/null
}
wait_http() {   # ждёт, пока сервис начнёт отвечать; печатает код /health
  local code="000"
  for _ in $(seq 1 30); do
    code="$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/health" || true)"
    [[ "$code" != "000" ]] && break
    sleep 1
  done
  echo "$code"
}
code_of() { curl -s -o /dev/null -w '%{http_code}' "$@"; }

echo "== Контейнер с корректной моделью"
run lab4-smoke
health_code="$(wait_http)"
health="$(curl -sS "http://127.0.0.1:${PORT}/health")"
echo "/health: HTTP $health_code $health"
[[ "$health_code" == "200" ]] && grep -q '"status":"healthy"' <<< "$health"

uid="$(docker exec lab4-smoke id -u)"
echo "UID процесса в контейнере: $uid"
[[ "$uid" != "0" ]]

spam="$(curl -sS -X POST "http://127.0.0.1:${PORT}/predict" \
  -H "Authorization: Bearer $api_token" -H 'Content-Type: application/json' \
  --data '{"text": "Congratulations! You have won a FREE prize. Call 09061701461 to claim"}')"
echo "/predict (spam): $spam"
grep -q '"prediction":"spam"' <<< "$spam"

no_token="$(code_of -X POST "http://127.0.0.1:${PORT}/predict" -H 'Content-Type: application/json' --data '{"text":"hi"}')"
echo "/predict без токена: HTTP $no_token"
[[ "$no_token" == "401" ]]

metrics_code="$(code_of -H "Authorization: Bearer $metrics_token" "http://127.0.0.1:${PORT}/metrics")"
echo "/metrics с токеном сборщика: HTTP $metrics_code"
[[ "$metrics_code" == "200" ]]

docker logs lab4-smoke > "$OUT/container_api.log" 2>&1
docker rm -f lab4-smoke >/dev/null

echo "== Контейнер с неверным эталоном SHA-256 модели"
printf '%064d  models/spam_tfidf_logreg.skops\n' 0 > "$WORK/BAD_MODEL_SHA256SUMS"
chmod 0444 "$WORK/BAD_MODEL_SHA256SUMS"
run lab4-badhash -v "$WORK/BAD_MODEL_SHA256SUMS:/app/integrity/MODEL_SHA256SUMS:ro"
bad_health="$(wait_http)"
bad_predict="$(code_of -X POST "http://127.0.0.1:${PORT}/predict" \
  -H "Authorization: Bearer $api_token" -H 'Content-Type: application/json' --data '{"text":"hi"}')"
echo "/health: HTTP $bad_health; /predict: HTTP $bad_predict"
docker logs lab4-badhash > "$OUT/container_badhash.log" 2>&1
grep -q '"model_load_failed"' "$OUT/container_badhash.log"
[[ "$bad_health" == "503" && "$bad_predict" == "503" ]]

{
  echo "image: $IMAGE"
  echo "uid: $uid"
  echo "health: $health_code $health"
  echo "predict_spam: $spam"
  echo "predict_without_token: $no_token"
  echo "metrics_with_token: $metrics_code"
  echo "bad_hash_health: $bad_health"
  echo "bad_hash_predict: $bad_predict"
} | tee "$OUT/smoke.txt"
echo "SMOKE: PASSED"

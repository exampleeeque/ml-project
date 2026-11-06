"""Правило обнаружения по структурированным логам API (ЛР4, раздел 8).

Правило: число ответов /predict с кодами из --statuses (по умолчанию 401 и
429) за скользящее окно --window секунд >= --threshold -> состояние FIRING.
Состояние снимается (RESOLVED), когда число таких ответов в окне
опускается ниже порога. Окно — полуинтервал (t - window; t].

Источник — JSON-строки лога сервиса (событие "request"); прочие строки
(например, сообщения uvicorn) пропускаются. Логи не содержат токенов и
текстов, поэтому правило работает на обезличенных данных.

Запуск из корня проекта:
    python -m src.detect logs/api.jsonl --window 60 --threshold 10
    docker logs ml-api | python -m src.detect -        # чтение из stdin
Код возврата: 0 — правило не срабатывало, 1 — срабатывало хотя бы раз.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime, timedelta


def read_events(lines, route: str, statuses: set[int]) -> list[datetime]:
    times = []
    for line in lines:
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(rec, dict) or rec.get("event") != "request":
            continue
        if rec.get("route") == route and rec.get("status") in statuses:
            times.append(datetime.fromisoformat(rec["ts"]))
    return sorted(times)


def evaluate(times: list[datetime], window: float, threshold: int, now: datetime) -> list[dict]:
    """Переходы состояния правила: [{"state": "FIRING"|"RESOLVED", "at": ..., "count": ...}]."""
    span = timedelta(seconds=window)
    points = [(t, 1) for t in times] + [(t + span, -1) for t in times if t + span <= now]
    points.sort(key=lambda p: (p[0], p[1]))           # истечение раньше поступления в тот же момент
    count, firing, transitions = 0, False, []
    for at, delta in points:
        count += delta
        if not firing and count >= threshold:
            firing = True
            transitions.append({"state": "FIRING", "at": at, "count": count})
        elif firing and count < threshold:
            firing = False
            transitions.append({"state": "RESOLVED", "at": at, "count": count})
    return transitions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Правило обнаружения 401/429 по логам API")
    parser.add_argument("log", help="файл JSON-логов сервиса или - для stdin")
    parser.add_argument("--route", default="/predict")
    parser.add_argument("--statuses", default="401,429")
    parser.add_argument("--window", type=float, default=60.0, help="окно, с")
    parser.add_argument("--threshold", type=int, default=10, help="порог числа ответов в окне")
    args = parser.parse_args(argv)

    statuses = {int(s) for s in args.statuses.split(",")}
    if args.log == "-":
        times = read_events(sys.stdin, args.route, statuses)
    else:
        with open(args.log, encoding="utf-8") as f:
            times = read_events(f, args.route, statuses)

    now = datetime.now(UTC)
    transitions = evaluate(times, args.window, args.threshold, now)
    print(f"Правило: route={args.route}, статусы {sorted(statuses)}, >= {args.threshold} за {args.window:g} с")
    print(f"Подходящих ответов в логе: {len(times)}" +
          (f" ({times[0].isoformat()} — {times[-1].isoformat()})" if times else ""))
    for t in transitions:
        print(f"{t['state']:<9}{t['at'].isoformat()}  (ответов в окне: {t['count']})")
    state = transitions[-1]["state"] if transitions else "OK"
    print(f"Текущее состояние на {now.isoformat(timespec='seconds')}: {'OK' if state == 'RESOLVED' else state}")
    return 1 if transitions else 0


if __name__ == "__main__":
    sys.exit(main())

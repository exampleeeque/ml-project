"""Model quality gate (ЛР3): сравнение кандидатной модели с baseline ЛР2.

Правило (config/model_gate.toml):
  * direction = "higher": кандидат проходит при candidate >= baseline * (1 - tol);
  * direction = "lower":  кандидат проходит при candidate <= baseline * (1 + tol);
  * при baseline == 0 используется абсолютный порог absolute_threshold.
Отклоняются: отсутствующая, нечисловая, NaN/Inf или вне шкалы метрика,
другая метрика, другая версия данных, другой размер контрольной выборки.
Baseline никогда не перезаписывается. Результат и код возврата сохраняются
в reports/model_gate.json. Код возврата: 0 — прошёл, 1 — отказ.

Запуск из корня проекта:  python -m src.model_gate
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path

from src import config

POLICY_PATH = config.CONFIG_DIR / "model_gate.toml"


class GateError(ValueError):
    """Входные данные gate некорректны — это отказ, а не прохождение."""


def _number(value, name: str) -> float:
    # bool — подкласс int, но метрикой не является; строки и None отклоняются
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GateError(f"{name}: нечисловое значение {value!r}")
    if math.isnan(value) or math.isinf(value):
        raise GateError(f"{name}: недопустимое значение {value!r}")
    return float(value)


def _load_json(path: Path, name: str) -> dict:
    if not path.is_file():
        raise GateError(f"{name}: файл {path} отсутствует")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise GateError(f"{name}: некорректный JSON ({e})") from None
    if not isinstance(data, dict):
        raise GateError(f"{name}: ожидался объект JSON")
    return data


def compare(baseline: dict, candidate: dict, policy: dict) -> dict:
    """Чистая логика сравнения. Возвращает результат; при некорректных данных — GateError."""
    metric = policy["metric"]
    if baseline.get("primary_metric") != metric:
        raise GateError(f"baseline: метрика {baseline.get('primary_metric')!r}, политика требует {metric!r}")
    if candidate.get("metric") != metric:
        raise GateError(f"candidate: метрика {candidate.get('metric')!r}, политика требует {metric!r}")
    if "value" not in candidate:
        raise GateError("candidate: значение метрики отсутствует")
    base = _number(baseline.get("value"), "baseline.value")
    cand = _number(candidate.get("value"), "candidate.value")
    lo, hi = policy["value_range"]
    for name, v in (("baseline.value", base), ("candidate.value", cand)):
        if not lo <= v <= hi:
            raise GateError(f"{name}={v} вне шкалы [{lo}; {hi}]")
    if policy["require_same_data_id"] and candidate.get("data_id") != baseline.get("data_id"):
        raise GateError(f"несовместимая версия данных: {candidate.get('data_id')!r} != {baseline.get('data_id')!r}")
    if policy["require_same_size"] and candidate.get("n") != baseline.get("test_size"):
        raise GateError(f"размер выборки {candidate.get('n')!r} != {baseline.get('test_size')!r}")

    tol = policy["relative_tolerance"]
    if base == 0:
        threshold, rule = float(policy["absolute_threshold"]), "absolute"
    elif policy["direction"] == "higher":
        threshold, rule = base * (1 - tol), "relative"
    elif policy["direction"] == "lower":
        threshold, rule = base * (1 + tol), "relative"
    else:
        raise GateError(f"неизвестное направление {policy['direction']!r}")
    threshold = round(threshold, 6)
    passed = cand >= threshold if policy["direction"] == "higher" else cand <= threshold
    return {"metric": metric, "direction": policy["direction"], "baseline": base,
            "candidate": cand, "threshold": threshold, "rule": rule,
            "relative_change": round((cand - base) / base, 6) if base else None,
            "passed": passed}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Model quality gate")
    parser.add_argument("--policy", type=Path, default=POLICY_PATH)
    parser.add_argument("--candidate", type=Path, help="файл метрик кандидата (по умолчанию из политики)")
    args = parser.parse_args(argv)

    with args.policy.open("rb") as f:
        policy = tomllib.load(f)
    baseline_path = config.ROOT / policy["baseline_path"]
    candidate_path = args.candidate or (config.ROOT / policy["candidate_path"])
    report = {"gate": "model", "checked_utc": datetime.now(UTC).isoformat(timespec="seconds"),
              "baseline_file": str(baseline_path), "candidate_file": str(candidate_path)}
    try:
        baseline = _load_json(baseline_path, "baseline")
        candidate = _load_json(candidate_path, "candidate")
        result = compare(baseline, candidate, policy)
        report.update(result, data_id=candidate.get("data_id"),
                      model_sha256=candidate.get("model_sha256"), commit=candidate.get("commit"))
        status = "passed" if result["passed"] else "failed"
        reason = "" if result["passed"] else "метрика кандидата ниже допустимого порога"
    except GateError as e:
        status, reason = "failed", str(e)
    report.update(status=status, reason=reason, exit_code=0 if status == "passed" else 1)

    config.REPORTS_DIR.mkdir(exist_ok=True)
    (config.REPORTS_DIR / "model_gate.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if "threshold" in report:
        print(f"{report['metric']}: baseline {report['baseline']}, кандидат {report['candidate']}, "
              f"порог {report['threshold']} ({report['rule']})")
    print(f"MODEL GATE: {status.upper()}" + (f" — {reason}" if reason else ""))
    return report["exit_code"]


if __name__ == "__main__":
    sys.exit(main())

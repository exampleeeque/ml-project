"""Security gate (ЛР3): проверка отчётов pip-audit и Trivy по политике проекта.

Политика: config/security_policy.toml.
  * pip-audit (JSON): любая уязвимость или непроверенный пакет -> отказ;
  * Trivy image (JSON): уязвимость уровня CRITICAL или UNKNOWN -> отказ;
    для UNKNOWN допускаются точечные исключения (идентификатор + пакет +
    обоснование + срок); CRITICAL не исключается никогда.
Отсутствующий, пустой или некорректный отчёт -> отказ: ошибка сканирования
не трактуется как отсутствие уязвимостей.
Код возврата: 0 — политика соблюдена, 1 — нарушение или некорректный отчёт.

Запуск из корня проекта:
    python -m src.security_gate --pip-audit reports/pip-audit.json
    python -m src.security_gate --trivy reports/trivy-image.json
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from datetime import UTC, date, datetime
from pathlib import Path

from src import config

POLICY_PATH = config.CONFIG_DIR / "security_policy.toml"


class ReportError(ValueError):
    """Отчёт отсутствует или не соответствует ожидаемому формату."""


def _load(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size == 0:
        raise ReportError(f"отчёт {path} отсутствует или пуст")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ReportError(f"отчёт {path} — некорректный JSON: {e}") from None
    if not isinstance(data, dict):
        raise ReportError(f"отчёт {path}: ожидался объект JSON")
    return data


def check_pip_audit(report: dict, policy: dict) -> tuple[list[dict], list[str]]:
    """Возвращает (находки, нарушения политики)."""
    deps = report.get("dependencies")
    if not isinstance(deps, list) or not deps:
        raise ReportError("pip-audit: в отчёте нет списка dependencies")
    findings, violations, seen = [], [], set()
    for dep in deps:
        name, version = dep.get("name"), dep.get("version")
        if (name, version) in seen:          # пакет может встретиться в нескольких файлах -r
            continue
        seen.add((name, version))
        if "skip_reason" in dep:
            findings.append({"package": name, "version": version, "id": "SKIPPED",
                             "fix": "", "severity": "UNKNOWN", "note": dep["skip_reason"]})
            if policy["block_skipped"]:
                violations.append(f"{name}: пакет не проверен ({dep['skip_reason']})")
            continue
        ids = set()
        for v in dep.get("vulns", []):
            if v.get("id") in ids:           # pip-audit может повторять запись из разных баз
                continue
            ids.add(v.get("id"))
            fix = ", ".join(v.get("fix_versions", [])) or "нет"
            findings.append({"package": name, "version": version, "id": v.get("id"),
                             "fix": fix, "severity": "не сообщается pip-audit"})
            if policy["block_any_vulnerability"]:
                violations.append(f"{name} {version}: {v.get('id')} (исправление: {fix})")
    return findings, violations


def _exception_for(item: dict, exceptions: list[dict], today: date) -> dict | None:
    """Точечное исключение: только UNKNOWN, точный идентификатор и пакет, срок не истёк."""
    if item["severity"] != "UNKNOWN":
        return None                      # CRITICAL исключать нельзя
    for exc in exceptions:
        if exc["id"] == item["id"] and item["package"] in exc["packages"] and today <= exc["until"]:
            return exc
    return None


def check_trivy(report: dict, policy: dict, today: date | None = None) -> tuple[list[dict], list[str]]:
    if report.get("SchemaVersion") != 2 or "ArtifactName" not in report:
        raise ReportError("Trivy: неизвестный формат отчёта (нет SchemaVersion=2 или ArtifactName)")
    results = report.get("Results")
    if not isinstance(results, list):
        raise ReportError("Trivy: в отчёте нет Results — сканирование не выполнено")
    today = today or datetime.now(UTC).date()
    block = {s.upper() for s in policy["block_severities"]}
    exceptions = policy.get("exceptions", [])
    findings, violations = [], []
    for res in results:
        for v in res.get("Vulnerabilities") or []:
            sev = str(v.get("Severity") or "UNKNOWN").upper()
            item = {"package": v.get("PkgName"), "version": v.get("InstalledVersion"),
                    "id": v.get("VulnerabilityID"), "fix": v.get("FixedVersion") or "нет",
                    "severity": sev, "target": res.get("Target")}
            findings.append(item)
            if sev not in block:
                continue
            exc = _exception_for(item, exceptions, today)
            if exc:
                item["exception"] = f"до {exc['until']}: {exc['reason']}"
            else:
                violations.append(f"{item['package']} {item['version']}: {item['id']} [{sev}] "
                                  f"(исправление: {item['fix']})")
    return findings, violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Security gate")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pip-audit", type=Path, dest="pip_audit")
    group.add_argument("--trivy", type=Path)
    parser.add_argument("--policy", type=Path, default=POLICY_PATH)
    args = parser.parse_args(argv)
    with args.policy.open("rb") as f:
        policy = tomllib.load(f)

    kind, path = ("pip-audit", args.pip_audit) if args.pip_audit else ("trivy", args.trivy)
    out = {"gate": "security", "source": kind, "report": str(path),
           "checked_utc": datetime.now(UTC).isoformat(timespec="seconds")}
    try:
        data = _load(path)
        if kind == "pip-audit":
            findings, violations = check_pip_audit(data, policy["pip_audit"])
        else:
            findings, violations = check_trivy(data, policy["trivy"])
            out["artifact"] = data.get("ArtifactName")
            out["trivy_db_updated"] = (data.get("Metadata") or {}).get("DB", {}).get("UpdatedAt")
        out.update(findings=findings, violations=violations)
        status = "failed" if violations else "passed"
    except ReportError as e:
        out.update(findings=[], violations=[str(e)])
        status = "failed"
    out.update(status=status, exit_code=0 if status == "passed" else 1)

    config.REPORTS_DIR.mkdir(exist_ok=True)
    (config.REPORTS_DIR / f"security_gate_{kind}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print(f"{'Пакет':<28}{'Версия':<20}{'Идентификатор':<24}{'Severity':<14}Исправление")
    for f_ in out["findings"]:
        print(f"{str(f_['package']):<28}{str(f_['version']):<20}{str(f_['id']):<24}"
              f"{str(f_['severity'])[:13]:<14}{f_['fix']}")
    excepted = [f_ for f_ in out["findings"] if "exception" in f_]
    print(f"Находок всего: {len(out['findings'])}; нарушений политики: {len(out['violations'])}; "
          f"по исключениям: {len(excepted)}")
    for f_ in excepted:
        print(f"  ИСКЛЮЧЕНИЕ: {f_['package']} {f_['version']}: {f_['id']} [{f_['severity']}] {f_['exception']}")
    for v in out["violations"]:
        print(f"  НАРУШЕНИЕ: {v}")
    print(f"SECURITY GATE ({kind}): {status.upper()}")
    return out["exit_code"]


if __name__ == "__main__":
    sys.exit(main())

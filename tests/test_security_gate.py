"""Тесты проверки отчётов pip-audit и Trivy по политике проекта (ЛР3).

Отчёты — искусственные минимальные примеры формата JSON сканеров.
"""

import json
from datetime import date

import pytest

from src.security_gate import ReportError, check_pip_audit, check_trivy

PIP_POLICY = {"block_any_vulnerability": True, "block_skipped": True}
TRIVY_POLICY = {"block_severities": ["CRITICAL", "UNKNOWN"], "exceptions": []}


def trivy_report(*severities):
    vulns = [{"VulnerabilityID": f"CVE-0000-{i}", "PkgName": "pkg", "InstalledVersion": "1.0",
              "FixedVersion": "1.1", "Severity": s} for i, s in enumerate(severities)]
    return {"SchemaVersion": 2, "ArtifactName": "ml-project:test",
            "Results": [{"Target": "test", "Vulnerabilities": vulns}]}


def test_pip_audit_clean_passes():
    report = {"dependencies": [{"name": "numpy", "version": "2.5.4", "vulns": []}]}
    assert check_pip_audit(report, PIP_POLICY) == ([], [])


def test_pip_audit_any_vulnerability_fails_with_fields():
    report = {"dependencies": [{"name": "pip", "version": "25.1.1",
                                "vulns": [{"id": "PYSEC-2026-1795", "fix_versions": ["26.2.0"]}]}]}
    findings, violations = check_pip_audit(report, PIP_POLICY)
    f = findings[0]
    assert (f["package"], f["version"], f["id"], f["fix"]) == ("pip", "25.1.1", "PYSEC-2026-1795", "26.2.0")
    assert len(violations) == 1


def test_pip_audit_skipped_package_is_not_safe():
    report = {"dependencies": [{"name": "local-pkg", "skip_reason": "not on PyPI"}]}
    assert check_pip_audit(report, PIP_POLICY)[1]


@pytest.mark.parametrize("severities, blocked", [
    ((), False),
    (("LOW", "MEDIUM", "HIGH"), False),
    (("CRITICAL",), True),
    (("UNKNOWN",), True),          # неизвестная severity не считается безопасной
])
def test_trivy_policy(severities, blocked):
    assert bool(check_trivy(trivy_report(*severities), TRIVY_POLICY)[1]) is blocked


def _exc(**over):
    e = {"id": "CVE-0000-0", "packages": ["pkg"], "until": date(2026, 12, 5), "reason": "test"}
    e.update(over)
    return {**TRIVY_POLICY, "exceptions": [e]}


def test_unknown_exception_by_exact_id_and_package_within_term():
    today = date(2026, 11, 5)
    assert check_trivy(trivy_report("UNKNOWN"), _exc(), today)[1] == []
    assert check_trivy(trivy_report("UNKNOWN", "UNKNOWN"), _exc(), today)[1]        # CVE-0000-1 не исключён
    assert check_trivy(trivy_report("UNKNOWN"), _exc(packages=["other"]), today)[1]  # другой пакет


def test_expired_exception_blocks_again():
    assert check_trivy(trivy_report("UNKNOWN"), _exc(), date(2026, 12, 6))[1]


def test_critical_cannot_be_excepted():
    assert check_trivy(trivy_report("CRITICAL"), _exc(), date(2026, 11, 5))[1]


@pytest.mark.parametrize("report", [{}, {"SchemaVersion": 2, "ArtifactName": "x"}, {"Results": []}])
def test_malformed_trivy_report_is_failure(report):
    with pytest.raises(ReportError):
        check_trivy(report, TRIVY_POLICY)


def test_simulated_report_is_blocked():
    from src import config
    data = json.loads((config.ROOT / "tests" / "fixtures" / "trivy_simulated_critical.json").read_text())
    assert "SIMULATED" in data["ArtifactName"]
    assert check_trivy(data, TRIVY_POLICY)[1]

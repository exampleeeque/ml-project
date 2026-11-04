"""Тесты проверки отчётов pip-audit и Trivy по политике проекта (ЛР3).

Отчёты — искусственные минимальные примеры формата JSON сканеров.
"""

import json

import pytest

from src.security_gate import ReportError, check_pip_audit, check_trivy

PIP_POLICY = {"block_any_vulnerability": True, "block_skipped": True}
TRIVY_POLICY = {"block_severities": ["CRITICAL", "UNKNOWN"], "ignore": []}


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


def test_trivy_ignore_is_by_exact_id_only():
    policy = {**TRIVY_POLICY, "ignore": ["CVE-0000-0"]}
    assert check_trivy(trivy_report("CRITICAL"), policy)[1] == []
    assert check_trivy(trivy_report("CRITICAL", "CRITICAL"), policy)[1]


@pytest.mark.parametrize("report", [{}, {"SchemaVersion": 2, "ArtifactName": "x"}, {"Results": []}])
def test_malformed_trivy_report_is_failure(report):
    with pytest.raises(ReportError):
        check_trivy(report, TRIVY_POLICY)


def test_simulated_report_is_blocked():
    from src import config
    data = json.loads((config.ROOT / "tests" / "fixtures" / "trivy_simulated_critical.json").read_text())
    assert "SIMULATED" in data["ArtifactName"]
    assert check_trivy(data, TRIVY_POLICY)[1]

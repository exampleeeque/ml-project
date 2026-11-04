"""Тесты ЛОГИКИ СРАВНЕНИЯ model gate на искусственных метриках (ЛР3, раздел 6).

Это не оценка реальной модели: значения метрик заданы вручную,
чтобы проверить правило порога и отклонение некорректных данных.
"""

import math
import tomllib

import pytest

from src import config
from src.model_gate import GateError, compare

POLICY = {"metric": "f1_spam", "direction": "higher", "relative_tolerance": 0.05,
          "absolute_threshold": 0.5, "value_range": [0.0, 1.0],
          "require_same_data_id": True, "require_same_size": True}
BASE = {"primary_metric": "f1_spam", "value": 0.80, "data_id": "d1", "test_size": 100}


def cand(value, **over):
    c = {"metric": "f1_spam", "value": value, "data_id": "d1", "n": 100}
    c.update(over)
    return c


def test_example_from_assignment_threshold_076():
    # baseline 0,80 -> порог 0,76 (пример из условия)
    assert compare(BASE, cand(0.76), POLICY)["threshold"] == pytest.approx(0.76)
    assert compare(BASE, cand(0.76), POLICY)["passed"] is True
    assert compare(BASE, cand(0.7599), POLICY)["passed"] is False


def test_improvement_passes_and_constant_predictor_fails():
    assert compare(BASE, cand(0.95), POLICY)["passed"] is True
    assert compare(BASE, cand(0.0), POLICY)["passed"] is False


def test_lower_is_better_direction():
    policy = {**POLICY, "direction": "lower", "value_range": [0.0, 100.0]}
    base = {**BASE, "value": 10.0}
    assert compare(base, cand(10.5), policy)["passed"] is True        # 10 * 1,05 = 10,5
    assert compare(base, cand(10.51), policy)["passed"] is False


def test_zero_baseline_uses_absolute_threshold():
    res = compare({**BASE, "value": 0.0}, cand(0.4), POLICY)
    assert res["rule"] == "absolute" and res["passed"] is False


@pytest.mark.parametrize("bad", [None, "0.95", True, math.nan, math.inf, 1.5, -0.1])
def test_invalid_candidate_metric_rejected(bad):
    with pytest.raises(GateError):
        compare(BASE, cand(bad), POLICY)


def test_missing_metric_and_other_metric_rejected():
    c = cand(0.9)
    del c["value"]
    with pytest.raises(GateError):
        compare(BASE, c, POLICY)
    with pytest.raises(GateError):
        compare(BASE, cand(0.9, metric="accuracy"), POLICY)


def test_incompatible_data_version_or_size_rejected():
    with pytest.raises(GateError, match="версия данных"):
        compare(BASE, cand(0.9, data_id="d2"), POLICY)
    with pytest.raises(GateError):
        compare(BASE, cand(0.9, n=50), POLICY)


def test_project_policy_matches_assignment():
    with (config.CONFIG_DIR / "model_gate.toml").open("rb") as f:
        policy = tomllib.load(f)
    assert policy["relative_tolerance"] == 0.05 and policy["direction"] == "higher"
    assert policy["baseline_path"] != policy["candidate_path"]       # baseline отделён от результата

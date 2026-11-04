"""Тесты data quality gate на маленьких искусственных файлах (ЛР3, раздел 5)."""

import hashlib

import pytest

from src.data_gate import load_rules, run_gate

COMMON = {"allowed_labels": ["ham", "spam"], "known_artifact_chars": ["U+0092"]}
GOOD_ROWS = [(i, "spam" if i % 4 == 0 else "ham", f"message number {i}") for i in range(20)]


def _write(tmp_path, rows, header="row_id\tlabel\ttext"):
    path = tmp_path / "control.tsv"
    lines = [header] + ["\t".join(map(str, r)) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _rules(path, **over):
    rules = {"columns": ["row_id", "label", "text"], "header": True, "min_rows": 10, "max_rows": 100,
             "spam_share": [0.2, 0.3], "max_normalized_duplicates": 0,
             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    rules.update(over)
    return rules


def _failed(results):
    return {r["check"] for r in results if not r["ok"]}


def test_valid_file_passes(tmp_path):
    path = _write(tmp_path, GOOD_ROWS)
    assert _failed(run_gate(path, _rules(path), COMMON)) == set()


def test_missing_required_field_fails(tmp_path):
    path = _write(tmp_path, GOOD_ROWS, header="row_id\tcategory\ttext")
    assert "schema_required_fields" in _failed(run_gate(path, _rules(path), COMMON))


def test_wrong_shape_fails(tmp_path):
    rows = [*GOOD_ROWS, (99, "ham")]                      # строка из двух полей
    path = _write(tmp_path, rows)
    assert "shape_fields_per_row" in _failed(run_gate(path, _rules(path), COMMON))


@pytest.mark.parametrize("bad_id", ["nan", "inf", "1.5", ""])
def test_nan_inf_and_non_integer_ids_fail(tmp_path, bad_id):
    rows = [*GOOD_ROWS[:-1], (bad_id, "ham", "valid text")]
    path = _write(tmp_path, rows)
    assert "type_row_id_int" in _failed(run_gate(path, _rules(path), COMMON))


def test_unknown_label_and_empty_text_fail(tmp_path):
    rows = [*GOOD_ROWS[:-2], (18, "phishing", "text"), (19, "ham", "")]
    path = _write(tmp_path, rows)
    failed = _failed(run_gate(path, _rules(path), COMMON))
    assert {"allowed_labels", "no_missing_values", "text_policy_lr2"} <= failed


def test_control_characters_fail_by_lr2_policy(tmp_path):
    rows = [*GOOD_ROWS[:-1], (19, "ham", "bad\x1b[31m text")]
    path = _write(tmp_path, rows)
    assert "text_policy_lr2" in _failed(run_gate(path, _rules(path), COMMON))


def test_known_encoding_artifacts_allowed_only_within_limit(tmp_path):
    # U+0092 — известный артефакт cp1252 (апостроф); допускается не более чем в 1 тексте
    rows = [*GOOD_ROWS[:-2], (18, "ham", "it\x92s fine"), (19, "ham", "don\x92t")]
    path = _write(tmp_path, rows)
    assert _failed(run_gate(path, _rules(path, max_texts_with_known_artifacts=2), COMMON)) == set()
    assert "known_encoding_artifacts" in _failed(run_gate(path, _rules(path, max_texts_with_known_artifacts=1), COMMON))


def test_unknown_c1_character_is_not_tolerated(tmp_path):
    rows = [*GOOD_ROWS[:-1], (19, "ham", "bad \x85 next line")]   # U+0085 не входит в список
    path = _write(tmp_path, rows)
    assert "text_policy_lr2" in _failed(run_gate(path, _rules(path, max_texts_with_known_artifacts=5), COMMON))


def test_volume_and_class_share_bounds(tmp_path):
    path = _write(tmp_path, GOOD_ROWS)                    # 20 строк, доля spam 0,25
    failed = _failed(run_gate(path, _rules(path, min_rows=50, spam_share=[0.3, 0.5]), COMMON))
    assert {"min_rows", "spam_share_bounds"} <= failed


def test_duplicates_fail(tmp_path):
    rows = [*GOOD_ROWS[:-1], (19, "ham", "MESSAGE   number 1")]
    path = _write(tmp_path, rows)
    assert "normalized_duplicates" in _failed(run_gate(path, _rules(path), COMMON))


def test_hash_mismatch_and_missing_file_fail(tmp_path):
    path = _write(tmp_path, GOOD_ROWS)
    assert "sha256" in _failed(run_gate(path, _rules(path, sha256="0" * 64), COMMON))
    assert "file_exists" in _failed(run_gate(tmp_path / "nope.tsv", _rules(path), COMMON))


def test_project_rules_file_is_complete():
    rules = load_rules()
    for name in ("raw", "control"):
        ds = rules["datasets"][name]
        assert {"path", "columns", "min_rows", "spam_share", "sha256"} <= set(ds)

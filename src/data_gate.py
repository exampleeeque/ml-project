"""Data quality gate (ЛР3): обязательная проверка данных до оценки и обучения.

Правила читаются из config/data_gate.toml. Проверяются наличие файла,
SHA-256 и версия данных, схема (обязательные поля и их порядок), форма
(число полей в каждой строке), типы, допустимые значения, пропуски,
политика текстов ЛР2, объём, доля класса spam и дубликаты.
Все нарушения собираются в один отчёт reports/data_gate_<имя>.json.
Код возврата: 0 — данные соответствуют политике, 1 — нарушение.

Запуск из корня проекта:
    python -m src.data_gate --dataset raw
    python -m src.data_gate --dataset control
    python -m src.data_gate --dataset control --input КОПИЯ.tsv   # проверка другого файла по тем же правилам
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
import unicodedata
from datetime import UTC, datetime
from pathlib import Path

from src import config
from src.data import sha256_of
from src.predict import InputValidationError, validate_input

RULES_PATH = config.CONFIG_DIR / "data_gate.toml"


def load_rules(path: Path = RULES_PATH) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


def _check(results: list, name: str, ok: bool, detail: str) -> None:
    results.append({"check": name, "ok": bool(ok), "detail": detail})


def _parse_int(value: str) -> int | None:
    """Целое число в десятичной записи; NaN, Inf, дробные и пустые значения -> None."""
    try:
        return int(value)          # int("nan"), int("inf"), int("1.5"), int("") -> ValueError
    except ValueError:
        return None


def run_gate(path: Path, rules: dict, common: dict) -> list[dict]:
    results: list[dict] = []
    columns = list(rules["columns"])

    if not path.is_file():
        _check(results, "file_exists", False, f"файл {path} не найден")
        return results
    _check(results, "file_exists", True, str(path))

    expected_sha = str(rules.get("sha256", ""))
    actual_sha = sha256_of(path)
    if len(expected_sha) != 64:
        _check(results, "sha256", False, f"ожидаемый хэш не задан в правилах; фактический {actual_sha}")
    else:
        _check(results, "sha256", actual_sha == expected_sha,
               f"фактический {actual_sha}, ожидаемый {expected_sha}")

    if "data_id" in rules and "manifest" in rules:
        manifest = config.ROOT / rules["manifest"]
        got = json.loads(manifest.read_text(encoding="utf-8")).get("data_id") if manifest.exists() else None
        _check(results, "data_version", got == rules["data_id"],
               f"версия данных {got!r}, ожидается {rules['data_id']!r}")

    try:
        lines = path.read_text(encoding="utf-8").rstrip("\n").split("\n")  # не splitlines(): режет по \x85
    except UnicodeDecodeError as e:
        _check(results, "encoding_utf8", False, f"файл не является корректным UTF-8: {e}")
        return results
    _check(results, "encoding_utf8", True, "UTF-8")

    # Схема: обязательные поля и их порядок
    if rules.get("header", False):
        header = lines[0].split("\t") if lines else []
        missing = [c for c in columns if c not in header]
        _check(results, "schema_required_fields", header == columns,
               f"заголовок {header}, ожидается {columns}; отсутствуют {missing}")
        if header != columns:
            return results       # без корректной схемы остальные проверки не имеют смысла
        lines = lines[1:]

    # Форма: число полей в каждой строке
    rows, bad_shape = [], []
    for i, line in enumerate(lines, start=1):
        parts = line.split("\t")
        if len(parts) != len(columns):
            bad_shape.append(i)
        else:
            rows.append(dict(zip(columns, parts, strict=True)))
    _check(results, "shape_fields_per_row", not bad_shape,
           f"строк с числом полей != {len(columns)}: {len(bad_shape)} (первые: {bad_shape[:5]})")

    n = len(rows)
    _check(results, "min_rows", n >= rules["min_rows"], f"{n} строк, минимум {rules['min_rows']}")
    _check(results, "max_rows", n <= rules["max_rows"], f"{n} строк, максимум {rules['max_rows']}")

    # Пропуски: пустые значения обязательных полей
    missing_values = sum(1 for r in rows for c in columns if r[c].strip() == "")
    _check(results, "no_missing_values", missing_values == 0, f"пустых значений: {missing_values}")

    # Типы: row_id — конечное целое, уникальное (NaN/Inf отклоняются)
    if "row_id" in columns:
        ids = [_parse_int(r["row_id"]) for r in rows]
        bad = sum(1 for x in ids if x is None)
        _check(results, "type_row_id_int", bad == 0, f"нецелых/NaN/Inf row_id: {bad}")
        valid_ids = [x for x in ids if x is not None]
        _check(results, "row_id_unique", len(valid_ids) == len(set(valid_ids)),
               f"повторяющихся row_id: {len(valid_ids) - len(set(valid_ids))}")

    # Допустимые значения метки
    allowed = set(common["allowed_labels"])
    bad_labels = sorted({r["label"] for r in rows} - allowed)
    _check(results, "allowed_labels", not bad_labels, f"недопустимые метки: {bad_labels}")

    # Тексты: та же политика, что и для входа модели (ЛР2). Исключение — известные
    # артефакты кодировки источника (C1-символы из правил), и только в пределах
    # зафиксированного числа текстов: любой другой символ или рост числа — отказ.
    known = {chr(int(c.removeprefix("U+"), 16)) for c in common.get("known_artifact_chars", [])}
    max_known = rules.get("max_texts_with_known_artifacts", 0)
    violations: dict[str, int] = {}
    with_known = 0
    for r in rows:
        try:
            validate_input(r["text"])
        except InputValidationError as e:
            bad = {ch for ch in r["text"] if unicodedata.category(ch) in ("Cc", "Cs") and ch not in "\t\n\r"}
            if e.code == "bad_char" and bad and bad <= known:
                with_known += 1
            else:
                violations[e.code] = violations.get(e.code, 0) + 1
    _check(results, "text_policy_lr2", not violations, f"нарушения по кодам ЛР2: {violations}")
    _check(results, "known_encoding_artifacts", with_known <= max_known,
           f"текстов с известными C1-символами: {with_known}, допустимо {max_known}")

    # Распределение классов
    spam = sum(1 for r in rows if r["label"] == "spam")
    share = spam / n if n else 0.0
    lo, hi = rules["spam_share"]
    _check(results, "spam_share_bounds", lo <= share <= hi, f"доля spam {share:.4f}, допустимо [{lo}; {hi}]")

    # Дубликаты по ключу ЛР2 (нижний регистр + схлопнутые пробелы)
    keys = [" ".join(r["text"].lower().split()) for r in rows]
    dups = len(keys) - len(set(keys))
    limit = rules["max_normalized_duplicates"]
    _check(results, "normalized_duplicates", dups <= limit, f"дубликатов {dups}, максимум {limit}")
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Data quality gate")
    parser.add_argument("--dataset", required=True, help="имя набора в config/data_gate.toml")
    parser.add_argument("--input", type=Path, help="проверить другой файл по правилам набора")
    parser.add_argument("--rules", type=Path, default=RULES_PATH)
    args = parser.parse_args(argv)

    all_rules = load_rules(args.rules)
    rules = all_rules["datasets"][args.dataset]
    path = args.input or (config.ROOT / rules["path"])
    results = run_gate(path, rules, all_rules["common"])
    passed = all(r["ok"] for r in results)

    report = {"gate": "data", "dataset": args.dataset, "file": str(path),
                "checked_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                "status": "passed" if passed else "failed", "exit_code": 0 if passed else 1,
                "checks": results}
    config.REPORTS_DIR.mkdir(exist_ok=True)
    (config.REPORTS_DIR / f"data_gate_{args.dataset}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    for r in results:
        print(f"[{'OK ' if r['ok'] else 'FAIL'}] {r['check']}: {r['detail']}")
    print(f"DATA GATE ({args.dataset}): {'PASSED' if passed else 'FAILED'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

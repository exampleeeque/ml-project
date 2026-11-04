"""Фиксированная контрольная выборка для CI (ЛР3).

Контрольная выборка = отложенная тестовая выборка ЛР2 (1 032 сообщения).
Она воспроизводится детерминированно из проверенного сырого файла:
SHA-256 сырого файла -> удаление дубликатов -> train_test_split(seed 42,
stratify) -> test. Поэтому её SHA-256 постоянен и записан в
config/data_gate.toml, а метрика кандидата сравнима с baseline ЛР2
(та же версия данных, та же предобработка, то же определение метрики).

Формат data/ci/control.tsv: UTF-8, разделитель TAB, строка заголовка
"row_id<TAB>label<TAB>text", без кавычек и экранирования.

Запуск из корня проекта:
    python -m src.control_set                          # создать data/ci/control.tsv
    python -m src.control_set --demo-corrupt OUT.tsv   # ТОЛЬКО для демонстрации отказа data gate:
                                                       # испорченная КОПИЯ, исходные файлы не меняются
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src import config
from src.data import data_version, load_raw, prepare, sha256_of, verify_raw_data
from src.train import split

COLUMNS = ("row_id", "label", "text")


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    """Пишет TSV без кавычек. Табуляция и перевод строки в тексте недопустимы."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["\t".join(COLUMNS)]
    for row_id, label, text in zip(df["row_id"], df["label"], df["text"], strict=True):
        if "\t" in text or "\n" in text or "\r" in text:
            raise ValueError(f"row_id={row_id}: текст содержит TAB или перевод строки")
        lines.append(f"{int(row_id)}\t{label}\t{text}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_control(path: Path = config.CONTROL_PATH) -> dict:
    raw_sha = verify_raw_data()
    prepared = prepare(load_raw())
    _, test_df = split(prepared)
    test_df = test_df.sort_values("row_id")          # стабильный порядок строк
    write_tsv(test_df, path)
    info = {
        "path": path.relative_to(config.ROOT).as_posix(),
        "sha256": sha256_of(path),
        "rows": int(len(test_df)),
        "spam": int(test_df["y"].sum()),
        "data_id": data_version(raw_sha, prepared)["data_id"],
        "source": "UCI SMS Spam Collection (CC BY 4.0), тестовая выборка ЛР2",
    }
    return info


def make_corrupted_copy(src: Path, dst: Path) -> None:
    """Демонстрация (ЛР3, раздел 10): копия с нарушенной схемой и недопустимыми значениями.

    Исходный файл не изменяется. Нарушения: недопустимая метка, пустой текст,
    идентификатор NaN и строка с неверным числом полей.
    """
    lines = src.read_text(encoding="utf-8").rstrip("\n").split("\n")
    rid, _, text = lines[1].split("\t", 2)
    lines[1] = f"{rid}\tphishing\t{text}"          # недопустимое значение метки
    rid, label, _ = lines[2].split("\t", 2)
    lines[2] = f"{rid}\t{label}\t"                 # пропуск: пустой текст
    _, label, text = lines[3].split("\t", 2)
    lines[3] = f"nan\t{label}\t{text}"             # NaN вместо целого row_id
    lines.append("99999\tspam")                    # нарушение формы: 2 поля вместо 3
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Контрольная выборка CI")
    parser.add_argument("--demo-corrupt", metavar="OUT", type=Path,
                        help="создать испорченную копию контрольной выборки для демонстрации")
    args = parser.parse_args(argv)
    info = build_control()
    if args.demo_corrupt:
        make_corrupted_copy(config.CONTROL_PATH, args.demo_corrupt)
        print(f"ДЕМОНСТРАЦИЯ: создана испорченная копия {args.demo_corrupt} (исходные данные не изменены)")
    config.REPORTS_DIR.mkdir(exist_ok=True)
    (config.REPORTS_DIR / "control_set.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

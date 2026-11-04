"""Эксперимент с устойчивостью к небольшим изменениям текста и проверка нетипичных входов.

Берутся 20 объектов тестовой выборки (10 spam + 10 ham, seed 42) и четыре вида
изменений, сохраняющих смысл сообщения и его истинную метку:
  * case    — смена регистра (весь текст в верхнем регистре);
  * spaces  — лишние пробелы между словами и по краям;
  * typos   — 2 опечатки (перестановка соседних букв) в случайных словах;
  * leet    — замена букв цифрами/символами в 2 словах (FREE -> FR33), типичный приём спамеров;
  * leet_all — усиленный вариант leet во всех словах из 4+ букв (уже не «небольшое»
               изменение, а целенаправленная обфускация; показывает предел устойчивости).
Предсказания получаются через ту же функцию src.predict.predict.

Запуск из корня проекта (после python -m src.train):
    python -m src.robustness
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src import config
from src.data import load_raw, prepare, verify_raw_data
from src.predict import load_model, predict, preprocess
from src.train import split

N_PER_CLASS = 10
LEET = {"a": "@", "e": "3", "i": "1", "o": "0", "s": "5", "A": "@", "E": "3", "I": "1", "O": "0", "S": "5"}


def _words(text: str) -> list[str]:
    return text.split(" ")


def _pick(rng: np.random.Generator, words: list[str], k: int) -> list[int]:
    idx = [i for i, w in enumerate(words) if sum(c.isalpha() for c in w) >= 4]
    if not idx:
        return []
    return sorted(rng.choice(idx, size=min(k, len(idx)), replace=False).tolist())


def perturb_case(text: str, rng: np.random.Generator) -> str:
    return text.upper()


def perturb_spaces(text: str, rng: np.random.Generator) -> str:
    return "  " + "   ".join(text.split()) + "  "


def perturb_typos(text: str, rng: np.random.Generator, k: int = 2) -> str:
    words = _words(text)
    for i in _pick(rng, words, k):
        w = list(words[i])
        letters = [j for j in range(len(w) - 1) if w[j].isalpha() and w[j + 1].isalpha() and w[j] != w[j + 1]]
        if letters:
            j = int(rng.choice(letters))
            w[j], w[j + 1] = w[j + 1], w[j]
        words[i] = "".join(w)
    return " ".join(words)


def perturb_leet(text: str, rng: np.random.Generator, k: int = 2) -> str:
    words = _words(text)
    for i in _pick(rng, words, k):
        words[i] = "".join(LEET.get(c, c) for c in words[i])
    return " ".join(words)


def perturb_leet_all(text: str, rng: np.random.Generator) -> str:
    """Усиленный вариант: замена во всех словах из 4+ букв (целенаправленная обфускация)."""
    return perturb_leet(text, rng, k=10**6)


PERTURBATIONS = {"case": perturb_case, "spaces": perturb_spaces,
                 "typos": perturb_typos, "leet": perturb_leet,
                 "leet_all": perturb_leet_all}


def oov_share(model, texts: list[str]) -> list[float]:
    """Доля токенов, отсутствующих в словаре TF-IDF, обученном на train."""
    vec = model.named_steps["tfidf"]
    analyze = vec.build_analyzer()
    vocab = vec.vocabulary_
    out = []
    for t in preprocess(texts):
        tokens = [tok for tok in analyze(t) if " " not in tok]   # только униграммы
        out.append(1.0 if not tokens else sum(tok not in vocab for tok in tokens) / len(tokens))
    return out


ATYPICAL = {
    "ru_text": "Поздравляем! Вы выиграли приз 10000 рублей, позвоните по номеру 88001234567",
    "emoji_only": "\U0001F381\U0001F4B0\U0001F4F1\U0001F525\U0001F525",
    "url_only": "http://bit.ly/3xYz9Qw",
    "digits_only": "09061701461",
}


def main() -> int:
    verify_raw_data()
    model = load_model()
    df = prepare(load_raw())
    _, test_df = split(df)

    sample = pd.concat([
        test_df[test_df["y"] == 1].sample(N_PER_CLASS, random_state=config.SEED),
        test_df[test_df["y"] == 0].sample(N_PER_CLASS, random_state=config.SEED),
    ])
    texts, y = sample["text"].tolist(), sample["y"].to_numpy()

    def run(batch):
        res = predict(batch, model=model)
        return np.array([r["label"] == "spam" for r in res], int), np.array([r["spam_probability"] for r in res])

    y0, p0 = run(texts)
    summary = {"n": len(texts), "original": {"accuracy": round(accuracy_score(y, y0), 4),
                                             "f1_spam": round(f1_score(y, y0, zero_division=0), 4)}}
    rows = []
    for name, fn in PERTURBATIONS.items():
        rng = np.random.default_rng(config.SEED)
        changed_texts = [fn(t, rng) for t in texts]
        y1, p1 = run(changed_texts)
        flips = y0 != y1
        summary[name] = {
            "flip_rate": round(float(flips.mean()), 4),
            "flips_spam_to_ham": int(((y0 == 1) & (y1 == 0)).sum()),
            "flips_ham_to_spam": int(((y0 == 0) & (y1 == 1)).sum()),
            "accuracy": round(accuracy_score(y, y1), 4),
            "f1_spam": round(f1_score(y, y1, zero_division=0), 4),
            "mean_abs_delta_p": round(float(np.abs(p1 - p0).mean()), 4),
            "mean_p_spam_class_before": round(float(p0[y == 1].mean()), 4),
            "mean_p_spam_class_after": round(float(p1[y == 1].mean()), 4),
        }
        for i in range(len(texts)):
            rows.append({"perturbation": name, "row_id": int(sample["row_id"].iloc[i]),
                         "label": config.LABELS[y[i]], "p_before": p0[i], "p_after": p1[i],
                         "flip": bool(flips[i]), "original": texts[i], "changed": changed_texts[i]})

    # Нетипичные входы: сравнение доли незнакомых токенов с тестовой выборкой
    test_oov = np.array(oov_share(model, test_df["text"].tolist()))
    atypical = {}
    for name, text in ATYPICAL.items():
        res = predict(text, model=model)
        atypical[name] = {"text": text, **res, "oov_share": round(oov_share(model, [text])[0], 4)}
    summary["test_oov_share"] = {"mean": round(float(test_oov.mean()), 4),
                                 "p95": round(float(np.quantile(test_oov, 0.95)), 4),
                                 "max": round(float(test_oov.max()), 4),
                                 "share_of_test_with_oov_1.0": round(float((test_oov == 1.0).mean()), 4)}
    zero_vec = model.predict_proba([""])[:, 1][0]
    summary["empty_vector_spam_probability"] = round(float(zero_vec), 4)
    summary["atypical"] = atypical

    config.METRICS_DIR.mkdir(exist_ok=True)
    (config.METRICS_DIR / "robustness.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                                        encoding="utf-8")
    pd.DataFrame(rows).to_csv(config.METRICS_DIR / "robustness_details.csv", index=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

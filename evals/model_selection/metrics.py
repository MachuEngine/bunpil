"""지표 계산 — 신뢰구간·짝 비교·Judge 일치도·편향·Pareto·가중 점수 민감도.

신뢰구간은 **입력(클러스터) 단위 부트스트랩**이다. 같은 입력에서 나온 문항·반복은 독립이 아니므로
문항 단위로 뽑으면 구간이 실제보다 좁아진다.
"""
import itertools
import random
from collections import defaultdict

import numpy as np


def cluster_bootstrap_ci(values: list[float], clusters: list[str], n_boot: int = 1000, seed: int = 0) -> tuple:
    """(평균, 하한, 상한). values와 clusters는 같은 길이. None 값은 제외."""
    pairs = [(v, c) for v, c in zip(values, clusters) if v is not None]
    if not pairs:
        return (None, None, None)
    by = defaultdict(list)
    for v, c in pairs:
        by[c].append(float(v))
    keys = list(by)
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        sample = [x for k in (rng.choice(keys) for _ in keys) for x in by[k]]
        means.append(sum(sample) / len(sample))
    point = sum(v for v, _ in pairs) / len(pairs)
    return (point, float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def mcnemar(a: list[bool], b: list[bool]) -> dict:
    """같은 입력에 대한 두 모델의 성공/실패 짝 비교(정확 이항 검정)."""
    from scipy.stats import binomtest

    b01 = sum(1 for x, y in zip(a, b) if x and not y)
    b10 = sum(1 for x, y in zip(a, b) if y and not x)
    n = b01 + b10
    p = binomtest(b01, n, 0.5).pvalue if n else 1.0
    return {"a_only": b01, "b_only": b10, "p_value": p}


def holm(pvalues: dict) -> dict:
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m, adjusted, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[k] = running
    return adjusted


def percentile(values: list[float], q: float) -> float | None:
    vals = [v for v in values if v is not None]
    return float(np.percentile(vals, q)) if vals else None


# ── Judge 일치도 ─────────────────────────────────────────────────────

ORDINAL = {"no": 0, "unsure": 1, "yes": 2}


def weighted_kappa(a: list[str], b: list[str]) -> float | None:
    from sklearn.metrics import cohen_kappa_score

    pairs = [(ORDINAL[x], ORDINAL[y]) for x, y in zip(a, b) if x in ORDINAL and y in ORDINAL]
    if len(pairs) < 2 or len({p for pair in pairs for p in pair}) < 2:
        return None
    x, y = zip(*pairs)
    return float(cohen_kappa_score(x, y, weights="quadratic", labels=[0, 1, 2]))


def macro_f1(truth: list[bool], pred: list[bool]) -> float | None:
    from sklearn.metrics import f1_score

    if not truth:
        return None
    return float(f1_score(truth, pred, average="macro", labels=[True, False], zero_division=0))


def spearman(a: list[float], b: list[float]) -> float | None:
    from scipy.stats import spearmanr

    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    rho = spearmanr(*zip(*pairs)).statistic
    return None if rho != rho else float(rho)  # NaN(분산 0) → None


def recall(truth: list[bool], pred: list[bool]) -> float | None:
    """truth=True(치명적 오류 있음)인 항목 중 pred=True로 잡은 비율."""
    positives = [p for t, p in zip(truth, pred) if t]
    return sum(positives) / len(positives) if positives else None


def false_reject_rate(truth_pass: list[bool], pred_pass: list[bool]) -> float | None:
    normals = [p for t, p in zip(truth_pass, pred_pass) if t]
    return sum(1 for p in normals if not p) / len(normals) if normals else None


def position_consistency(ab: list[str], ba_swapped_back: list[str]) -> float | None:
    """A/B로 물은 결과와 B/A로 물어 되돌린 결과가 같은 비율."""
    pairs = list(zip(ab, ba_swapped_back))
    return sum(1 for x, y in pairs if x == y) / len(pairs) if pairs else None


def length_bias(len_a: list[int], len_b: list[int], judge: list[str], human: list[str]) -> float | None:
    """(Judge가 더 긴 쪽을 고른 비율) − (사람이 더 긴 쪽을 고른 비율). A/B 결정이 있는 항목만."""
    def longer_pick_rate(picks):
        rows = [(la, lb, w) for la, lb, w in zip(len_a, len_b, picks) if w in ("A", "B") and la != lb]
        if not rows:
            return None
        return sum(1 for la, lb, w in rows if (w == "A") == (la > lb)) / len(rows)

    j, h = longer_pick_rate(judge), longer_pick_rate(human)
    return None if j is None or h is None else j - h


# ── 선정 단계 ────────────────────────────────────────────────────────

def pareto_front(rows: list[dict], maximize: tuple, minimize: tuple) -> list[str]:
    """다른 후보에게 모든 축에서 같거나 뒤지고 한 축 이상 뒤지는(지배당하는) 후보를 뺀 나머지."""
    def dominates(x, y):
        ge = all(x[k] >= y[k] for k in maximize) and all(x[k] <= y[k] for k in minimize)
        gt = any(x[k] > y[k] for k in maximize) or any(x[k] < y[k] for k in minimize)
        return ge and gt

    return [r["model"] for r in rows if not any(dominates(o, r) for o in rows if o is not r)]


def weighted_scores(rows: list[dict], weights: dict) -> dict:
    """rows[i][criterion]은 0~1로 정규화된 값(높을수록 좋음). 반환 {model: score}."""
    total = sum(weights.values())
    return {r["model"]: sum(r[k] * w for k, w in weights.items()) / total for r in rows}


def sensitivity(rows: list[dict], weights: dict, delta: float = 0.2) -> dict:
    """가중치를 하나씩 ±delta 바꿔 1위가 바뀌는지 본다."""
    base = weighted_scores(rows, weights)
    base_winner = max(base, key=base.get)
    flips = []
    for k, sign in itertools.product(weights, (-1, 1)):
        w = dict(weights)
        w[k] = max(0.0, w[k] * (1 + sign * delta))
        scores = weighted_scores(rows, w)
        winner = max(scores, key=scores.get)
        if winner != base_winner:
            flips.append({"criterion": k, "change": f"{'+' if sign > 0 else '-'}{int(delta * 100)}%", "winner": winner})
    return {"base_winner": base_winner, "base_scores": base, "flips": flips, "stable": not flips}

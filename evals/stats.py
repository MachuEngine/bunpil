"""부트스트랩 신뢰구간(CI)·가중 κ — 8a3a543(모델 선정 v1, 롤백됨)의
`evals/model_selection/metrics.py`에서 2026-10 복원. 거기서 쓰던 mcnemar/holm/pareto 등
선정 전용 함수는 가져오지 않는다.

CI가 0을 포함하면 "판정 불가"(차이가 있다고 말할 근거 부족)로 취급하는 용도로 쓴다.
"""
from collections import defaultdict
from typing import Callable, Sequence

import numpy as np


def cluster_bootstrap_ci(
    rows: Sequence,
    clusters: Sequence,
    stat_fn: Callable[[list], float | None],
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float | None, float | None, float | None]:
    """클러스터 단위 부트스트랩 CI. (원 표본 통계량, 하한, 상한)을 반환한다.

    rows와 clusters는 같은 길이 — clusters[i]는 rows[i]가 속한 클러스터 id(예: 지문 id).
    같은 입력에서 나온 항목은 독립이 아니므로, 항목이 아니라 **클러스터**를 복원추출로
    뽑아 리샘플을 만든다(뽑힌 클러스터의 행을 전부 모음 — 같은 클러스터가 두 번 뽑히면
    그 행도 두 번 들어감). stat_fn은 행 리스트를 받아 통계량을 반환하며, 정의 불가(예:
    분산이 0인 κ)일 때 None을 반환할 수 있다 — 이런 리샘플은 퍼센타일 계산에서 제외한다.

    짝지은 비교(같은 지문에서 모델 A−B 차이)는 rows에 (a, b) 쌍을 넣고 stat_fn에서
    차이의 평균을 내도록 작성하면 된다 — 별도 함수 불필요.
    """
    point = stat_fn(list(rows))

    by_cluster = defaultdict(list)
    for row, c in zip(rows, clusters):
        by_cluster[c].append(row)
    keys = list(by_cluster)
    if not keys:
        return (point, None, None)

    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        picked = rng.choice(keys, size=len(keys), replace=True)
        sample = [row for k in picked for row in by_cluster[k]]
        s = stat_fn(sample)
        if s is not None:
            stats.append(s)

    if not stats:
        return (point, None, None)
    lo = float(np.percentile(stats, 100 * alpha / 2))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return (point, lo, hi)


def mean_ci(
    values: Sequence,
    clusters: Sequence,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float | None, float | None, float | None]:
    """cluster_bootstrap_ci의 편의 함수 — 평균의 CI. None 값은 제외."""
    def _mean(rows: list[float]) -> float | None:
        vals = [v for v in rows if v is not None]
        return sum(vals) / len(vals) if vals else None

    return cluster_bootstrap_ci(values, clusters, _mean, n_boot=n_boot, seed=seed, alpha=alpha)


# ── Judge 일치도 ─────────────────────────────────────────────────────

def weighted_kappa(
    a: Sequence[int],
    b: Sequence[int],
    labels: Sequence[int],
    weights: str = "quadratic",
) -> float | None:
    """순서형 점수(예: 1~5)용 가중 Cohen's κ. sklearn 미사용(numpy만 — CI 의존성 최소화).

    labels는 필수 — 척도 전체(예: [1,2,3,4,5])를 명시해야 한다. 관측값(a∪b)만으로
    추정하면 예를 들어 실제로는 {1,2,5}만 관측됐을 때 2와 5가 인접 라벨로 취급돼
    가중치가 왜곡되고, cluster_bootstrap_ci처럼 리샘플마다 관측 집합이 달라지는
    맥락에서는 리샘플별로 척도 자체가 바뀌어버린다. labels에 없는 값이 a나 b에
    있으면 조용히 건너뛰지 않고 ValueError를 낸다(척도 밖 값은 입력 오류).

    기대 불일치가 0이면(예: 모든 값이 동일) 정의할 수 없어 None을 반환한다.
    """
    pairs = list(zip(a, b))
    if len(pairs) < 2:
        return None

    n_labels = len(labels)
    if n_labels < 2:
        return None
    index = {label: i for i, label in enumerate(labels)}

    unknown = (set(a) | set(b)) - set(labels)
    if unknown:
        raise ValueError(f"labels에 없는 값이 입력에 있습니다: {sorted(unknown)!r}")

    # 가중치 행렬 W[i][j] — 라벨 i, j 사이의 불일치 정도(대각선은 0)
    idx = np.arange(n_labels)
    diff = np.abs(idx[:, None] - idx[None, :])
    if weights == "linear":
        w = diff
    else:  # quadratic(기본)
        w = diff ** 2

    # 관측(Po) · 기대(Pe) 분포 행렬
    o = np.zeros((n_labels, n_labels))
    for x, y in pairs:
        o[index[x], index[y]] += 1
    n = o.sum()
    if n == 0:
        return None
    o = o / n

    row_marg = o.sum(axis=1)
    col_marg = o.sum(axis=0)
    e = np.outer(row_marg, col_marg)

    observed_disagreement = float((w * o).sum())
    expected_disagreement = float((w * e).sum())
    if expected_disagreement == 0:
        return None
    return 1 - observed_disagreement / expected_disagreement
